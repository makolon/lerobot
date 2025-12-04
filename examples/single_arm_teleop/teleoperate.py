import contextlib
import json
import socket
import time
from dataclasses import dataclass

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.teleoperators.so100_leader.so100_leader import SO100Leader
from lerobot.teleoperators.so100_leader.config_so100_leader import SO100LeaderConfig
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.rotation import Rotation


@dataclass
class PoseData:
    """Structure to hold pose data with position, orientation, and gripper state"""
    position: np.ndarray  # [x, y, z]
    orientation: np.ndarray  # [qx, qy, qz, qw]
    gripper_command: float  # Gripper joint command (0 or 1 based on threshold)
    timestamp: float


class SingleArmTeleopSocketSender:
    """Sends single arm SO100 leader End Effector poses via socket at high frequency"""

    def __init__(
        self,
        arm_port: str,
        socket_host: str = "localhost",
        socket_port: int = 12345,
        urdf_path: str = "./SO101/so101_new_calib.urdf",
        frequency: float = 100.0,
        calibration_dir: str = None,
        gripper_threshold: float = 30.0,
    ):
        """
        Initialize the single arm teleop socket sender

        Args:
            arm_port: Serial port for SO100 leader arm
            socket_host: Host address for socket connection
            socket_port: Port for socket connection
            urdf_path: Path to the robot URDF file
            frequency: Transmission frequency in Hz
            calibration_dir: Directory for calibration files
            gripper_threshold: Threshold value for gripper command (above=1, below=0)
        """
        self.frequency = frequency
        self.socket_host = socket_host
        self.socket_port = socket_port
        self.gripper_threshold = gripper_threshold

        # Initialize single arm leader configuration
        self.teleop_config = SO100LeaderConfig(
            arm_port=arm_port,
            calibration_dir=calibration_dir,
            id="single_arm_leader"
        )

        # Initialize single arm leader
        self.teleop = SO100Leader(self.teleop_config)

        # Initialize kinematics solvers for the arm
        arm_joint_names = [name for name in self.teleop.arm.bus.motors if name != "gripper"]

        self.arm_kinematics = RobotKinematics(
            urdf_path=urdf_path,
            target_frame_name="gripper_frame_link",
            joint_names=arm_joint_names,
        )

        # Socket connection
        self.socket = None
        self.connected_clients = []

    def connect(self):
        """Connect to the teleoperator and setup socket server"""
        print("Connecting to single arm SO100 leader...")
        self.teleop.connect()

        if not self.teleop.is_connected:
            raise RuntimeError("Failed to connect to single arm leader arm")
        print("Connected to single arm SO100 leader")

        # Setup socket server
        self.setup_socket_server()

    def setup_socket_server(self):
        """Setup TCP socket server for pose transmission"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.socket_host, self.socket_port))
        self.socket.listen(5)
        self.socket.settimeout(0.001)  # Non-blocking with short timeout
        print(f"Socket server listening on {self.socket_host}:{self.socket_port}")

    def accept_new_clients(self):
        """Accept new client connections (non-blocking)"""
        try:
            client_socket, address = self.socket.accept()
            client_socket.settimeout(0.001)  # Non-blocking
            self.connected_clients.append(client_socket)
            print(f"New client connected from {address}")
        except TimeoutError:
            pass  # No new connections
        except Exception as e:
            print(f"Error accepting client: {e}")

    def get_joint_positions_and_gripper(self):
        """Get current joint positions and gripper commands from both arms"""
        action_dict = self.teleop.get_action()

        # Get joint names (excluding gripper) in the same order as kinematics
        arm_joint_names = [name for name in self.teleop.arm.bus.motors if name != "gripper"]

        # Extract joint positions for left arm in correct order
        arm_joints = []
        gripper = None
        for motor_name in arm_joint_names:
            key = f"arm_{motor_name}.pos"
            if key in action_dict:
                arm_joints.append(action_dict[key])

        # Get gripper
        gripper_key = "arm_gripper.pos"
        if gripper_key in action_dict:
            gripper = action_dict[gripper_key]

        return np.array(arm_joints), gripper

    def compute_end_effector_poses(self, arm_joints, gripper):
        """Compute end effector poses from joint positions and process gripper commands"""
        if self.arm_kinematics is None:
            return None

        try:
            # Ensure joint arrays have correct size
            if arm_joints.size == 0:
                print("Warning: Empty joint arrays received")
                return None, None

            # Compute forward kinematics for arm
            ee_transform = self.arm_kinematics.forward_kinematics(arm_joints)
        except Exception as e:
            print(f"Error in forward kinematics: {e}")
            print(f"Arm joints size: {arm_joints.size}")
            return None, None

        # Process gripper commands using threshold
        gripper_cmd = 1.0 if gripper is not None and gripper >= self.gripper_threshold else 0.0

        # Extract position and orientation
        ee_pose = PoseData(
            position=ee_transform[:3, 3],
            orientation=Rotation.from_matrix(ee_transform[:3, :3]).as_quat(),
            gripper_command=gripper_cmd,
            timestamp=time.time()
        )

        return ee_pose

    def create_pose_message(self, arm_pose):
        """Create JSON message with arm pose and gripper command"""
        message = {
            "timestamp": time.time(),
            "arm": {
                "position": {
                    "px": float(arm_pose.position[0]),
                    "py": float(arm_pose.position[1]),
                    "pz": float(arm_pose.position[2])
                },
                "orientation": {
                    "qx": float(arm_pose.orientation[0]),
                    "qy": float(arm_pose.orientation[1]),
                    "qz": float(arm_pose.orientation[2]),
                    "qw": float(arm_pose.orientation[3])
                },
                "gripper": float(arm_pose.gripper_command)
            },
        }
        return json.dumps(message) + "\n"

    def send_to_clients(self, message):
        """Send message to all connected clients"""
        disconnected_clients = []

        for client in self.connected_clients:
            try:
                client.send(message.encode('utf-8'))
            except (OSError, BrokenPipeError):
                disconnected_clients.append(client)

        # Remove disconnected clients
        for client in disconnected_clients:
            with contextlib.suppress(Exception):
                client.close()
            self.connected_clients.remove(client)
            print("Client disconnected")

    def run_teleop_loop(self):
        """Main teleoperation loop"""
        print(f"Starting single arm teleop loop at {self.frequency} Hz...")
        print("Waiting for client connections...")

        loop_duration = 1.0 / self.frequency

        while True:
            loop_start = time.perf_counter()

            try:
                # Accept new client connections
                self.accept_new_clients()

                # Get joint positions and gripper commands from arm
                arm_joints, gripper = self.get_joint_positions_and_gripper()

                # Debug: Print joint array sizes occasionally
                if int(time.time() * 10) % 50 == 0:  # Print every 5 seconds
                    print(f"Debug: Arm joints shape: {arm_joints.shape}")
                    print(f"Debug: Gripper: {gripper}")

                if arm_joints.size > 0:
                    # Compute end effector poses with gripper commands
                    arm_pose = self.compute_end_effector_poses(arm_joints, gripper)

                    if arm_pose is not None:
                        # Create and send message
                        message = self.create_pose_message(arm_pose)
                        if self.connected_clients:
                            self.send_to_clients(message)

                        # Debug output (reduce frequency for readability)
                        if int(time.time() * 2) % 2 == 0:  # Print every 0.5 seconds
                            print(f"Arm - Pos: {arm_pose.position}, Rot: {arm_pose.orientation}, Gripper: {arm_pose.gripper_command}")

            except KeyboardInterrupt:
                print("\nShutting down...")
                break
            except Exception as e:
                print(f"Error in main loop: {e}")

            # Maintain loop frequency
            elapsed = time.perf_counter() - loop_start
            busy_wait(max(loop_duration - elapsed, 0.0))

    def disconnect(self):
        """Disconnect from devices and close socket"""
        if self.teleop:
            self.teleop.disconnect()

        # Close all client connections
        for client in self.connected_clients:
            with contextlib.suppress(Exception):
                client.close()

        # Close server socket
        if self.socket:
            self.socket.close()

        print("Disconnected successfully")


def main():
    """Main function to run the single arm teleop socket sender"""

    # Configuration - Update this port according to your setup
    arm_port = "/dev/tty.usbmodem5A7A0178511"

    # Socket configuration
    socket_host = "localhost"
    socket_port = 12345

    # URDF path - Download from https://github.com/TheRobotStudio/SO-ARM100
    urdf_path = "./SO101/so101_new_calib.urdf"

    # Transmission frequency
    frequency = 100.0  # Hz

    # Gripper threshold (adjust based on your gripper's range)
    gripper_threshold = 50.0

    try:
        # Initialize single arm teleop socket sender
        sender = SO100LeaderConfig(
            arm_port=arm_port,
            socket_host=socket_host,
            socket_port=socket_port,
            urdf_path=urdf_path,
            frequency=frequency,
            gripper_threshold=gripper_threshold
        )
        # Connect and start
        sender.connect()
        sender.run_teleop_loop()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'sender' in locals():
            sender.disconnect()


if __name__ == "__main__":
    main()
