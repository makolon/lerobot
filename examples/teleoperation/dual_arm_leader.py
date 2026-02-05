import contextlib
import json
import socket
import time
from dataclasses import dataclass

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.teleoperators.bi_so100_leader.bi_so100_leader import BiSO100Leader
from lerobot.teleoperators.bi_so100_leader.config_bi_so100_leader import BiSO100LeaderConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.rotation import Rotation


@dataclass
class PoseData:
    """Structure to hold pose data with position, orientation, and gripper state"""
    position: np.ndarray  # [x, y, z]
    orientation: np.ndarray  # [qx, qy, qz, qw]
    gripper_command: float  # Gripper joint command (0 or 1 based on threshold)
    timestamp: float


class DualArmLeader:
    """Sends dual arm SO100 leader End Effector poses via socket at high frequency"""

    def __init__(
        self,
        left_arm_port: str,
        right_arm_port: str,
        socket_host: str = "localhost",
        socket_port: int = 12345,
        urdf_path: str = "./SO101/so101_new_calib.urdf",
        frequency: float = 100.0,
        calibration_dir: str = None,
        gripper_threshold: float = 30.0,
    ):
        """
        Initialize the dual arm leader

        Args:
            left_arm_port: Serial port for left SO100 leader arm
            right_arm_port: Serial port for right SO100 leader arm
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

        # Initialize dual arm leader configuration
        self.leader_config = BiSO100LeaderConfig(
            left_arm_port=left_arm_port,
            right_arm_port=right_arm_port,
            calibration_dir=calibration_dir,
            id="dual_arm_leader"
        )

        # Initialize dual arm leader
        self.leader = BiSO100Leader(self.leader_config)

        # Initialize kinematics solvers for both arms
        # Exclude gripper from joint names for kinematics
        self.left_joint_names = [name for name in self.leader.left_arm.bus.motors if name != "gripper"]
        self.right_joint_names = [name for name in self.leader.right_arm.bus.motors if name != "gripper"]
        print(f"Left leader joint names: {self.left_joint_names}")
        print(f"Right leader joint names: {self.right_joint_names}")

        self.left_kinematics = RobotKinematics(
            urdf_path=urdf_path,
            target_frame_name="gripper_frame_link",
            joint_names=self.left_joint_names,
        )

        self.right_kinematics = RobotKinematics(
            urdf_path=urdf_path,
            target_frame_name="gripper_frame_link",
            joint_names=self.right_joint_names,
        )

        # Socket connection
        self.socket = None
        self.connected_clients = []

    def connect(self):
        """Connect to the leader arms and setup socket server"""
        print("Connecting to dual arm SO100 leader...")
        self.leader.connect()

        if not self.leader.is_connected:
            raise RuntimeError("Failed to connect to dual arm leader arms")

        print("Connected to dual arm SO100 leader")

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
        action_dict = self.leader.get_action()

        # Extract joint positions for left arm in correct order
        left_joints = []
        left_gripper = None
        for motor_name in self.left_joint_names:
            key = f"left_{motor_name}.pos"
            if key in action_dict:
                left_joints.append(action_dict[key])

        # Get left gripper
        left_gripper_key = "left_gripper.pos"
        if left_gripper_key in action_dict:
            left_gripper = action_dict[left_gripper_key]

        # Extract joint positions for right arm in correct order
        right_joints = []
        right_gripper = None
        for motor_name in self.right_joint_names:
            key = f"right_{motor_name}.pos"
            if key in action_dict:
                right_joints.append(action_dict[key])

        # Get right gripper
        right_gripper_key = "right_gripper.pos"
        if right_gripper_key in action_dict:
            right_gripper = action_dict[right_gripper_key]

        return np.array(left_joints), np.array(right_joints), left_gripper, right_gripper

    def compute_end_effector_poses(self, left_joints, right_joints, left_gripper, right_gripper):
        """Compute end effector poses from joint positions and process gripper commands"""
        if self.left_kinematics is None or self.right_kinematics is None:
            return None, None

        try:
            # Ensure joint arrays have correct size
            if left_joints.size == 0 or right_joints.size == 0:
                print("Warning: Empty joint arrays received")
                return None, None

            # Compute forward kinematics for both arms
            left_transform = self.left_kinematics.forward_kinematics(left_joints)
            right_transform = self.right_kinematics.forward_kinematics(right_joints)
        except Exception as e:
            print(f"Error in forward kinematics: {e}")
            print(f"Left joints size: {left_joints.size}, Right joints size: {right_joints.size}")
            return None, None

        # Process gripper commands using threshold
        left_gripper_cmd = 1.0 if left_gripper is not None and left_gripper >= self.gripper_threshold else 0.0
        right_gripper_cmd = 1.0 if right_gripper is not None and right_gripper >= self.gripper_threshold else 0.0

        # Extract position and orientation
        left_pose = PoseData(
            position=left_transform[:3, 3],
            orientation=Rotation.from_matrix(left_transform[:3, :3]).as_quat(),
            gripper_command=left_gripper_cmd,
            timestamp=time.time()
        )

        right_pose = PoseData(
            position=right_transform[:3, 3],
            orientation=Rotation.from_matrix(right_transform[:3, :3]).as_quat(),
            gripper_command=right_gripper_cmd,
            timestamp=time.time()
        )

        return left_pose, right_pose

    def create_pose_message(self, left_pose, right_pose):
        """Create JSON message with both arm poses and gripper commands"""
        message = {
            "timestamp": time.time(),
            "left_arm": {
                "position": {
                    "px": float(left_pose.position[0]),
                    "py": float(left_pose.position[1]),
                    "pz": float(left_pose.position[2])
                },
                "orientation": {
                    "qx": float(left_pose.orientation[0]),
                    "qy": float(left_pose.orientation[1]),
                    "qz": float(left_pose.orientation[2]),
                    "qw": float(left_pose.orientation[3])
                },
                "gripper": float(left_pose.gripper_command)
            },
            "right_arm": {
                "position": {
                    "px": float(right_pose.position[0]),
                    "py": float(right_pose.position[1]),
                    "pz": float(right_pose.position[2])
                },
                "orientation": {
                    "qx": float(right_pose.orientation[0]),
                    "qy": float(right_pose.orientation[1]),
                    "qz": float(right_pose.orientation[2]),
                    "qw": float(right_pose.orientation[3])
                },
                "gripper": float(right_pose.gripper_command)
            }
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

    def run_leader_loop(self):
        """Main leader loop"""
        print(f"Starting dual arm leader loop at {self.frequency} Hz...")
        print("Waiting for client connections...")

        loop_duration = 1.0 / self.frequency

        while True:
            loop_start = time.perf_counter()

            try:
                # Accept new client connections
                self.accept_new_clients()

                # Get joint positions and gripper commands from both arms
                left_joints, right_joints, left_gripper, right_gripper = self.get_joint_positions_and_gripper()

                # Debug: Print joint array sizes occasionally
                if int(time.time() * 10) % 50 == 0:  # Print every 5 seconds
                    print(f"Debug: Left joints shape: {left_joints.shape}, Right joints shape: {right_joints.shape}")
                    print(f"Debug: Left gripper: {left_gripper}, Right gripper: {right_gripper}")

                if left_joints.size > 0 and right_joints.size > 0:
                    # Compute end effector poses with gripper commands
                    left_pose, right_pose = self.compute_end_effector_poses(left_joints, right_joints, left_gripper, right_gripper)

                    if left_pose is not None and right_pose is not None:
                        # Create and send message
                        message = self.create_pose_message(left_pose, right_pose)

                        if self.connected_clients:
                            self.send_to_clients(message)

                        # Debug output (reduce frequency for readability)
                        if int(time.time() * 2) % 2 == 0:  # Print every 0.5 seconds
                            print(f"Left - Pos: {left_pose.position}, Rot: {left_pose.orientation}, Gripper: {left_pose.gripper_command}")
                            print(f"Right - Pos: {right_pose.position}, Rot: {right_pose.orientation}, Gripper: {right_pose.gripper_command}")

            except KeyboardInterrupt:
                print("\nShutting down...")
                break
            except Exception as e:
                print(f"Error in main loop: {e}")

            # Maintain loop frequency
            elapsed = time.perf_counter() - loop_start
            precise_sleep(max(loop_duration - elapsed, 0.0))

    def disconnect(self):
        """Disconnect from leader arms and close socket"""
        if self.leader:
            self.leader.disconnect()

        # Close all client connections
        for client in self.connected_clients:
            with contextlib.suppress(Exception):
                client.close()

        # Close server socket
        if self.socket:
            self.socket.close()

        print("Disconnected successfully")


def main():
    """Main function to run the dual arm teleop socket sender"""

    # Configuration - Update these ports according to your setup
    left_arm_port = "/dev/tty.usbmodem5A7A0178511"  # Update with your left arm port
    right_arm_port = "/dev/tty.usbmodem5A7A0181491"  # Update with your right arm port

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
        # Initialize dual arm leader
        leader = DualArmLeader(
            left_arm_port=left_arm_port,
            right_arm_port=right_arm_port,
            socket_host=socket_host,
            socket_port=socket_port,
            urdf_path=urdf_path,
            frequency=frequency,
            gripper_threshold=gripper_threshold
        )
        # Connect and start
        leader.connect()
        leader.run_leader_loop()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'leader' in locals():
            leader.disconnect()


if __name__ == "__main__":
    main()
