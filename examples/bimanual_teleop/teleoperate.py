import contextlib
import json
import socket
import time
from dataclasses import dataclass

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.teleoperators.bi_so100_leader.bi_so100_leader import BiSO100Leader
from lerobot.teleoperators.bi_so100_leader.config_bi_so100_leader import BiSO100LeaderConfig
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.rotation import Rotation


@dataclass
class PoseData:
    """Structure to hold pose data with position and orientation"""
    position: np.ndarray  # [x, y, z]
    orientation: np.ndarray  # [wx, wy, wz] - rotation vector
    timestamp: float


class BimanualTeleopSocketSender:
    """Sends bimanual SO100 leader End Effector poses via socket at high frequency"""

    def __init__(
        self,
        left_arm_port: str,
        right_arm_port: str,
        socket_host: str = "localhost",
        socket_port: int = 12345,
        urdf_path: str = "./SO101/so101_new_calib.urdf",
        frequency: float = 100.0,
        calibration_dir: str = None,
    ):
        """
        Initialize the bimanual teleop socket sender

        Args:
            left_arm_port: Serial port for left SO100 leader arm
            right_arm_port: Serial port for right SO100 leader arm
            socket_host: Host address for socket connection
            socket_port: Port for socket connection
            urdf_path: Path to the robot URDF file
            frequency: Transmission frequency in Hz
            calibration_dir: Directory for calibration files
        """
        self.frequency = frequency
        self.socket_host = socket_host
        self.socket_port = socket_port

        # Initialize bimanual leader configuration
        self.teleop_config = BiSO100LeaderConfig(
            left_arm_port=left_arm_port,
            right_arm_port=right_arm_port,
            calibration_dir=calibration_dir,
            id="bimanual_leader"
        )

        # Initialize bimanual leader
        self.teleop = BiSO100Leader(self.teleop_config)

        # Initialize kinematics solvers for both arms
        self.left_kinematics = RobotKinematics(
            urdf_path=urdf_path,
            target_frame_name="gripper_frame_link",
            joint_names=list(self.teleop.left_arm.bus.motors.keys()),
        )

        self.right_kinematics = RobotKinematics(
            urdf_path=urdf_path,
            target_frame_name="gripper_frame_link",
            joint_names=list(self.teleop.right_arm.bus.motors.keys()),
        )

        # Socket connection
        self.socket = None
        self.connected_clients = []

    def connect(self):
        """Connect to the teleoperator and setup socket server"""
        print("Connecting to bimanual SO100 leader...")
        self.teleop.connect()

        if not self.teleop.is_connected:
            raise RuntimeError("Failed to connect to bimanual leader arms")

        print("Connected to bimanual SO100 leader")

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

    def get_joint_positions(self):
        """Get current joint positions from both arms"""
        action_dict = self.teleop.get_action()

        # Extract joint positions for left arm (remove 'left_' prefix and '.pos' suffix)
        left_joints = []
        for motor_name in self.teleop.left_arm.bus.motors:
            key = f"left_{motor_name}.pos"
            if key in action_dict:
                left_joints.append(action_dict[key])

        # Extract joint positions for right arm (remove 'right_' prefix and '.pos' suffix)
        right_joints = []
        for motor_name in self.teleop.right_arm.bus.motors:
            key = f"right_{motor_name}.pos"
            if key in action_dict:
                right_joints.append(action_dict[key])

        return np.array(left_joints), np.array(right_joints)

    def compute_end_effector_poses(self, left_joints, right_joints):
        """Compute end effector poses from joint positions"""
        if self.left_kinematics is None or self.right_kinematics is None:
            return None, None

        # Compute forward kinematics for both arms
        left_transform = self.left_kinematics.forward_kinematics(left_joints)
        right_transform = self.right_kinematics.forward_kinematics(right_joints)

        # Extract position and orientation
        left_pose = PoseData(
            position=left_transform[:3, 3],
            orientation=Rotation.from_matrix(left_transform[:3, :3]).as_rotvec(),
            timestamp=time.time()
        )

        right_pose = PoseData(
            position=right_transform[:3, 3],
            orientation=Rotation.from_matrix(right_transform[:3, :3]).as_rotvec(),
            timestamp=time.time()
        )

        return left_pose, right_pose

    def create_pose_message(self, left_pose, right_pose):
        """Create JSON message with both arm poses"""
        message = {
            "timestamp": time.time(),
            "left_arm": {
                "position": {
                    "x": float(left_pose.position[0]),
                    "y": float(left_pose.position[1]),
                    "z": float(left_pose.position[2])
                },
                "orientation": {
                    "wx": float(left_pose.orientation[0]),
                    "wy": float(left_pose.orientation[1]),
                    "wz": float(left_pose.orientation[2])
                }
            },
            "right_arm": {
                "position": {
                    "x": float(right_pose.position[0]),
                    "y": float(right_pose.position[1]),
                    "z": float(right_pose.position[2])
                },
                "orientation": {
                    "wx": float(right_pose.orientation[0]),
                    "wy": float(right_pose.orientation[1]),
                    "wz": float(right_pose.orientation[2])
                }
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

    def run_teleop_loop(self):
        """Main teleoperation loop"""
        print(f"Starting bimanual teleop loop at {self.frequency} Hz...")
        print("Waiting for client connections...")

        loop_duration = 1.0 / self.frequency

        while True:
            loop_start = time.perf_counter()

            try:
                # Accept new client connections
                self.accept_new_clients()

                # Get joint positions from both arms
                left_joints, right_joints = self.get_joint_positions()

                if left_joints is not None and right_joints is not None:
                    # Compute end effector poses
                    left_pose, right_pose = self.compute_end_effector_poses(left_joints, right_joints)

                    if left_pose is not None and right_pose is not None:
                        # Create and send message
                        message = self.create_pose_message(left_pose, right_pose)

                        if self.connected_clients:
                            self.send_to_clients(message)

                        # Debug output (reduce frequency for readability)
                        if int(time.time() * 10) % 10 == 0:  # Print every 0.1 seconds
                            print(f"Sent poses - Left: {left_pose.position}, Right: {right_pose.position}")

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
    """Main function to run the bimanual teleop socket sender"""

    # Configuration - Update these ports according to your setup
    left_arm_port = "/dev/tty.usbmodem585A0077581"  # Update with your left arm port
    right_arm_port = "/dev/tty.usbmodem585A0077582"  # Update with your right arm port

    # Socket configuration
    socket_host = "localhost"
    socket_port = 12345

    # URDF path - Download from https://github.com/TheRobotStudio/SO-ARM100
    urdf_path = "./SO101/so101_new_calib.urdf"

    # Transmission frequency
    frequency = 100.0  # Hz

    try:
        # Initialize bimanual teleop socket sender
        sender = BimanualTeleopSocketSender(
            left_arm_port=left_arm_port,
            right_arm_port=right_arm_port,
            socket_host=socket_host,
            socket_port=socket_port,
            urdf_path=urdf_path,
            frequency=frequency
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
