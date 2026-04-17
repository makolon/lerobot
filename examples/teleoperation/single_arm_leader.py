import contextlib
import json
import socket
import time
from dataclasses import dataclass
from pathlib import Path

import lerobot
import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.teleoperators.so100_leader.so100_leader import SO100Leader
from lerobot.teleoperators.so100_leader.config_so100_leader import SO100LeaderConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.rotation import Rotation


@dataclass
class JointData:
    """Structure to hold received joint state data"""

    joint_positions: dict  # Joint name to position mapping
    timestamp: float


@dataclass
class PoseData:
    """Structure to hold pose data with position, orientation, and joint states"""

    position: np.ndarray  # [x, y, z]
    orientation: np.ndarray  # [qx, qy, qz, qw]
    timestamp: float


REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_ROOT = REPO_ROOT / "src" / "lerobot" / "assets"


def resolve_urdf_path(urdf_path: str) -> str:
    path = Path(urdf_path)
    if path.is_absolute():
        return str(path)
    repo_candidate = REPO_ROOT / path
    if repo_candidate.exists():
        return str(repo_candidate)
    assets_candidate = ASSETS_ROOT / path
    if assets_candidate.exists():
        return str(assets_candidate)
    if path.exists():
        return str(path)
    raise FileNotFoundError(f"URDF file not found: {urdf_path}")


class SingleArmLeader:
    """Sends single arm SO100 leader End Effector poses via socket at high frequency"""

    def __init__(
        self,
        port: str,
        socket_host: str = "localhost",
        socket_port: int = 12345,
        frequency: float = 100.0,
        calibration_dir: str = None,
    ):
        """
        Initialize the single arm leader

        Args:
            port: Serial port for SO100 leader arm
            socket_host: Host address for socket connection
            socket_port: Port for socket connection
            frequency: Transmission frequency in Hz
            calibration_dir: Directory for calibration files
        """
        self.frequency = frequency
        self.socket_host = socket_host
        self.socket_port = socket_port

        # Resolve URDF path with a packaged default when none is provided
        base_dir = Path(lerobot.__file__).resolve().parent
        resolved_urdf_path = base_dir / "assets" / "so101" / "robot.urdf"

        if not resolved_urdf_path.exists():
            raise FileNotFoundError(f"URDF file not found: {resolved_urdf_path}")

        # Initialize single arm leader configuration
        self.leader_config = SO100LeaderConfig(
            port=port, calibration_dir=calibration_dir, id="single_arm_leader"
        )

        # Initialize single arm leader
        self.leader = SO100Leader(self.leader_config)

        # Initialize kinematics solvers for the arm
        self.leader_joint_names = list(self.leader.bus.motors)
        print(f"Leader joint names: {self.leader_joint_names}")

        # Resolve URDF path to an existing file
        urdf_path_resolved = resolve_urdf_path(urdf_path)
        print(f"Loading URDF from: {urdf_path_resolved}")

        self.arm_kinematics = RobotKinematics(
            urdf_path=str(resolved_urdf_path),
            target_frame_name="gripper_frame_link",
            joint_names=self.leader_joint_names,
        )

        # Socket connection
        self.socket = None
        self.connected_clients = []

    def connect(self):
        """Connect to the teleoperator and setup socket server"""
        print("Connecting to single arm SO100 leader...")
        self.leader.connect()

        if not self.leader.is_connected:
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

    def get_joint_positions(self) -> JointData:
        """Get current joint positions from the arm"""
        action_dict = self.leader.get_action()

        # Extract joint positions for arm in correct order
        joint_positions = {}
        for motor_name in self.leader_joint_names:
            key = f"{motor_name}.pos"
            if key in action_dict:
                joint_positions[motor_name] = action_dict[key]
            else:
                print(f"Warning: Key '{key}' not found in action_dict")

        return JointData(joint_positions=joint_positions, timestamp=time.time())

    def compute_end_effector_poses(self, joint_data: JointData) -> PoseData | None:
        """Compute end effector poses from joint positions"""
        if self.arm_kinematics is None:
            return None

        # Convert joint_positions dict to array in correct order
        arm_joints = np.array([joint_data.joint_positions[name] for name in self.leader_joint_names])

        # Ensure joint arrays have correct size
        if arm_joints.size == 0:
            print("Warning: Empty joint arrays received")
            return None

        # Compute forward kinematics for arm
        ee_transform = self.arm_kinematics.forward_kinematics(arm_joints)

        # Extract position and orientation
        ee_pose = PoseData(
            position=ee_transform[:3, 3],
            orientation=Rotation.from_matrix(ee_transform[:3, :3]).as_quat(),
            timestamp=time.time(),
        )

        return ee_pose

    def create_leader_message(self, ee_data: PoseData, joint_data: JointData):
        """Create JSON message with arm pose and joint positions"""
        message = {
            "timestamp": time.time(),
            "end_effector": {
                "position": {
                    "px": float(ee_data.position[0]),
                    "py": float(ee_data.position[1]),
                    "pz": float(ee_data.position[2]),
                },
                "orientation": {
                    "qx": float(ee_data.orientation[0]),
                    "qy": float(ee_data.orientation[1]),
                    "qz": float(ee_data.orientation[2]),
                    "qw": float(ee_data.orientation[3]),
                },
            },
            "joints": {"arm": joint_data.joint_positions},
        }
        return json.dumps(message) + "\n"

    def send_to_clients(self, message):
        """Send message to all connected clients"""
        disconnected_clients = []

        for client in self.connected_clients:
            try:
                client.send(message.encode("utf-8"))
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
        print(f"Starting single arm leader loop at {self.frequency} Hz...")
        print("Waiting for client connections...")

        loop_duration = 1.0 / self.frequency

        while True:
            loop_start = time.perf_counter()

            try:
                # Accept new client connections
                self.accept_new_clients()

                # Get joint positions from arm
                joint_data = self.get_joint_positions()

                # Compute end effector poses
                ee_data = self.compute_end_effector_poses(joint_data)

                if ee_data is not None:
                    # Create and send message
                    message = self.create_leader_message(ee_data, joint_data)
                    if self.connected_clients:
                        self.send_to_clients(message)
            except KeyboardInterrupt:
                print("\nShutting down...")
                break
            except Exception as e:
                print(f"Error in main loop: {e}")

            # Maintain loop frequency
            elapsed = time.perf_counter() - loop_start
            precise_sleep(max(loop_duration - elapsed, 0.0))

    def disconnect(self):
        """Disconnect from devices and close socket"""
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
    """Main function to run the single arm leader"""

    # Configuration - Update this port according to your setup
    port = "/dev/tty.usbmodem5A7A0181491"
    port = "/dev/tty.usbmodem5AF71735431"

    # Socket configuration
    socket_host = "localhost"
    socket_port = 12345

    # Transmission frequency
    frequency = 100.0  # Hz

    try:
        # Initialize leader
        leader = SingleArmLeader(
            port=port,
            socket_host=socket_host,
            socket_port=socket_port,
            frequency=frequency,
            calibration_dir=None,
        )

        # Connect and start
        leader.connect()
        leader.run_leader_loop()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if "leader" in locals():
            leader.disconnect()


if __name__ == "__main__":
    main()
