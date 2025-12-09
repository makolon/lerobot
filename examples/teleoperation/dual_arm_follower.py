import json
import socket
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.bi_so100_follower.bi_so100_follower import BiSO100Follower
from lerobot.robots.bi_so100_follower.config_bi_so100_follower import BiSO100FollowerConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.rotation import Rotation


@dataclass
class PoseData:
    """Structure to hold pose data with position, orientation, and gripper state"""
    position: np.ndarray  # [x, y, z]
    orientation: np.ndarray  # [qx, qy, qz, qw]
    gripper: float  # Gripper command (0 or 1)
    timestamp: float


@dataclass
class DualArmState:
    """Structure to hold received dual arm state data"""
    left_position: np.ndarray
    left_orientation: np.ndarray
    left_gripper: float
    right_position: np.ndarray
    right_orientation: np.ndarray
    right_gripper: float
    timestamp: float
    receive_time: float


class DualArmFollower:
    """Receives poses from Leader and controls dual arm follower"""

    def __init__(
        self,
        left_arm_port: str,
        right_arm_port: str,
        socket_host: str = "localhost",
        socket_port: int = 12345,
        urdf_path: str = "./SO101/so101_new_calib.urdf",
        frequency: float = 100.0,
        calibration_dir: str = None,
        gripper_threshold: float = 0.5,
    ):
        """
        Initialize the dual arm follower

        Args:
            left_arm_port: Serial port for left SO100 follower arm
            right_arm_port: Serial port for right SO100 follower arm
            socket_host: Host address of the Leader socket server
            socket_port: Port of the Leader socket server
            urdf_path: Path to the robot URDF file
            frequency: Control frequency in Hz
            calibration_dir: Directory for calibration files
            gripper_threshold: Threshold for gripper open/close (0-1)
        """
        self.frequency = frequency
        self.socket_host = socket_host
        self.socket_port = socket_port
        self.gripper_threshold = gripper_threshold

        # Initialize dual arm follower configuration
        self.follower_config = BiSO100FollowerConfig(
            left_arm_port=left_arm_port,
            right_arm_port=right_arm_port,
            calibration_dir=calibration_dir,
            id="dual_arm_follower"
        )

        # Initialize dual arm follower
        self.follower = BiSO100Follower(self.follower_config)

        # Get joint names (excluding gripper) for kinematics
        self.left_joint_names = [name for name in self.follower.left_arm.bus.motors if name != "gripper"]
        self.right_joint_names = [name for name in self.follower.right_arm.bus.motors if name != "gripper"]
        print(f"Left follower joint names: {self.left_joint_names}")
        print(f"Right follower joint names: {self.right_joint_names}")

        # Resolve URDF path
        urdf_path_obj = Path(urdf_path)
        urdf_path_resolved = str(urdf_path_obj)
        print(f"Loading URDF from: {urdf_path_resolved}")

        # Initialize kinematics solvers for both arms
        self.left_kinematics = RobotKinematics(
            urdf_path=urdf_path_resolved,
            target_frame_name="gripper_frame_link",
            joint_names=self.left_joint_names,
        )

        self.right_kinematics = RobotKinematics(
            urdf_path=urdf_path_resolved,
            target_frame_name="gripper_frame_link",
            joint_names=self.right_joint_names,
        )

        # Socket connection
        self.socket = None
        self.connected = False

        # Buffer for incomplete messages
        self.receive_buffer = ""

        # Latest received state and computed poses
        self.latest_state: DualArmState | None = None
        self.latest_left_pose: PoseData | None = None
        self.latest_right_pose: PoseData | None = None

        # Statistics
        self.message_count = 0
        self.last_stats_time = time.time()

    def connect(self):
        """Connect to the follower arms and Leader socket server"""
        print("Connecting to dual arm SO100 follower...")
        self.follower.connect()

        if not self.follower.is_connected:
            raise RuntimeError("Failed to connect to dual arm follower")
        print("Connected to dual arm SO100 follower")

        # Connect to Leader socket server
        self.connect_to_leader()

    def connect_to_leader(self):
        """Connect to the Leader socket server"""
        print(f"Connecting to Leader at {self.socket_host}:{self.socket_port}...")

        max_retries = 10
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((self.socket_host, self.socket_port))
                self.socket.settimeout(0.1)  # 100ms timeout for non-blocking receive
                self.connected = True
                print("Connected to Leader successfully!")
                return
            except ConnectionRefusedError:
                print(f"Connection attempt {attempt + 1}/{max_retries} failed. "
                      f"Make sure Leader is running. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            except Exception as e:
                print(f"Connection error: {e}")
                time.sleep(retry_delay)

        raise RuntimeError(f"Failed to connect to Leader after {max_retries} attempts")

    def receive_state(self) -> DualArmState | None:
        """Receive dual arm state from Leader (non-blocking)"""
        if not self.connected:
            return None

        try:
            # Receive data
            data = self.socket.recv(4096).decode('utf-8')
            if not data:
                print("Leader disconnected")
                self.connected = False
                return None

            self.receive_buffer += data
            receive_time = time.time()

            # Process complete messages (newline-delimited JSON)
            messages = self.receive_buffer.split('\n')
            self.receive_buffer = messages[-1]  # Keep incomplete message

            # Process the latest complete message
            latest_state = None
            for msg in messages[:-1]:
                if msg.strip():
                    try:
                        pose_data = json.loads(msg)
                        self.message_count += 1

                        # Extract left arm data
                        left_arm = pose_data.get("left_arm", {})
                        left_pos = left_arm.get("position", {})
                        left_orient = left_arm.get("orientation", {})

                        # Extract right arm data
                        right_arm = pose_data.get("right_arm", {})
                        right_pos = right_arm.get("position", {})
                        right_orient = right_arm.get("orientation", {})

                        latest_state = DualArmState(
                            left_position=np.array([
                                left_pos.get("px", 0),
                                left_pos.get("py", 0),
                                left_pos.get("pz", 0)
                            ]),
                            left_orientation=np.array([
                                left_orient.get("qx", 0),
                                left_orient.get("qy", 0),
                                left_orient.get("qz", 0),
                                left_orient.get("qw", 1)
                            ]),
                            left_gripper=left_arm.get("gripper", 0.0),
                            right_position=np.array([
                                right_pos.get("px", 0),
                                right_pos.get("py", 0),
                                right_pos.get("pz", 0)
                            ]),
                            right_orientation=np.array([
                                right_orient.get("qx", 0),
                                right_orient.get("qy", 0),
                                right_orient.get("qz", 0),
                                right_orient.get("qw", 1)
                            ]),
                            right_gripper=right_arm.get("gripper", 0.0),
                            timestamp=pose_data.get("timestamp", time.time()),
                            receive_time=receive_time
                        )
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {e}")

            return latest_state

        except TimeoutError:
            return None
        except Exception as e:
            print(f"Error receiving data: {e}")
            self.connected = False
            return None

    def compute_inverse_kinematics(self, position: np.ndarray, orientation: np.ndarray,
                                    kinematics: RobotKinematics) -> np.ndarray | None:
        """Compute inverse kinematics to get joint positions from end effector pose"""
        try:
            # Create target transformation matrix
            rotation_matrix = Rotation.from_quat(orientation).as_matrix()
            target_transform = np.eye(4)
            target_transform[:3, :3] = rotation_matrix
            target_transform[:3, 3] = position

            # Compute inverse kinematics
            joint_positions = kinematics.inverse_kinematics(target_transform)
            return joint_positions

        except Exception as e:
            print(f"Error in inverse kinematics: {e}")
            return None

    def apply_state(self, state: DualArmState):
        """Apply received state to the follower arms using inverse kinematics"""
        if state is None:
            return

        try:
            # Compute inverse kinematics for left arm
            left_joints = self.compute_inverse_kinematics(
                state.left_position,
                state.left_orientation,
                self.left_kinematics
            )

            # Compute inverse kinematics for right arm
            right_joints = self.compute_inverse_kinematics(
                state.right_position,
                state.right_orientation,
                self.right_kinematics
            )

            if left_joints is None or right_joints is None:
                print("Warning: Failed to compute inverse kinematics")
                return

            # Create action dict for follower
            action_dict = {}

            # Add left arm joint positions
            for i, joint_name in enumerate(self.left_joint_names):
                action_key = f"left_{joint_name}.pos"
                action_dict[action_key] = float(left_joints[i])

            # Add left gripper
            left_gripper_value = 1.0 if state.left_gripper >= self.gripper_threshold else 0.0
            action_dict["left_gripper.pos"] = left_gripper_value * 100.0  # Scale to gripper range

            # Add right arm joint positions
            for i, joint_name in enumerate(self.right_joint_names):
                action_key = f"right_{joint_name}.pos"
                action_dict[action_key] = float(right_joints[i])

            # Add right gripper
            right_gripper_value = 1.0 if state.right_gripper >= self.gripper_threshold else 0.0
            action_dict["right_gripper.pos"] = right_gripper_value * 100.0  # Scale to gripper range

            # Send action to follower
            self.follower.send_action(action_dict)

        except Exception as e:
            print(f"Error applying state: {e}")

    def compute_end_effector_poses(self) -> tuple[PoseData | None, PoseData | None]:
        """Compute current end effector poses from follower arm positions"""
        try:
            # Get current joint positions from follower
            state = self.follower.get_observation()

            # Extract left arm joints
            left_joints = []
            for joint_name in self.left_joint_names:
                key = f"left_{joint_name}.pos"
                if key in state:
                    left_joints.append(state[key])
            left_joints = np.array(left_joints)

            # Extract right arm joints
            right_joints = []
            for joint_name in self.right_joint_names:
                key = f"right_{joint_name}.pos"
                if key in state:
                    right_joints.append(state[key])
            right_joints = np.array(right_joints)

            # Compute forward kinematics for left arm
            left_pose = None
            if left_joints.size > 0:
                left_transform = self.left_kinematics.forward_kinematics(left_joints)
                left_pose = PoseData(
                    position=left_transform[:3, 3],
                    orientation=Rotation.from_matrix(left_transform[:3, :3]).as_quat(),
                    gripper=state.get("left_gripper.pos", 0.0),
                    timestamp=time.time()
                )

            # Compute forward kinematics for right arm
            right_pose = None
            if right_joints.size > 0:
                right_transform = self.right_kinematics.forward_kinematics(right_joints)
                right_pose = PoseData(
                    position=right_transform[:3, 3],
                    orientation=Rotation.from_matrix(right_transform[:3, :3]).as_quat(),
                    gripper=state.get("right_gripper.pos", 0.0),
                    timestamp=time.time()
                )

            return left_pose, right_pose

        except Exception as e:
            print(f"Error computing end effector poses: {e}")
            return None, None

    def print_statistics(self):
        """Print connection and performance statistics"""
        current_time = time.time()
        elapsed = current_time - self.last_stats_time

        if elapsed >= 5.0:  # Print every 5 seconds
            rate = self.message_count / elapsed
            print("\n=== Statistics ===")
            print(f"Messages received: {self.message_count} ({rate:.1f} Hz)")

            if self.latest_state is not None:
                latency = (self.latest_state.receive_time - self.latest_state.timestamp) * 1000
                print(f"Latency: {latency:.2f} ms")
                print(f"Left gripper: {self.latest_state.left_gripper}")
                print(f"Right gripper: {self.latest_state.right_gripper}")

            self.message_count = 0
            self.last_stats_time = current_time

    def run_follower_loop(self):
        """Main follower control loop"""
        print(f"Starting dual arm follower control loop at {self.frequency} Hz...")
        print("Receiving poses from Leader and applying to Follower...")

        loop_duration = 1.0 / self.frequency

        while True:
            loop_start = time.perf_counter()

            try:
                # Receive state from Leader
                state = self.receive_state()

                if state is not None:
                    self.latest_state = state
                    # Apply state to follower using inverse kinematics
                    self.apply_state(state)
                    # Update current poses
                    self.latest_left_pose, self.latest_right_pose = self.compute_end_effector_poses()

                # Print statistics periodically
                self.print_statistics()

                # Check connection status
                if not self.connected:
                    print("Lost connection to Leader. Attempting to reconnect...")
                    try:
                        self.connect_to_leader()
                    except RuntimeError:
                        print("Reconnection failed. Exiting...")
                        break

            except KeyboardInterrupt:
                print("\nShutting down...")
                break
            except Exception as e:
                print(f"Error in follower loop: {e}")

            # Maintain loop frequency
            elapsed = time.perf_counter() - loop_start
            precise_sleep(max(loop_duration - elapsed, 0.0))

    def disconnect(self):
        """Disconnect from follower arms and socket"""
        if self.follower:
            self.follower.disconnect()

        if self.socket:
            self.socket.close()

        print("Disconnected successfully")


def main():
    """Main function to run the dual arm follower"""

    # Configuration - Update these ports according to your setup
    left_arm_port = "/dev/tty.usbmodem5A7A0178512"  # Update with your left follower port
    right_arm_port = "/dev/tty.usbmodem5A7A0181492"  # Update with your right follower port

    # Socket configuration
    socket_host = "localhost"
    socket_port = 12345

    # URDF path
    urdf_path = "./SO101/so101_new_calib.urdf"

    # Control frequency
    frequency = 100.0  # Hz

    # Gripper threshold
    gripper_threshold = 0.5

    try:
        # Initialize follower
        follower = DualArmFollower(
            left_arm_port=left_arm_port,
            right_arm_port=right_arm_port,
            socket_host=socket_host,
            socket_port=socket_port,
            urdf_path=urdf_path,
            frequency=frequency,
            calibration_dir=None,
            gripper_threshold=gripper_threshold,
        )

        # Connect and start
        follower.connect()
        follower.run_follower_loop()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'follower' in locals():
            follower.disconnect()


if __name__ == "__main__":
    main()
