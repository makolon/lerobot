import argparse
import json
import time
from dataclasses import dataclass

import numpy as np
import zmq

from lerobot.model.kinematics import RobotKinematics
from lerobot.teleoperators.bi_so100_leader.bi_so100_leader import BiSO100Leader
from lerobot.teleoperators.bi_so100_leader.config_bi_so100_leader import BiSO100LeaderConfig
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.rotation import Rotation


@dataclass
class PoseData:
    """Structure to hold pose data with position, orientation, gripper state, and joint positions"""
    position: np.ndarray  # [x, y, z]
    orientation: np.ndarray  # [qx, qy, qz, qw] - quaternion
    joint_positions: np.ndarray  # Joint positions in degrees
    timestamp: float


class BimanualTeleopZeroMQSender:
    """Sends bimanual SO100 leader End Effector poses via ZeroMQ at high frequency"""

    def __init__(
        self,
        left_arm_port: str,
        right_arm_port: str,
        server_ip: str = "localhost",
        zmq_port: int = 5555,
        urdf_path: str = "./SO101/so101_new_calib.urdf",
        frequency: float = 100.0,
        calibration_dir: str = None,
        gripper_threshold: float = 30.0,
    ):
        """
        Initialize the bimanual teleop socket sender

        Args:
            left_arm_port: Serial port for left SO100 leader arm
            right_arm_port: Serial port for right SO100 leader arm
            server_ip: IP address of the server to send data to
            zmq_port: ZeroMQ port number
            urdf_path: Path to the robot URDF file
            frequency: Transmission frequency in Hz
            calibration_dir: Directory for calibration files
            gripper_threshold: Threshold value for gripper command (above=1, below=0)
        """
        self.frequency = frequency
        self.server_ip = server_ip
        self.zmq_port = zmq_port
        self.gripper_threshold = gripper_threshold

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
        # Exclude gripper from joint names for kinematics
        left_joint_names = [name for name in self.teleop.left_arm.bus.motors if name != "gripper"]
        right_joint_names = [name for name in self.teleop.right_arm.bus.motors if name != "gripper"]

        self.left_kinematics = RobotKinematics(
            urdf_path=urdf_path,
            target_frame_name="gripper_frame_link",
            joint_names=left_joint_names,
        )

        self.right_kinematics = RobotKinematics(
            urdf_path=urdf_path,
            target_frame_name="gripper_frame_link",
            joint_names=right_joint_names,
        )

        # ZeroMQ connection
        self.zmq_context = None
        self.zmq_socket = None

    def connect(self):
        """Connect to the teleoperator and setup ZeroMQ publisher"""
        print("Connecting to bimanual SO100 leader...")
        self.teleop.connect()

        if not self.teleop.is_connected:
            raise RuntimeError("Failed to connect to bimanual leader arms")

        print("Connected to bimanual SO100 leader")

        # Setup ZeroMQ communication
        self.setup_zeromq()

    def setup_zeromq(self):
        """Setup ZeroMQ publisher for pose transmission"""
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.PUB)

        # Connect to server
        zmq_address = f"tcp://{self.server_ip}:{self.zmq_port}"
        self.zmq_socket.connect(zmq_address)

        # Give ZeroMQ time to establish connection
        time.sleep(0.1)

        print(f"ZeroMQ publisher connected to {zmq_address}")

    def get_joint_positions(self):
        """Get current joint positions from both arms"""
        action_dict = self.teleop.get_action()

        # Get joint names in the same order as kinematics
        left_joint_names = self.teleop.left_arm.bus.motors
        right_joint_names = self.teleop.right_arm.bus.motors

        # Extract joint positions for left arm in correct order
        left_joints = []
        for motor_name in left_joint_names:
            key = f"left_{motor_name}.pos"
            if key in action_dict:
                left_joints.append(action_dict[key])

        # Extract joint positions for right arm in correct order
        right_joints = []
        for motor_name in right_joint_names:
            key = f"right_{motor_name}.pos"
            if key in action_dict:
                right_joints.append(action_dict[key])

        return np.array(left_joints), np.array(right_joints)

    def compute_end_effector_poses(self, left_joints, right_joints):
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

        # Extract position and orientation
        left_pose = PoseData(
            position=left_transform[:3, 3],
            orientation=Rotation.from_matrix(left_transform[:3, :3]).as_quat(),
            joint_positions=left_joints,
            timestamp=time.time()
        )

        right_pose = PoseData(
            position=right_transform[:3, 3],
            orientation=Rotation.from_matrix(right_transform[:3, :3]).as_quat(),
            joint_positions=right_joints,
            timestamp=time.time()
        )

        return left_pose, right_pose

    def create_pose_message(self, left_pose, right_pose):
        """Create JSON message with both arm poses, gripper commands, and joint positions"""
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
                "joint_positions": [float(jp) for jp in left_pose.joint_positions]
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
                "joint_positions": [float(jp) for jp in right_pose.joint_positions]
            }
        }
        return json.dumps(message) + "\n"

    def send_to_clients(self, message):
        """Send message via ZeroMQ publisher"""
        try:
            # Send with topic "pose" for filtering on subscriber side
            self.zmq_socket.send_string(f"pose {message}", zmq.NOBLOCK)
        except zmq.Again:
            # Queue is full, skip this message
            pass
        except Exception as e:
            print(f"Error sending via ZeroMQ: {e}")

    def run_teleop_loop(self):
        """Main teleoperation loop"""
        print(f"Starting bimanual teleop loop at {self.frequency} Hz using ZeroMQ...")

        loop_duration = 1.0 / self.frequency

        while True:
            loop_start = time.perf_counter()

            try:

                # Get joint positions from both arms
                left_joints, right_joints = self.get_joint_positions()

                # Debug: Print joint array sizes occasionally
                if int(time.time() * 10) % 50 == 0:  # Print every 5 seconds
                    print(f"Debug: Left joints shape: {left_joints.shape}, Right joints shape: {right_joints.shape}")

                if left_joints.size > 0 and right_joints.size > 0:
                    # Compute end effector poses with gripper commands
                    left_pose, right_pose = self.compute_end_effector_poses(left_joints, right_joints)

                    if left_pose is not None and right_pose is not None:
                        # Create and send message
                        message = self.create_pose_message(left_pose, right_pose)

                        # Send via ZeroMQ
                        self.send_to_clients(message)

                        # Debug output (reduce frequency for readability)
                        if int(time.time() * 2) % 2 == 0:  # Print every 0.5 seconds
                            print(f"Left - Pos: {left_pose.position}, Rot: {left_pose.orientation}")
                            print(f"Right - Pos: {right_pose.position}, Rot: {right_pose.orientation}")

            except KeyboardInterrupt:
                print("\nShutting down...")
                break
            except Exception as e:
                print(f"Error in main loop: {e}")

            # Maintain loop frequency
            elapsed = time.perf_counter() - loop_start
            busy_wait(max(loop_duration - elapsed, 0.0))

    def disconnect(self):
        """Disconnect from devices and close ZeroMQ"""
        if self.teleop:
            self.teleop.disconnect()

        # Close ZeroMQ
        if self.zmq_socket:
            self.zmq_socket.close()
        if self.zmq_context:
            self.zmq_context.term()

        print("Disconnected successfully")


def main():
    """Main function to run the bimanual teleop ZeroMQ sender"""

    parser = argparse.ArgumentParser(
        description="Bimanual SO100 Leader Teleoperation with ZeroMQ Transmission"
    )

    # Required arguments
    parser.add_argument(
        "--left-arm-port",
        type=str,
        required=True,
        help="Serial port for left SO100 leader arm (e.g., /dev/tty.usbmodem5A7A0178511)"
    )
    parser.add_argument(
        "--right-arm-port",
        type=str,
        required=True,
        help="Serial port for right SO100 leader arm (e.g., /dev/tty.usbmodem5A7A0181491)"
    )
    parser.add_argument(
        "--server-ip",
        type=str,
        required=True,
        help="IP address of the server to send data to (e.g., 192.168.1.100 or localhost)"
    )

    # Optional arguments
    parser.add_argument(
        "--zmq-port",
        type=int,
        default=5555,
        help="ZeroMQ port number (default: 5555)"
    )
    parser.add_argument(
        "--urdf-path",
        type=str,
        default="./SO101/so101_new_calib.urdf",
        help="Path to the robot URDF file (default: ./SO101/so101_new_calib.urdf)"
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=100.0,
        help="Transmission frequency in Hz (default: 100.0)"
    )
    parser.add_argument(
        "--gripper-threshold",
        type=float,
        default=50.0,
        help="Threshold value for gripper command (default: 50.0)"
    )
    parser.add_argument(
        "--calibration-dir",
        type=str,
        default=None,
        help="Directory for calibration files (optional)"
    )

    args = parser.parse_args()

    try:
        # Initialize bimanual teleop ZeroMQ sender
        sender = BimanualTeleopZeroMQSender(
            left_arm_port=args.left_arm_port,
            right_arm_port=args.right_arm_port,
            server_ip=args.server_ip,
            zmq_port=args.zmq_port,
            urdf_path=args.urdf_path,
            frequency=args.frequency,
            calibration_dir=args.calibration_dir,
            gripper_threshold=args.gripper_threshold
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
