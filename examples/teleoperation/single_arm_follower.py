#!/usr/bin/env python
"""
Single Arm Follower - Receives joint positions from Leader via socket and controls follower arm.

This script connects to single_arm_teleop.py (Leader) and replicates the joint positions
on a follower SO100 robot arm.

Usage:
    1. First, start the Leader:
       python examples/teleoperation/single_arm_teleop.py

    2. Then, start this Follower:
       python examples/teleoperation/single_arm_follower.py
"""

import argparse
import json
import socket
import time
from dataclasses import dataclass

import numpy as np

from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.utils.robot_utils import precise_sleep


@dataclass
class ReceivedJointState:
    """Structure to hold received joint state data"""
    joint_positions: dict  # Joint name to position mapping
    timestamp: float
    receive_time: float


class SingleArmFollower:
    """Receives joint positions from Leader and controls follower arm"""

    def __init__(
        self,
        port: str,
        socket_host: str = "localhost",
        socket_port: int = 12345,
        frequency: float = 100.0,
        calibration_dir: str = None,
    ):
        """
        Initialize the single arm follower

        Args:
            port: Serial port for SO100 follower arm
            socket_host: Host address of the Leader socket server
            socket_port: Port of the Leader socket server
            frequency: Control frequency in Hz
            calibration_dir: Directory for calibration files
        """
        self.frequency = frequency
        self.socket_host = socket_host
        self.socket_port = socket_port

        # Initialize follower arm configuration
        self.follower_config = SO100FollowerConfig(
            port=port,
            calibration_dir=calibration_dir,
            id="single_arm_follower"
        )

        # Initialize follower arm
        self.follower = SO100Follower(self.follower_config)

        # Socket connection
        self.socket = None
        self.connected = False

        # Buffer for incomplete messages
        self.receive_buffer = ""

        # Latest received joint state
        self.latest_joint_state: ReceivedJointState | None = None

        # Statistics
        self.message_count = 0
        self.last_stats_time = time.time()

    def connect(self):
        """Connect to the follower arm and Leader socket server"""
        print("Connecting to SO100 follower arm...")
        self.follower.connect()

        if not self.follower.is_connected:
            raise RuntimeError("Failed to connect to follower arm")
        print("Connected to SO100 follower arm")

        # Get follower joint names for verification
        self.follower_joint_names = list(self.follower.bus.motors)
        print(f"Follower joint names: {self.follower_joint_names}")

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

    def receive_joint_state(self) -> ReceivedJointState | None:
        """Receive joint state from Leader (non-blocking)"""
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

                        # Extract joint positions from the "joints" field
                        if "joints" in pose_data and "arm" in pose_data["joints"]:
                            joint_positions = pose_data["joints"]["arm"]
                            latest_state = ReceivedJointState(
                                joint_positions=joint_positions,
                                timestamp=pose_data.get("timestamp", time.time()),
                                receive_time=receive_time
                            )
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {e}")

            return latest_state

        except socket.timeout:
            return None
        except Exception as e:
            print(f"Error receiving data: {e}")
            self.connected = False
            return None

    def apply_joint_positions(self, joint_state: ReceivedJointState):
        """Apply received joint positions to the follower arm"""
        if joint_state is None:
            return

        try:
            # Create action dict for follower
            action_dict = {}
            for joint_name, position in joint_state.joint_positions.items():
                # Map joint name to follower action key
                action_key = f"{joint_name}.pos"
                action_dict[action_key] = position

            # Send action to follower
            self.follower.send_action(action_dict)

        except Exception as e:
            print(f"Error applying joint positions: {e}")

    def print_statistics(self):
        """Print connection and performance statistics"""
        current_time = time.time()
        elapsed = current_time - self.last_stats_time

        if elapsed >= 5.0:  # Print every 5 seconds
            rate = self.message_count / elapsed
            print("\n=== Statistics ===")
            print(f"Messages received: {self.message_count} ({rate:.1f} Hz)")

            if self.latest_joint_state is not None:
                latency = (self.latest_joint_state.receive_time -
                          self.latest_joint_state.timestamp) * 1000
                print(f"Latency: {latency:.2f} ms")
                print(f"Latest joints: {self.latest_joint_state.joint_positions}")

            self.message_count = 0
            self.last_stats_time = current_time

    def run_follower_loop(self):
        """Main follower control loop"""
        print(f"Starting follower control loop at {self.frequency} Hz...")
        print("Receiving joint positions from Leader and applying to Follower...")

        loop_duration = 1.0 / self.frequency

        while True:
            loop_start = time.perf_counter()

            try:
                # Receive joint state from Leader
                joint_state = self.receive_joint_state()

                if joint_state is not None:
                    self.latest_joint_state = joint_state
                    # Apply joint positions to follower
                    self.apply_joint_positions(joint_state)

                # If we have a latest state, keep applying it (hold position)
                elif self.latest_joint_state is not None:
                    self.apply_joint_positions(self.latest_joint_state)

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
        """Disconnect from follower arm and socket"""
        if self.follower:
            self.follower.disconnect()

        if self.socket:
            self.socket.close()

        print("Disconnected successfully")


def main():
    """Main function to run the single arm follower"""

    # Configuration - Update this port according to your setup
    port = "/dev/tty.usbmodem5A7A0182351"

    # Socket configuration
    socket_host = "localhost"
    socket_port = 12345

    # URDF path - relative to lerobot/assets or absolute path
    urdf_path = "so101/so101_new_calib.urdf"

    # Transmission frequency
    frequency = 100.0  # Hz

    try:
        # Initialize follower
        follower = SingleArmFollower(
            port=port,
            socket_host=socket_host,
            socket_port=socket_port,
            frequency=frequency,
            calibration_dir=None,
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
