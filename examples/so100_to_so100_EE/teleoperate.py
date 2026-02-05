# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import socket
import time
from pathlib import Path

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    robot_action_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    ForwardKinematicsJointsToEE,
    InverseKinematicsEEToJoints,
)
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.teleoperators.so100_leader.config_so100_leader import SO100LeaderConfig
from lerobot.teleoperators.so100_leader.so100_leader import SO100Leader
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FPS = 30
LEROBOT_ROOT = Path(__file__).parent.parent.parent / "src" / "lerobot"
SIM_JAW_LIMITS = (-0.174, 1.75)


def resolve_urdf_path(urdf_path: str) -> str:
    path = Path(urdf_path)
    if path.is_absolute():
        return str(path)
    candidate = LEROBOT_ROOT / "assets" / urdf_path
    if candidate.exists():
        return str(candidate)
    if path.exists():
        return str(path)
    raise FileNotFoundError(f"URDF file not found: {urdf_path}")


class SimSocketSender:
    """Stream joint positions to a simulator over a socket."""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.socket: socket.socket | None = None

    def connect(self, retries: int, wait_s: float) -> None:
        last_err: Exception | None = None
        for attempt in range(retries + 1):
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((self.host, self.port))
                return
            except ConnectionRefusedError as exc:
                last_err = exc
                if attempt < retries:
                    time.sleep(wait_s)
                    continue
                raise RuntimeError(
                    f"Simulator socket not accepting connections at {self.host}:{self.port}. "
                    "Start the simulator socket server first."
                ) from exc
        if last_err:
            raise last_err

    def send(self, payload: dict) -> None:
        if self.socket is None:
            return
        message = json.dumps(payload) + "\n"
        self.socket.sendall(message.encode("utf-8"))

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close()
            self.socket = None


def gripper_to_jaw(gripper_pos: float) -> float:
    g = max(0.0, min(100.0, gripper_pos))
    j_min, j_max = SIM_JAW_LIMITS
    return j_min + (g / 100.0) * (j_max - j_min)


def main():
    parser = argparse.ArgumentParser(description="SO100 leader to follower teleoperation")
    parser.add_argument("--sim-socket", action="store_true", help="Send joint positions to a simulator over socket")
    parser.add_argument("--sim-socket-host", default="127.0.0.1", help="Simulator socket host")
    parser.add_argument("--sim-socket-port", type=int, default=12345, help="Simulator socket port")
    parser.add_argument("--sim-socket-retries", type=int, default=10, help="Simulator socket connect retries")
    parser.add_argument("--sim-socket-wait", type=float, default=0.5, help="Seconds to wait between retries")
    parser.add_argument("--log-path", type=str, default="teleop_log.jsonl", help="Write JSONL logs to this path")
    args = parser.parse_args()

    # Initialize the robot and teleoperator config
    follower_config = SO100FollowerConfig(
        port="/dev/tty.usbmodem5A7A0182351", id="my_awesome_follower_arm", use_degrees=True
    )
    leader_config = SO100LeaderConfig(port="/dev/tty.usbmodem5A7A0181491", id="my_awesome_leader_arm")

    # Initialize the robot and teleoperator
    follower = SO100Follower(follower_config)
    leader = SO100Leader(leader_config)

    # NOTE: It is highly recommended to use the urdf in the SO-ARM100 repo: https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
    urdf_path = resolve_urdf_path("so101/so101_new_calib.urdf")

    follower_kinematics_solver = RobotKinematics(
        urdf_path=urdf_path,
        target_frame_name="gripper_frame_link",
        joint_names=list(follower.bus.motors.keys()),
    )

    # NOTE: It is highly recommended to use the urdf in the SO-ARM100 repo: https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
    leader_kinematics_solver = RobotKinematics(
        urdf_path=urdf_path,
        target_frame_name="gripper_frame_link",
        joint_names=list(leader.bus.motors.keys()),
    )

    # Build pipeline to convert teleop joints to EE action
    leader_to_ee = RobotProcessorPipeline[RobotAction, RobotAction](
        steps=[
            ForwardKinematicsJointsToEE(
                kinematics=leader_kinematics_solver, motor_names=list(leader.bus.motors.keys())
            ),
        ],
        to_transition=robot_action_to_transition,
        to_output=transition_to_robot_action,
    )

    # build pipeline to convert EE action to robot joints
    ee_to_follower_joints = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        [
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
                max_ee_step_m=0.10,
            ),
            InverseKinematicsEEToJoints(
                kinematics=follower_kinematics_solver,
                motor_names=list(follower.bus.motors.keys()),
                initial_guess_current_joints=False,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # Connect to the robot and teleoperator
    follower.connect()
    leader.connect()

    # Init rerun viewer
    init_rerun(session_name="so100_so100_EE_teleop")

    sim_sender = None
    if args.sim_socket:
        sim_sender = SimSocketSender(args.sim_socket_host, args.sim_socket_port)
        sim_sender.connect(args.sim_socket_retries, args.sim_socket_wait)

    log_path = args.log_path
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    log_file = open(log_path, "a", encoding="utf-8")

    print("Starting teleop loop...")
    try:
        while True:
            t0 = time.perf_counter()

            # Get robot observation
            robot_obs = follower.get_observation()

            # Get teleop observation
            leader_joints_obs = leader.get_action()

            # teleop joints -> teleop EE action
            leader_ee_act = leader_to_ee(leader_joints_obs)
            print(f"Leader EE Action: {leader_ee_act}")

            # teleop EE -> robot joints
            follower_joints_act = ee_to_follower_joints((leader_ee_act, robot_obs))

            # Send action to robot
            _ = follower.send_action(follower_joints_act)

            log_payload = {
                "timestamp": time.time(),
                "leader_joints": leader_joints_obs,
                "leader_ee": leader_ee_act,
                "follower_obs": robot_obs,
                "follower_joints_cmd": follower_joints_act,
            }
            log_file.write(json.dumps(log_payload) + "\n")
            log_file.flush()

            if sim_sender is not None:
                sim_joint_positions = [
                    float(follower_joints_act["shoulder_pan.pos"]) * 0.017453292519943295,
                    float(follower_joints_act["shoulder_lift.pos"]) * 0.017453292519943295,
                    float(follower_joints_act["elbow_flex.pos"]) * 0.017453292519943295,
                    float(follower_joints_act["wrist_flex.pos"]) * 0.017453292519943295,
                    float(follower_joints_act["wrist_roll.pos"]) * 0.017453292519943295,
                    float(gripper_to_jaw(follower_joints_act["gripper.pos"])),
                ]
                payload = {
                    "timestamp": time.time(),
                    "joint_positions": sim_joint_positions,
                }
                sim_sender.send(payload)

            # Visualize
            log_rerun_data(observation=leader_ee_act, action=follower_joints_act)

            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
    finally:
        if sim_sender is not None:
            sim_sender.close()
        log_file.close()


if __name__ == "__main__":
    main()
