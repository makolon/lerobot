#!/usr/bin/env python3
"""
Load a joint trajectory JSON file and execute it on a SO100 follower arm.

The JSON should contain a joint_trajectory list. Each step should contain motor
name keys, or a joint_positions dict with those keys.
"""

import argparse
import json
import logging
import time
from typing import Any

from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.utils.robot_utils import precise_sleep

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)
DEFAULT_SIM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
DEFAULT_REAL_CMD_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def extract_steps(trajectory_data: dict[str, Any]) -> list[Any]:
    if "joint_trajectory" in trajectory_data and isinstance(trajectory_data["joint_trajectory"], list):
        return trajectory_data["joint_trajectory"]
    raise ValueError("Trajectory data missing joint_trajectory")


def extract_action_payload(step: Any, motor_names: list[str]) -> dict[str, float]:
    if not isinstance(step, dict):
        raise ValueError("Joint trajectory step must be a dict")
    if "joint_positions" in step:
        payload = step["joint_positions"]
        if isinstance(payload, dict):
            return {str(k): float(v) for k, v in payload.items()}
        if isinstance(payload, list):
            if len(payload) != len(motor_names):
                raise ValueError("joint_positions list length does not match motor count")
            return {
                f"{name}.pos": float(val)
                for name, val in zip(motor_names, payload, strict=True)
            }
        raise ValueError("joint_positions must be a dict of motor keys")
    return {str(k): float(v) for k, v in step.items() if k != "timestamp"}


def normalize_joint_step(step: Any) -> list[float]:
    if isinstance(step, (list, tuple)):
        if len(step) != 6:
            raise ValueError(f"Expected 6 joints, got {len(step)}")
        return [float(v) for v in step]
    if isinstance(step, dict):
        if all(k in step for k in DEFAULT_SIM_JOINT_NAMES):
            return [float(step[k]) for k in DEFAULT_SIM_JOINT_NAMES]
        if all(f"{k}.pos" in step for k in DEFAULT_SIM_JOINT_NAMES):
            return [float(step[f"{k}.pos"]) for k in DEFAULT_SIM_JOINT_NAMES]
        raise ValueError("Joint dict must include sim joint keys or .pos keys")
    raise ValueError("Joint step must be a list/tuple or dict")


def normalize_input_units(joints: list[list[float]], mode: str) -> list[list[float]]:
    if mode == "deg":
        return [[val * 0.017453292519943295 for val in step] for step in joints]
    if mode == "auto":
        mags = [max(abs(val) for val in step[:5]) for step in joints if step]
        if mags and max(mags) > 6.5:
            return [[val * 0.017453292519943295 for val in step] for step in joints]
    return joints


def parse_csv_floats(text: str, expected: int, label: str) -> list[float]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != expected:
        raise ValueError(f"{label} must have {expected} values")
    return [float(p) for p in parts]


def sim_to_real_rad(q_sim: list[float], offsets: list[float]) -> list[float]:
    q_real = [q_sim[idx] + offsets[idx] for idx in range(6)]
    q_real[0] = -q_real[0]
    return q_real


def real_rad_to_cmd(q_real: list[float], jaw_limits: tuple[float, float]) -> dict[str, float]:
    j_min, j_max = jaw_limits
    if j_max <= j_min:
        raise ValueError("jaw limits invalid: j_max must be > j_min")
    deg = [val * 57.29577951308232 for val in q_real[:5]]
    jaw = q_real[5]
    gripper = (jaw - j_min) / (j_max - j_min) * 100.0
    gripper = float(max(min(gripper, 100.0), 0.0))
    cmd = dict(zip(DEFAULT_REAL_CMD_KEYS[:5], deg, strict=True))
    cmd["gripper.pos"] = gripper
    return cmd


def load_trajectory(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def execute_trajectory(
    robot: SO100Follower,
    trajectory_data: dict[str, Any],
    repeat: int,
    repeat_delay: float,
    input_units: str,
    joint_offsets: list[float],
    jaw_limits: tuple[float, float],
) -> dict[str, Any]:
    steps = extract_steps(trajectory_data)
    joints = []
    for step in steps:
        payload = step.get("joint_positions", step) if isinstance(step, dict) else step
        joints.append(normalize_joint_step(payload))
    joints_rad = normalize_input_units(joints, input_units)
    joint_steps = []
    for q_sim in joints_rad:
        q_real = sim_to_real_rad(q_sim, joint_offsets)
        joint_steps.append(real_rad_to_cmd(q_real, jaw_limits))

    log.info("Executing %d steps x %d", len(joint_steps), repeat)
    t_start = time.perf_counter()
    executed = 0

    for step_idx, joint_action in enumerate(joint_steps, start=1):
        for repeat_idx in range(repeat):
            print(f"Step {step_idx}/{len(joint_steps)} repeat {repeat_idx + 1}/{repeat}")
            print(f"Joint command: {joint_action}")

            _ = robot.send_action(joint_action)
            executed += 1

            if repeat_delay > 0 and repeat_idx < repeat - 1:
                precise_sleep(repeat_delay)

    return {
        "status": "success",
        "steps_executed": executed,
        "execution_time": time.perf_counter() - t_start,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Receive a joint trajectory over socket and execute it on a SO100 follower arm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--port", required=True, help="Serial port for SO100 follower arm")
    parser.add_argument("--id", default="single_arm_follower", help="Robot ID used for calibration files")
    parser.add_argument(
        "--use-degrees",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use degrees for joint values",
    )
    parser.add_argument("--no-calibrate", action="store_true", help="Skip calibration on connect")
    parser.add_argument("--trajectory", required=True, help="Path to trajectory JSON file")
    parser.add_argument("--repeat", type=int, default=20, help="Repeat the trajectory N times")
    parser.add_argument(
        "--repeat-delay",
        type=float,
        default=0.02,
        help="Delay in seconds between repeats",
    )
    parser.add_argument(
        "--input-units",
        choices=["rad", "deg", "auto"],
        default="rad",
        help="Units of joints stored in robot_trajectory.json",
    )
    parser.add_argument(
        "--joint-offsets",
        type=str,
        default="0.0,1.6,-1.4,0.0,0.0,-0.3",
        help="Comma-separated joint offsets in radians",
    )
    parser.add_argument(
        "--jaw-limits",
        type=str,
        default="-0.174,1.75",
        help="Comma-separated jaw limits in radians: j_min,j_max",
    )
    parser.add_argument(
        "--force-enable-torque",
        action="store_true",
        help="Force enable torque before executing the trajectory",
    )
    parser.add_argument(
        "--keep-torque",
        action="store_true",
        help="Keep torque enabled after finishing by not disabling it on disconnect",
    )

    args = parser.parse_args()

    if args.repeat < 1:
        raise ValueError("--repeat must be >= 1")

    robot_config = SO100FollowerConfig(
        port=args.port,
        id=args.id,
        use_degrees=args.use_degrees,
        disable_torque_on_disconnect=not args.keep_torque,
    )
    robot = SO100Follower(robot_config)

    try:
        log.info("Connecting to follower arm...")
        robot.connect(calibrate=not args.no_calibrate)
        if not robot.is_connected:
            raise RuntimeError("Failed to connect to follower arm")
        log.info("Follower arm connected")

        trajectory_data = load_trajectory(args.trajectory)
        if args.force_enable_torque:
            robot.bus.enable_torque()
        offsets = parse_csv_floats(args.joint_offsets, 6, "--joint-offsets")
        jaw_limits_raw = parse_csv_floats(args.jaw_limits, 2, "--jaw-limits")
        jaw_limits = (float(jaw_limits_raw[0]), float(jaw_limits_raw[1]))
        execute_trajectory(
            robot,
            trajectory_data,
            args.repeat,
            args.repeat_delay,
            args.input_units,
            offsets,
            jaw_limits,
        )
    except KeyboardInterrupt:
        log.warning("Interrupted by user")
        return 1
    except Exception as exc:
        log.error("Server error: %s", exc)
        return 1
    finally:
        if robot.is_connected:
            try:
                robot.disconnect()
            except ConnectionError as exc:
                log.warning("Disconnect failed: %s", exc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
