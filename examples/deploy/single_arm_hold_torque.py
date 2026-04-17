#!/usr/bin/env python3
"""
Hold the SO100 follower arm at a fixed target posture with torque enabled.
"""

import argparse
import logging

from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.utils.robot_utils import precise_sleep

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)
DEFAULT_REAL_CMD_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Keep torque ON and hold the arm at a fixed target posture."
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
    parser.add_argument(
        "--target-joints",
        type=str,
        default="0,-1.57,1.57,1.57,-1.57,0",
        help="Target SIM joints (6 values) to hold, default: 0,-1.57,1.57,1.57,-1.57,0",
    )
    parser.add_argument(
        "--input-units",
        choices=["rad", "deg"],
        default="rad",
        help="Units for --target-joints",
    )
    parser.add_argument(
        "--joint-offsets",
        type=str,
        default="0.06,1.6,-1.5,0.0,0.0,0.0",
        help="Comma-separated joint offsets in radians",
    )
    parser.add_argument(
        "--jaw-limits",
        type=str,
        default="-0.174,1.75",
        help="Comma-separated jaw limits in radians: j_min,j_max",
    )
    parser.add_argument(
        "--refresh-interval",
        type=float,
        default=0.5,
        help="Seconds between re-sending hold command. Set 0 to send only once.",
    )
    parser.add_argument(
        "--keep-torque-on-exit",
        action="store_true",
        help="Do not disable torque when disconnecting after Ctrl+C.",
    )

    args = parser.parse_args()
    if args.refresh_interval < 0:
        raise ValueError("--refresh-interval must be >= 0")
    if not args.use_degrees:
        raise ValueError("This script requires --use-degrees to be enabled")

    target_joints = parse_csv_floats(args.target_joints, 6, "--target-joints")
    if args.input_units == "deg":
        target_joints = [val * 0.017453292519943295 for val in target_joints]
    joint_offsets = parse_csv_floats(args.joint_offsets, 6, "--joint-offsets")
    jaw_limits_raw = parse_csv_floats(args.jaw_limits, 2, "--jaw-limits")
    jaw_limits = (float(jaw_limits_raw[0]), float(jaw_limits_raw[1]))
    hold_action = real_rad_to_cmd(sim_to_real_rad(target_joints, joint_offsets), jaw_limits)

    robot_config = SO100FollowerConfig(
        port=args.port,
        id=args.id,
        use_degrees=args.use_degrees,
        disable_torque_on_disconnect=not args.keep_torque_on_exit,
    )
    robot = SO100Follower(robot_config)

    try:
        log.info("Connecting to follower arm...")
        robot.connect(calibrate=not args.no_calibrate)
        if not robot.is_connected:
            raise RuntimeError("Failed to connect to follower arm")
        log.info("Follower arm connected")

        robot.bus.enable_torque()
        robot.send_action(hold_action)
        log.info("Holding target posture with torque ON. Press Ctrl+C to stop.")
        log.info("Hold command: %s", hold_action)

        while True:
            if args.refresh_interval > 0:
                precise_sleep(args.refresh_interval)
                robot.send_action(hold_action)
            else:
                precise_sleep(1.0)
    except KeyboardInterrupt:
        log.info("Interrupted by user")
        return 0
    except Exception as exc:
        log.error("Failed to hold posture: %s", exc)
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
