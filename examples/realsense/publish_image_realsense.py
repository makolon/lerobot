#!/usr/bin/env python3
"""
RealSense RGB+Depth socket sender (depth aligned to color).

This script captures synchronized frames from a RealSense camera,
aligns depth to color (rs.align), and streams them as base64-encoded
JPEG (color) and PNG (depth) over a TCP socket.
"""

import argparse
import base64
import contextlib
import json
import logging
import socket
import time

import cv2
import numpy as np
import pyrealsense2 as rs

from lerobot.cameras import ColorMode
from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
from lerobot.utils.robot_utils import precise_sleep

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


class RealSenseImageSender:
    """Sends aligned RealSense RGB + depth frames via socket."""

    def __init__(
        self,
        socket_host: str = "localhost",
        socket_port: int = 12346,
        serial_number_or_name: str = "",
        fps: int = 10,
        width: int = 640,
        height: int = 480,
        align_depth: bool = True,
    ):
        self.fps = fps
        self.socket_host = socket_host
        self.socket_port = socket_port
        self.align_depth = align_depth

        self.camera_config = RealSenseCameraConfig(
            serial_number_or_name=serial_number_or_name,
            fps=fps,
            width=width,
            height=height,
            color_mode=ColorMode.BGR,
            use_depth=True,
        )

        self.camera = RealSenseCamera(self.camera_config)
        self.rs_align = None

        self.socket: socket.socket | None = None
        self.connected_clients: list[socket.socket] = []

    def connect(self):
        log.info("Connecting to RealSense camera...")
        self.camera.connect()
        log.info("Connected to RealSense camera")
        self.print_intrinsics()

        if self.align_depth:
            self.rs_align = rs.align(rs.stream.color)
            log.info("Depth alignment to color enabled")
        else:
            self.rs_align = None

        self.setup_socket_server()

    def print_intrinsics(self):
        if self.camera.rs_profile is None:
            log.warning("RealSense profile is not available yet.")
            return

        color_profile = self.camera.rs_profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_profile = self.camera.rs_profile.get_stream(rs.stream.depth).as_video_stream_profile()
        color_intr = color_profile.get_intrinsics()
        depth_intr = depth_profile.get_intrinsics()

        log.info("Color intrinsics: %s", color_intr)
        log.info("Depth intrinsics: %s", depth_intr)

    def setup_socket_server(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.socket_host, self.socket_port))
        self.socket.listen(5)
        self.socket.settimeout(0.001)
        log.info("Socket server listening on %s:%s", self.socket_host, self.socket_port)

    def accept_new_clients(self):
        if self.socket is None:
            return
        try:
            client_socket, address = self.socket.accept()
            client_socket.settimeout(0.001)
            self.connected_clients.append(client_socket)
            log.info("New client connected from %s", address)
        except TimeoutError:
            pass
        except Exception as exc:
            log.warning("Error accepting client: %s", exc)

    def read_aligned_frames(self):
        if self.camera.rs_pipeline is None:
            return None, None

        frames = self.camera.rs_pipeline.wait_for_frames()
        if self.rs_align is not None:
            frames = self.rs_align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame:
            return None, None

        color_image = np.asanyarray(color_frame.get_data())
        color_image = self.camera._postprocess_image(color_image, depth_frame=False)

        depth_image = None
        if depth_frame:
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_image = self.camera._postprocess_image(depth_image, depth_frame=True)

        return color_image, depth_image

    def create_image_message(self, image, depth_image=None):
        ret, buffer = cv2.imencode(".jpg", image)
        if not ret:
            return None
        jpg_as_text = base64.b64encode(buffer).decode("utf-8")

        message = {
            "timestamp": time.time(),
            "image": jpg_as_text,
            "format": "jpeg",
            "shape": image.shape,
            "has_depth": depth_image is not None,
        }

        if depth_image is not None:
            ret_d, buffer_d = cv2.imencode(".png", depth_image)
            if not ret_d:
                return None
            depth_as_text = base64.b64encode(buffer_d).decode("utf-8")
            message.update(
                {
                    "depth": depth_as_text,
                    "depth_format": "png",
                    "depth_shape": depth_image.shape,
                    "depth_dtype": str(depth_image.dtype),
                }
            )

        return json.dumps(message) + "\n"

    def send_to_clients(self, message):
        disconnected_clients = []
        for client in self.connected_clients:
            try:
                client.send(message.encode("utf-8"))
            except (OSError, BrokenPipeError):
                disconnected_clients.append(client)

        for client in disconnected_clients:
            with contextlib.suppress(Exception):
                client.close()
            self.connected_clients.remove(client)
            log.info("Client disconnected")

    def run_loop(self):
        log.info("Starting image sender loop at %s Hz...", self.fps)
        log.info("Waiting for client connections...")

        loop_duration = 1.0 / self.fps

        while True:
            loop_start = time.perf_counter()
            try:
                self.accept_new_clients()

                image, depth_image = self.read_aligned_frames()
                if image is not None:
                    message = self.create_image_message(image, depth_image=depth_image)
                    if message and self.connected_clients:
                        self.send_to_clients(message)
            except KeyboardInterrupt:
                log.info("Shutting down...")
                break
            except Exception as exc:
                log.error("Error in main loop: %s", exc)

            elapsed = time.perf_counter() - loop_start
            precise_sleep(max(loop_duration - elapsed, 0.0))

    def disconnect(self):
        if self.camera:
            self.camera.disconnect()
        for client in self.connected_clients:
            with contextlib.suppress(Exception):
                client.close()
        if self.socket:
            self.socket.close()
        log.info("Disconnected successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Send RealSense RGB+Depth frames over a socket (depth aligned to color)",
    )
    parser.add_argument("--socket-host", type=str, default="localhost", help="Socket host (default: localhost)")
    parser.add_argument("--socket-port", type=int, default=12346, help="Socket port (default: 12346)")
    parser.add_argument("--serial", type=str, required=True, help="RealSense serial number or unique name")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    parser.add_argument("--width", type=int, default=640, help="Frame width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Frame height (default: 480)")
    parser.add_argument(
        "--no-align",
        action="store_true",
        help="Disable depth-to-color alignment (not recommended)",
    )
    args = parser.parse_args()

    sender = None
    try:
        sender = RealSenseImageSender(
            socket_host=args.socket_host,
            socket_port=args.socket_port,
            serial_number_or_name=args.serial,
            fps=args.fps,
            width=args.width,
            height=args.height,
            align_depth=not args.no_align,
        )
        sender.connect()
        sender.run_loop()
    except KeyboardInterrupt:
        log.info("Interrupted by user")
    except Exception as exc:
        log.error("Error: %s", exc)
    finally:
        if sender is not None:
            sender.disconnect()


if __name__ == "__main__":
    main()
