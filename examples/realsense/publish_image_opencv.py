#!/usr/bin/env python3
"""
OpenCV RGB socket sender.

This script captures frames from an OpenCV camera and streams them as
base64-encoded JPEG over a TCP socket.
"""

import argparse
import base64
import contextlib
import json
import logging
import socket
import time

import cv2

from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.utils.robot_utils import precise_sleep

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


class OpenCVImageSender:
    """Sends OpenCV RGB frames via socket."""

    def __init__(
        self,
        socket_host: str = "localhost",
        socket_port: int = 12346,
        camera_index: int = 0,
        fps: int = 10,
        width: int = 640,
        height: int = 480,
    ):
        self.fps = fps
        self.socket_host = socket_host
        self.socket_port = socket_port

        self.camera_config = OpenCVCameraConfig(
            index_or_path=camera_index,
            fps=fps,
            width=width,
            height=height,
        )

        self.camera = OpenCVCamera(self.camera_config)

        self.socket: socket.socket | None = None
        self.connected_clients: list[socket.socket] = []

    def connect(self):
        log.info("Connecting to OpenCV camera...")
        self.camera.connect()
        log.info("Connected to OpenCV camera")

        self.setup_socket_server()

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

    def read_frame(self):
        return self.camera.read()

    def create_image_message(self, image):
        ret, buffer = cv2.imencode(".jpg", image)
        if not ret:
            return None
        jpg_as_text = base64.b64encode(buffer).decode("utf-8")

        message = {
            "timestamp": time.time(),
            "image": jpg_as_text,
            "format": "jpeg",
            "shape": image.shape,
            "has_depth": False,
        }

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

                image = self.read_frame()
                if image is not None:
                    message = self.create_image_message(image)
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
        description="Send OpenCV RGB frames over a socket",
    )
    parser.add_argument("--socket-host", type=str, default="localhost", help="Socket host (default: localhost)")
    parser.add_argument("--socket-port", type=int, default=12346, help="Socket port (default: 12346)")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index (default: 0)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    parser.add_argument("--width", type=int, default=640, help="Frame width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Frame height (default: 480)")
    args = parser.parse_args()

    sender = None
    try:
        sender = OpenCVImageSender(
            socket_host=args.socket_host,
            socket_port=args.socket_port,
            camera_index=args.camera_index,
            fps=args.fps,
            width=args.width,
            height=args.height,
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
