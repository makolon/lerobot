import contextlib
import json
import socket
import time
import base64
import cv2
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

from lerobot.utils.robot_utils import precise_sleep


class OpenCVImageSender:
    """Sends OpenCV camera images via socket at a specified frequency"""

    def __init__(
        self,
        socket_host: str = "localhost",
        socket_port: int = 12346,
        camera_index: int = 0,
        fps: int = 30,
        width: int = 640,
        height: int = 480,
    ):
        """
        Initialize the image sender

        Args:
            socket_host: Host address for socket connection
            socket_port: Port for socket connection
            camera_index: Index of the OpenCV camera
            fps: Frames per second
            width: Image width
            height: Image height
        """
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

        # Socket connection
        self.socket = None
        self.connected_clients = []

    def connect(self):
        """Connect to the camera and setup socket server"""
        print("Connecting to OpenCV camera...")
        self.camera.connect()
        print("Connected to OpenCV camera")

        # Setup socket server
        self.setup_socket_server()

    def setup_socket_server(self):
        """Setup TCP socket server for image transmission"""
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

    def create_image_message(self, image):
        """Create JSON message with base64 encoded image"""
        # Encode image to JPEG
        ret, buffer = cv2.imencode('.jpg', image)
        if not ret:
            return None

        # Convert to base64
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        message = {
            "timestamp": time.time(),
            "image": jpg_as_text,
            "format": "jpeg",
            "shape": image.shape
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

    def run_loop(self):
        """Main loop"""
        print(f"Starting image sender loop at {self.fps} Hz...")
        print("Waiting for client connections...")

        loop_duration = 1.0 / self.fps

        while True:
            loop_start = time.perf_counter()

            try:
                # Accept new client connections
                self.accept_new_clients()

                # Capture frame
                image = self.camera.read()

                if image is not None:
                    # Create and send message
                    message = self.create_image_message(image)
                    if message and self.connected_clients:
                        self.send_to_clients(message)

                        # Debug output
                        if int(time.time() * 2) % 2 == 0:  # Print every 0.5 seconds
                            print(f"Sent image: {image.shape} at {time.time():.2f}")

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
        if self.camera:
            self.camera.disconnect()

        # Close all client connections
        for client in self.connected_clients:
            with contextlib.suppress(Exception):
                client.close()

        # Close server socket
        if self.socket:
            self.socket.close()

        print("Disconnected successfully")


def main():
    """Main function to run the image sender"""

    # Configuration
    socket_host = "localhost"
    socket_port = 12346 # Different port from teleop
    fps = 6

    # Camera settings
    # Use the camera index found via `lerobot-find-cameras opencv`
    # In your case, Camera #0 or Camera #1
    camera_index = 0

    try:
        sender = OpenCVImageSender(
            socket_host=socket_host,
            socket_port=socket_port,
            camera_index=camera_index,
            fps=fps
        )
        sender.connect()
        sender.run_loop()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'sender' in locals():
            sender.disconnect()


if __name__ == "__main__":
    main()
