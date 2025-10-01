#!/usr/bin/env python

"""
Simple client to receive bimanual SO100 leader poses via socket.
This script connects to the pose sender and displays the received poses.
"""

import contextlib
import json
import socket
import time


def main():
    """Connect to the pose sender and display received poses"""

    # Connection settings
    host = "localhost"
    port = 12345

    try:
        # Create socket connection
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((host, port))

        print(f"Connected to pose sender at {host}:{port}")
        print("Receiving poses... (Press Ctrl+C to stop)")

        # Buffer for received data
        buffer = ""

        while True:
            try:
                # Receive data
                data = client_socket.recv(1024).decode('utf-8')
                if not data:
                    print("Connection closed by server")
                    break

                # Add to buffer
                buffer += data

                # Process complete messages (separated by newlines)
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        try:
                            # Parse JSON message
                            pose_data = json.loads(line)

                            # Extract poses
                            timestamp = pose_data["timestamp"]
                            left_pos = pose_data["left_arm"]["position"]
                            left_rot = pose_data["left_arm"]["orientation"]
                            right_pos = pose_data["right_arm"]["position"]
                            right_rot = pose_data["right_arm"]["orientation"]

                            # Display poses (reduce frequency for readability)
                            current_time = time.time()
                            if int(current_time * 2) % 2 == 0:  # Print every 0.5 seconds
                                print(f"\n--- Timestamp: {timestamp:.3f} ---")
                                print(f"Left arm  - Pos: [{left_pos['x']:.3f}, {left_pos['y']:.3f}, {left_pos['z']:.3f}], "
                                      f"Rot: [{left_rot['wx']:.3f}, {left_rot['wy']:.3f}, {left_rot['wz']:.3f}]")
                                print(f"Right arm - Pos: [{right_pos['x']:.3f}, {right_pos['y']:.3f}, {right_pos['z']:.3f}], "
                                      f"Rot: [{right_rot['wx']:.3f}, {right_rot['wy']:.3f}, {right_rot['wz']:.3f}]")

                        except json.JSONDecodeError as e:
                            print(f"Error parsing JSON: {e}")
                            continue
            except TimeoutError:
                continue
            except KeyboardInterrupt:
                print("\nStopping client...")
                break
            except Exception as e:
                print(f"Error receiving data: {e}")
                break
    except ConnectionRefusedError:
        print(f"Could not connect to {host}:{port}. Make sure the pose sender is running.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        with contextlib.suppress(Exception):
            client_socket.close()
        print("Client disconnected")


if __name__ == "__main__":
    main()