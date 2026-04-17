import pyrealsense2 as rs
import numpy as np
import cv2
from cv2 import aruco

# =========================
# 設定
# =========================
MARKER_LENGTH = 0.05  # マーカーの実辺長[m]
ARUCO_DICT = aruco.DICT_4X4_50

WIDTH, HEIGHT, FPS = 640, 480, 30

# =========================
# RealSense 初期化
# =========================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
profile = pipeline.start(config)

# 内部パラメータ取得
color_stream = profile.get_stream(rs.stream.color)
intr = rs.video_stream_profile(color_stream).get_intrinsics()
camera_matrix = np.array([
    [intr.fx, 0, intr.ppx],
    [0, intr.fy, intr.ppy],
    [0, 0, 1]
], dtype=np.float32)
dist_coeffs = np.array(intr.coeffs)

print("Camera matrix:\n", camera_matrix)
print("Distortion coeffs:", dist_coeffs)

# =========================
# ArUco 準備
# =========================
dictionary = aruco.getPredefinedDictionary(ARUCO_DICT)

# DetectorParameters の作り方
# OpenCV のバージョンによって create() がない場合があるので分岐
try:
    parameters = aruco.DetectorParameters_create()
except AttributeError:
    parameters = aruco.DetectorParameters()

# OpenCV >=4.7 形式（cv2.aruco.ArucoDetector）
if hasattr(aruco, 'ArucoDetector'):
    detector = aruco.ArucoDetector(dictionary, parameters)
else:
    detector = None

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 検出（新/旧 API どちらにも対応）
        if detector:
            corners, ids, rejected = detector.detectMarkers(gray)
        else:
            corners, ids, rejected = aruco.detectMarkers(gray, dictionary, parameters=parameters)

        output = image.copy()

        if ids is not None and len(ids) > 0:
            aruco.drawDetectedMarkers(output, corners, ids)

            # 姿勢推定
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, MARKER_LENGTH, camera_matrix, dist_coeffs
            )

            for i, marker_id in enumerate(ids.flatten()):
                rvec = rvecs[i]
                tvec = tvecs[i]

                # Draw axis
                aruco.drawAxis(output, camera_matrix, dist_coeffs, rvec, tvec, MARKER_LENGTH * 0.5)

                # marker -> camera
                R_cm, _ = cv2.Rodrigues(rvec)
                t_cm = tvec.reshape(3, 1)
                # camera -> marker
                R_mc = R_cm.T
                t_mc = -R_cm.T @ t_cm

                print(f"ID {marker_id} cam in marker frame = {t_mc.ravel()}")

        cv2.imshow("ArUco RealSense", output)
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

