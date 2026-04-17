import cv2
from cv2 import aruco
import numpy as np

# ======== カメラ内部パラメータ（さっき取れた値をそのまま使用） ========
camera_matrix = np.array(
    [[604.10004, 0.0, 326.65164], [0.0, 604.1955, 239.55637], [0.0, 0.0, 1.0]], dtype=np.float32
)

# 歪み係数（RealSense から全部 0 で返ってきた）
dist_coeffs = np.zeros(5, dtype=np.float32)

# ======== ArUco 設定 ========
MARKER_LENGTH = 0.05  # [m] マーカー一辺の長さ（自分のマーカーに合わせて変更）
ARUCO_DICT = aruco.DICT_4X4_50

dictionary = aruco.getPredefinedDictionary(ARUCO_DICT)
try:
    parameters = aruco.DetectorParameters_create()
except AttributeError:
    parameters = aruco.DetectorParameters()

# OpenCV 4.7+ の新APIがあれば使う
if hasattr(aruco, "ArucoDetector"):
    detector = aruco.ArucoDetector(dictionary, parameters)
else:
    detector = None

# ======== カメラオープン ========
# D435i が内蔵カメラとは別デバイス番号になっている場合は、
# 0 -> 1,2,... に変えて試してください
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Failed to open camera. Try changing VideoCapture index (0,1,2,...)")

print("Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ======== マーカー検出 ========
    if detector is not None:
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        corners, ids, rejected = aruco.detectMarkers(gray, dictionary, parameters=parameters)

    out = frame.copy()

    if ids is not None and len(ids) > 0:
        aruco.drawDetectedMarkers(out, corners, ids)

        # marker -> camera の姿勢推定
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_LENGTH, camera_matrix, dist_coeffs)

        for i, marker_id in enumerate(ids.flatten()):
            rvec = rvecs[i]
            tvec = tvecs[i]

            # 座標軸を描画（カメラ座標系でのマーカー姿勢）
            aruco.drawAxis(out, camera_matrix, dist_coeffs, rvec, tvec, MARKER_LENGTH * 0.5)

            # ======== marker -> camera (R_cm, t_cm) ========
            R_cm, _ = cv2.Rodrigues(rvec)
            t_cm = tvec.reshape(3, 1)

            # ======== camera -> marker に変換 ========
            R_mc = R_cm.T
            t_mc = -R_mc @ t_cm

            # 表示用テキスト
            pos = t_mc.ravel()
            text = f"ID:{marker_id} Cam in marker: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) m"
            cv2.putText(
                out, text, (10, 30 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
            )

            # コンソール出力
            print(f"[ID {marker_id}] camera position in marker frame [m]: {pos}")

    cv2.imshow("D435i RGB (OpenCV)", out)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
