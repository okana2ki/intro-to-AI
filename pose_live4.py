import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# モデルのパスを設定
model_path = 'C:/Users/oka/pose_landmarker_lite.task'

# グローバル変数で画像フレームを保持
output_frame = None

# コールバック関数の定義
def print_result(result, output_image, timestamp_ms):
    global output_frame
    if result.pose_landmarks:
        for landmarks in result.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                output_image.mat, landmarks, mp.solutions.pose.POSE_CONNECTIONS)
    # 結果のフレームをグローバル変数に保存
    output_frame = output_image.mat

# PoseLandmarkerオプションの作成
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_poses=2,
    result_callback=print_result
)
landmarker = vision.PoseLandmarker.create_from_options(options)

def main():
    global output_frame
    # カメラのセットアップ
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 画像を反転し、BGRからRGBに変換
        rgb_frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

        # Mediapipe Imageオブジェクトを作成
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # タイムスタンプを取得
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        # ポーズランドマークを検出
        landmarker.detect_async(mp_image, timestamp_ms)

        # 結果のフレームを表示
        if output_frame is not None:
            cv2.imshow('MediaPipe Pose', output_frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()