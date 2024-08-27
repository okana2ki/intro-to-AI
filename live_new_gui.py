import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tkinter as tk
from tkinter import ttk
import threading
import time

# モデルのパスを指定(★自分の環境に合わせて変更してください★)
# ここからダウンロード可能：https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
# lite, full, heavyの3つのモデルがある
# 検出精度は落ちると思うが、ノートPCでダンスのように動きが速い場合はliteが良さそう
# 検出人数を増やし過ぎない方が良さそう
# モデルのパスを現在のスクリプトと同じディレクトリに設定
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'pose_landmarker_lite.task')

class PoseLandmarker:
    def __init__(self):
        self.frame = None
        self.processed_frame = None
        self.running = False
        self.landmarker = None
        self.cap = None

    # 画像上にランドマークを描画する関数
    def draw_landmarks_on_image(self, rgb_image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)

        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
                for landmark in pose_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style()
            )
        return annotated_image

    # 結果を処理するコールバック関数
    def process_result(self, result: vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        if self.frame is None:
            # print("Frame is None")  # デバッグ用
            return
        
        self.processed_frame = self.draw_landmarks_on_image(self.frame, result)
        # print("Landmarks processed")  # デバッグ用

    # カメラの一覧を取得する関数
    def get_camera_list(self):
        camera_list = []
        for i in range(10):  # 0から9までのカメラインデックスをチェック
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                camera_list.append(f"Camera {i}")
                cap.release()
        return camera_list

    # メイン処理を開始する関数
    def start_processing(self, num_poses, camera_index):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_poses=num_poses,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
            result_callback=self.process_result
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

        self.cap = cv2.VideoCapture(camera_index)
        self.running = True

        self.main_loop_thread = threading.Thread(target=self.main_loop)
        self.main_loop_thread.start()

    def main_loop(self):
        while self.running:
            ret, frame_bgr = self.cap.read()
            if not ret:
                # print("Failed to capture frame")  # デバッグ用
                break

            self.frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            # print("Frame captured")  # デバッグ用
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=self.frame)

            timestamp_ms = int(time.time() * 1000)

            self.landmarker.detect_async(mp_image, timestamp_ms)

            self.display_frame()

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESCキーで終了
                self.running = False
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def display_frame(self):
        if self.processed_frame is not None:
            display_frame = cv2.cvtColor(self.processed_frame, cv2.COLOR_RGB2BGR)
            # print("Display processed frame")  # デバッグ用
        else:
            display_frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
            # print("Display original frame")  # デバッグ用

        cv2.imshow('Pose Landmarker', display_frame)

def main():
    pose_landmarker = PoseLandmarker()

    # GUI作成
    root = tk.Tk()
    root.title("Pose Landmarker Settings")

    # 検出人数の選択
    ttk.Label(root, text="Select number of poses to detect:").pack(pady=5)
    num_poses_var = tk.StringVar(value="4")
    ttk.Spinbox(root, from_=1, to=10, textvariable=num_poses_var, width=5).pack()

    # カメラの選択
    ttk.Label(root, text="Select camera:").pack(pady=5)
    camera_var = tk.StringVar(value="Camera 0")
    ttk.Combobox(root, textvariable=camera_var, values=pose_landmarker.get_camera_list(), state="readonly").pack()

    # スタートボタン
    start_button = ttk.Button(root, text="Start", command=lambda: pose_landmarker.start_processing(
        int(num_poses_var.get()), 
        int(camera_var.get().split()[1])
    ))
    start_button.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()