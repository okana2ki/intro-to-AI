import cv2
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# モデルのパスを指定(★自分の環境に合わせて変更してください★)
# ここからダウンロード可能：https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
# lite, full, heavyの3つのモデルがある
# 検出精度は落ちると思うが、ノートPCでダンスのように動きが速い場合はliteが良さそう
# 検出人数を増やし過ぎない方が良さそう
model_path = 'C:/Briefcase/__python/dance/pose_landmarker_lite.task'

# 画像上にランドマークを描画する関数
def draw_landmarks_on_image(rgb_image, detection_result):
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

# グローバル変数
frame = None
processed_frame = None

# 結果を処理するコールバック関数
def process_result(result: vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global frame, processed_frame
    if frame is None:
        return
    
    processed_frame = draw_landmarks_on_image(frame, result)

# PoseLandmarkerを作成
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_poses=4,  # 検出する人数を設定
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_segmentation_masks=False,
    result_callback=process_result
)
landmarker = vision.PoseLandmarker.create_from_options(options)

# カメラキャプチャの設定
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(2)

# メインループ
while cap.isOpened():
    ret, frame_bgr = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # BGR画像をRGBに変換←OpenCVはBGR形式だが、MediaPipeはRGB形式
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # タイムスタンプを生成（ミリ秒単位）
    timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

    # ポーズランドマーク検出を非同期で実行
    landmarker.detect_async(mp_image, timestamp_ms)

    # 処理済みフレームがあれば表示
    if processed_frame is not None:
        display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
    else:
        display_frame = frame_bgr

    cv2.imshow('Pose Landmarker', display_frame)

    # ESCキーで終了
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # 27はESCキーのASCIIコード
        break

# リソースの解放
cap.release()
cv2.destroyAllWindows()