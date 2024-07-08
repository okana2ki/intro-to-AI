import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Specify the path to the new model
model_path = 'C:/Briefcase/__python/dance/pose_landmarker_lite.task'

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a pose landmarker instance with the live stream mode:
def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('pose landmarker result: {}'.format(result))

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

with PoseLandmarker.create_from_options(options) as landmarker:
  # The landmarker is initialized. Use it here.
  # ...



# Use OpenCV’s VideoCapture to start capturing from the webcam.

# Create a loop to read the latest frame from the camera using VideoCapture#read()

# Convert the frame received from OpenCV to a MediaPipe’s Image object.
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)



# Send live image data to perform pose landmarking.
# The results are accessible via the `result_callback` provided in
# the `PoseLandmarkerOptions` object.
# The pose landmarker must be created with the live stream mode.
landmarker.detect_async(mp_image, frame_timestamp_ms)



# 動画モードまたはライブ ストリーム モードで実行する場合は、
# ポーズ ランドマーク タスクに入力フレームのタイムスタンプも指定します。
# ライブ ストリーム モードで実行すると、ポーズのランドマーク タスクはすぐに返され、
# 現在のスレッドはブロックされません。入力フレームの処理が完了するたびに、
# 検出結果とともに結果リスナーを呼び出します。
# ポーズのランドマーク タスクが別のフレームの処理でビジー状態のときに検出関数が呼び出された場合、
# タスクは新しい入力フレームを無視します


# ポーズのランドマークは、検出を実行するたびに poseLandmarkerResult オブジェクトを返します。
# 結果オブジェクトには、各ポーズ ランドマークの座標が含まれます。
# このタスクからの出力データの例を次に示します。

'''
PoseLandmarkerResult:
  Landmarks:
    Landmark #0:
      x            : 0.638852
      y            : 0.671197
      z            : 0.129959
      visibility   : 0.9999997615814209
      presence     : 0.9999984502792358
    Landmark #1:
      x            : 0.634599
      y            : 0.536441
      z            : -0.06984
      visibility   : 0.999909
      presence     : 0.999958
    ... (33 landmarks per pose)
  WorldLandmarks:
    Landmark #0:
      x            : 0.067485
      y            : 0.031084
      z            : 0.055223
      visibility   : 0.9999997615814209
      presence     : 0.9999984502792358
    Landmark #1:
      x            : 0.063209
      y            : -0.00382
      z            : 0.020920
      visibility   : 0.999976
      presence     : 0.999998
    ... (33 world landmarks per pose)
  SegmentationMasks:
    ... (pictured below)
'''

