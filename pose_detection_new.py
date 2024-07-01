import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# model_path = '/absolute/path/to/pose_landmarker.task'

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



# Initialize the MediaPipe pose module
mp_drawing = mp.solutions.drawing_utils

# Initialize the holistic model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Initialize the video capture object
cap = cv2.VideoCapture(0)  # Change to camera index if needed

while cap.isOpened():
  # Read each frame from the camera
  success, image = cap.read()

  # Process the image for pose landmark detection
  image = cv2.flip(image, 1)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  results = pose.process(image)

  # Convert the image back to BGR format
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

  # Draw the pose landmarks on the image
  if results.pose_landmarks is not None:
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

  # Display the image
  cv2.imshow('MediaPipe Pose', image)

  # Exit the loop when the user presses the 'q' key
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()