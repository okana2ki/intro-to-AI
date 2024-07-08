import cv2
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Specify the path to the model
model_path = 'C:/Briefcase/__python/dance/pose_landmarker_lite.task'

# Visualization Utilities
def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(
                x=landmark.x, y=landmark.y, z=landmark.z
            ) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style()
        )
    return annotated_image

# Create PoseLandmarker object with new specifications
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=2,
    output_segmentation_masks=True
)
landmarker = vision.PoseLandmarker.create_from_options(options)

# Input and output video paths
input_video_path = 'C:/Briefcase/__python/dance/input_video.mp4'
output_video_path = 'C:/Briefcase/__python/dance/annotated_video.mp4'

# Load input video
cap = cv2.VideoCapture(input_video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Output video settings
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Frame counter for generating timestamps
frame_counter = 0

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Generate timestamp in microseconds
    frame_timestamp_us = int(frame_counter * (1000000 / fps))
    frame_counter += 1

    # Perform pose landmark detection
    pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_us)

    # Draw landmarks on the image
    annotated_frame = draw_landmarks_on_image(rgb_frame, pose_landmarker_result)

    # Convert RGB image back to BGR
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
    out.write(annotated_frame)

# Release resources
cap.release()
out.release()