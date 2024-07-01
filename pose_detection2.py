import cv2
import mediapipe as mp
import time

# Initialize the MediaPipe pose module
mp_drawing = mp.solutions.drawing_utils

# Initialize the holistic model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Initialize the video capture object
cap = cv2.VideoCapture(0)  # Change to camera index if needed

# Open a file to write landmark data
with open('landmark_data.txt', 'w') as f:
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

            # Get the timestamp
            timestamp = time.time()

            # Write landmark data to the file
            for i in range(len(results.pose_landmarks.landmark)):
                landmark = results.pose_landmarks.landmark[i]
                f.write(f"{timestamp},{i},{landmark.x},{landmark.y},{landmark.z}\n")

        # Display the image
        cv2.imshow('MediaPipe Pose', image)

        # Exit the loop when the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()