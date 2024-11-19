import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# Initialize mediapipe holistic model and drawing utilities
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Number of landmarks
num_pose_landmarks = 33
num_face_landmarks = 468

# Path to video folder
video_folder = "recorded_videos/"
video_files = [f for f in os.listdir(video_folder) if f.endswith(('.avi', '.mp4'))]  # Filter video files

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    # Create header for CSV
    landmarks = ['class']
    for val in range(1, num_pose_landmarks + 1):
        landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
        
    for val in range(1, num_face_landmarks + 1):
        landmarks += ['x{}'.format(num_pose_landmarks + val), 'y{}'.format(num_pose_landmarks + val), 
                      'z{}'.format(num_pose_landmarks + val), 'v{}'.format(num_pose_landmarks + val)]

    # Check if the file exists to decide whether to write header
    file_exists = os.path.isfile('coords.csv')

    # Write header to CSV if the file does not exist
    with open('coords.csv', mode='a', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if not file_exists:
            csv_writer.writerow(landmarks)

    # Loop through all videos in the folder
    for video_filename in video_files:
        cap = cv2.VideoCapture(os.path.join(video_folder, video_filename))  # Open video from file

        while cap.isOpened():
            ret, frame = cap.read()  # Read each frame from video

            if not ret:
                break  # Break loop if unable to read frame (video ends)

            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False        

            # Make Detections
            results = holistic.process(image)

            # Recolor image back to BGR (not used but necessary for processing)
            image.flags.writeable = True   
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Export coordinates
            try:
                # Extract Pose landmarks
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                # Extract Face landmarks
                face = results.face_landmarks.landmark
                face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

                # Concatenate rows
                row = pose_row + face_row
                

                # Determine class name dynamically from video filename
                class_name = os.path.splitext(video_filename)[0] 
                class_name = class_name.split('_')[0]  ## Use filename without extension as class
                row.insert(0, class_name)

                # Append to CSV
                with open('coords.csv', mode='a', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(row) 

            except Exception as e:
                print(f"Error processing video {video_filename}: {e}")

        cap.release()  # Release video capture after processing the video

cv2.destroyAllWindows()  # Close all OpenCV windows (not needed anymore)
