import os
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Load the trained pose classifier model
model = tf.keras.models.load_model('pose_classifier_final.h5')

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to preprocess images before extracting keypoints
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 480))  # Standardize size
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

# Function to extract keypoints from an image with normalization
def extract_keypoints_from_image(image_path):
    image = preprocess_image(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_image)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        keypoints = [(landmark.x, landmark.y, landmark.z) for landmark in landmarks]
        return np.array(keypoints)
    else:
        return None

# Function to load ideal keypoints from both orientations
def load_ideal_keypoints(dataset_path):
    ideal_keypoints = {}
    for pose_name in os.listdir(dataset_path):
        pose_folder = os.path.join(dataset_path, pose_name)
        
        if os.path.isdir(pose_folder):
            images = sorted(os.listdir(pose_folder))[:2]  # Ensure two images
            if len(images) < 2:
                continue
            
            keypoints_default = extract_keypoints_from_image(os.path.join(pose_folder, images[0]))
            keypoints_mirrored = extract_keypoints_from_image(os.path.join(pose_folder, images[1]))
            
            if keypoints_default is not None and keypoints_mirrored is not None:
                ideal_keypoints[pose_name] = {"default": keypoints_default, "mirrored": keypoints_mirrored}
    
    return ideal_keypoints

# Load ideal keypoints
dataset_path = "idealdata"
ideal_keypoints = load_ideal_keypoints(dataset_path)

# Debug: Verify ideal image landmarks
for pose_name, orientations in ideal_keypoints.items():
    for orientation, keypoints in orientations.items():
        print(f"Pose: {pose_name}, Orientation: {orientation}, Keypoints: {keypoints.shape}")

# Display available poses
pose_names = list(ideal_keypoints.keys())
print("Available poses:")
for i, pose_name in enumerate(pose_names):
    print(f"{i + 1}. {pose_name}")

# User selects a pose
while True:
    try:
        selected_pose = int(input(f"Select the pose number you want to perform (1-{len(pose_names)}): "))
        if 1 <= selected_pose <= len(pose_names):
            selected_pose_name = pose_names[selected_pose - 1]
            print(f"\nYou selected: {selected_pose_name}")
            break
        else:
            print("Invalid choice. Try again.")
    except ValueError:
        print("Invalid input. Enter a number.")

# Pose detection in real-time
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    if results.pose_landmarks:
        user_keypoints = np.array([(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark])
        
        # Determine best orientation match
        best_match = "default"
        best_score = float('inf')
        for orientation in ["default", "mirrored"]:
            ideal_pose_keypoints = ideal_keypoints[selected_pose_name][orientation]
            score = np.linalg.norm(user_keypoints - ideal_pose_keypoints)
            if score < best_score:
                best_score = score
                best_match = orientation
        
        ideal_pose_keypoints = ideal_keypoints[selected_pose_name][best_match]
        
        # Draw corrective pose lines
        correct_count = 0
        for connection in mp_pose.POSE_CONNECTIONS:
            start_idx, end_idx = list(connection)
            
            x_start, y_start, _ = user_keypoints[start_idx]
            x_end, y_end, _ = user_keypoints[end_idx]
            
            ideal_start = ideal_pose_keypoints[start_idx]
            ideal_end = ideal_pose_keypoints[end_idx]
            
            distance_start = np.linalg.norm(np.array([x_start, y_start]) - np.array(ideal_start[:2]))
            distance_end = np.linalg.norm(np.array([x_end, y_end]) - np.array(ideal_end[:2]))
            
            threshold = 0.25  # Adjust threshold for accuracy
            color = (0, 0, 255)  # Default red for incorrect
            if distance_start < threshold and distance_end < threshold:
                color = (0, 255, 0)  # Change to green for correct
                correct_count += 1
            
            start_coords = (int(x_start * frame.shape[1]), int(y_start * frame.shape[0]))
            end_coords = (int(x_end * frame.shape[1]), int(y_end * frame.shape[0]))
            cv2.line(frame, start_coords, end_coords, color, 2)
        
        # Display correctness feedback
        if correct_count / len(mp_pose.POSE_CONNECTIONS) > 0.8:
            cv2.putText(frame, "Correct", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Incorrect", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Display detected orientation
        cv2.putText(frame, f"Orientation: {best_match.capitalize()}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Not Found", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    cv2.imshow('Yoga Pose Detection & Correction', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
