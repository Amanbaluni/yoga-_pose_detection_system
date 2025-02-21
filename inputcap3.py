import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
from sklearn.preprocessing import LabelEncoder
from keras import backend as K

# Define focal loss function
def focal_loss(gamma=1., alpha=0.75):
    def focal_loss_fixed(y_true, y_pred):
        y_true = K.cast(y_true, K.floatx())
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.sum(loss, axis=1)
    return focal_loss_fixed

# Register the custom loss function
from keras.utils import get_custom_objects
get_custom_objects()['focal_loss_fixed'] = focal_loss()

# Initialize mediapipe pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)  # Increased confidence

# Load the pre-trained pose classification model (replace with your model path)
model = tf.keras.models.load_model('./pose_classifier_final.h5')

# Function to extract pose landmarks from an image
def extract_pose_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])  # x, y, z coordinates
        return np.array(landmarks).flatten(), results.pose_landmarks  # Flatten to a 1D array
    else:
        return None, None

# Load pose class names from dataset folder names (assumes folder names represent pose classes)
pose_class_names = sorted(os.listdir('./dataset'))  # Replace with the actual path to your dataset
# Filter out any non-directory items in case there are files
pose_class_names = [name for name in pose_class_names if os.path.isdir(os.path.join('./dataset', name))]

# Label Encoder for class names
label_encoder = LabelEncoder()
label_encoder.fit(pose_class_names)

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

# Pose landmarks connection lines (body connections for drawing)
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Extract pose landmarks
    pose_landmarks, pose_landmarks_obj = extract_pose_landmarks(frame)
    
    if pose_landmarks is not None and pose_landmarks_obj is not None:
        # Normalize the pose landmarks (optional, depending on the model)
        pose_landmarks = (pose_landmarks - np.min(pose_landmarks)) / (np.max(pose_landmarks) - np.min(pose_landmarks))  # Normalize keypoints
        
        # Prepare the data for classification (reshape if necessary for your model)
        pose_landmarks = np.expand_dims(pose_landmarks, axis=0)  # Adding batch dimension
        
        # Predict the pose class using the trained model
        pose_class = model.predict(pose_landmarks)
        pose_class_label = np.argmax(pose_class, axis=1)  # Assuming it's a classification task
        
        # Get the pose name using the class label
        predicted_pose_name = label_encoder.inverse_transform([pose_class_label[0]])[0]
        
        # Display the predicted pose class name
        cv2.putText(frame, f"Pose: {predicted_pose_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw the landmarks (dots)
        for landmark in pose_landmarks_obj.landmark:
            # Convert to pixel values
            h, w, _ = frame.shape
            x, y = int(landmark.x * w), int(landmark.y * h)
            
            # Draw a circle for each landmark
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Green dot
        # Draw the connections (lines between key points)
        
        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            start = pose_landmarks_obj.landmark[start_idx]
            end = pose_landmarks_obj.landmark[end_idx]
            
            # Convert to pixel values
            h, w, _ = frame.shape
            start_x, start_y = int(start.x * w), int(start.y * h)
            end_x, end_y = int(end.x * w), int(end.y * h)
            
            # Draw a line between the connected landmarks
            cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)  # Green line
    
    # Display the frame with pose label
    cv2.imshow("Pose Classification", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
