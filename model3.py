import cv2
import numpy as np
import mediapipe as mp
import os
from sklearn.preprocessing import LabelEncoder
from keras import layers, models
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from keras import backend as K
import random

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Function to extract keypoints from an image
def extract_keypoints(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks).flatten()
    else:
        return None

# Function to visualize keypoints on the image
def visualize_keypoints(image, keypoints):
    for i in range(33):  # 33 keypoints in the pose
        x, y, z = keypoints[i*3:(i+1)*3]
        x = int(x * image.shape[1])
        y = int(y * image.shape[0])
        x = min(max(x, 0), image.shape[1] - 1)
        y = min(max(y, 0), image.shape[0] - 1)
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Green circle
    return image

# Augmentation Function for Keypoints
def augment_keypoints(keypoints, image):
    keypoints_reshaped = keypoints.reshape(-1, 3)

    # Random rotation angle
    rotation_angle = random.uniform(-30, 30)

    # Random translation
    tx = random.uniform(-10, 10)
    ty = random.uniform(-10, 10)

    # Random mirroring
    flip_prob = random.uniform(0, 1)
    flip = False
    if flip_prob > 0.5:
        flip = True
        image = cv2.flip(image, 1)

    center = (keypoints_reshaped[:, 0].mean(), keypoints_reshaped[:, 1].mean())
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

    rotated_keypoints = []
    for point in keypoints_reshaped:
        x, y, z = point
        rotated_x, rotated_y = np.dot(rotation_matrix, np.array([x, y, 1]))[:2]
        rotated_keypoints.append([rotated_x, rotated_y, z])

    rotated_keypoints = np.array(rotated_keypoints)
    rotated_keypoints[:, 0] += tx
    rotated_keypoints[:, 1] += ty

    if flip:
        image_width = image.shape[1]
        rotated_keypoints[:, 0] = image_width - rotated_keypoints[:, 0]

    return rotated_keypoints.flatten(), image

# Adjusted Focal Loss with Customizable Alpha and Gamma
def focal_loss(gamma=3., alpha=0.5):
    def focal_loss_fixed(y_true, y_pred):
        y_true = K.cast(y_true, K.floatx())
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.sum(loss, axis=1)
    return focal_loss_fixed

# Prepare dataset function with normalization
def prepare_data(dataset_dir):
    keypoints = []
    labels = []
    skipped_images = 0

    for subdir in os.listdir(dataset_dir):
        subdir_path = os.path.join(dataset_dir, subdir)
        if os.path.isdir(subdir_path):
            for image_name in os.listdir(subdir_path):
                image_path = os.path.join(subdir_path, image_name)

                image = cv2.imread(image_path)
                if image is None:
                    skipped_images += 1
                    continue

                keypoint = extract_keypoints(image)
                if keypoint is not None:
                    # Normalize the image and keypoints
                    image = image / 255.0  # Normalize image
                    keypoints.append(keypoint)
                    labels.append(subdir)

                    augmented_keypoint, augmented_image = augment_keypoints(keypoint, image)
                    keypoints.append(augmented_keypoint)
                    labels.append(subdir)
                else:
                    skipped_images += 1

    keypoints = np.array(keypoints)
    labels = np.array(labels)

    # Normalize keypoints (between 0 and 1)
    keypoints = (keypoints - np.min(keypoints)) / (np.max(keypoints) - np.min(keypoints))

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    print(f"Total skipped images: {skipped_images}")
    return keypoints, labels, label_encoder

# Prepare the dataset
dataset_dir = './dataset'  # Change this to your dataset directory
keypoints, labels, label_encoder = prepare_data(dataset_dir)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Build the model
model = models.Sequential()

# Add deeper CNN layers with BatchNormalization
model.add(layers.Reshape((33, 3), input_shape=(keypoints.shape[1],)))
model.add(layers.Conv1D(64, 3, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Conv1D(128, 3, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Conv1D(256, 3, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling1D(pool_size=2))

# Add LSTM layers
model.add(layers.LSTM(128, return_sequences=True))
model.add(layers.LSTM(128))

# Dropout for regularization
model.add(layers.Dropout(0.5))

# Fully connected layers
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))

# Output layer
model.add(layers.Dense(len(np.unique(labels)), activation='softmax'))

# Compile the model with Focal Loss
model.compile(optimizer='adam', loss=focal_loss(gamma=2., alpha=0.25), metrics=['accuracy'])

# Define callbacks
checkpoint = ModelCheckpoint('pose_classifier.h5', save_best_only=True, monitor='val_loss', mode='min')
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)

# Train the model with class weights
model.fit(keypoints, labels, epochs=20, batch_size=32, validation_split=0.2, 
          callbacks=[checkpoint, lr_scheduler], class_weight=class_weight_dict)

# Save the final model
model.save('pose_classifier_final.h5')

# Print class names
print(f"Classes: {label_encoder.classes_}")

# Predict pose from an image
def predict_pose(image_path, model, label_encoder):
    image = cv2.imread(image_path)
    keypoint = extract_keypoints(image)

    if keypoint is not None:
        keypoint = np.expand_dims(keypoint, axis=0)
        prediction = model.predict(keypoint)
        predicted_class = np.argmax(prediction, axis=1)
        predicted_pose = label_encoder.inverse_transform([predicted_class[0]])

        return predicted_pose[0]
    else:
        print("No keypoints detected.")
        return None

# Example usage: Predict pose from an image
image_path = './sample2.jpg'  # Replace with your image path
predicted_pose = predict_pose(image_path, model, label_encoder)
print(f"Predicted Pose: {predicted_pose}")
