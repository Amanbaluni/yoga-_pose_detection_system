import os
import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
from collections import Counter
from imblearn.over_sampling import RandomOverSampler

# Ensure dataset exists and create keypoint files if missing
def create_dataset():
    if not os.path.exists("X_data.npy") or not os.path.exists("y_labels.npy"):
        print("[ERROR] Data files not found! Running keypoint extraction...")
        
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()
        DATASET_PATH = "./dataset"
        classes = sorted(os.listdir(DATASET_PATH))
        classes = [cls for cls in classes if os.path.isdir(os.path.join(DATASET_PATH, cls))]
        X, y = [], []
        
        for class_label in classes:
            class_folder = os.path.join(DATASET_PATH, class_label)
            for img_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                results = pose.process(img_rgb)
                if results.pose_landmarks:
                    keypoints = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
                    
                    # Data Augmentation: Mirror Flip
                    if np.random.rand() > 0.5:
                        keypoints[::3] = -keypoints[::3]  # Flip x-coordinates
                    
                    # Data Augmentation: Random Noise & Scaling
                    keypoints += np.random.normal(0, 0.02, keypoints.shape)  # Slightly higher noise
                    
                    X.append(keypoints)
                    y.append(classes.index(class_label))
        
        X = np.array(X)
        y = np.array(y)
        np.save("X_data.npy", X)
        np.save("y_labels.npy", y)
        print("[INFO] Keypoints extracted and saved!")

# Visualization functions
def plot_class_distribution(y, classes):
    counts = Counter(y)
    plt.bar(classes, [counts[i] for i in range(len(classes))])
    plt.xlabel("Pose Classes")
    plt.ylabel("Number of Samples")
    plt.title("Class Distribution")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

def plot_tsne(X, y):
    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(X.reshape(X.shape[0], -1))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap="rainbow")
    plt.xlabel("TSNE Component 1")
    plt.ylabel("TSNE Component 2")
    plt.title("TSNE Visualization of Pose Data")
    plt.colorbar()
    plt.show()

# Define a custom Attention Layer
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], input_shape[-1]), 
            initializer="random_normal", 
            trainable=True,
            name="attention_W"
        )
        self.b = self.add_weight(
            shape=(input_shape[-1],), 
            initializer="zeros", 
            trainable=True,
            name="attention_b"
        )

    def call(self, inputs):
        attention_scores = tf.nn.softmax(
            tf.linalg.matmul(inputs, self.W) + self.b, axis=1
        )
        return inputs * attention_scores

    def get_config(self):  # Ensure serialization works
        config = super(AttentionLayer, self).get_config()
        return config

# Main script execution
create_dataset()
X = np.load("X_data.npy")
y = np.load("y_labels.npy")

num_samples, num_features = X.shape
num_keypoints = num_features // 3  # 33 keypoints, each with x, y, z
X = X.reshape((num_samples, num_keypoints, 3))

# Normalize keypoints per sample
X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Balance dataset with RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train.reshape(X_train.shape[0], -1), y_train)
X_train = X_train_resampled.reshape(X_train_resampled.shape[0], num_keypoints, 3)
y_train = y_train_resampled

# Define CNN-LSTM model with improvements
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(256, kernel_size=5, activation='relu', input_shape=(num_keypoints, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv1D(128, kernel_size=5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
    tf.keras.layers.LayerNormalization(),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
    tf.keras.layers.LayerNormalization(),
    tf.keras.layers.LSTM(256, return_sequences=True),
    AttentionLayer(),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(set(y)), activation='softmax')
])

# Use learning rate scheduling with Cosine Decay with Warm Restarts (SGDR)
lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=0.0005, first_decay_steps=2000, t_mul=2.0, m_mul=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Compute class weights for imbalance handling
class_weights = {i: 1 / (y_train.tolist().count(i) / len(y_train)) for i in set(y_train)}

# Train with early stopping and learning rate reduction
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
model.fit(X_train, y_train, epochs=150, validation_data=(X_test, y_test), class_weight=class_weights, callbacks=[early_stop, reduce_lr])
tf.keras.models.save_model(model, "yoga_pose_model.h5", overwrite=True)

print("[INFO] Model trained and saved!")

y_pred = np.argmax(model.predict(X_test), axis=1)
print(classification_report(y_test, y_pred))

# Call visualization functions
plot_class_distribution(y, sorted(set(y)))
plot_confusion_matrix(y_test, y_pred, sorted(set(y)))
plot_tsne(X_test, y_test)

print("[INFO] All tasks completed successfully!")
