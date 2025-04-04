import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ensure TensorFlow uses the correct execution mode
tf.keras.backend.clear_session()

# Define dataset path and labels
DATA_DIR = "dataset"
LABELS = ["awesome","fine","hello","help","i love you","no","okay","peace","thanks","yes"]

X, y = [], []

# Load dataset
for label in LABELS:
    label_dir = os.path.join(DATA_DIR, label)
    label_idx = LABELS.index(label)

    for file in os.listdir(label_dir):
        file_path = os.path.join(label_dir, file)
        landmarks = np.load(file_path)

        X.append(landmarks)
        y.append(label_idx)

# Convert lists to NumPy arrays
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

# Normalize input data
X = X / np.max(X)

# Apply data augmentation (Gaussian Noise & Scaling)
def augment_data(data):
    noise = np.random.normal(0, 0.02, data.shape)  # Small noise
    scale_factor = np.random.uniform(0.9, 1.1)  # Random scaling
    return (data + noise) * scale_factor

X_augmented = np.array([augment_data(sample) for sample in X])
X = np.concatenate((X, X_augmented))
y = np.concatenate((y, y))  # Duplicate labels for augmented data

# Split dataset into training & validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define the improved model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, input_shape=(X.shape[1],)),
    tf.keras.layers.LeakyReLU(alpha=0.1),  # Better activation function
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(256),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(128),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(len(LABELS), activation="softmax")
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Define callbacks for better training
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.3, min_lr=1e-5)

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val), callbacks=[early_stopping, reduce_lr])

# Save the model
model.save("sign_language_model.keras")

print("Optimized model training complete and saved successfully.")

# Plot training & validation accuracy values
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()