import cv2
import numpy as np
import os
import mediapipe as mp
import time
from threading import Thread

# Define Mediapipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8, max_num_hands=2)

# Define dataset path
DATA_PATH = "dataset"
os.makedirs(DATA_PATH, exist_ok=True)

# Define labels
LABELS = ["hello", "no", "okay", "thanks", "yes", "peace"]  # Modify as needed

# Define number of samples per gesture
NUM_SAMPLES = 2000  # Increased dataset size for better model accuracy

# Frame skipping for diverse samples
FRAME_SKIP = 3  # Adjust based on movement speed

def preprocess_frame(frame):
    """Apply preprocessing techniques for better hand detection."""
    frame = cv2.flip(frame, 1)  # Flip horizontally
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Adaptive Histogram Equalization
    equalized = clahe.apply(gray)
    processed_frame = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)  # Convert back to 3-channel BGR
    return processed_frame

def collect_data(label):
    """Capture and store hand landmark data for a specific label."""
    label_path = os.path.join(DATA_PATH, label)
    os.makedirs(label_path, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    
    count = 0
    frame_counter = 0  # Track frames to apply skipping
    
    start_time = time.time()
    
    while count < NUM_SAMPLES:
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        if frame_counter % FRAME_SKIP != 0:
            continue  # Skip frames to ensure diverse samples
        
        frame = preprocess_frame(frame)  # Apply preprocessing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_list = [lm.x for lm in hand_landmarks.landmark] + \
                                [lm.y for lm in hand_landmarks.landmark] + \
                                [lm.z for lm in hand_landmarks.landmark]

                # Normalize landmarks (to ensure consistent scale)
                landmark_array = np.array(landmark_list, dtype=np.float32)
                landmark_array = landmark_array / np.linalg.norm(landmark_array)

                # Save landmark data
                npy_path = os.path.join(label_path, f"{count}.npy")
                np.save(npy_path, landmark_array)
                count += 1

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Display progress
        elapsed_time = time.time() - start_time
        cv2.putText(frame, f"Collecting {label}: {count}/{NUM_SAMPLES}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Time: {int(elapsed_time)}s", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow("Collecting Data", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    threads = []
    for label in LABELS:
        print(f"Collecting data for: {label}")
        thread = Thread(target=collect_data, args=(label,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
    
    print("Data collection complete!")
