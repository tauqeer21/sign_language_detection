import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
import threading

LABELS = ["awesome", "fine", "hello", "help", "i love you", "no", "okay", "peace", "thanks", "yes"]
TEXT_SIZE = 1.2  # Adjust text size here
TEXT_COLOR = (0, 0, 0)  # Black text
BG_COLOR = (255, 255, 255)  # White background
PADDING = 5  # Space around text

# Load model only once
model = tf.keras.models.load_model("sign_language_model.keras")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Threaded video capture class
class VideoCaptureThread:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.ret = False
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()

cap = VideoCaptureThread(0)  # Change to "rtsp://..." for IP camera

prev_time = time.time()
pred_buffers = {}

display_text = ""

with mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=4) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        start_time = time.time()
        frame = cv2.resize(frame, (1280, 720))  # Resize to optimize processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        h, w, _ = frame.shape

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32).flatten()
                
                if landmarks.shape[0] != model.input_shape[1]:
                    continue
                
                landmarks /= np.max(landmarks)  # Normalize in place
                prediction = model.predict_on_batch(landmarks[None, :])
                predicted_index = int(np.argmax(prediction))
                confidence = float(prediction[0][predicted_index])
                
                pred_buffers.setdefault(idx, []).append(predicted_index)
                if len(pred_buffers[idx]) > 10:
                    pred_buffers[idx].pop(0)

                final_prediction = max(set(pred_buffers[idx]), key=pred_buffers[idx].count)
                display_text = f"{LABELS[final_prediction]} ({confidence * 100:.1f}%)"
                
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        curr_time = time.time()
        fps = int(1 / (curr_time - prev_time))
        prev_time = curr_time
        
        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, 2)
        
        # Define rectangle coordinates
        rect_x1, rect_y1 = 25, 60 - text_height - PADDING
        rect_x2, rect_y2 = 35 + text_width, 60 + PADDING
        
        # Draw filled rectangle (white background)
        cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), BG_COLOR, -1)
        
        # Put text on top of rectangle
        cv2.putText(frame, display_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, TEXT_COLOR, 2)
        cv2.putText(frame, f"FPS: {fps}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow("Sign Language Detection", frame)

        print(f"Processing time per frame: {time.time() - start_time:.3f}s")

        if cv2.waitKey(1) == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
