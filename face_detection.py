import cv2                          # OpenCV for video + image processing
import numpy as np                  # Numerical operations (arrays, math)
import time                         # For FPS calculation
from collections import deque       # For smoothing predictions over time
from tensorflow.keras.models import load_model  # Load pretrained model

# =========================
# LOAD MODEL + LABELS
# =========================

# Load pretrained emotion model
# compile=False avoids compatibility issues with older Keras models
model = load_model("emotion_model.hdf5", compile=False)

# Emotion labels corresponding to model output indices
emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# =========================
# FACE DETECTION SETUP
# =========================

# Load Haar Cascade face detector (pretrained by OpenCV)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Start webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# =========================
# STABILITY + TRACKING SETUP
# =========================

# Buffer to store recent predictions (for smoothing)
emotion_buffer = deque(maxlen=10)

# Variables to stabilize emotion output (reduce flickering)
stable_emotion = None
stable_count = 0

# For FPS calculation
prev_time = time.time()

# =========================
# MAIN LOOP
# =========================

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        break  # Exit if frame not captured

    # Resize frame for faster processing (performance optimization)
    frame = cv2.resize(frame, (640, 480))

    # Convert frame to grayscale (face detection works better in grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,   # How much image size is reduced at each scale
        minNeighbors=3,    # Higher = fewer false positives
        minSize=(30, 30)   # Minimum face size
    )

    # =========================
    # FPS CALCULATION
    # =========================

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # =========================
    # FACE PROCESSING
    # =========================

    if len(faces) > 0:
        # Select the largest face (main subject)
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])

        # Draw bounding box around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # -------------------------
        # PREPROCESS FACE IMAGE
        # -------------------------

        # Crop face region from grayscale image
        face_img = gray[y:y+h, x:x+w]

        # Improve contrast (helps with lighting variations)
        face_img = cv2.equalizeHist(face_img)

        # Resize to match model input size (64x64)
        face_img = cv2.resize(face_img, (64, 64))

        # Normalize pixel values to range [0,1]
        face_img = face_img / 255.0

        # Reshape to match model input shape (batch_size, width, height, channels)
        face_img = np.reshape(face_img, (1, 64, 64, 1))

        # -------------------------
        # EMOTION PREDICTION
        # -------------------------

        # Get prediction probabilities from model
        predictions = model.predict(face_img, verbose=0)[0]

        # Add prediction to buffer for smoothing
        emotion_buffer.append(predictions)

        # Average predictions over recent frames (reduces noise)
        avg_predictions = np.mean(emotion_buffer, axis=0)

        # Get most likely emotion index
        emotion_index = np.argmax(avg_predictions)
        emotion_text = emotions[emotion_index]

        # Get confidence (highest probability)
        confidence = np.max(avg_predictions)

        # -------------------------
        # STABILITY LOGIC (ANTI-FLICKER)
        # -------------------------

        # Check if emotion is consistent across frames
        if emotion_text == stable_emotion:
            stable_count += 1
        else:
            stable_count = 0

        # Only update display if stable for several frames
        if stable_count > 3:
            display_emotion = emotion_text
        else:
            display_emotion = stable_emotion if stable_emotion else emotion_text

        # Update tracked emotion
        stable_emotion = emotion_text

        # -------------------------
        # CONFIDENCE FILTER
        # -------------------------

        # If confidence is too low, mark as uncertain
        if confidence < 0.4:
            display_emotion = "Uncertain"

        # -------------------------
        # DISPLAY RESULTS
        # -------------------------

        # Show main emotion label above face
        label = f"{display_emotion} ({confidence:.2f})"
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # Show all emotion probabilities (for debugging/demo)
        for i, emo in enumerate(emotions):
            text = f"{emo}: {avg_predictions[i]:.2f}"
            cv2.putText(frame, text, (10, 60 + i*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # -------------------------
        # LOGGING (FOR EVALUATION)
        # -------------------------

        # Save emotion + confidence to file
        with open("emotion_log.txt", "a") as f:
            f.write(f"{display_emotion},{confidence:.2f}\n")

    else:
        # If no face detected, show message
        cv2.putText(frame, "No face detected", (180, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    # =========================
    # DISPLAY FPS
    # =========================

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Show final frame
    cv2.imshow("Emotion Detection (Robust)", frame)

    # Exit when ESC key is pressed
    if cv2.waitKey(1) == 27:
        break

# =========================
# CLEANUP
# =========================

cap.release()            # Release webcam
cv2.destroyAllWindows() # Close all OpenCV windows

# Note, comments were generated by ChatGPT to explain the code in detail