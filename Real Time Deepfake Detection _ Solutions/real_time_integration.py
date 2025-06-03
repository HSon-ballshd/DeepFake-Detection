import cv2
import numpy as np
from tensorflow.keras.models import load_model

def real_time_detection(model_path):
    model = load_model(model_path)
    cap = cv2.VideoCapture(0)  # Webcam feed
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (128, 128)) / 255.0
        prediction = model.predict(np.expand_dims(resized_frame, axis=0))
        label = "Fake" if prediction > 0.5 else "Real"
        cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Real-Time Deepfake Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
