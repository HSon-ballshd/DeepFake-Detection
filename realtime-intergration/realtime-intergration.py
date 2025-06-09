import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('cnn_model.h5')  # Update with your model path

# Define any preprocessing required by your model
def preprocess_frame(frame):
    # Example: resize and normalize
    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    input_img = preprocess_frame(frame)

    # Predict
    prediction = model.predict(input_img)
    label = "Fake" if prediction[0][0] > 0.5 else "Real"

    # Display result
    cv2.putText(frame, f"Prediction: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('DeepFake Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()