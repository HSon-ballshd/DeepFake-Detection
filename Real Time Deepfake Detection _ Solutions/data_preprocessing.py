import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def extract_frames(video_path, output_dir, fps=1):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count, frame_rate = 0, int(cap.get(cv2.CAP_PROP_FPS) / fps)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_rate == 0:
            output_path = os.path.join(output_dir, f"frame_{count}.jpg")
            cv2.imwrite(output_path, frame)
        count += 1
    cap.release()

def preprocess_frames(input_dir, output_size=(128, 128)):
    processed_images = []
    for img_file in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, output_size)
        processed_images.append(img / 255.0)  # Normalize to [0, 1]
    return np.array(processed_images)
