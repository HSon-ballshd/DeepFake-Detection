import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
import random


def mass_extract(vid_file_path,output_dir): #extract frames from each video in folder
    for vid in os.listdir(vid_file_path):
        full_path = os.path.join(vid_file_path, vid)
        extract_frames(full_path,output_dir)


def extract_frames(video_path, output_dir, output_size=(256, 256), fps=1, noise_ratio=0.1):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count, frame_rate = 0, int(cap.get(cv2.CAP_PROP_FPS) / fps)
    vid_name = os.path.splitext(os.path.basename(video_path))[0]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_rate == 0:
            output_path = os.path.join(output_dir, f"{vid_name}_frame_{count}.jpg")
            img = cv2.resize(frame, output_size)
            cv2.imwrite(output_path, img)

            # augment ảnh
            augment_data(img, output_path, datagen)

            # tạo ảnh noisy với một xác suất
            if random.random() < noise_ratio:
                std = random.uniform(0.5,1)  # randomize the noise intensity
                noisy_img = gaussian_noise(img, std= std)
                noisy_output = output_path.replace('.jpg', '_noisy.jpg')
                cv2.imwrite(noisy_output, noisy_img)

        count += 1
    cap.release()

"""Insert Function Here, pray this works"""


datagen = ImageDataGenerator(   #datagen for ez image augmenting
    rotation_range = 10,
    width_shift_range = 0.075,
    height_shift_range = 0.075,
    zoom_range = 0.1,
    horizontal_flip = True,
    brightness_range = (0.9,1.1),
    fill_mode = 'nearest'
)


def augment_data(img, output_path,datagen,aug_count = 2): #augment more data
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    i = 0
    for batch in datagen.flow(x, batch_size=1):
        aug_img = array_to_img(batch[0])
        aug_img.save(f"{output_path}_aug_{i}.jpg")
        i += 1
        if i >= aug_count:
            break


def gaussian_noise(img, mean=0, std = 0):
    noise = np.random.normal(mean, std, img.shape).astype(np.uint8)
    return cv2.add(img, noise)


def add_noise(folder_path, ratio=0.1):
    images = os.listdir(folder_path)
    sample_size = int(len(images) * ratio)
    selected = random.sample(images, sample_size)

    for img_file in selected:
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path)
        noisy_img = gaussian_noise(img)
        output_path = img_path.replace('.jpg', '_noisy.jpg')
        cv2.imwrite(output_path, noisy_img)


# How many images to take in one second? That could be a parameter
# (Mainly how well the hardwares can tank)

if __name__ == "__main__":
    mass_extract('newvid','outputimg') #Example usage
    add_noise('outputimg')