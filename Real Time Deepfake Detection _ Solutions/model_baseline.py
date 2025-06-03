import tensorflow as tf
from tensorflow.keras import models, layers

def build_baseline_model(input_shape=(128, 128, 3)):
    model = models.Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
