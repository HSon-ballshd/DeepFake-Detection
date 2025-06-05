import os
from tensorflow.keras.models import load_model

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)

def load_saved_model(path):
    return load_model(path)
