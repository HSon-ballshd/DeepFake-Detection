from data_preprocessing import extract_frames, preprocess_frames
from model_cnn import build_cnn
from evaluation import evaluate_model
from real_time_integration import real_time_detection

if __name__ == "__main__":
    # Preprocess Data
    extract_frames('input_video.mp4', 'frames/')
    X = preprocess_frames('frames/')
    y = [0, 1] * (len(X) // 2)  # Dummy labels, replace with actual labels

    # Train Model
    cnn_model = build_cnn()
    cnn_model.fit(X, y, epochs=5, validation_split=0.2)

    # Save and Evaluate
    cnn_model.save('saved_model/cnn_model.h5')
    evaluate_model(cnn_model, X, y)

    # Real-Time Detection
    real_time_detection('saved_model/cnn_model.h5')
