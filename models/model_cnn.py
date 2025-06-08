# model_cnn.py
from pathlib import Path
import json, matplotlib.pyplot as plt, tensorflow as tf

# ── paths ─────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
LIST_DIR   = REPO_ROOT / "train_test_data"
OUT_DIR    = SCRIPT_DIR / "model_cnn_artifacts"
OUT_DIR.mkdir(exist_ok=True)

IMG_SIZE, BATCH, EPOCHS = 128, 32, 8          # nhỏ hơn để debug nhanh
AUTOTUNE = tf.data.AUTOTUNE

# ── dataset ────────────────────────────────────────────────────────
def load_list(txt_name: str):
    paths, labels = [], []
    with open(LIST_DIR / txt_name) as f:
        for rel in f:
            abs_path = REPO_ROOT / rel.strip()
            paths.append(str(abs_path))
            labels.append(0.0 if 'original' in abs_path.parts else 1.0)
    return paths, labels

def make_dataset(txt_name: str, subset_frac=0.3):
    paths, labels = load_list(txt_name)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if subset_frac < 1.0:
        ds = ds.take(int(len(paths) * subset_frac))

    def _pre(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.cast(img, tf.float32) / 255.0
        return img, tf.expand_dims(label, -1)

    return (ds.shuffle(1000)
              .map(_pre, num_parallel_calls=AUTOTUNE)
              .batch(BATCH)
              .prefetch(AUTOTUNE))

# ── cnn model ─────────────────────────────────────────────────────
def build_model():
    from tensorflow.keras import layers
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    m = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    

    m.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return m

def export_architecture(model):
    from keras.utils import plot_model	
    (OUT_DIR / "cnn_arch.json").write_text(model.to_json(indent=2))
    try:
        plot_model(model,
                   to_file=str(OUT_DIR / "cnn_arch.png"),
                   show_shapes=True,
                   dpi=120,
                   rankdir="LR")
    except Exception as e:
        print("PNG graph skipped:", e)

def plot_history(hist):
    for metric, fname in [("accuracy", "acc_curve.png"), ("loss", "loss_curve.png")]:
        plt.figure(figsize=(5,3))
        plt.plot(hist[metric],     label=f"train {metric}")
        plt.plot(hist[f"val_{metric}"], label=f"val {metric}")
        plt.xlabel("epoch"); plt.ylabel(metric); plt.legend(); plt.tight_layout()
        plt.savefig(OUT_DIR / fname); plt.close()

# ── main ───────────────────────────────────────────────────────────
def main():
    train_ds = make_dataset("train.txt", subset_frac=0.3)  # nhỏ để debug
    val_ds   = make_dataset("val.txt",   subset_frac=0.3)
    test_ds  = make_dataset("test.txt",  subset_frac=1.0)

    model = build_model()
    export_architecture(model)

    hist = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)

    model.save(OUT_DIR / "cnn_model.h5")
    json.dump(hist.history, open(OUT_DIR / "hist.json", "w"))
    plot_history(hist.history)

    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"\n🔎  Test accuracy = {acc:.3f} | loss = {loss:.3f}")
    print("📂  outputs →", OUT_DIR.resolve())

if __name__ == "__main__":
    main()

#  Test accuracy = 0.433 | loss = 0.812