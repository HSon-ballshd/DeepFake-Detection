# mode_baseline.py
from pathlib import Path
import json, matplotlib.pyplot as plt, tensorflow as tf

SCRIPT_DIR = Path(__file__).resolve().parent          
REPO_ROOT  = SCRIPT_DIR.parent                        
LIST_DIR   = REPO_ROOT / "train_test_data"            
OUT_DIR    = SCRIPT_DIR / "model_baseline_artifacts"          
OUT_DIR.mkdir(exist_ok=True)

IMG_SIZE, BATCH, EPOCHS = 256, 64, 10
AUTOTUNE = tf.data.AUTOTUNE

def load_list(txt_name: str):
    paths, labels = [], []
    with open(LIST_DIR / txt_name) as f:
        for rel in f:
            abs_path = REPO_ROOT / rel.strip()
            paths.append(str(abs_path))
            labels.append(0.0 if 'original' in abs_path.parts else 1.0)
    return paths, labels

def make_dataset(txt_name: str):
    paths, labels = load_list(txt_name)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _pre(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.cast(img, tf.float32) / 255.0
        return tf.reshape(img, [-1]), tf.expand_dims(label, -1)

    return (ds.shuffle(len(paths))
              .map(_pre, num_parallel_calls=AUTOTUNE)
              .batch(BATCH).prefetch(AUTOTUNE))

def build_model():
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE*IMG_SIZE*3,)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1,   activation='sigmoid')
    ])
    m.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return m

def export_architecture(model):
    from keras.utils import plot_model	
    # JSON
    (OUT_DIR / "baseline_arch.json").write_text(model.to_json(indent=2))

    plot_model(
        model,
        to_file=str(OUT_DIR / "baseline_arch.png"),
        show_shapes=True,
        show_layer_names=True,
        dpi=120,
        rankdir="LR"
    )   

def plot_history(hist_dict):
    plt.figure(figsize=(5,3))
    plt.plot(hist_dict["accuracy"], label="train")
    plt.plot(hist_dict["val_accuracy"], label="val")
    plt.xlabel("epoch"); plt.ylabel("acc"); plt.legend(); plt.tight_layout()
    plt.savefig(OUT_DIR / "acc_curve.png"); plt.close()

    plt.figure(figsize=(5,3))
    plt.plot(hist_dict["loss"], label="train")
    plt.plot(hist_dict["val_loss"], label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
    plt.savefig(OUT_DIR / "loss_curve.png"); plt.close()

def main():
    train_ds = make_dataset("train.txt")
    val_ds   = make_dataset("val.txt")
    test_ds  = make_dataset("test.txt")

    model = build_model()
    model.summary()
    export_architecture(model)

    hist = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)

    # save artifacts
    model.save(OUT_DIR / "baseline_nn.h5")
    json.dump(hist.history, open(OUT_DIR / "hist.json", "w"))
    plot_history(hist.history)

    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"\n  Test accuracy = {acc:.3f} | loss = {loss:.3f}")
    print("  All outputs →", OUT_DIR.resolve())

if __name__ == "__main__":
    main()

#    Test accuracy = 0.500 | loss = 0.693