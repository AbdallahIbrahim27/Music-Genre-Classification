"""
cnn_approach.py
---------------
Approach 2: Mel spectrogram images → CNN (Keras/TensorFlow)

Two sub-approaches:
  A) Custom CNN from scratch
  B) Transfer Learning with MobileNetV2 (bonus)

Outputs:
  - Training curves (loss + accuracy)
  - Confusion matrix
  - Best model saved as cnn_best_model.keras
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # suppress TF info logs

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                         ModelCheckpoint)
from sklearn.metrics import classification_report, confusion_matrix

from spectrogram_gen import generate_spectrograms

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_ROOT    = 'data'
IMG_DIR      = 'images'
IMG_SIZE     = 128
BATCH_SIZE   = 32
EPOCHS       = 50
SEED         = 42
NUM_CLASSES  = 10
GENRES       = ['blues', 'classical', 'country', 'disco', 'hiphop',
                'jazz', 'metal', 'pop', 'reggae', 'rock']
# ──────────────────────────────────────────────────────────────────────────────


# ── 1. Data preparation ───────────────────────────────────────────────────────
def get_data_generators(img_dir: str):
    """Return (train_gen, val_gen, test_gen) using ImageDataGenerator."""
    # Augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.20,
        rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=False,          # spectrograms are time-ordered
        zoom_range=0.05,
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        img_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        seed=SEED,
        shuffle=True,
    )
    val_gen = train_datagen.flow_from_directory(
        img_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        seed=SEED,
        shuffle=False,
    )
    return train_gen, val_gen


# ── 2. Custom CNN ─────────────────────────────────────────────────────────────
def build_custom_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                     num_classes=NUM_CLASSES) -> keras.Model:
    inp = keras.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inp, out, name='CustomCNN')
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ── 3. Transfer Learning (MobileNetV2) ────────────────────────────────────────
def build_transfer_model(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                         num_classes=NUM_CLASSES) -> keras.Model:
    base = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
    )
    base.trainable = False          # freeze base initially

    inp = keras.Input(shape=input_shape)
    x   = keras.applications.mobilenet_v2.preprocess_input(inp * 255)
    x   = base(x, training=False)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dense(256, activation='relu')(x)
    x   = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inp, out, name='MobileNetV2_Transfer')
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ── 4. Training ───────────────────────────────────────────────────────────────
def train_model(model: keras.Model, train_gen, val_gen,
                model_path: str, epochs: int = EPOCHS):
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint(model_path, save_best_only=True, verbose=1),
    ]
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    return history


# ── 5. Evaluation helpers ─────────────────────────────────────────────────────
def plot_history(history, title: str, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['loss'],     label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title(f'{title} — Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot(history.history['accuracy'],     label='Train Acc')
    axes[1].plot(history.history['val_accuracy'], label='Val Acc')
    axes[1].set_title(f'{title} — Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Training curves → {out_path}")


def evaluate_model(model: keras.Model, val_gen, class_names, title: str, tag: str):
    val_gen.reset()
    y_true = val_gen.classes
    y_pred = np.argmax(model.predict(val_gen, verbose=0), axis=1)

    acc = (y_true == y_pred).mean()
    print(f"\n  {title}")
    print(f"  Val accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'{title} — Confusion Matrix')
    plt.tight_layout()
    path = f'cm_{tag}.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Confusion matrix → {path}")
    return acc


# ── 6. Main ───────────────────────────────────────────────────────────────────
def main():
    # Generate spectrograms if needed
    if not os.path.isdir(IMG_DIR) or not any(
            os.scandir(os.path.join(IMG_DIR, GENRES[0]))
            if os.path.isdir(os.path.join(IMG_DIR, GENRES[0])) else []):
        print("Generating spectrograms ...")
        generate_spectrograms(DATA_ROOT, IMG_DIR)

    train_gen, val_gen = get_data_generators(IMG_DIR)
    class_names = list(train_gen.class_indices.keys())
    print(f"\nClasses: {class_names}")
    print(f"Train: {train_gen.n}  |  Val: {val_gen.n}")

    results = {}

    # ── A) Custom CNN ──────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  A) Custom CNN")
    print("═"*60)
    cnn = build_custom_cnn()
    cnn.summary()
    hist_cnn = train_model(cnn, train_gen, val_gen, 'custom_cnn.keras')
    plot_history(hist_cnn, 'Custom CNN', 'cnn_history.png')
    acc_cnn = evaluate_model(cnn, val_gen, class_names, 'Custom CNN', 'custom_cnn')
    results['Custom CNN'] = acc_cnn

    # ── B) Transfer Learning ──────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  B) MobileNetV2 Transfer Learning (Bonus)")
    print("═"*60)
    tl = build_transfer_model()
    tl.summary()
    # Phase 1: Train head only
    hist_tl = train_model(tl, train_gen, val_gen, 'transfer_cnn.keras', epochs=30)

    # Phase 2: Fine-tune last 20 layers
    print("\nFine-tuning last 20 layers ...")
    tl.layers[3].trainable = True   # base model is layer index 3
    for layer in tl.layers[3].layers[:-20]:
        layer.trainable = False
    tl.compile(optimizer=keras.optimizers.Adam(1e-5),
                loss='categorical_crossentropy', metrics=['accuracy'])
    hist_ft = train_model(tl, train_gen, val_gen, 'transfer_cnn_finetuned.keras', epochs=20)

    plot_history(hist_ft, 'MobileNetV2 Fine-tuned', 'transfer_history.png')
    acc_tl = evaluate_model(tl, val_gen, class_names,
                             'MobileNetV2 Transfer Learning', 'transfer_cnn')
    results['MobileNetV2 Transfer'] = acc_tl

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  CNN Results Summary")
    print("═"*60)
    for name, acc in results.items():
        print(f"  {name:30s}: {acc*100:.2f}%")

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ['#9C27B0', '#E91E63']
    bars = ax.bar(results.keys(), [v*100 for v in results.values()], color=colors)
    ax.set_ylim(0, 100); ax.set_ylabel('Accuracy (%)')
    ax.set_title('CNN Approach — Model Comparison')
    for bar, val in zip(bars, results.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val*100:.1f}%', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig('cnn_comparison.png', dpi=150)
    plt.close()
    print("  Comparison chart → cnn_comparison.png")

    return results


if __name__ == '__main__':
    main()
