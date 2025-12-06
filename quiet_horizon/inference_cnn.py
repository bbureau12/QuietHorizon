# inference_cnn.py
import argparse
import os
import sys
from typing import Dict

import cnn_generation.audio_standardizer as audio_standardizer
import librosa
import cv2 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Must match training setup
IMG_SIZE = (128, 128)
CLASS_NAMES = ["anthro", "nature"]  # 0=anthro, 1=nature

def load_melspec_from_audio(path: str) -> np.ndarray:
    """
    Load raw audio (wav/mp3/flac/ogg), standardize it for CNN,
    compute mel-spectrogram, and return a CNN-ready image array.

    Output shape: (1, 128, 128, 3)
    """
    # 1. Standardize audio
    y, sr = audio_standardizer.load_and_standardize(path)

    # 2. Compute mel-spectrogram (must match training!)
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        power=2.0,
    )
    S = librosa.power_to_db(S, ref=np.max)

    # Normalize 0–255 for CNN image input
    S_norm = 255 * (S - S.min()) / (S.max() - S.min())
    S_norm = S_norm.astype(np.uint8)

    # Resize to (128,128) if necessary
    img = cv2.resize(S_norm, (128, 128))

    # Convert to RGB (stack 3 channels)
    img_rgb = np.stack([img, img, img], axis=-1)

    # Add batch dimension
    return np.expand_dims(img_rgb, axis=0)


def predict_audio(model: keras.Model, audio_path: str, threshold: float = 0.5):
    x = load_melspec_from_audio(audio_path)
    probs = model.predict(x, verbose=0)
    prob_nature = float(probs[0][0])
    prob_anthro = 1.0 - prob_nature
    label = "nature" if prob_nature >= threshold else "anthro"
    return {
        "audio_path": audio_path,
        "prob_nature": prob_nature,
        "prob_anthro": prob_anthro,
        "pred_label": label,
    }

def load_spectrogram_image(image_path: str) -> np.ndarray:
    """
    Load a spectrogram PNG/JPG and prepare it for the CNN.

    Returns:
      np.ndarray with shape (1, 128, 128, 3)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    img = keras.utils.load_img(image_path, target_size=IMG_SIZE)
    arr = keras.utils.img_to_array(img)  # float32, shape (H, W, 3), values 0–255
    arr = np.expand_dims(arr, axis=0)    # shape (1, H, W, 3)

    # NOTE: We DO NOT rescale here because the model's first layer
    # is layers.Rescaling(1./255).
    return arr


def predict_image(
    model: keras.Model,
    image_path: str,
    threshold: float = 0.5,
) -> Dict:
    """
    Run inference on a single spectrogram image.

    Returns a dict with:
      - image_path
      - prob_nature
      - prob_anthro
      - predicted_label
      - threshold
    """
    x = load_spectrogram_image(image_path)

    # Model outputs a single sigmoid probability:
    #   p = P(label == 1) = P("nature")
    probs = model.predict(x, verbose=0)
    prob_nature = float(probs[0][0])
    prob_anthro = 1.0 - prob_nature

    # Decide class based on threshold on "nature" probability
    if prob_nature >= threshold:
        pred_idx = 1  # "nature"
    else:
        pred_idx = 0  # "anthro"

    pred_label = CLASS_NAMES[pred_idx]

    return {
        "image_path": image_path,
        "prob_nature": prob_nature,
        "prob_anthro": prob_anthro,
        "pred_idx": pred_idx,
        "pred_label": pred_label,
        "threshold": threshold,
    }


IMG_SIZE = (128, 128)
CLASS_NAMES = ["anthro", "nature"]  # 0=anthro, 1=nature

def build_model() -> tf.keras.Model:
    """
    Rebuild the CNN architecture exactly as in training_cnn.
    """
    input_shape = IMG_SIZE + (3,)

    model = keras.Sequential([
        layers.Rescaling(1./255, input_shape=input_shape),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])

    # Compile isn't strictly needed for inference, but it's harmless
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )

    return model


def load_model(weights_path: str) -> tf.keras.Model:
    """
    Build a fresh model and load weights from .weights.h5.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weight file not found: {weights_path}")

    model = build_model()
    model.load_weights(weights_path)
    return model

def main():
    # ------------------------------------
    # DEBUG MODE: run without argparse
    # ------------------------------------
    # if len(sys.argv) == 1:   # No command-line args → debug mode
    #     print("Running in DEBUG mode...\n")

    #     model_path = "models/quiet_horizon_cnn.weights.h5"
    #     audio_path = r"D:\Projects\QuietHorizon\quiet_horizon\dataset_cnn\anthro\home_improvement\240916_0319_19_09_59.wav"
    #     threshold = 0.5

    #     model = load_model(model_path)
    #     result = predict_audio(model, audio_path, threshold)

    #     print(f"Input (audio): {result['audio_path']}")
    #     print(f"  P(nature) = {result['prob_nature']:.3f}")
    #     print(f"  P(anthro) = {result['prob_anthro']:.3f}")
    #     print(f"  Predicted label = {result['pred_label'].upper()}")
    #     return

    parser = argparse.ArgumentParser(
        description="Run QuietHorizon CNN inference on a spectrogram image or audio file."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to input spectrogram image OR audio file.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/quiet_horizon_cnn.weights.h5",
        help="Path to the trained weights file (.weights.h5).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold on P(nature). Default: 0.5",
    )
    parser.add_argument(
        "--audio",
        action="store_true",
        help="Interpret input as audio instead of image.",
    )

    args = parser.parse_args()

    model = load_model(args.model_path)

    if args.audio:
        result = predict_audio(model, args.input_path, threshold=args.threshold)
        print(f"\nInput (audio): {result['audio_path']}")
    else:
        result = predict_image(model, args.input_path, threshold=args.threshold)
        print(f"\nInput (image): {result['image_path']}")

    print(f"  P(nature) = {result['prob_nature']:.3f}")
    print(f"  P(anthro) = {result['prob_anthro']:.3f}")
    print(f"  Threshold = {args.threshold:.2f}")
    print(f"  → Predicted label: {result['pred_label'].upper()}")

if __name__ == "__main__":
    main()
