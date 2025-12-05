# inference_cnn.py
import argparse
import os
from typing import Dict

import cnn_generation.audio_standardizer as audio_standardizer
import librosa
import cv2 
import numpy as np
import tensorflow as tf
from tensorflow import keras

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


def load_model(model_path: str) -> keras.Model:
    """
    Load the trained QuietHorizon CNN model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = keras.models.load_model(model_path)
    return model

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


def main():
    parser = argparse.ArgumentParser(
        description="Run QuietHorizon CNN inference on a spectrogram image."
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to input spectrogram image (PNG/JPG).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/quiet_horizon_cnn.keras",
        help="Path to the trained .keras model file.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold on P(nature). Default: 0.5",
    )

    parser.add_argument(
    "input_path",
    type=str,
    help="Path to input spectrogram image OR audio file."
    )
    parser.add_argument(
        "--audio",
        action="store_true",
        help="Interpret input as audio instead of image."
    )


    args = parser.parse_args()

    model = load_model(args.model_path)
    if args.audio:
     result = predict_audio(model, args.input_path, threshold=args.threshold)
    else:
     result = predict_image(model, args.input_path, threshold=args.threshold)


    print(f"\nImage: {result['image_path']}")
    print(f"  P(nature) = {result['prob_nature']:.3f}")
    print(f"  P(anthro) = {result['prob_anthro']:.3f}")
    print(f"  Threshold = {result['threshold']:.2f}")
    print(f"  → Predicted label: {result['pred_label'].upper()}")


if __name__ == "__main__":
    main()
