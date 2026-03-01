"""
Model loading utilities with caching for QuietHorizon
"""
import streamlit as st
from pathlib import Path
import tensorflow as tf
from huggingface_hub import hf_hub_download
import sys

# Add frontend to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


@st.cache_resource
def load_model(model_path=None):
    """
    Load the QuietHorizon CNN model with Streamlit caching.

    Args:
        model_path: Path to local model file. If None, uses default local model.

    Returns:
        Loaded TensorFlow Keras model
    """
    if model_path is None:
        selected_path = Path(config.DEFAULT_MODEL_PATH)
    else:
        selected_path = Path(model_path)

    # Auto mode fallback resolution.
    if not selected_path.exists() and model_path is None:
        if config.DEFAULT_WEIGHTS_PATH.exists():
            selected_path = config.DEFAULT_WEIGHTS_PATH
            st.info(f"Using local weights model: {selected_path}")
        elif config.DEFAULT_KERAS_PATH.exists():
            selected_path = config.DEFAULT_KERAS_PATH
            st.info(f"Using local keras model: {selected_path}")
        else:
            st.info("Model not found locally. Downloading from Hugging Face...")
            selected_path = Path(download_model_from_hf())
    elif not selected_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {selected_path}")

    try:
        # Support both full .keras models and .weights.h5 checkpoints.
        if str(selected_path).lower().endswith(".weights.h5"):
            try:
                from quiet_horizon.inference_cnn import load_model as load_weights_model
                model = load_weights_model(str(selected_path))
            except Exception as weights_error:
                if config.DEFAULT_KERAS_PATH.exists():
                    st.warning(
                        f"Failed to load weights model ({weights_error}). "
                        f"Falling back to keras model: {config.DEFAULT_KERAS_PATH}"
                    )
                    model = tf.keras.models.load_model(str(config.DEFAULT_KERAS_PATH))
                else:
                    raise
        else:
            model = tf.keras.models.load_model(str(selected_path))

        st.success("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        raise


def download_model_from_hf():
    """
    Download the pretrained model from Hugging Face Hub.

    Returns:
        Path to downloaded model file
    """
    try:
        model_path = hf_hub_download(
            repo_id=config.HF_REPO_ID,
            filename=config.HF_MODEL_FILENAME,
            cache_dir=str(config.MODELS_DIR),
        )
        return model_path
    except Exception as e:
        st.error(f"Error downloading model from Hugging Face: {e}")
        st.info(
            f"Please ensure internet access and that the model exists at {config.HF_REPO_ID}"
        )
        raise


def predict_from_spectrogram(model, spectrogram_image):
    """
    Make a prediction from a spectrogram image.

    Args:
        model: Loaded TensorFlow model
        spectrogram_image: Numpy array of shape (128, 128, 3) or (1, 128, 128, 3)

    Returns:
        dict with prediction results
    """
    import numpy as np

    # Ensure correct shape
    if len(spectrogram_image.shape) == 3:
        spectrogram_image = np.expand_dims(spectrogram_image, axis=0)

    # Make prediction
    prediction = model.predict(spectrogram_image, verbose=0)
    prob_nature = float(prediction[0][0])
    prob_anthro = 1 - prob_nature

    # Keep API backward compatible for tests and existing callers.
    predicted_label = "nature" if prob_nature >= 0.5 else "anthro"
    confidence = prob_nature if prob_nature >= 0.5 else prob_anthro

    return {
        "prob_nature": prob_nature,
        "prob_anthro": prob_anthro,
        "predicted_label": predicted_label,
        "confidence": confidence,
    }
