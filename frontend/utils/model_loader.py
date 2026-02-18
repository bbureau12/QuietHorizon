"""
Model loading utilities with caching for QuietHorizon
"""
import streamlit as st
from pathlib import Path
import tensorflow as tf
from huggingface_hub import hf_hub_download
import config


@st.cache_resource
def load_model(model_path=None):
    """
    Load the QuietHorizon CNN model with Streamlit caching.
    
    Args:
        model_path: Path to local model file. If None, downloads from HuggingFace.
    
    Returns:
        Loaded TensorFlow Keras model
    """
    if model_path is None:
        model_path = config.DEFAULT_MODEL_PATH
    
    # Check if model exists locally
    if not Path(model_path).exists():
        st.info("Model not found locally. Downloading from Hugging Face...")
        model_path = download_model_from_hf()
    
    try:
        model = tf.keras.models.load_model(model_path)
        st.success("✓ Model loaded successfully!")
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
            cache_dir=str(config.MODELS_DIR)
        )
        return model_path
    except Exception as e:
        st.error(f"Error downloading model from Hugging Face: {e}")
        st.info(f"Please ensure you have internet connection and the model exists at {config.HF_REPO_ID}")
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
    prob_anthro = float(prediction[0][0])
    prob_nature = 1 - prob_anthro
    
    # Determine label
    predicted_label = "ANTHRO" if prob_anthro >= config.ANTHRO_THRESHOLD else "NATURE"
    confidence = "HIGH" if max(prob_anthro, prob_nature) >= config.HIGH_CONFIDENCE_THRESHOLD else "MEDIUM"
    
    return {
        "prob_nature": prob_nature,
        "prob_anthro": prob_anthro,
        "predicted_label": predicted_label,
        "confidence": confidence
    }
