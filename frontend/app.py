"""
QuietHorizon - Environmental Audio Classifier
Main Streamlit Application

Detects anthropogenic (human-made) noise in natural soundscapes using deep learning.
"""
import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path to import quiet_horizon modules if needed
sys.path.append(str(Path(__file__).parent.parent))
# Add frontend directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

import config
from utils import (
    load_model,
    process_audio_file,
    validate_audio_file,
    predict_from_spectrogram,
)
from components import (
    render_upload_section,
    render_results_section,
    render_batch_processing,
)

DEMO_TEST_FILES = [
    {
        "name": "Northern Cardinal (should be natural)",
        "path": config.PROJECT_ROOT
        / "quiet_horizon"
        / "test_data"
        / "240404__itinerantmonk108__northern-cardinal-closeup.wav",
        "expected_label": "nature",
        "expected_text": "natural",
    },
    {
        "name": "Heavy Traffic (should be anthropogenic)",
        "path": config.PROJECT_ROOT
        / "quiet_horizon"
        / "test_data"
        / "691513__ania635__heavy_traffic_03.wav",
        "expected_label": "anthro",
        "expected_text": "anthropogenic",
    },
]


# Page configuration
st.set_page_config(
    page_title="QuietHorizon - Audio Classifier",
    page_icon="QH",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    """Main application logic"""

    # Header
    st.title("QuietHorizon")
    st.markdown(
        """
    ### Environmental Audio Classifier
    Detecting human noise intrusion in natural soundscapes using deep learning.
    """
    )

    # Sidebar
    with st.sidebar:
        st.markdown("## About QuietHorizon")
        st.markdown(
            """
        QuietHorizon uses a Convolutional Neural Network (CNN) to classify audio 
        as either:
        - **Nature**: Clean wildlife calls, natural ambience
        - **Anthropogenic**: Human-made noise (vehicles, machinery, construction)

        **Model Performance:**
        - Accuracy: ~95%
        - AUC: ~0.99
        - Model Size: ~4 MB
        """
        )

        st.markdown("---")
        st.markdown("### Settings")

        # Model loading
        model_source = st.radio(
            "Model Source:",
            ["Auto (HuggingFace or Local)", "Local Path"],
            help="Choose where to load the model from",
        )

        if model_source == "Local Path":
            model_path = st.text_input("Model Path:", value=str(config.DEFAULT_MODEL_PATH))
        else:
            model_path = None

        st.markdown("---")
        st.markdown(
            f"""
        **Classification Threshold:** {config.ANTHRO_THRESHOLD}  
        **High Confidence:** >= {config.HIGH_CONFIDENCE_THRESHOLD}
        """
        )

        st.markdown("---")
        st.markdown(
            """
        ### Resources
        - [Hugging Face Model](https://huggingface.co/bbureau12/QuietHorizon)
        - [GitHub Repository](#)
        - [Documentation](../quiet_horizon/README.md)
        """
        )

    # Load model
    try:
        with st.spinner("Loading model..."):
            model = load_model(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["Single File", "Batch Processing", "How It Works"])

    with tab1:
        render_single_file_tab(model)

    with tab2:
        render_batch_processing(model)

    with tab3:
        render_info_tab()


def render_single_file_tab(model):
    """Render the single file processing tab"""
    render_demo_test_data(model)
    st.markdown("---")

    # File upload
    uploaded_file = render_upload_section()

    if uploaded_file is not None:
        # Validate file
        validation = validate_audio_file(uploaded_file)

        if not validation["valid"]:
            st.error(f"Error: {validation['error']}")
            return

        # Process button
        if st.button("Classify Audio", type="primary"):
            run_inference_for_audio(model, uploaded_file, uploaded_file.name)


def run_inference_for_audio(model, audio_source, display_name, expected_label=None, expected_text=None):
    """Run inference and render results for an uploaded or demo audio source."""
    with st.spinner("Processing audio..."):
        try:
            processed = process_audio_file(audio_source)
            result = predict_from_spectrogram(model, processed["spectrogram_image"])
            render_results_section(result, processed, display_name)

            predicted_label = str(result.get("predicted_label", "")).strip().lower()
            if expected_label is not None and predicted_label != expected_label.lower():
                st.warning(
                    f"Demo expectation mismatch for `{display_name}`: expected `{expected_text}` "
                    f"but predicted `{predicted_label}`."
                )
            elif expected_label is not None:
                st.success(f"Expected demo label matched: `{expected_text}`.")
        except Exception as e:
            st.error(f"Error processing audio: {e}")
            with st.expander("View error details"):
                st.exception(e)


def render_demo_test_data(model):
    """Render fixed demo samples from quiet_horizon/test_data."""
    st.markdown("### Demo Test Data")
    st.caption("Play either clip and run inference directly, without uploading.")

    for idx, sample in enumerate(DEMO_TEST_FILES):
        sample_path = sample["path"]
        st.markdown(f"#### {sample['name']}")

        if not sample_path.exists():
            st.error(f"Missing demo file: {sample_path}")
            continue

        st.audio(sample_path.read_bytes(), format="audio/wav")
        st.caption(f"Expected class for demo: {sample['expected_text']}")

        if st.button("Run Inference", key=f"demo_inference_{idx}"):
            run_inference_for_audio(
                model=model,
                audio_source=str(sample_path),
                display_name=sample_path.name,
                expected_label=sample["expected_label"],
                expected_text=sample["expected_text"],
            )


def render_info_tab():
    """Render the information tab"""

    st.markdown(
        """
    ## How QuietHorizon Works

    ### The Process

    1. **Audio Input**
       - Upload your audio file (WAV, MP3, OGG, FLAC, M4A)
       - Audio is automatically resampled to 22,050 Hz

    2. **Spectrogram Generation**
       - Audio is converted to a mel-spectrogram (visual representation of sound)
       - Uses 128 mel-frequency bins
       - Resized to 128x128 RGB image

    3. **CNN Classification**
       - The spectrogram image is fed to a Convolutional Neural Network
       - Model architecture: Conv layers (32->64->128->256) + Global Average Pooling + Dense layers
       - Output: Probability score for Nature vs Anthropogenic

    4. **Results**
       - Classification: NATURE or ANTHRO
       - Confidence level: HIGH (>=85%) or MEDIUM (<85%)
       - Detailed probability scores

    ### Model Architecture

    ```
    Input (128x128x3)
      ->
    Rescaling
      ->
    Conv2D (32 filters) -> MaxPool
      ->
    Conv2D (64 filters) -> MaxPool
      ->
    Conv2D (128 filters) -> MaxPool
      ->
    Conv2D (256 filters) -> MaxPool
      ->
    Global Average Pooling
      ->
    Dense (128) + Dropout
      ->
    Dense (1, sigmoid)
      ->
    Nature vs Anthro Probability
    ```

    ### Training Data

    - **Nature Class**: ~70 species including birds, frogs, mammals, plus natural ambience (rain, thunder, waterfalls)
    - **Anthropogenic Class**: Vehicles, aircraft, construction tools, machinery
    - **Augmentation**: 25% blend of anthro sounds with nature for robustness
    - **Total**: ~20,000 labeled spectrograms

    ### Use Cases

    - **Conservation Research**: Filter noisy recordings from wildlife monitoring
    - **Bioacoustics**: Identify clean recordings for species analysis
    - **Environmental Studies**: Quantify human noise intrusion in ecosystems
    - **Field Recording**: Quality control for nature recordings

    ### Limitations

    - Binary classification only (no multi-class for specific noise types yet)
    - Optimized for North American wildlife (Minnesota dataset)
    - Best performance on clear recordings (some noise blending is tolerated)
    - Short clips (<5s) may have reduced accuracy

    ### Future Enhancements

    - Multi-class classification (road vs plane vs construction)
    - Noise suppression using U-Net architecture
    - Real-time stream processing
    - Mobile/edge deployment with TensorFlow Lite
    """
    )

    st.markdown("---")

    # Example images
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Nature Spectrogram Example")
        st.markdown("Clean, distinct patterns from bird calls or natural sounds")

    with col2:
        st.markdown("#### Anthropogenic Spectrogram Example")
        st.markdown("Harmonic structures from engines, machinery, or tools")


if __name__ == "__main__":
    main()
