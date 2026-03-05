"""
QuietHorizon - Environmental Audio Classifier
Main Streamlit Application

Detects anthropogenic (human-made) noise in natural soundscapes using deep learning.
"""
import streamlit as st
import sys
import json
from pathlib import Path
from uuid import uuid4

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
from quiet_horizon.evaluation.evaluate_cnn import (
    Sample,
    collect_samples_from_dataset_root,
    collect_samples_from_manifest,
    evaluate_dataset,
    write_confusion_matrix_image,
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Single File", "Batch Processing", "How It Works", "Evaluation", "Model Card"]
    )

    with tab1:
        render_single_file_tab(model)

    with tab2:
        render_batch_processing(model)

    with tab3:
        render_info_tab()

    with tab4:
        render_evaluation_tab(model)

    with tab5:
        render_model_card_tab()


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


def render_evaluation_tab(model):
    """Render evaluation workflow and confusion matrix display."""
    st.markdown("## Model Evaluation")
    st.caption("Run dataset evaluation and generate a confusion matrix image.")
    st.info(
        "This confusion matrix is computed on the dataset you provide. "
        "Results depend on labeling quality and class distribution."
    )

    input_mode = st.radio(
        "Input Source",
        ["Dataset Root", "Manifest CSV"],
        horizontal=True,
        help="Dataset root infers labels from paths containing 'nature' or 'anthro'.",
    )

    dataset_root = None
    manifest_path = None
    recursive = False

    if input_mode == "Dataset Root":
        dataset_root = st.text_input(
            "Dataset Root",
            value=str(config.PROJECT_ROOT / "quiet_horizon" / "dataset_cnn"),
        )
        recursive = st.checkbox("Scan recursively", value=True)
    else:
        manifest_path = st.text_input(
            "Manifest CSV Path",
            value=str(config.PROJECT_ROOT / "tmp_eval_cardinal.csv"),
        )

    threshold = st.number_input(
        "Nature Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
    )
    max_files = st.number_input(
        "Max Files (0 = no limit)",
        min_value=0,
        max_value=1_000_000,
        value=0,
        step=1,
    )

    col_run, col_demo = st.columns(2)

    with col_run:
        run_eval_clicked = st.button("Run Evaluation", type="primary")
    with col_demo:
        run_demo_eval_clicked = st.button("Evaluate Demo Files")

    if run_eval_clicked:
        with st.spinner("Running evaluation..."):
            try:
                samples = []
                if input_mode == "Dataset Root":
                    samples = collect_samples_from_dataset_root(
                        Path(dataset_root), recursive=recursive
                    )
                else:
                    samples = collect_samples_from_manifest(Path(manifest_path))

                if max_files > 0:
                    samples = samples[:max_files]

                if not samples:
                    st.error("No labeled samples found for evaluation.")
                    return

                report = evaluate_dataset(
                    model=model,
                    samples=samples,
                    threshold=float(threshold),
                )

                cm_path = (
                    config.PROJECT_ROOT
                    / "reports"
                    / f"confusion_matrix_{uuid4().hex[:8]}.png"
                )
                write_confusion_matrix_image(
                    report["confusion_matrix_anthro"],
                    cm_path,
                )

                st.success("Evaluation complete.")
                render_evaluation_results(report, cm_path)
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
                with st.expander("View error details"):
                    st.exception(e)

    if run_demo_eval_clicked:
        with st.spinner("Running demo evaluation..."):
            try:
                samples = []
                for item in DEMO_TEST_FILES:
                    sample_path = item["path"]
                    if not sample_path.exists():
                        st.error(f"Missing demo file: {sample_path}")
                        return
                    samples.append(
                        Sample(
                            path=sample_path.resolve(),
                            label=item["expected_label"],
                        )
                    )

                report = evaluate_dataset(
                    model=model,
                    samples=samples,
                    threshold=float(threshold),
                )

                cm_path = (
                    config.PROJECT_ROOT
                    / "reports"
                    / f"demo_confusion_matrix_{uuid4().hex[:8]}.png"
                )
                write_confusion_matrix_image(
                    report["confusion_matrix_anthro"],
                    cm_path,
                )

                st.success("Demo evaluation complete.")
                render_evaluation_results(report, cm_path)
            except Exception as e:
                st.error(f"Demo evaluation failed: {e}")
                with st.expander("View error details"):
                    st.exception(e)


def render_evaluation_results(report, cm_path):
    """Render evaluation metrics and outputs."""
    summary = report["summary"]
    metrics = report["metrics"]
    per_class = report.get("per_class", {})
    confusion = report["confusion_matrix_anthro"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Evaluated", summary["evaluated_samples"])
    with col2:
        st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    with col3:
        st.metric("F1 (Anthro)", f"{metrics['f1_anthro']:.2%}")
    with col4:
        roc_auc = metrics["roc_auc_anthro"]
        st.metric("ROC-AUC (Anthro)", "n/a" if roc_auc is None else f"{roc_auc:.3f}")
    st.caption(
        "ROC-AUC shown above is one-vs-rest with anthro as the positive class. "
        "For benchmarking claims, report this on a held-out test split."
    )

    st.markdown("### Per-Class Metrics")
    class_rows = []
    for class_name in ("anthro", "nature"):
        class_data = per_class.get(class_name, {})
        class_rows.append(
            {
                "class": class_name,
                "precision": f"{class_data.get('precision', 0.0):.3f}",
                "recall": f"{class_data.get('recall', 0.0):.3f}",
                "f1": f"{class_data.get('f1', 0.0):.3f}",
                "support": class_data.get("support", 0),
            }
        )
    st.table(class_rows)

    st.markdown("### Confusion Matrix")
    st.image(str(cm_path), caption=str(cm_path), width="stretch")
    st.caption(
        f"TP: {confusion['tp']} | FP: {confusion['fp']} | FN: {confusion['fn']} | TN: {confusion['tn']}"
    )

    with st.expander("Show Evaluation Report JSON"):
        st.json(report)
        st.download_button(
            "Download JSON Report",
            data=json.dumps(report, indent=2),
            file_name="evaluation_report.json",
            mime="application/json",
        )

    if report["failed_files"]:
        with st.expander("Failed Files"):
            st.write(report["failed_files"])


def render_model_card_tab():
    """Render model card information in-app."""
    st.markdown("## QuietHorizon Model Card")
    st.caption("Summary information for responsible model usage.")

    st.markdown("### Model Details")
    st.markdown(
        """
- **Model type**: Binary CNN classifier (Nature vs Anthropogenic)
- **Input**: 128x128 RGB mel-spectrogram image derived from audio
- **Output**: Probability of nature class (`P(nature)`), with anthro as `1 - P(nature)`
- **Primary artifact**: `quiet_horizon_cnn.weights.h5` / `quiet_horizon_cnn.keras`
- **Hosted model**: https://huggingface.co/bbureau12/QuietHorizon
"""
    )

    st.markdown("### Intended Use")
    st.markdown(
        """
- Environmental audio quality filtering
- Conservation and bioacoustic preprocessing
- Detecting anthropogenic noise intrusion in recordings
"""
    )

    st.markdown("### Out of Scope")
    st.markdown(
        """
- Safety-critical decision making
- Legal or regulatory enforcement
- Fine-grained source identification (e.g., exact machine type)
"""
    )

    st.markdown("### Training Data Summary")
    st.markdown(
        """
- **Nature class**: Wildlife calls and natural ambience
- **Anthropogenic class**: Vehicles, machinery, construction-like sources
- **Approx dataset size**: ~20,000 labeled spectrograms
- **Augmentation**: Anthro/nature blending used for robustness
"""
    )

    st.markdown("### Evaluation Summary")
    st.markdown(
        """
- **Reported performance**: ~95% accuracy, ~0.99 AUC (project documentation)
- Use the **Evaluation** tab for reproducible, local dataset-specific metrics
- Confusion matrix generation is supported for each evaluation run
- AUC should be interpreted as **ROC-AUC on a held-out test set** with split strategy documented
"""
    )

    st.markdown("### Benchmark Snapshot (dataset_cnn)")
    benchmark_json = config.PROJECT_ROOT / "reports" / "dataset_cnn_benchmark.json"
    benchmark_cm = config.PROJECT_ROOT / "reports" / "dataset_cnn_benchmark_cm.png"
    if benchmark_json.exists():
        try:
            report = json.loads(benchmark_json.read_text(encoding="utf-8"))
            summary = report.get("summary", {})
            metrics = report.get("metrics", {})
            st.write(
                {
                    "evaluated_samples": summary.get("evaluated_samples"),
                    "threshold_nature": summary.get("threshold_nature"),
                    "accuracy": metrics.get("accuracy"),
                    "roc_auc_anthro": metrics.get("roc_auc_anthro"),
                }
            )
            if benchmark_cm.exists():
                st.image(str(benchmark_cm), caption=str(benchmark_cm), width="stretch")
            else:
                st.caption("No benchmark confusion matrix image found yet.")
        except Exception as e:
            st.warning(f"Could not read benchmark snapshot: {e}")
    else:
        st.caption(
            "No precomputed dataset_cnn benchmark found yet. "
            "Generate one with the evaluation CLI and save it to reports/."
        )

    st.markdown("### Known Limitations")
    st.markdown(
        """
- Binary classification only (`nature` vs `anthro`)
- Performance can vary by recording quality, geography, species mix, and mic setup
- Cross-version model artifacts (`.keras` vs `.weights.h5`) may behave differently in incompatible TF/Keras environments
"""
    )

    st.markdown("### Known Tricky Cases")
    st.markdown(
        """
- Wind plus distant highway noise
- Rain hitting the microphone housing
- Footsteps on snow
- Low, persistent HVAC hum
"""
    )

    st.markdown("### Ethical and Risk Considerations")
    st.markdown(
        """
- Risk of false positives (natural sounds flagged as anthro) and false negatives (missed anthro noise)
- Should be used as decision support, not sole authority
- Recommended to validate on representative local data before deployment
"""
    )


if __name__ == "__main__":
    main()
