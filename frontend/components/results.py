"""
Results display component for QuietHorizon
"""
import streamlit as st
from utils.visualization import (
    plot_waveform,
    plot_spectrogram,
    plot_prediction_gauge,
    create_results_summary,
)


def render_results_section(result, processed_audio, filename):
    """
    Render the prediction results section.

    Args:
        result: Dictionary with prediction results
        processed_audio: Dictionary with audio processing results
        filename: Name of the audio file
    """
    # Display summary
    st.markdown("---")
    st.markdown("## Prediction Results")

    summary_html = create_results_summary(result, filename)
    st.markdown(summary_html, unsafe_allow_html=True)

    # Probability gauge
    st.markdown("### Probability Distribution")
    gauge_fig = plot_prediction_gauge(result["prob_nature"], result["prob_anthro"])
    st.pyplot(gauge_fig)

    # Audio player
    st.markdown("### Audio Playback")
    audio_data = processed_audio["audio_data"]
    sample_rate = processed_audio["sample_rate"]

    # Convert to bytes for playback
    import io
    import scipy.io.wavfile

    buffer = io.BytesIO()
    scipy.io.wavfile.write(buffer, sample_rate, audio_data)
    buffer.seek(0)

    st.audio(buffer, format="audio/wav")
    st.caption(f"Duration: {processed_audio['duration']:.2f} seconds")

    # Visualizations in tabs
    tab1, tab2, tab3 = st.tabs(["Waveform", "Spectrogram", "Model Input"])

    with tab1:
        waveform_fig = plot_waveform(audio_data, sample_rate)
        st.pyplot(waveform_fig)

    with tab2:
        spec_fig = plot_spectrogram(processed_audio["mel_spectrogram"], sample_rate)
        st.pyplot(spec_fig)

    with tab3:
        st.markdown("**This is the spectrogram image fed to the CNN:**")
        st.image(
            processed_audio["spectrogram_image"],
            caption="128x128 RGB Mel-Spectrogram",
            width="stretch",
        )

    confidence_value = result.get("confidence", 0.0)
    if isinstance(confidence_value, str):
        confidence_level = confidence_value.upper()
    else:
        confidence_level = "HIGH" if float(confidence_value) >= 0.85 else "MEDIUM"
    predicted_label = str(result.get("predicted_label", "")).upper()

    # Additional info
    with st.expander("About the Classification"):
        st.markdown(
            f"""
        **How it works:**
        - The audio is converted to a mel-spectrogram (visual representation of sound)
        - The spectrogram is resized to 128x128 pixels and converted to RGB
        - A Convolutional Neural Network (CNN) analyzes the patterns
        - The model outputs a probability score for each class

        **Classification threshold:** {result['prob_anthro']:.1%} vs {(1 - result['prob_anthro'])*100:.1f}%

        **Confidence level:**
        - HIGH: Prediction probability >= 85%
        - MEDIUM: Prediction probability < 85%

        **Current prediction:**
        - The model is **{confidence_level.lower()} confidence** that this is **{predicted_label}** sound.
        """
        )
