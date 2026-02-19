"""
File upload component for QuietHorizon
"""
import streamlit as stimport sys
from pathlib import Path

# Add frontend to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))import config


def render_upload_section():
    """
    Render the file upload section of the UI.
    
    Returns:
        UploadedFile object or None
    """
    st.markdown("### 📁 Upload Audio File")
    
    # Format the supported formats for display
    formats_display = ", ".join([f".{fmt}" for fmt in config.SUPPORTED_AUDIO_FORMATS])
    
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=config.SUPPORTED_AUDIO_FORMATS,
        help=f"Supported formats: {formats_display}. Max size: {config.MAX_FILE_SIZE_MB} MB"
    )
    
    if uploaded_file is not None:
        # Display file info
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.info(f"📄 **{uploaded_file.name}** ({file_size_mb:.2f} MB)")
    
    return uploaded_file
