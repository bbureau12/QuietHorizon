"""
Batch processing component for QuietHorizon
"""
import streamlit as st
import pandas as pd
from utils.audio_processor import process_audio_file, validate_audio_file
from utils.model_loader import predict_from_spectrogram
from utils.visualization import plot_batch_results
import config


def render_batch_processing(model):
    """
    Render the batch processing section.
    
    Args:
        model: Loaded TensorFlow model
    """
    st.markdown("---")
    st.markdown("## 📦 Batch Processing")
    st.markdown("Upload multiple audio files for batch classification.")
    
    uploaded_files = st.file_uploader(
        "Choose audio files",
        type=config.SUPPORTED_AUDIO_FORMATS,
        accept_multiple_files=True,
        key="batch_upload"
    )
    
    if uploaded_files:
        num_files = len(uploaded_files)
        
        if num_files > config.BATCH_PROCESSING_LIMIT:
            st.warning(f"⚠️ Too many files! Maximum: {config.BATCH_PROCESSING_LIMIT}. "
                      f"Please upload fewer files.")
            return
        
        st.info(f"📊 Processing {num_files} file(s)...")
        
        if st.button("🚀 Process All Files", type="primary"):
            process_batch(model, uploaded_files)


def process_batch(model, uploaded_files):
    """
    Process multiple audio files and display results.
    
    Args:
        model: Loaded TensorFlow model
        uploaded_files: List of UploadedFile objects
    """
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, uploaded_file in enumerate(uploaded_files):
        # Update progress
        progress = (idx + 1) / len(uploaded_files)
        progress_bar.progress(progress)
        status_text.text(f"Processing {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")
        
        try:
            # Validate file
            is_valid, error_msg = validate_audio_file(uploaded_file)
            if not is_valid:
                results.append({
                    'filename': uploaded_file.name,
                    'status': 'Error',
                    'error': error_msg,
                    'predicted_label': None,
                    'prob_nature': None,
                    'prob_anthro': None,
                    'confidence': None
                })
                continue
            
            # Process audio
            processed = process_audio_file(uploaded_file)
            
            # Make prediction
            result = predict_from_spectrogram(model, processed['spectrogram_image'])
            
            results.append({
                'filename': uploaded_file.name,
                'status': 'Success',
                'error': None,
                'predicted_label': result['predicted_label'],
                'prob_nature': result['prob_nature'],
                'prob_anthro': result['prob_anthro'],
                'confidence': result['confidence'],
                'duration': processed['duration']
            })
            
        except Exception as e:
            results.append({
                'filename': uploaded_file.name,
                'status': 'Error',
                'error': str(e),
                'predicted_label': None,
                'prob_nature': None,
                'prob_anthro': None,
                'confidence': None
            })
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Display results
    display_batch_results(results)


def display_batch_results(results):
    """
    Display batch processing results.
    
    Args:
        results: List of result dictionaries
    """
    st.success("✅ Batch processing complete!")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Files", len(df))
    
    with col2:
        success_count = (df['status'] == 'Success').sum()
        st.metric("Successful", success_count)
    
    with col3:
        nature_count = (df['predicted_label'] == 'NATURE').sum()
        st.metric("Nature", nature_count)
    
    with col4:
        anthro_count = (df['predicted_label'] == 'ANTHRO').sum()
        st.metric("Anthropogenic", anthro_count)
    
    # Visualizations (only for successful predictions)
    successful_df = df[df['status'] == 'Success']
    
    if len(successful_df) > 0:
        st.markdown("### 📊 Results Overview")
        results_fig = plot_batch_results(successful_df)
        st.pyplot(results_fig)
    
    # Detailed results table
    st.markdown("### 📋 Detailed Results")
    
    # Format display DataFrame
    display_df = df.copy()
    if 'prob_nature' in display_df.columns:
        display_df['prob_nature'] = display_df['prob_nature'].apply(
            lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"
        )
        display_df['prob_anthro'] = display_df['prob_anthro'].apply(
            lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"
        )
    
    st.dataframe(
        display_df,
        width='stretch',
        hide_index=True
    )
    
    # Download results as CSV
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="quiet_horizon_batch_results.csv",
        mime="text/csv"
    )
    
    # Show errors if any
    error_df = df[df['status'] == 'Error']
    if len(error_df) > 0:
        with st.expander("⚠️ View Errors"):
            st.dataframe(
                error_df[['filename', 'error']],
                width='stretch',
                hide_index=True
            )
