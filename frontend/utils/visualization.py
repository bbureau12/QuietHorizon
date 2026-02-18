"""
Visualization utilities for QuietHorizon frontend
"""
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import config


def plot_waveform(audio_data, sample_rate, title="Audio Waveform"):
    """
    Create a waveform plot of audio data.
    
    Args:
        audio_data: Audio time series
        sample_rate: Sample rate of audio
        title: Plot title
    
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    
    librosa.display.waveshow(audio_data, sr=sample_rate, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_spectrogram(mel_spec_db, sample_rate, title="Mel-Spectrogram"):
    """
    Create a mel-spectrogram visualization.
    
    Args:
        mel_spec_db: Mel-spectrogram in dB scale
        sample_rate: Sample rate of audio
        title: Plot title
    
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    
    img = librosa.display.specshow(
        mel_spec_db,
        x_axis='time',
        y_axis='mel',
        sr=sample_rate,
        hop_length=config.HOP_LENGTH,
        ax=ax,
        cmap='viridis'
    )
    
    ax.set_title(title)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    
    plt.tight_layout()
    return fig


def plot_prediction_gauge(prob_nature, prob_anthro):
    """
    Create a horizontal gauge showing prediction probabilities.
    
    Args:
        prob_nature: Probability of nature class
        prob_anthro: Probability of anthropogenic class
    
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 2))
    
    # Create horizontal bar
    bar_height = 0.5
    ax.barh(0, prob_nature, height=bar_height, color=config.COLOR_NATURE, 
            label=f'Nature: {prob_nature:.1%}')
    ax.barh(0, prob_anthro, left=prob_nature, height=bar_height, 
            color=config.COLOR_ANTHRO, label=f'Anthropogenic: {prob_anthro:.1%}')
    
    # Add threshold line
    ax.axvline(x=config.ANTHRO_THRESHOLD, color='black', linestyle='--', 
               linewidth=2, alpha=0.7, label='Decision Threshold')
    
    # Styling
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('Probability', fontsize=12, fontweight='bold')
    ax.set_yticks([])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5), ncol=3)
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add percentage labels
    if prob_nature > 0.15:
        ax.text(prob_nature/2, 0, f'{prob_nature:.1%}', 
                ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    if prob_anthro > 0.15:
        ax.text(prob_nature + prob_anthro/2, 0, f'{prob_anthro:.1%}', 
                ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    plt.tight_layout()
    return fig


def create_results_summary(result, filename):
    """
    Create a formatted summary of prediction results.
    
    Args:
        result: Dictionary with prediction results
        filename: Name of the audio file
    
    Returns:
        Formatted HTML string
    """
    label = result['predicted_label']
    confidence = result['confidence']
    prob_nature = result['prob_nature']
    prob_anthro = result['prob_anthro']
    
    # Color based on prediction
    color = config.COLOR_ANTHRO if label == "ANTHRO" else config.COLOR_NATURE
    
    # Confidence emoji
    confidence_emoji = "🎯" if confidence == "HIGH" else "⚠️"
    
    html = f"""
    <div style='background-color: {color}22; padding: 20px; border-radius: 10px; 
                border-left: 5px solid {color};'>
        <h2 style='color: {color}; margin-top: 0;'>
            {confidence_emoji} Prediction: <strong>{label}</strong>
        </h2>
        <p style='font-size: 16px; margin: 10px 0;'>
            <strong>File:</strong> {filename}<br>
            <strong>Confidence:</strong> {confidence}<br>
            <strong>Nature Probability:</strong> {prob_nature:.2%}<br>
            <strong>Anthropogenic Probability:</strong> {prob_anthro:.2%}
        </p>
    </div>
    """
    
    return html


def plot_batch_results(results_df):
    """
    Create visualization for batch processing results.
    
    Args:
        results_df: Pandas DataFrame with batch results
    
    Returns:
        matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pie chart of classifications
    class_counts = results_df['predicted_label'].value_counts()
    colors = [config.COLOR_NATURE if label == 'NATURE' else config.COLOR_ANTHRO 
              for label in class_counts.index]
    
    ax1.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax1.set_title('Classification Distribution', fontsize=14, fontweight='bold')
    
    # Histogram of probabilities
    ax2.hist(results_df['prob_anthro'], bins=20, color=config.COLOR_ANTHRO, 
             alpha=0.6, edgecolor='black', label='Anthropogenic Probability')
    ax2.axvline(x=config.ANTHRO_THRESHOLD, color='black', linestyle='--', 
                linewidth=2, label='Threshold')
    ax2.set_xlabel('Probability', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Probability Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
