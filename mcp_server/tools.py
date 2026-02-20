"""MCP Tool implementations for QuietHorizon audio classification."""

import base64
import json
import sys
from pathlib import Path
from typing import Any
import logging

import numpy as np
import librosa
from mcp.types import TextContent, ImageContent

# Add parent directory to path to import QuietHorizon modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from quiet_horizon.audio.preprocessing import audio_to_spectrogram

logger = logging.getLogger(__name__)

# Lazy-load model
_MODEL = None


def get_model():
    """Get or load the model singleton."""
    global _MODEL
    if _MODEL is None:
        logger.info("Loading QuietHorizon model...")
        # Import here to avoid loading TensorFlow on module import
        import tensorflow as tf
        from pathlib import Path
        
        # Try local model first
        local_model = Path(__file__).parent.parent / "models" / "quiet_horizon_cnn.keras"
        
        if local_model.exists():
            _MODEL = tf.keras.models.load_model(str(local_model))
        else:
            # Download from HuggingFace
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(
                repo_id="bbureau12/QuietHorizon",
                filename="quiet_horizon_cnn.keras"
            )
            _MODEL = tf.keras.models.load_model(model_path)
            
    return _MODEL


async def classify_audio_tool(arguments: dict[str, Any]) -> list[TextContent]:
    """
    Classify a single audio file.
    
    Args:
        arguments: Dict with 'file_path' and optional 'threshold'
        
    Returns:
        List of TextContent with classification results
    """
    file_path = arguments.get("file_path")
    threshold = arguments.get("threshold", 0.5)
    
    if not file_path:
        return [TextContent(type="text", text="Error: file_path is required")]
    
    file_path = Path(file_path)
    if not file_path.exists():
        return [TextContent(type="text", text=f"Error: File not found: {file_path}")]
    
    try:
        # Load and preprocess audio
        spec_data = audio_to_spectrogram(str(file_path))
        spectrogram = spec_data["spectrogram"]
        
        # Run inference
        model = get_model()
        prediction = model.predict(np.expand_dims(spectrogram, axis=0), verbose=0)
        prob_nature = float(prediction[0][0])
        prob_anthro = 1.0 - prob_nature
        
        # Determine classification
        predicted_label = "nature" if prob_nature >= threshold else "anthro"
        confidence = max(prob_nature, prob_anthro)
        
        # Format result
        result = {
            "file": str(file_path.name),
            "classification": predicted_label,
            "confidence": round(confidence, 4),
            "probabilities": {
                "nature": round(prob_nature, 4),
                "anthropogenic": round(prob_anthro, 4)
            },
            "threshold": threshold,
            "audio_info": {
                "duration_seconds": spec_data.get("duration", "N/A"),
                "sample_rate": spec_data.get("sample_rate", 22050)
            }
        }
        
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
        
    except Exception as e:
        logger.error(f"Classification failed for {file_path}", exc_info=True)
        return [TextContent(
            type="text",
            text=f"Error classifying {file_path.name}: {str(e)}"
        )]


async def batch_classify_tool(arguments: dict[str, Any]) -> list[TextContent]:
    """
    Classify multiple audio files from a directory.
    
    Args:
        arguments: Dict with 'directory', optional 'recursive' and 'threshold'
        
    Returns:
        List of TextContent with batch results
    """
    directory = arguments.get("directory")
    recursive = arguments.get("recursive", False)
    threshold = arguments.get("threshold", 0.5)
    
    if not directory:
        return [TextContent(type="text", text="Error: directory is required")]
    
    dir_path = Path(directory)
    if not dir_path.exists() or not dir_path.is_dir():
        return [TextContent(type="text", text=f"Error: Directory not found: {dir_path}")]
    
    # Supported formats
    supported_formats = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    
    # Find audio files
    if recursive:
        files = [f for f in dir_path.rglob("*") if f.suffix.lower() in supported_formats]
    else:
        files = [f for f in dir_path.glob("*") if f.suffix.lower() in supported_formats]
    
    if not files:
        return [TextContent(
            type="text",
            text=f"No audio files found in {dir_path}"
        )]
    
    # Process each file
    model = get_model()
    results = []
    nature_count = 0
    anthro_count = 0
    
    for file_path in files:
        try:
            spec_data = audio_to_spectrogram(str(file_path))
            spectrogram = spec_data["spectrogram"]
            
            prediction = model.predict(np.expand_dims(spectrogram, axis=0), verbose=0)
            prob_nature = float(prediction[0][0])
            predicted_label = "nature" if prob_nature >= threshold else "anthro"
            
            if predicted_label == "nature":
                nature_count += 1
            else:
                anthro_count += 1
            
            results.append({
                "file": str(file_path.name),
                "classification": predicted_label,
                "confidence": round(max(prob_nature, 1 - prob_nature), 4),
                "prob_nature": round(prob_nature, 4)
            })
            
        except Exception as e:
            logger.warning(f"Failed to process {file_path.name}: {e}")
            results.append({
                "file": str(file_path.name),
                "error": str(e)
            })
    
    # Summary
    summary = {
        "directory": str(dir_path),
        "total_files": len(files),
        "processed": len([r for r in results if "error" not in r]),
        "failed": len([r for r in results if "error" in r]),
        "summary": {
            "nature": nature_count,
            "anthropogenic": anthro_count,
            "nature_percentage": round(100 * nature_count / len(files), 1) if files else 0
        },
        "threshold": threshold,
        "results": results
    }
    
    return [TextContent(
        type="text",
        text=json.dumps(summary, indent=2)
    )]


async def analyze_soundscape_tool(arguments: dict[str, Any]) -> list[TextContent | ImageContent]:
    """
    Perform detailed spectral analysis of an audio file.
    
    Args:
        arguments: Dict with 'file_path' and optional 'include_spectrogram'
        
    Returns:
        List of TextContent and optionally ImageContent with analysis
    """
    file_path = arguments.get("file_path")
    include_spectrogram = arguments.get("include_spectrogram", True)
    
    if not file_path:
        return [TextContent(type="text", text="Error: file_path is required")]
    
    file_path = Path(file_path)
    if not file_path.exists():
        return [TextContent(type="text", text=f"Error: File not found: {file_path}")]
    
    try:
        # Load audio
        y, sr = librosa.load(str(file_path), sr=22050)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        
        # RMS energy
        rms = librosa.feature.rms(y=y)
        
        # Create analysis result
        analysis = {
            "file": str(file_path.name),
            "duration_seconds": round(duration, 2),
            "sample_rate": sr,
            "spectral_features": {
                "centroid_mean_hz": round(float(np.mean(spectral_centroid)), 2),
                "centroid_std_hz": round(float(np.std(spectral_centroid)), 2),
                "rolloff_mean_hz": round(float(np.mean(spectral_rolloff)), 2),
                "zero_crossing_rate": round(float(np.mean(zero_crossing_rate)), 4)
            },
            "energy": {
                "rms_mean": round(float(np.mean(rms)), 4),
                "rms_std": round(float(np.std(rms)), 4)
            }
        }
        
        # Get classification
        spec_data = audio_to_spectrogram(str(file_path))
        spectrogram = spec_data["spectrogram"]
        
        model = get_model()
        prediction = model.predict(np.expand_dims(spectrogram, axis=0), verbose=0)
        prob_nature = float(prediction[0][0])
        
        analysis["classification"] = {
            "predicted_label": "nature" if prob_nature >= 0.5 else "anthro",
            "prob_nature": round(prob_nature, 4),
            "prob_anthro": round(1 - prob_nature, 4)
        }
        
        response = [TextContent(
            type="text",
            text=json.dumps(analysis, indent=2)
        )]
        
        # Optionally include spectrogram image
        if include_spectrogram:
            # Convert spectrogram to base64 PNG
            from PIL import Image
            import io
            
            img = Image.fromarray(spectrogram)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            response.append(ImageContent(
                type="image",
                data=img_base64,
                mimeType="image/png"
            ))
        
        return response
        
    except Exception as e:
        logger.error(f"Soundscape analysis failed for {file_path}", exc_info=True)
        return [TextContent(
            type="text",
            text=f"Error analyzing {file_path.name}: {str(e)}"
        )]


async def get_audio_info_tool(arguments: dict[str, Any]) -> list[TextContent]:
    """
    Get technical information about an audio file.
    
    Args:
        arguments: Dict with 'file_path'
        
    Returns:
        List of TextContent with audio metadata
    """
    file_path = arguments.get("file_path")
    
    if not file_path:
        return [TextContent(type="text", text="Error: file_path is required")]
    
    file_path = Path(file_path)
    if not file_path.exists():
        return [TextContent(type="text", text=f"Error: File not found: {file_path}")]
    
    try:
        y, sr = librosa.load(str(file_path), sr=None)  # Keep original sample rate
        duration = librosa.get_duration(y=y, sr=sr)
        
        info = {
            "file": str(file_path.name),
            "path": str(file_path),
            "format": file_path.suffix.lower(),
            "size_bytes": file_path.stat().st_size,
            "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
            "duration_seconds": round(duration, 2),
            "sample_rate": sr,
            "samples": len(y),
            "channels": 1 if y.ndim == 1 else y.shape[0]
        }
        
        return [TextContent(
            type="text",
            text=json.dumps(info, indent=2)
        )]
        
    except Exception as e:
        logger.error(f"Failed to get info for {file_path}", exc_info=True)
        return [TextContent(
            type="text",
            text=f"Error reading {file_path.name}: {str(e)}"
        )]
