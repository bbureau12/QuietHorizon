"""MCP Resource handlers for QuietHorizon metadata."""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def get_model_info() -> str:
    """Get QuietHorizon model metadata."""
    info = {
        "name": "QuietHorizon CNN",
        "version": "1.0.0",
        "type": "Binary Classifier",
        "architecture": "Convolutional Neural Network",
        "framework": "TensorFlow/Keras",
        "classes": ["nature", "anthropogenic"],
        "input_shape": [128, 128, 3],
        "input_type": "Mel-spectrogram (RGB image)",
        "performance": {
            "accuracy": "~95%",
            "training_dataset": "Nature vs Anthropogenic Sounds",
            "training_samples": "1000+ audio clips",
            "validation_split": "20%"
        },
        "preprocessing": {
            "sample_rate": 22050,
            "n_mels": 128,
            "n_fft": 2048,
            "hop_length": 512,
            "spectrogram_size": [128, 128]
        },
        "hosted_on": "HuggingFace Hub",
        "repo_id": "bbureau12/QuietHorizon",
        "threshold_default": 0.5,
        "description": (
            "QuietHorizon classifies environmental audio as either natural "
            "(bird calls, rain, wind, etc.) or anthropogenic (traffic, machinery, "
            "human activity). Trained on diverse soundscapes for ecological monitoring."
        )
    }
    
    return json.dumps(info, indent=2)


async def get_dataset_statistics() -> str:
    """Get training dataset composition."""
    stats = {
        "dataset_name": "Environmental Audio Classification Dataset",
        "classes": {
            "nature": {
                "description": "Natural environmental sounds",
                "examples": [
                    "Bird calls and songs (50+ species)",
                    "Animal vocalizations (frogs, mammals)",
                    "Natural phenomena (rain, thunder, waves, waterfalls)",
                    "Insect sounds (crickets, katydids)"
                ],
                "sample_count": "~60+ categories",
                "acoustic_characteristics": {
                    "frequency_range": "Highly variable (200 Hz - 8 kHz typical)",
                    "temporal_patterns": "Often rhythmic or pulsed",
                    "spectral_complexity": "Harmonic structure common in bird songs"
                }
            },
            "anthropogenic": {
                "description": "Human-generated sounds",
                "examples": [
                    "Vehicular noise (cars, planes, roads)",
                    "Construction and tools",
                    "Human activity and voices",
                    "Machinery and equipment"
                ],
                "sample_count": "3 main categories",
                "acoustic_characteristics": {
                    "frequency_range": "Broadband noise common",
                    "temporal_patterns": "Continuous or mechanical rhythms",
                    "spectral_complexity": "Less harmonic, more noise-like"
                }
            }
        },
        "data_augmentation": {
            "techniques_used": [
                "Time stretching",
                "Pitch shifting",
                "Background noise addition",
                "Volume normalization"
            ],
            "purpose": "Improve model robustness to variations"
        },
        "dataset_location": "dataset_cnn/",
        "spectrogram_dataset": "dataset_cnn_specs/",
        "preprocessing_pipeline": "quiet_horizon/cnn_generation/",
        "quality_control": "Manual verification and filtering applied"
    }
    
    return json.dumps(stats, indent=2)


async def get_supported_formats() -> str:
    """Get supported audio formats and limits."""
    formats = {
        "supported_formats": [
            {
                "extension": ".wav",
                "description": "Waveform Audio File Format",
                "recommended": True,
                "lossless": True
            },
            {
                "extension": ".mp3",
                "description": "MPEG Audio Layer III",
                "recommended": False,
                "lossless": False,
                "note": "Lossy compression may affect accuracy"
            },
            {
                "extension": ".flac",
                "description": "Free Lossless Audio Codec",
                "recommended": True,
                "lossless": True
            },
            {
                "extension": ".ogg",
                "description": "Ogg Vorbis",
                "recommended": False,
                "lossless": False
            },
            {
                "extension": ".m4a",
                "description": "MPEG-4 Audio",
                "recommended": False,
                "lossless": False
            }
        ],
        "limits": {
            "max_file_size_mb": 50,
            "recommended_duration_seconds": "3-60",
            "min_duration_seconds": 0.5,
            "sample_rates": "Any (resampled to 22050 Hz)"
        },
        "recommendations": {
            "optimal_format": "WAV or FLAC",
            "optimal_duration": "5-30 seconds",
            "recording_quality": "16-bit or higher",
            "channels": "Mono or stereo (converted to mono)"
        },
        "preprocessing": {
            "resampling": "All audio resampled to 22050 Hz",
            "mono_conversion": "Stereo files converted to mono",
            "normalization": "Applied during spectrogram generation"
        }
    }
    
    return json.dumps(formats, indent=2)
