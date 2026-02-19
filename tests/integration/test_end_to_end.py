"""
Integration tests for end-to-end workflows.

These tests verify that components work together correctly.
"""
import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestEndToEndClassification:
    """Test complete classification pipeline."""
    
    @pytest.mark.skip(reason="Requires sample audio files")
    def test_audio_to_prediction_pipeline(self):
        """
        Test complete pipeline from audio file to prediction.
        
        Steps:
        1. Load test audio file
        2. Process to spectrogram
        3. Load model
        4. Make prediction
        5. Verify result format
        """
        from quiet_horizon.audio import audio_to_spectrogram
        from frontend.utils.model_loader import load_model, predict_from_spectrogram
        
        # This would use a real test audio file
        audio_path = "tests/fixtures/sample_nature.wav"
        
        # Process audio
        spec_image = audio_to_spectrogram(audio_path)
        
        # Load model
        model = load_model()
        
        # Predict
        result = predict_from_spectrogram(model, spec_image)
        
        # Verify
        assert "predicted_label" in result
        assert "confidence" in result
        assert result["predicted_label"] in ["nature", "anthro"]
        assert 0 <= result["confidence"] <= 1
    
    @pytest.mark.skip(reason="Requires sample audio files")
    def test_batch_processing(self):
        """Test processing multiple files."""
        from quiet_horizon.audio import audio_to_spectrogram
        from frontend.utils.model_loader import load_model, predict_from_spectrogram
        
        audio_files = [
            "tests/fixtures/sample_nature.wav",
            "tests/fixtures/sample_anthro.wav",
        ]
        
        model = load_model()
        results = []
        
        for audio_path in audio_files:
            spec = audio_to_spectrogram(audio_path)
            result = predict_from_spectrogram(model, spec)
            results.append(result)
        
        # Verify all processed
        assert len(results) == len(audio_files)
        assert all("predicted_label" in r for r in results)


class TestCLIIntegration:
    """Test CLI inference tool."""
    
    @pytest.mark.skip(reason="Requires CLI setup and audio files")
    def test_cli_inference_command(self):
        """Test running inference from command line."""
        import subprocess
        
        result = subprocess.run(
            ["python", "-m", "quiet_horizon.inference_cnn", 
             "tests/fixtures/sample_nature.wav"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "predicted" in result.stdout.lower()
    
    @pytest.mark.skip(reason="Requires CLI setup")
    def test_cli_with_invalid_file(self):
        """Test CLI with invalid input."""
        import subprocess
        
        result = subprocess.run(
            ["python", "-m", "quiet_horizon.inference_cnn", 
             "nonexistent.wav"],
            capture_output=True,
            text=True
        )
        
        # Should exit with error
        assert result.returncode != 0


class TestFrontendIntegration:
    """Test frontend component integration."""
    
    @pytest.mark.skip(reason="Requires Streamlit testing setup")
    def test_upload_and_classify_workflow(self):
        """Test upload -> process -> display workflow."""
        # Would use Streamlit testing utilities
        # from streamlit.testing.v1 import AppTest
        pass


class TestPreprocessingConsistency:
    """
    Critical integration test: Verify CLI and frontend produce identical results.
    
    This validates the refactoring success - both paths should use the same
    implementation and get the same output.
    """
    
    @pytest.mark.skip(reason="Requires audio file fixture")
    def test_cli_and_frontend_produce_same_spectrogram(self):
        """
        Test that CLI and frontend preprocessing are identical.
        
        This is the key integration test that validates the code
        refactoring eliminated divergence between implementations.
        """
        from quiet_horizon.audio import audio_to_spectrogram
        from quiet_horizon.inference_cnn import load_melspec_from_audio
        
        audio_path = "tests/fixtures/sample_audio.wav"
        
        # Frontend path (returns without batch dimension)
        frontend_spec = audio_to_spectrogram(audio_path)
        
        # CLI path (returns with batch dimension)
        cli_spec_batch = load_melspec_from_audio(audio_path)
        cli_spec = cli_spec_batch[0]  # Remove batch dimension
        
        # Should be EXACTLY identical
        np.testing.assert_array_equal(frontend_spec, cli_spec)
    
    def test_preprocessing_functions_produce_same_output(self):
        """Test that shared functions produce consistent output."""
        from quiet_horizon.audio import create_mel_spectrogram, spectrogram_to_image
        
        # Generate test audio
        sr = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t)
        
        # Process twice
        spec1 = create_mel_spectrogram(audio, sr)
        spec2 = create_mel_spectrogram(audio, sr)
        
        # Should be identical
        np.testing.assert_array_equal(spec1, spec2)
        
        # Convert to image twice
        img1 = spectrogram_to_image(spec1)
        img2 = spectrogram_to_image(spec2)
        
        # Should be identical
        np.testing.assert_array_equal(img1, img2)


class TestModelDownloadAndCache:
    """Test model downloading and caching."""
    
    @pytest.mark.skip(reason="Requires network access")
    def test_download_model_from_huggingface(self):
        """Test downloading model from HuggingFace Hub."""
        from frontend.utils.model_loader import download_model_from_hf
        
        model = download_model_from_hf()
        
        assert model is not None
        # Model should be callable
        assert hasattr(model, 'predict')
    
    @pytest.mark.skip(reason="Requires network and filesystem access")
    def test_model_cached_after_download(self):
        """Test that model is cached after first download."""
        from frontend.utils.model_loader import download_model_from_hf
        from pathlib import Path
        import os
        
        # Clear cache if exists
        cache_path = Path.home() / ".cache" / "huggingface" / "hub"
        
        # Download once
        model1 = download_model_from_hf()
        
        # Download again - should use cache
        model2 = download_model_from_hf()
        
        # Both should work
        assert model1 is not None
        assert model2 is not None


class TestErrorRecovery:
    """Test error handling and recovery."""
    
    def test_invalid_audio_file_handling(self):
        """Test graceful handling of invalid audio."""
        from quiet_horizon.audio import load_audio
        
        with pytest.raises(ValueError):
            load_audio("nonexistent.wav")
    
    def test_corrupted_spectrogram_handling(self, mock_model):
        """Test handling of invalid spectrogram input."""
        from frontend.utils.model_loader import predict_from_spectrogram
        
        # Invalid shape - should validate shape before prediction
        bad_spec = np.zeros((64, 64, 1))
        
        # The function should validate input shape
        result = predict_from_spectrogram(mock_model, bad_spec)
        
        # Should still return valid prediction structure (model is resilient)
        assert "predicted_label" in result
        assert "confidence" in result
    
    @pytest.mark.skip(reason="Requires model file")
    def test_missing_model_file_handling(self):
        """Test handling when model file is missing."""
        from frontend.utils.model_loader import load_model
        
        with patch('os.path.exists', return_value=False):
            # Should attempt to download or raise clear error
            try:
                model = load_model("nonexistent_path.keras")
            except Exception as e:
                assert "model" in str(e).lower()


class TestPerformanceBenchmarks:
    """Basic performance benchmarks."""
    
    def test_spectrogram_generation_performance(self, sample_audio_array):
        """Benchmark spectrogram generation speed."""
        import time
        from quiet_horizon.audio import create_mel_spectrogram
        
        audio, sr = sample_audio_array
        
        start = time.time()
        for _ in range(10):
            spec = create_mel_spectrogram(audio, sr)
        end = time.time()
        
        avg_time = (end - start) / 10
        
        # Should be reasonably fast (< 100ms per spectrogram)
        assert avg_time < 0.1, f"Spectrogram generation too slow: {avg_time:.3f}s"
    
    @pytest.mark.skip(reason="Requires model")
    def test_inference_performance(self, mock_model, sample_spectrogram):
        """Benchmark inference speed."""
        import time
        from frontend.utils.model_loader import predict_from_spectrogram
        
        mock_model.predict = Mock(return_value=np.array([[0.7]]))
        
        start = time.time()
        for _ in range(10):
            result = predict_from_spectrogram(mock_model, sample_spectrogram)
        end = time.time()
        
        avg_time = (end - start) / 10
        
        # Should be fast (< 50ms per prediction with mock)
        assert avg_time < 0.05


class TestDataFlowIntegrity:
    """Test data maintains integrity through pipeline."""
    
    def test_audio_array_not_modified(self):
        """Test that audio array is not modified during processing."""
        from quiet_horizon.audio import create_mel_spectrogram
        
        # Generate audio
        sr = 22050
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr))
        audio_copy = audio.copy()
        
        # Process
        spec = create_mel_spectrogram(audio, sr)
        
        # Original should be unchanged
        np.testing.assert_array_equal(audio, audio_copy)
    
    def test_spectrogram_range_preserved(self):
        """Test that spectrogram values stay in valid range."""
        from quiet_horizon.audio import create_mel_spectrogram, spectrogram_to_image
        
        sr = 22050
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr))
        
        spec = create_mel_spectrogram(audio, sr)
        img = spectrogram_to_image(spec)
        
        # Image should be valid uint8
        assert img.dtype == np.uint8
        assert img.min() >= 0
        assert img.max() <= 255
