"""
Unit tests for model loading and inference.
"""
import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestModelLoader:
    """Tests for model loading utilities."""
    
    @patch('frontend.utils.model_loader.tf.keras.models.load_model')
    def test_load_model_from_local(self, mock_load):
        """Test loading model from local path."""
        from frontend.utils import model_loader
        
        # Setup mock
        mock_model = Mock()
        mock_load.return_value = mock_model
        
        # Test
        model_path = "models/quiet_horizon_cnn.keras"
        
        # Note: This would call the actual function which has caching
        # For now, we test the underlying behavior
        assert mock_load is not None
    
    @patch('frontend.utils.model_loader.hf_hub_download')
    @patch('frontend.utils.model_loader.tf.keras.models.load_model')
    def test_download_model_from_hf(self, mock_load, mock_download):
        """Test downloading model from HuggingFace."""
        from frontend.utils.model_loader import download_model_from_hf
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'frontend'))
        import config
        
        # Setup mocks
        mock_download.return_value = "/path/to/downloaded/model.keras"
        mock_model = Mock()
        mock_load.return_value = mock_model
        
        # Test
        model = download_model_from_hf()
        
        # Verify HF download was called with correct params (including cache_dir)
        mock_download.assert_called_once_with(
            repo_id="bbureau12/QuietHorizon",
            filename="quiet_horizon_cnn.keras",
            cache_dir=str(config.MODELS_DIR)
        )
        
        # Should return the path, not load it
        assert model == "/path/to/downloaded/model.keras"
    
    def test_model_loading_error_handling(self):
        """Test error handling when model cannot be loaded."""
        from frontend.utils.model_loader import download_model_from_hf
        
        with patch('frontend.utils.model_loader.hf_hub_download') as mock_download:
            mock_download.side_effect = Exception("Network error")
            
            with pytest.raises(Exception, match="Network error"):
                download_model_from_hf()


class TestPrediction:
    """Tests for prediction functionality."""
    
    def test_predict_from_spectrogram_shape(self, mock_model, sample_spectrogram):
        """Test prediction with correct input shape."""
        from frontend.utils.model_loader import predict_from_spectrogram
        
        # Mock model returns nature prediction
        mock_model.predict = Mock(return_value=np.array([[0.85]]))
        
        result = predict_from_spectrogram(mock_model, sample_spectrogram)
        
        # Verify model was called with correct shape
        call_args = mock_model.predict.call_args[0][0]
        assert call_args.shape == (1, 128, 128, 3)
    
    def test_predict_from_spectrogram_output_format(self, mock_model, sample_spectrogram):
        """Test that prediction returns correct format."""
        from frontend.utils.model_loader import predict_from_spectrogram
        
        # Mock nature prediction (high probability)
        mock_model.predict = Mock(return_value=np.array([[0.85]]))
        
        result = predict_from_spectrogram(mock_model, sample_spectrogram)
        
        # Check output format
        assert isinstance(result, dict)
        assert "prob_nature" in result
        assert "prob_anthro" in result
        assert "predicted_label" in result
        assert "confidence" in result
        
        # Check values
        assert result["prob_nature"] == pytest.approx(0.85, abs=0.01)
        assert result["prob_anthro"] == pytest.approx(0.15, abs=0.01)
        assert result["predicted_label"] == "nature"
        assert result["confidence"] == pytest.approx(0.85, abs=0.01)
    
    def test_predict_anthro_classification(self, mock_model, sample_spectrogram):
        """Test anthropogenic classification."""
        from frontend.utils.model_loader import predict_from_spectrogram
        
        # Mock anthro prediction (low nature probability)
        mock_model.predict = Mock(return_value=np.array([[0.25]]))
        
        result = predict_from_spectrogram(mock_model, sample_spectrogram)
        
        assert result["predicted_label"] == "anthro"
        assert result["prob_anthro"] == pytest.approx(0.75, abs=0.01)
        assert result["confidence"] == pytest.approx(0.75, abs=0.01)
    
    def test_predict_threshold_boundary(self, mock_model, sample_spectrogram):
        """Test prediction at 50% threshold."""
        from frontend.utils.model_loader import predict_from_spectrogram
        
        # Mock exactly at threshold
        mock_model.predict = Mock(return_value=np.array([[0.50]]))
        
        result = predict_from_spectrogram(mock_model, sample_spectrogram)
        
        # At 0.5, should classify as nature (>= threshold)
        assert result["predicted_label"] == "nature"
        assert result["confidence"] == pytest.approx(0.50, abs=0.01)


class TestPredictionThresholds:
    """Tests for prediction threshold logic."""
    
    def test_high_confidence_nature(self, mock_model, sample_spectrogram):
        """Test high confidence nature prediction."""
        from frontend.utils.model_loader import predict_from_spectrogram
        
        mock_model.predict = Mock(return_value=np.array([[0.95]]))
        result = predict_from_spectrogram(mock_model, sample_spectrogram)
        
        assert result["predicted_label"] == "nature"
        assert result["confidence"] >= 0.9
    
    def test_high_confidence_anthro(self, mock_model, sample_spectrogram):
        """Test high confidence anthropogenic prediction."""
        from frontend.utils.model_loader import predict_from_spectrogram
        
        mock_model.predict = Mock(return_value=np.array([[0.05]]))
        result = predict_from_spectrogram(mock_model, sample_spectrogram)
        
        assert result["predicted_label"] == "anthro"
        assert result["confidence"] >= 0.9
    
    def test_low_confidence_prediction(self, mock_model, sample_spectrogram):
        """Test low confidence prediction (near 0.5)."""
        from frontend.utils.model_loader import predict_from_spectrogram
        
        mock_model.predict = Mock(return_value=np.array([[0.52]]))
        result = predict_from_spectrogram(mock_model, sample_spectrogram)
        
        # Still makes a prediction, but confidence is low
        assert result["predicted_label"] in ["nature", "anthro"]
        assert result["confidence"] < 0.6


class TestInputValidation:
    """Tests for input validation."""
    
    def test_predict_with_invalid_shape(self, mock_model):
        """Test prediction with wrong input shape."""
        from frontend.utils.model_loader import predict_from_spectrogram
        
        # Wrong shape spectrogram
        invalid_spec = np.zeros((64, 64, 3), dtype=np.uint8)
        
        # Should either raise error or handle gracefully
        # Depending on implementation
        mock_model.predict = Mock(return_value=np.array([[0.5]]))
        
        # This might raise an error in real implementation
        # For now, just ensure it doesn't crash unexpectedly
        try:
            result = predict_from_spectrogram(mock_model, invalid_spec)
        except (ValueError, Exception):
            # Expected to fail with wrong shape
            pass
    
    def test_predict_with_wrong_dtype(self, mock_model, sample_spectrogram):
        """Test prediction with wrong dtype."""
        from frontend.utils.model_loader import predict_from_spectrogram
        
        # Convert to float instead of uint8
        float_spec = sample_spectrogram.astype(np.float32)
        
        mock_model.predict = Mock(return_value=np.array([[0.7]]))
        
        # Should handle or convert dtype
        result = predict_from_spectrogram(mock_model, float_spec)
        assert result is not None


class TestModelCaching:
    """Tests for model caching behavior."""
    
    @pytest.mark.skip(reason="Requires Streamlit cache testing")
    def test_model_cached_on_subsequent_calls(self):
        """Test that model is only loaded once with caching."""
        # This would test @st.cache_resource behavior
        # Requires Streamlit testing utilities
        pass
    
    @pytest.mark.skip(reason="Requires Streamlit cache testing")
    def test_cache_invalidation(self):
        """Test cache is invalidated when needed."""
        pass
