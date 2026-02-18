# Contributing to QuietHorizon

Thank you for considering contributing to QuietHorizon! This document provides guidelines and instructions for contributing.

## Code of Conduct

Be respectful, inclusive, and constructive. We're all here to build better tools for environmental research.

## Getting Started

### 1. Fork and Clone

```bash
git clone https://github.com/yourusername/QuietHorizon.git
cd QuietHorizon
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e .[dev]
```

### 3. Install Pre-commit Hooks (Optional but Recommended)

```bash
pre-commit install
```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 2. Make Your Changes

- Write clear, descriptive commit messages
- Follow existing code style (Black formatting, type hints)
- Add tests for new functionality
- Update documentation as needed

### 3. Run Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=quiet_horizon

# Run specific test file
pytest tests/test_audio_processor.py
```

### 4. Code Quality Checks

```bash
# Format code
black quiet_horizon/ frontend/ tests/

# Lint
ruff check quiet_horizon/ frontend/ tests/

# Type checking
mypy quiet_horizon/
```

### 5. Commit Your Changes

```bash
git add .
git commit -m "feat: add real-time audio streaming support"
# or
git commit -m "fix: correct mel-spectrogram normalization"
```

**Commit Message Format**:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Adding tests
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then open a Pull Request on GitHub with:
- Clear description of changes
- Link to related issues
- Screenshots/examples if applicable

## Code Style Guidelines

### Python Code

- **Formatting**: Use Black (line length 88)
- **Imports**: Absolute imports preferred, sorted with isort
- **Type Hints**: Add type hints for function signatures
- **Docstrings**: Use Google-style docstrings

```python
def process_audio(
    audio_file: str,
    target_sr: int = 22050
) -> np.ndarray:
    """
    Process audio file and return spectrogram.
    
    Args:
        audio_file: Path to audio file
        target_sr: Target sample rate in Hz
        
    Returns:
        Mel-spectrogram as numpy array
        
    Raises:
        ValueError: If audio file is invalid
    """
    pass
```

### Frontend Code

- Follow existing Streamlit patterns
- Separate UI (components/) from logic (utils/)
- Use type hints and docstrings

## Testing Guidelines

### Writing Tests

```python
# tests/test_audio_processor.py
import pytest
from quiet_horizon.audio import audio_to_spectrogram

def test_audio_to_spectrogram_shape():
    """Test that output has correct shape."""
    result = audio_to_spectrogram("tests/fixtures/sample.wav")
    assert result.shape == (128, 128, 3)

def test_audio_to_spectrogram_invalid_file():
    """Test error handling for invalid files."""
    with pytest.raises(ValueError):
        audio_to_spectrogram("nonexistent.wav")
```

### Test Structure

- `tests/unit/`: Unit tests for individual functions
- `tests/integration/`: End-to-end tests
- `tests/fixtures/`: Test data and sample files

### Coverage

- Aim for 80%+ coverage for new code
- Critical paths should have 100% coverage

## Documentation

### Update Documentation When:

- Adding new features
- Changing APIs
- Modifying configuration
- Adding dependencies

### Documentation Locations:

- Code: Docstrings and inline comments
- README.md: High-level overview
- docs/adr/: Architecture decisions
- frontend/: Frontend-specific docs

## Areas for Contribution

### High Priority

- [ ] **Tests**: Increase test coverage
- [ ] **Documentation**: Improve API docs
- [ ] **Performance**: Profile and optimize inference
- [ ] **Mobile**: TensorFlow Lite conversion

### Feature Requests

- [ ] Multi-class classification
- [ ] Real-time streaming
- [ ] Batch processing CLI
- [ ] REST API
- [ ] Noise suppression model
- [ ] Audio dataset cleanup tools

### Good First Issues

Look for issues tagged with `good-first-issue`:
- Documentation improvements
- Adding examples
- Writing tests
- Fixing typos

## Pull Request Process

1. **Update Tests**: Add/update tests for your changes
2. **Update Docs**: Update relevant documentation
3. **Run CI Checks**: Ensure all tests pass
4. **Code Review**: Address reviewer feedback
5. **Squash Commits**: Clean up commit history if needed
6. **Merge**: Maintainer will merge when approved

## Release Process

(For Maintainers)

1. Update version in `pyproject.toml` and `quiet_horizon/__init__.py`
2. Update `CHANGELOG.md`
3. Create git tag: `git tag -a v0.2.0 -m "Release v0.2.0"`
4. Push tag: `git push origin v0.2.0`
5. Publish to PyPI (if applicable)
6. Update Hugging Face model if needed

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions
- Check existing issues before creating new ones

## Attribution

Contributors will be recognized in:
- GitHub Contributors page
- CHANGELOG.md for significant contributions
- README.md acknowledgments

---

Thank you for contributing to QuietHorizon! 🌲
