# 🌲 QuietHorizon

**Environmental Audio Classifier — Detecting Human Noise Intrusion in Natural Soundscapes**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/bbureau12/QuietHorizon)

QuietHorizon is a machine-learning system designed to identify anthropogenic (human-made) noise in nature recordings. It supports environmental research, bioacoustics, and conservation work by enabling automated filtering of noisy audio.

**Live demo**: https://bbureau12.github.io/QuietHorizon/

**Model Performance**: 95% accuracy | 0.99 AUC | ~4 MB model size

---

## 🚀 Quick Start

### Option 1: Web Interface (Recommended)

The easiest way to get started:

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

Visit `http://localhost:8501` for a full-featured web interface with visualizations, batch processing, and audio playback.

### Option 2: Python API

```bash
pip install -e .
```

```python
from quiet_horizon import predict_from_audio

result = predict_from_audio("path/to/audio.wav")
print(f"Classification: {result['label']}")
print(f"Confidence: {result['probability']:.2%}")
```

### Option 3: Command Line

```bash
python -m quiet_horizon.inference_cnn path/to/audio.wav
```

### Option 4: Evaluation CLI

```bash
python -m quiet_horizon.evaluation.evaluate_cnn \
  --dataset-root quiet_horizon/dataset_cnn \
  --recursive \
  --model-path models/quiet_horizon_cnn.weights.h5
```

See [docs/evaluation.md](docs/evaluation.md) for manifest format and JSON reports.

### Option 5: Model Context Protocol (MCP)

Connect AI assistants like Claude Desktop to QuietHorizon:

```bash
# Configure in Claude Desktop
# See mcp_server/README.md for setup

# Then in Claude:
"Classify all audio files in my recordings folder"
```

Full documentation: [mcp_server/README.md](mcp_server/README.md)

---

## 📖 What's Inside

- **CNN Classifier**: Binary classification (Nature vs Anthropogenic) using mel-spectrogram images
- **Web Interface**: Streamlit app with visualizations and batch processing
- **MCP Server**: AI assistant integration (Claude Desktop, Cline, automation)
- **Training Pipeline**: Data augmentation, spectrogram generation, model training
- **CLI Tools**: Command-line inference and utilities
- **Comprehensive Docs**: Architecture Decision Records, API docs, deployment guides

---

## 🏗️ Project Structure

```
QuietHorizon/
├── quiet_horizon/          # Core Python library
│   ├── cnn_generation/    # Model training pipeline
│   ├── dsp/               # Signal processing (legacy)
│   └── inference_cnn.py   # CLI inference tool
│
├── frontend/              # Streamlit web application
│   ├── app.py            # Main web app
│   ├── utils/            # Audio processing, visualization
│   └── components/       # UI components
│
├── mcp_server/           # Model Context Protocol integration
│   ├── server.py         # MCP server for AI assistants
│   ├── tools.py          # Tool implementations
│   └── resources.py      # Resource providers
│
├── models/               # Pretrained models (download from HF)
├── docs/
│   └── adr/             # Architecture Decision Records
└── tests/               # Test suite (in progress)
```

---

## 🎯 Use Cases

- **🌳 Conservation Research**: Filter noisy recordings from wildlife monitoring
- **🦅 Bioacoustics**: Identify clean recordings for species analysis
- **📊 Environmental Studies**: Quantify human noise intrusion in ecosystems
- **🎙️ Field Recording**: Quality control for nature recordings

---

## 🤖 How It Works

1. **Audio Input**: Upload WAV, MP3, OGG, FLAC, or M4A
2. **Preprocessing**: Convert to mel-spectrogram (128×128 RGB image)
3. **Classification**: CNN analyzes patterns in spectrogram
4. **Results**: Binary classification with confidence scores

```
Audio → Mel-Spectrogram → CNN → Nature/Anthro (0-100%)
```

---

## 📦 Installation

### Requirements
- Python 3.10 or higher
- pip

### Install Core Library

```bash
# Clone the repository
git clone https://github.com/yourusername/QuietHorizon.git
cd QuietHorizon

# Install in development mode
pip install -e .
```

### Install with Frontend

```bash
pip install -e .
cd frontend
pip install -r requirements.txt
```

### Install with Development Tools

```bash
pip install -e .[dev]  # Includes pytest, black, ruff
```

---

## 📊 Model Details

| Metric    | Value  |
|-----------|--------|
| Accuracy  | ~95%   |
| AUC       | ~0.99  |
| Precision | ~0.95  |
| Recall    | ~0.96  |
| Model Size| ~4 MB  |

**Architecture**: Compact CNN with 4 convolutional layers + Global Average Pooling

**Training Data**: ~20,000 labeled spectrograms
- **Nature**: 70+ species (birds, frogs, mammals) + natural ambience (rain, thunder, waterfalls)
- **Anthropogenic**: Vehicles, aircraft, construction, machinery

**Pretrained Model**: Available on [Hugging Face](https://huggingface.co/bbureau12/QuietHorizon)

---

## 📚 Documentation

- **[Frontend README](frontend/README.md)**: Web interface documentation
- **[Quick Start Guide](frontend/QUICKSTART.md)**: Get up and running in 5 minutes
- **[Architecture](frontend/ARCHITECTURE.md)**: Technical architecture details
- **[ADRs](docs/adr/)**: Architecture Decision Records
- **[Evaluation CLI](docs/evaluation.md)**: Reproducible model evaluation workflow
- **[Code Review](docs/CODE_REVIEW.md)**: Development recommendations
- **[Main README](quiet_horizon/README.md)**: Detailed project documentation

---

## 🌐 GitHub Pages

The static site in `docs/` can be published automatically with GitHub Pages.
It is designed as a transparent sample-based demo, not a live hosted inference service.

1. Push the `docs/` files and `.github/workflows/pages.yml` to `main`.
2. In GitHub, open `Settings` → `Pages`.
3. Under `Source`, choose `GitHub Actions`.

After that, every push to `main` that changes `docs/` will deploy the site automatically.

The live page is available at:

```text
https://bbureau12.github.io/QuietHorizon/
```

The public page replays stored results for bundled repository samples.
For custom audio, users should clone the repository and run the Streamlit app or CLI locally.

---

## 🛠️ Development

### Running Tests

```bash
pytest tests/ --cov=quiet_horizon
```

### Code Quality

```bash
# Format code
black .

# Lint
ruff check .

# Type checking
mypy quiet_horizon/
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Areas for Contribution**:
- Add test coverage
- Improve documentation
- Implement multi-class classification
- Add real-time streaming support
- Mobile/edge deployment (TensorFlow Lite)

---

## 📈 Roadmap

- [x] Binary CNN classifier
- [x] Web interface
- [x] Architecture Decision Records
- [ ] Comprehensive test suite (in progress)
- [ ] Multi-class classification (road vs plane vs construction)
- [ ] Real-time audio streaming
- [ ] Mobile deployment (TFLite)
- [ ] Noise suppression using U-Net
- [ ] Public dataset release

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Training data: Minnesota wildlife recordings
- Frameworks: TensorFlow, Streamlit, Librosa
- Model hosting: Hugging Face Hub

---

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/QuietHorizon/issues)
- **Model**: [bbureau12/QuietHorizon on Hugging Face](https://huggingface.co/bbureau12/QuietHorizon)

---

**Made with 🌲 for conservation and environmental research**
