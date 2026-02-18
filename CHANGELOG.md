# Changelog

All notable changes to QuietHorizon will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Streamlit web frontend with single file and batch processing
- Comprehensive Architecture Decision Records (ADRs)
- Frontend architecture documentation
- Senior developer code review and recommendations
- Root README for project overview
- LICENSE file (MIT)
- CONTRIBUTING.md with development guidelines
- .env.example for environment configuration

### Changed
- Organized frontend with modular structure (utils/ and components/)
- Updated deprecated Streamlit parameters (use_container_width -> width)

### Fixed
- Import errors in frontend utils module

## [0.1.0] - 2026-01-15

### Added
- Initial CNN-based audio classifier
- Binary classification (Nature vs Anthropogenic)
- Mel-spectrogram preprocessing pipeline
- Model training notebook
- Audio augmentation scripts
- CLI inference tool
- DSP feature extraction (legacy)
- HuggingFace model hosting integration
- Basic documentation

### Performance
- 95% classification accuracy
- 0.99 AUC score
- ~4 MB model size
- Training on ~20,000 spectrograms

### Dataset
- 70+ wildlife species categories
- 10+ anthropogenic noise categories
- Balanced dataset with augmentation

---

## Version History

- **0.1.0** (2026-01-15): Initial release with CNN classifier
- **Unreleased**: Web frontend, documentation improvements

[Unreleased]: https://github.com/yourusername/QuietHorizon/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourusername/QuietHorizon/releases/tag/v0.1.0
