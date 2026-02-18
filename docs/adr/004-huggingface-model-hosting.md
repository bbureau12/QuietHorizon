# 4. HuggingFace Hub for Model Distribution

**Date**: 2026-02-18

**Status**: Accepted

## Context

The trained CNN model (~4 MB) needs to be distributed to users. Options include:
- Committing to git repository
- Hosting on cloud storage (AWS S3, Google Drive)
- Using ML model registries
- Self-hosted server

Requirements:
- Easy programmatic download
- Version control for models
- Free or low-cost
- Reliable availability
- Community trust

## Decision

Host the pretrained model on HuggingFace Hub at `bbureau12/QuietHorizon` and download programmatically using `huggingface-hub` library.

Implementation:
```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="bbureau12/QuietHorizon",
    filename="quiet_horizon_cnn.keras"
)
```

## Consequences

### Positive Consequences

- **Free Hosting**: Unlimited bandwidth for public models
- **Version Control**: Git-based versioning with commits and tags
- **Programmatic Access**: Simple Python API for downloading
- **Caching**: Automatic local caching prevents re-downloads
- **Community**: HuggingFace is trusted in ML community
- **Discoverability**: Model can be found via HuggingFace search
- **Documentation**: Can host model cards with usage examples
- **No Git LFS**: Model doesn't bloat the main repo
- **Metrics**: Download stats and analytics
- **Collaboration**: Easy sharing and collaboration

### Negative Consequences

- **External Dependency**: Requires internet connection for first download
- **Platform Lock-in**: Some coupling to HuggingFace ecosystem
- **Account Required**: Need HuggingFace account to upload models
- **Public by Default**: Private models have limits on free tier
- **Namespace**: Need to maintain HuggingFace repo separately

## Alternatives Considered

### Alternative 1: Commit Model to Git Repository

Include model file directly in the git repo.

**Rejected because**:
- Bloats repository size (every clone downloads model)
- Git not designed for large binary files
- Slows down git operations
- Hard to version models independently
- GitHub has file size limits (100 MB)

### Alternative 2: Git LFS (Large File Storage)

Use Git LFS for model versioning.

**Rejected because**:
- Bandwidth limits on free tier (1 GB/month GitHub)
- Complexity for users (need Git LFS installed)
- Still increases repo size
- Limited storage on free plans
- HuggingFace is better suited for ML models

### Alternative 3: Cloud Storage (S3, Google Drive)

Upload to AWS S3 or Google Drive.

**Rejected because**:
- Cost for bandwidth (S3) or storage limits (Google Drive)
- Manual version management
- Less discoverable
- No built-in Python integration
- Google Drive requires authentication for programmatic access

### Alternative 4: GitHub Releases

Attach model as release asset.

**Rejected because**:
- Not designed for continuous model updates
- Awkward API for downloading latest version
- No automatic caching
- Release assets less discoverable
- Not ML-specific

### Alternative 5: Self-Hosted Server

Host model on own server or file hosting.

**Rejected because**:
- Infrastructure costs
- Maintenance burden
- Bandwidth costs
- Reliability concerns
- No built-in versioning
- Less trusted source

## Implementation Notes

- Model automatically downloaded on first run
- Cached locally for subsequent uses
- Fallback to local model path if download fails
- Clear error messages for connection issues

## References

- [HuggingFace Hub Documentation](https://huggingface.co/docs/hub/index)
- [huggingface-hub Python library](https://github.com/huggingface/huggingface_hub)
- Model repository: https://huggingface.co/bbureau12/QuietHorizon
