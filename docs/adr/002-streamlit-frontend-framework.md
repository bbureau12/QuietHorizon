# 2. Streamlit for Web Frontend

**Date**: 2026-02-18

**Status**: Accepted

## Context

QuietHorizon needed a web interface to make the classifier accessible to non-technical users (conservationists, field researchers, educators). The frontend must:
- Be quick to develop and iterate on
- Provide rich visualizations (waveforms, spectrograms, charts)
- Support file uploads and batch processing
- Integrate seamlessly with Python ML stack
- Require minimal JavaScript/frontend expertise

## Decision

We will use Streamlit as the web framework for the QuietHorizon frontend.

## Consequences

### Positive Consequences

- **Rapid Development**: Built complete frontend in hours, not days
- **Pure Python**: No JavaScript required, leverages existing Python skills
- **Built-in Widgets**: File upload, charts, dataframes, audio player out-of-the-box
- **Auto-Reload**: Changes reflected immediately during development
- **Caching**: `@st.cache_resource` for model loading improves performance
- **Native ML Integration**: Seamless with TensorFlow, librosa, matplotlib, pandas
- **Deployment**: Easy to deploy with Streamlit Cloud or Docker
- **Professional Look**: Clean, modern UI without custom CSS

### Negative Consequences

- **Limited Customization**: Less control over UI/UX than React/Vue
- **Single Page Apps**: Best for simple navigation patterns
- **State Management**: Can be tricky for complex user interactions
- **Performance**: Not ideal for real-time streaming or very large datasets
- **Mobile**: Responsive but not mobile-optimized
- **API**: Not designed as a REST API (though can be done with separate FastAPI)

## Alternatives Considered

### Alternative 1: Flask/FastAPI + React

Traditional web stack with Python backend and React frontend.

**Rejected because**:
- Much longer development time (days to weeks)
- Requires JavaScript expertise
- More complex deployment
- Overhead for the use case
- Harder to iterate on visualizations

### Alternative 2: Gradio

Similar to Streamlit, specialized for ML demos.

**Rejected because**:
- Less flexible for custom layouts
- Fewer visualization options
- Smaller community and ecosystem
- Streamlit has better caching and state management
- Gradio better for quick demos, Streamlit better for apps

### Alternative 3: Desktop Application (PyQt/Tkinter)

Build a native desktop application.

**Rejected because**:
- No web access (requires installation)
- Harder to share and deploy
- Platform-specific issues
- Less accessible to broad audience
- More complex distribution

### Alternative 4: Jupyter Notebook/Voilà

Convert notebook to web app with Voilà.

**Rejected because**:
- Less polished UI
- Harder to structure as proper application
- Limited customization
- Not designed for production apps
- Voilà still relatively immature

## References

- [Streamlit Documentation](https://docs.streamlit.io)
- [Streamlit vs Gradio comparison](https://blog.streamlit.io/streamlit-vs-gradio/)
- Frontend QUICKSTART.md and README.md
