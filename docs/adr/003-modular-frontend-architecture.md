# 3. Modular Frontend Architecture

**Date**: 2026-02-18

**Status**: Accepted

## Context

The frontend needs to be maintainable, testable, and extensible. A monolithic `app.py` file would become unwieldy as features are added. We need clear separation between:
- UI presentation logic
- Business logic (audio processing, model inference)
- Configuration
- Utilities

## Decision

Organize the frontend with a modular structure:

```
frontend/
├── app.py                    # Main entry point, orchestration only
├── config.py                 # Centralized configuration
├── utils/                    # Business logic (UI-agnostic)
│   ├── model_loader.py
│   ├── audio_processor.py
│   └── visualization.py
└── components/               # UI components (Streamlit-specific)
    ├── upload.py
    ├── results.py
    └── batch.py
```

**Separation of Concerns**:
- `utils/` contains pure functions that don't depend on Streamlit
- `components/` contains Streamlit-specific rendering logic
- `config.py` holds all constants and settings
- `app.py` is thin orchestration layer

## Consequences

### Positive Consequences

- **Maintainability**: Easy to find and modify specific functionality
- **Testability**: Pure functions in `utils/` can be unit tested without Streamlit
- **Reusability**: Utility functions can be used in CLI tools, APIs, notebooks
- **Readability**: Clear responsibility for each module
- **Team Development**: Multiple developers can work on different modules
- **Extensibility**: New features can be added without touching existing code
- **Configuration**: Centralized settings make adjustments easy

### Negative Consequences

- **Initial Complexity**: More files to navigate initially
- **Import Overhead**: More import statements
- **Over-Engineering Risk**: Could be overkill for very small apps
- **Learning Curve**: New contributors need to understand the structure

## Alternatives Considered

### Alternative 1: Single File App

Put everything in `app.py`.

**Rejected because**:
- Becomes unmanageable beyond ~500 lines
- Difficult to test
- Hard to reuse logic
- Merge conflicts with multiple contributors
- Poor separation of concerns

### Alternative 2: Traditional MVC Pattern

Implement strict Model-View-Controller architecture.

**Rejected because**:
- Overkill for Streamlit's reactive paradigm
- Streamlit's state management doesn't fit traditional MVC
- More complexity than needed
- Harder to understand for Python/ML developers

### Alternative 3: Flat Module Structure

All modules at the same level (no `utils/` or `components/` directories).

**Rejected because**:
- Harder to distinguish UI from business logic
- Directory becomes cluttered with many files
- No clear organization principle
- Difficult to understand at a glance

## Design Principles Applied

1. **Single Responsibility**: Each module has one clear purpose
2. **Dependency Inversion**: High-level `app.py` depends on abstractions in `utils/`
3. **Don't Repeat Yourself (DRY)**: Common functionality extracted to utilities
4. **Separation of Concerns**: UI separate from business logic
5. **Interface Segregation**: Components expose minimal, focused APIs

## References

- Frontend ARCHITECTURE.md
- [Clean Code principles](https://www.amazon.com/Clean-Code-Handbook-Software-Craftsmanship/dp/0132350882)
- [Python Application Layouts](https://realpython.com/python-application-layouts/)
