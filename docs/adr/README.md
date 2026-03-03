# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) for QuietHorizon.

## What is an ADR?

An Architecture Decision Record (ADR) captures an important architectural decision made along with its context and consequences.

## ADR Format

Each ADR follows this structure:

- **Title**: Short noun phrase
- **Status**: Proposed, Accepted, Deprecated, Superseded
- **Context**: What is the issue we're facing?
- **Decision**: What are we doing about it?
- **Consequences**: What becomes easier or harder because of this decision?

## Index

| ADR | Title | Status |
|-----|-------|--------|
| [001](001-cnn-based-classification.md) | CNN-Based Audio Classification Using Spectrograms | Accepted |
| [002](002-streamlit-frontend-framework.md) | Streamlit for Web Frontend | Accepted |
| [003](003-modular-frontend-architecture.md) | Modular Frontend Architecture | Accepted |
| [004](004-huggingface-model-hosting.md) | HuggingFace Hub for Model Distribution | Accepted |
| [005](005-mel-spectrogram-preprocessing.md) | Mel-Spectrogram Audio Preprocessing | Accepted |
| [006](006-binary-classification-approach.md) | Binary Classification (Nature vs Anthropogenic) | Accepted |
| [007](ADR-007-mcp-server-implementation.md) | MCP Server Implementation | Accepted |
| [008](008-scripted-model-evaluation-and-adr-first-change-tracking.md) | Scripted Model Evaluation and ADR-First Change Tracking | Accepted |

## Creating New ADRs

1. Copy the template from `000-template.md`
2. Number sequentially (e.g., `007-your-decision.md`)
3. Fill in all sections
4. Update this index
5. Commit with other related changes
