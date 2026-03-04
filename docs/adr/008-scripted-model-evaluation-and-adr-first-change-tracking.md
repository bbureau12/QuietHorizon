# ADR-008: Scripted Model Evaluation and ADR-First Change Tracking

**Status:** Accepted  
**Date:** 2026-03-04  
**Deciders:** QuietHorizon Maintainers  
**Tags:** evaluation, quality, process, adr

## Context

QuietHorizon had model quality information in notebooks and ad hoc tests, but no dedicated CLI evaluation workflow that teams can run repeatedly across datasets. This made it harder to compare model artifacts, verify regressions, and share exact metrics in a reproducible way.

At the same time, architecture and process decisions were partially documented as ADRs, but not consistently extended when introducing new core workflows.

## Decision

1. Add a first-class evaluation CLI script at:
- `quiet_horizon/evaluation/evaluate_cnn.py`

2. Standardize evaluation output to include:
- Accuracy
- Precision/Recall/F1 (anthro as positive class)
- ROC-AUC (anthro)
- Confusion matrix
- Per-file predictions and failed-file diagnostics

3. Support two dataset input modes:
- Labeled folder scans (`--dataset-root`)
- CSV manifest (`--manifest`) with `path,label`

4. Add project documentation for evaluation usage:
- `docs/evaluation.md`
- README links and usage examples

5. Automatically generate a confusion matrix image (`.png`) for each evaluation run:
- Default output: `confusion_matrix.png`
- Optional override: `--output-confusion-matrix`

6. Integrate evaluation directly into the Streamlit app:
- Add an `Evaluation` tab in `frontend/app.py`
- Display metrics, confusion matrix image, JSON report, and failed-file details in-app

7. Add one-click demo evaluation in Streamlit:
- Evaluate fixed demo files (`Northern Cardinal`, `Heavy Traffic`)
- Use expected labels (`nature`, `anthro`) to provide a fast validation path

8. Add a Streamlit `Model Card` tab:
- Present intended use, data summary, metrics, limitations, and risk guidance in-app
- Improve discoverability of responsible-usage information for non-CLI users

9. Adopt ADR-first tracking for architecture/process changes:
- Significant workflow, architecture, and model-ops decisions should be captured as ADRs in `docs/adr/`.

## Consequences

### Positive Consequences

- Reproducible evaluation results that are easy to rerun and compare.
- Clear file-level diagnostics for bad/unsupported samples.
- Faster artifact validation when switching between `.weights.h5` and `.keras` models.
- Evaluation visibility for non-CLI users via the Streamlit UI.
- Immediate visual artifact (confusion matrix image) for sharing and reporting.
- Better transparency for model assumptions and limitations via in-app model card content.
- Better long-term project memory through explicit ADR records.

### Negative Consequences

- Additional maintenance burden for evaluation CLI and docs.
- Additional frontend complexity and UI testing surface area.
- Extra output artifact management (report and image files).
- Need to maintain consistency between model card content and evolving model artifacts.
- Slightly more process overhead when introducing significant architectural changes.

## Alternatives Considered

### Notebook-Only Evaluation

Rejected because notebooks are less reproducible in automated environments and harder to integrate into CI workflows.

### Test-Only Evaluation

Rejected because test assertions are not a substitute for rich metric reporting and confusion-matrix style model analysis.

### No ADR Process Expansion

Rejected because prior decisions were already captured as ADRs, and inconsistent usage reduced traceability over time.

## References

- `quiet_horizon/evaluation/evaluate_cnn.py`
- `frontend/app.py`
- `docs/evaluation.md`
- `docs/adr/README.md`
- `docs/adr/ADR-007-mcp-server-implementation.md`
