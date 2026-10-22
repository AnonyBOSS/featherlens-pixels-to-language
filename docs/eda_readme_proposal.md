# Proposed README contribution

These additions are incorporated in the coordinator documentation follow-up;
see the [updated README](../README.md). This retained proposal records the
contribution while preserving the original project overview and model roadmap.

Add statistical EDA to the development-status sentence. After the native-image
access section, add:

```markdown
### Available statistical EDA

Statistical EDA covers all 5,994 raw official-training images and 59,940 captions,
including native-image measurements, caption analysis, saved figures, and
manifest-aware train/validation summaries. See [the EDA handoff](docs/eda_handoff.md)
for setup, working commands, report and figure locations, findings, and limitations.
Official test images and captions are excluded from statistical summaries.
```

Add these entries to the existing project tree:

```text
│   ├── analysis/         # Statistical EDA and report generation
├── requirements-eda.txt # Pinned statistical EDA dependencies
```

Replace the sentence that calls visualization a placeholder with:

> Model, training and evaluation implementations remain planned. Statistical
> analysis and EDA visualization modules are available.

Near the installation instructions, add:

```markdown
For statistical EDA, use `requirements-eda.txt` and the
[EDA setup and reproduction guide](docs/eda_handoff.md). EDA-only tests run with
`python -m pytest -q tests/analysis`; the complete repository suite also needs
the [classification dependencies](docs/classification_data.md).
```

No training commands, model results, fitted normalization values or deferred
analyses should be described as implemented.
