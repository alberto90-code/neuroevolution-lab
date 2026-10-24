# NeuroEvolution Lab

A reproducible, CPU-only neural-network experiment in NumPy. It learns the synthetic function `y = x1 * x2` and compares hidden-layer widths 8, 16, and 32 using the same train/validation/test split.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python train.py
python quality_gate.py
```

Outputs are written to `artifacts/metrics.json` and `artifacts/best_weights.npz`. GitHub Actions uploads them as downloadable run artifacts (14-day retention). The workflow runs on code changes, manual dispatch, and Mondays at 03:17 UTC.

## Evaluation policy

Model selection uses **validation MSE**. The held-out **test MSE** is reported separately and checked against a preliminary threshold of 0.02. The model is not a general-purpose AI; it is a small controlled research example. Repeated scheduled runs with an unchanged seed and code reproduce the same experiment rather than automatically discovering new architectures.

## Safety and limits

Do not commit credentials, tokens, or private data. This repository is currently **public**; change its visibility in GitHub settings if you want private access. There is no automatic cross-run promotion or rollback yet.
