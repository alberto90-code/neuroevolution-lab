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

For closer comparisons with CI on compatible x86_64/OpenBLAS systems, set `OPENBLAS_NUM_THREADS=1` and `OPENBLAS_CORETYPE=Haswell` **before** starting Python (for example, `OPENBLAS_NUM_THREADS=1 OPENBLAS_CORETYPE=Haswell python train.py`). These are the settings used by the workflow; they reduce one source of numerical variation but do not guarantee identical metrics across machines.

## Evaluation policy

Model selection uses **validation MSE**. A preliminary quality gate requires **validation MSE** ≤ 0.02. The held-out **test MSE** is reported separately and checked for numeric validity only; it is not used to select or accept a model. The model is not a general-purpose AI; it is a small controlled research example. Repeated scheduled runs with an unchanged seed and code reproduce the same experiment rather than automatically discovering new architectures.

## Reproducibility scope

The experiment uses a fixed random seed, pinned Python dependencies, an Ubuntu 24.04 CI runner, and single-threaded OpenBLAS. Tests verify identical metrics and weights across repeated runs in the same environment. **Bit-for-bit agreement across different CI machines is not guaranteed**: runs [#25](https://github.com/alberto90-code/neuroevolution-lab/actions/runs/40322889731) and [#26](https://github.com/alberto90-code/neuroevolution-lab/actions/runs/40331197040) reported slightly different validation and test MSE despite the same training code. CPU/BLAS differences are a possible explanation, not a confirmed root cause. Compare the runtime diagnostics and metrics before interpreting small cross-run changes as model improvements.

## Safety and limits

Do not commit credentials, tokens, or private data. This repository is currently **public**; change its visibility in GitHub settings if you want private access. There is no automatic cross-run promotion or rollback yet.
