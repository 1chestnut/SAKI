# SAKI: paper-result implementation

Sample-Adaptive Relational Knowledge Integration for Zero-Shot Audio Classification.

This repository is prepared against the author-confirmed English CAS manuscript,
not against the most recently modified experimental branch. It contains the
executed source-run implementations, exact extracted scoring/AAKV functions,
sample manifests, configuration and CPU result-reconstruction tools.

## What is frozen

| Dataset | Test clips | Classification source | Relation-control source |
|---|---:|---|---|
| ESC-50 | 2,000 | Task 25 | Task 26 |
| UrbanSound8K | 8,732 | Task 25 | Task 26 |
| FSD50K | 10,231 | Task 31, corrected AAKV cache | Task 31 |
| AudioSet | 17,233 | Task 31, corrected AAKV cache | Task 31 |
| TUT2017 | 6,300 | Task 28, complete rerun | Task 28 |

DCASE17-T4 (419 accessible development clips) selects alpha=0.3 and N_r=5.
K=5, M=3 and aggregation scale=100. Random Top-5 reports the mean of seeds
42, 43 and 44. All other experimental choices follow their captured protocols.
The computational-cost table uses the corrected-cache serial measurement dated
2026-10-08: CLAP 21.42, iKnow-dagger 21.95 and SAKI 55.80 ms/sample.
The older Task-29 and concurrent timing trials are not reference results.

## Installation

For CPU result reconstruction:

```bash
python -m pip install -e .
```

For model execution, use the captured Linux/CUDA environment in
`provenance/python_packages.json` and install the inference dependencies:

```bash
python -m pip install -e ".[inference,test]"
```

The paper used PyTorch 2.4.0, CUDA 12.1, MS-CLAP 1.3.4, Transformers 4.57.6,
and PyKEEN 1.11.1. An installation command is not a claim that every platform
has been tested.

## Reconstruct the reported results without rerunning models

Keep the separately supplied `SAKI_artifacts` directory next to this repository:

```bash
python code/reproduce.py --artifacts ../SAKI_artifacts --bootstrap
```

This verifies exact sample/label alignment, archived scores and ranks, all
component branches, relation controls, development selection, 10,000-resample
paired bootstrap intervals, McNemar/Holm results and the final timing means.
Outputs are written to `SAKI_artifacts/rebuilt`, not over the frozen predictions.

## Run SAKI on the frozen evaluation manifest

Obtain the raw audio and MS-CLAP 2023 checkpoint under their original terms.
The artifact bundle supplies the exact retrieval/text resources used by the paper.

```bash
python code/predict.py --artifacts ../SAKI_artifacts --dataset 01_ESC50 \
  --audio-root /path/to/data --weights /path/to/CLAP_weights_2023.pth \
  --output outputs/esc50_saki.json --device cuda --limit 10
```

Omit `--limit` for the full manifest. Do not call a prefix test a full evaluation.
The portable adapter uses the exact extracted paper-scoring functions. A fresh
full GPU run of this adapter on a clean environment has not yet been completed;
the preserved source-run programs remain the provenance of the published results.

## Files

- `saki/`: exact scoring, relation selection, retrieval and AAKV mechanisms.
- `experiments/source_runs/`: result-producing implementations, with private
  path constants replaced by environment-configured paths; source hashes retained.
- `experiments/dataset_loaders/`: dataset-specific loaders and preprocessing context.
- `manifests/`: sample identity/order, labels, stored crop seeds and candidate order.
- `configs/paper.json`: paper-frozen settings; alpha is not sample-adaptive.
- `reference_results/`: canonical unrounded result tables and verification records.
- `code/`: CPU reconstruction, inference adapter, AAKV generation and validation.
- `docs/`: data sources, reproduction scope, result-to-code map and release checks.

## Repository and resource availability

Code repository: https://github.com/1chestnut/SAKI

This code snapshot corresponds to the paper-result verification described above.
The separate `SAKI_artifacts` bundle is not included in this GitHub repository;
a public artifact download location has not yet been assigned. Commands requiring
that bundle cannot be completed from this repository alone. No archive DOI has
been assigned. Raw audio, model weights, credentials and remote job-control scripts
are excluded.

A software license has not yet been selected. Third-party code attribution and
redistribution of graph-derived texts and caches require a separate review;
the local artifact bundle is not a declaration of redistribution permission.
See `docs/RELEASE_CHECKLIST.md` for the remaining reproducibility and release steps.
