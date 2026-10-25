# SAKI

**Sample-Adaptive Knowledge Integration for Zero-Shot Audio Classification**

Core code for sample-level relation selection, Audio-Aligned Knowledge
Verbalization (AAKV), and Hierarchical Fusion.

## Paper implementation

Scoring and AAKV functions are extracted from the confirmed result-producing
implementations. FSD50K and AudioSet use the corrected Task-31 pipeline;
TUT2017 uses the complete 6,300-clip Task-28 run. ESC-50 and UrbanSound8K
use Task 25. See [the implementation map](docs/RESULT_MAP.md).

Settings: K=5 original top-K classes, M=3 retrieved tails, aggregation scale=100,
fixed original-CLAP weight alpha=0.3, and N_r=5 selected relations out of 47.
The DCASE17-T4 development manifest has 419 clips. Random Top-5 uses seeds
42, 43, and 44.

## Installation and checks

Use Python 3.10:

```bash
python -m pip install -e ".[inference,test]"
python -m pytest tests -q
python code/check_data.py
```

The paper environment uses PyTorch 2.4.0, CUDA 12.1, MS-CLAP 1.3.4,
Transformers 4.57.6, and PyKEEN 1.11.1.

## Core files

- saki/scoring.py: aggregation, fusion, top-1 voting, top-2 score gaps,
  relation priority, and tail deduplication.
- saki/inference.py: relation selection and final class scoring.
- saki/aakv.py: generation instructions, checks, and constrained repair.
- saki/retrieval.py: RotatE tail retrieval, filtering, and text encoding.
- code/generate_aakv.py: offline sound description generation.
- code/predict.py: online inference from audio and prepared graph/text inputs.
- experiments/source_runs/: actual dataset-specific paper implementations.
- experiments/dataset_loaders/: loading and audio preprocessing.
- manifests/: sample IDs, labels, class/relation order, and stored crop seeds.
- configs/paper.json: fixed settings.
- tests/: focused tests of scoring and relation selection.

Only original top-K classes with evidence receive knowledge enhancement.
Each relation predicts over all candidate classes. Consensus-supporting
relations precede other relations; top-2 score gaps order each group.
The original-CLAP fusion weight is fixed across samples.

## Data and execution

Download data and pretrained models from [their original sources](docs/DATA.md).
The repository includes evaluation manifests but does not redistribute audio.
Offline graph and text resources must be prepared locally as described there.

```bash
python code/generate_aakv.py --triples /path/to/triples.json --model /path/to/Qwen2.5-7B-Instruct --output /path/to/aakv.json
python code/predict.py --artifacts /path/to/prepared_inputs --dataset 01_ESC50 --audio-root /path/to/data --weights /path/to/CLAP_weights_2023.pth --output outputs/esc50_saki.json --device cuda
```

The --artifacts argument names prepared input resources, not prediction results.
No reported result files are needed by the prediction command. The portable
audio entry point has not been rerun end to end on a clean GPU environment;
the preserved dataset-specific programs produced the paper results.

## Public release scope

This repository contains code, fixed configuration, manifests, and resource
instructions. Experimental result tables, predictions, logs, plotting data,
raw audio, pretrained weights, and frozen graph/text caches are not distributed
in the current tree. Previous public commits remain in Git history.

Regenerating retrieval/AAKV caches may produce different inputs. The repository
alone does not include all resources needed to reproduce the exact paper numbers.
Third-party resources retain their original terms. A software license has not
yet been selected; see THIRD_PARTY_NOTICES.md.

Repository: https://github.com/1chestnut/SAKI
