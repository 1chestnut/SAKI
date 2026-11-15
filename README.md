# SoNAR

**SoNAR: Sample-Adaptive Knowledge Integration for Zero-Shot Audio Classification**

Implementation of sample-level relation selection, Audio-Aligned Knowledge
Verbalization (AAKV), and Hierarchical Fusion. SoNAR adapts class-related graph
knowledge to each audio clip without updating CLAP parameters.

The Python package is named `sonar`.

## Install

Use Python 3.10 and install a CUDA-compatible PyTorch build on the machine used
for inference. The paper used PyTorch 2.4.0, CUDA 12.1, MS-CLAP 1.3.4,
Transformers 4.57.6, and PyKEEN 1.11.1.

```bash
python -m pip install -e ".[inference,test]"
python -m pytest tests -q
python code/check_data.py
```

## Data and method inputs

Download benchmark audio and model weights from [their original providers](docs/DATA.md).
The repository includes final sample manifests, candidate class/relation order,
stored audio-crop and random-relation seeds, retained graph triples, final AAKV
sound descriptions, and the fixed evidence used by the controlled baseline.
These are method inputs. Prediction outputs, metric tables, timing records,
generation audit logs, raw audio and pretrained weights are not included.

The development set contains **416** accessible DCASE17-T4 clips after excluding
three clips overlapping the AudioSet test set. This is the project's evaluation
subset, not the official dataset size. [Data preparation](docs/DATA.md) documents
the exclusions and audio directory layout.

## Run

First encode the supplied graph/text inputs offline. Then perform inference on
raw audio. Replace the example paths with your own:

```bash
python code/prepare_inputs.py --dataset 01_ESC50 --weights /models/CLAP_weights_2023.pth --output resources/01_ESC50/model_inputs.pt
python code/predict.py --dataset 01_ESC50 --prepared resources/01_ESC50/model_inputs.pt --audio-root /data --weights /models/CLAP_weights_2023.pth --output outputs/esc50_main
```

The prediction command runs CLAP, our controlled reimplementation of iKnow-audio
(`iKnow_dagger`), and SoNAR on the same manifest. It computes Hit@1/3/5, MRR, and
Macro-F1 for single-label datasets or mAP for multilabel datasets. It does not
read previously reported results. Ground-truth labels are used only for evaluation.

For offline generation from triples, including the exact prompts, checks,
one repair attempt and deterministic fallback, see [AAKV](docs/AAKV.md).

## Final manuscript experiments

All accuracy comparisons use the same implementation and immutable CLAP scores:

| Entry point | Purpose |
|---|---|
| `python -m experiments.run --suite main ...` | CLAP, iKnow dagger, SoNAR |
| `--suite components` | Eight graph-knowledge component combinations plus CLAP |
| `--suite relations` | FrozenRq, Frequency Top-5, Random Top-5 (seeds 42/43/44), SoNAR |
| `--suite text` | Direct concatenation, Raw triple, AAKV |
| `--suite pools` | 47/30/20 candidate relations; select five anew for each clip |
| `--suite grid` | alpha in {0.3, 0.5, 0.7}, N_r in {1, 3, 5, 10} |
| `python -m experiments.benchmark ...` | Serial CUDA latency using the final timing protocol |
| `python -m experiments.statistics ...` | Paired bootstrap, prediction transitions, McNemar/Holm |

See [complete commands and measurement boundaries](docs/REPRODUCTION.md).

## Fixed settings

`configs/paper.json`: K=5 initial top-K classes, M=3 retrieved tails per query,
normalized log-sum-exp scale=100, original-CLAP weight alpha=0.3, and N_r=5
selected relations from 47 candidates. Alpha and N_r stay fixed on the five test
datasets. The development selection rule compares Hit@1 first, MRR for exact
ties, then the smaller N_r. Test-set grids are sensitivity diagnostics.

For each relation, only the initial top-K classes with evidence are enhanced;
the top-1 prediction is obtained from scores for all candidate classes.
Relations predicting the consensus class precede other relations; top-2 score
gaps order each group. Duplicate tails keep the AAKV sound description from
the first selected relation.

## Reproducibility and scope

The published text inputs preserve the final evidence, avoiding a new LLM
generation during reproduction. Their source hashes are in `provenance/final_inputs.json`.
Regenerating retrieval or descriptions is a separate input variant. Re-encoding
the supplied texts or audio on a different software/hardware environment may
still affect floating-point scores and exact ties.

The release was checked against the original scoring functions and saved final
score archives. A complete raw-audio CUDA run of the reorganized entry points
has not yet been performed; see `docs/VALIDATION.md` for the exact checks.
Timing scripts implement the stated protocol; latency is hardware-dependent.

Third-party resources retain their original terms. No new software license has
been selected on the authors' behalf. See [third-party notices](THIRD_PARTY_NOTICES.md).

Repository: https://github.com/1chestnut/SoNAR
