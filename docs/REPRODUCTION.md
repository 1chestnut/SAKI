# Reproducing the final manuscript comparisons

All commands assume the package is installed with `pip install -e ".[inference,test]"`.
The examples use ESC-50; substitute one of the dataset IDs in `saki/data.py`.
Download the audio/model resources and prepare embeddings as described in `DATA.md`.

## Main results and class metrics

```bash
python code/prepare_inputs.py --dataset 01_ESC50 --weights /models/CLAP_weights_2023.pth --output resources/01_ESC50/model_inputs.pt
python -m experiments.run --dataset 01_ESC50 --prepared resources/01_ESC50/model_inputs.pt --audio-root /data --weights /models/CLAP_weights_2023.pth --suite main --output outputs/esc50_main
```

Run each dataset's complete manifest for the reported accuracy metrics. `--limit`
is only a smoke test. Output scores/rankings and summaries are generated locally
and excluded from Git. One audio encoding is shared across accuracy branches;
this runner is not used for online latency.

`code/evaluate.py` also evaluates a fresh prediction archive independently:

```bash
python code/evaluate.py --dataset 01_ESC50 --predictions outputs/esc50_main/SoNAR.npz --output outputs/esc50_metrics.json
```

Hit@k tests whether at least one true label is among the first k classes. MRR uses
the highest-ranked true label. Macro-F1 uses top-1 predictions with equal class
weight on single-label datasets; per-class precision/recall/F1 are also emitted.
Multilabel mAP averages non-interpolated per-class AP across all audio clips,
using all true labels and unmodified final class scores. No per-clip softmax,
threshold fitting, score normalization or calibration is introduced.

Stored class orders preserve inference-time score ties. When evaluating a
historical multi-method archive containing only scores, `--reference-samples`
can supply its original per-clip ranks. They are accepted only if consistent
with the score ordering, allowing alternative positions within exact ties.

## Ablations and sensitivity

Use the same arguments with one of these suites:

- `components`: CLAP and the eight T/F/S component combinations. The fixed
  baseline's exact evidence is preserved separately from per-relation evidence.
- `relations`: FrozenRq, Frequency Top-5, Random Top-5 (seeds 42/43/44), SoNAR.
  Random selection uses stored per-clip RNG keys, not a hash of the new local path.
  Each seed is evaluated separately; the summary reports their arithmetic mean.
- `text`: Direct concatenation, Raw triple, AAKV; selection/fusion stay fixed.
- `pools`: 47/30/20 candidate relations. For each dataset, rank relations by the
  number of retained cached class--tail entries, breaking ties by relation name.
  Retain the original relation iteration order within the chosen pool. Each
  clip's voting, relation ranking, five-relation selection and final scoring are
  recomputed from that pool. No new LLM generation is required.
- `grid`: alpha in {0.3,0.5,0.7}, N_r in {1,3,5,10}. On the 416-clip DCASE17-T4
  development set, choose highest Hit@1, exact-tie highest MRR, then smaller N_r.
  Test-set grids are sensitivity diagnostics and do not replace the fixed
  development-selected configuration used by the main results.

## Online latency and cache requirements

```bash
python -m experiments.benchmark --dataset 01_ESC50 --prepared resources/01_ESC50/model_inputs.pt --audio-root /data --weights /models/CLAP_weights_2023.pth --cpus 40 41 --device cuda:0 --output outputs/esc50_cost
```

Choose two CPU cores available on your machine; retain the same GPU and CPU
settings for all configurations and datasets. In the paper, GPU2 and CPU cores
40/41 were used. A different machine should record its own available core IDs.
Run datasets sequentially, with no competing experiment on the measurement GPU.

The timer uses 200 evenly spaced manifest positions, the same sample order,
batch size 1, five timed repetitions, 20 warm-up clips per configuration/repeat,
five cyclic configuration-order rotations, and CUDA synchronization before and
after both timed segments. Every valid repetition is included.

Online latency is the sum of live audio decoding/preprocessing/CLAP encoding and
matching/scoring/selection/fusion/ranking. Graph retrieval, AAKV generation and
text encoding are offline and excluded. Audio embeddings are recomputed during
timing; cached audio embeddings do not enter the timer. The accepted timing
scorer's arithmetic/allocation order is retained in `experiments/_timing_core.py`.
The Python adapter and hardware environment still affect measured latency.

Cache MiB counts class/text tensor storage plus serialized evidence indices.
It does not mean the file size of an entire experiment directory, or include raw
audio/model checkpoints. Peak CUDA allocated memory is measured for the active
configuration; inactive candidate pools remain on CPU. Full-test accuracy comes
from `--suite main` / `--suite pools`, not from the 200 timing clips.

## Paired statistics

Prepare a JSON list of objects with `dataset`, `baseline_npz` and `method_npz`.
Paths are relative to that JSON file. Use the aligned full-test outputs from
`--suite main`, then run:

```bash
python -m experiments.statistics --comparisons /path/to/comparisons.json --output outputs/paired_statistics.json
```

This computes 10,000 paired-bootstrap resamples for Hit@1/MRR differences, exact
two-sided McNemar p values from discordant Hit@1 outcomes, and Holm adjustment
over the supplied comparison family. For the paper's primary comparison, supply
all five datasets in one family. Conditional correction and degradation rates
use initially incorrect and initially correct clips as separate denominators.

## Scope and verification

`docs/VALIDATION.md` lists checks performed on this reorganized release.
The historical server/queue scripts are archived privately rather than used as
public entry points. Full raw-audio GPU validation remains distinct from CPU
scoring parity and score-archive metric verification.
