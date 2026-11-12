# Validation of the reorganized release

The following checks were completed before publishing version 0.2.0:

- 12 automated tests passed, covering missing evidence, all-class voting,
  consensus priority, first-occurrence tail deduplication, reduced candidate
  pools, stored random-selection seeds, non-refilling retrieval filters,
  Macro-F1, multilabel AP, exact score ties and Holm adjustment.
- All six final manifests passed identity, label-index, class/relation-order,
  exact AAKV-key and nonempty-evidence mapping checks. The development manifest
  contains 416 clips and has no source-ID overlap with the AudioSet test manifest.
- 54 scoring parity cases matched the original result-producing scoring
  functions and the retained timing scorer exactly. These use real class and
  direct-evidence embeddings with deterministic synthetic audio vectors, across
  six datasets and 47/30/20 relation pools. They test scoring/indexing logic,
  rather than real-audio model performance.
- The public metric implementation was run on the final saved score archives
  for CLAP, iKnow dagger and SoNAR on all five test datasets. All 15 evaluations
  matched the saved Hit@1/3/5 and MRR values and the independently checked
  Macro-F1 or mAP values. Historical ranks were retained only within valid exact
  score ties. No prediction archives or resulting metric tables are published.
- An offline-preparation/CLI smoke test passed for all six accuracy suites and
  independent evaluation, using a deterministic CPU substitute for CLAP. This
  verifies file interfaces and control flow; it is not a CLAP inference test.

Validation used Python 3.11, PyTorch 2.5.1+cpu, and the locally installed NumPy /
scikit-learn. The paper environment used Python 3.10 and PyTorch 2.4.0 with CUDA
12.1. A complete raw-audio CUDA run of the reorganized entry points, real Qwen
generation, and new latency measurements were not performed during this release.
The published paper results and timings were not replaced by these checks.

Source files and final method inputs are listed in SHA256SUMS.csv. To verify a
downloaded repository, run `python code/check_release.py`.
