# Reproduction levels and environment

## 1. Completed and model-free

The CPU reconstruction checks all 44,496 final test recordings and the 419
development recordings, full score/rank consistency, branch equivalences,
10,000 paired bootstrap resamples and McNemar/Holm statistics. It is an audit
of the actual archived results, not a new GPU inference run.

```bash
python code/reproduce.py --artifacts ../SAKI_artifacts --bootstrap
```

## 2. Portable inference interface

`code/predict.py` uses the frozen manifests and graph resources with raw audio.
It reproduces the SAKI mechanism through the exact extracted scoring functions.
It never receives ground-truth labels as a relation-selection input. New outputs
are saved separately and should be compared to the frozen predictions before
claiming exact reproduction. Use a new output filename; do not overwrite artifacts.

`code/generate_aakv.py` exposes the exact prompts, checks, one-repair rule and
deterministic fallback. Use frozen text resources for reconstruction of paper
results. Regeneration is provided for inspecting/reusing the method.

## 3. Executed source-run implementations

`experiments/source_runs` preserves the actual per-dataset versions that produced
the selected results. Only absolute path constants are transformed; original
source hashes and transformed hashes are recorded in `provenance/source_registry.json`.
The consolidated `saki/scoring.py`, `saki/aakv.py` and retrieval functions are
extracted verbatim from those sources.

The source-run programs retain historical directory roles. They are provenance
implementations, not all a second universal CLI. Their path tokens are resolved
by `saki.paths.release_path` using explicit environment variables:

- SAKI_RUN_ROOT: experiment workspace with the captured historical layout.
- SAKI_DATA_ROOT: original dataset metadata/audio layout.
- SAKI_MODEL_ROOT: CLAP/Qwen and tokenizer resources.
- SAKI_KGE_ROOT: captured RotatE model directory and training triples.
- SAKI_CLAP_SOURCE: tokenizer/model support resources used by the original loader.

Some loader paths have additional dataset-specific subdirectories; inspect the
captured loader rather than guessing a replacement. End-to-end portability of
all historical runners has not been asserted. The single-clip/full-manifest
adapter is the entry point for a newly configured machine.

## Important numerical details

- The relation vote is over all candidate classes, not only the original Top-5.
- Only original Top-5 classes with nonempty evidence receive fusion.
- Every other class retains its full original CLAP score.
- Consensus-supporting relations precede all other relations; gaps order each group.
- Missing evidence for one selected relation does not trigger a replacement relation.
- Tail deduplication retains the first selected relation's text.
- alpha=.3 is fixed across samples, not sample-adaptive.
- PyTorch sorting behavior is preserved in source runs. A new hardware/library
  stack may resolve exact score ties differently; compare saved class scores too.

Never load untrusted pickle/Torch caches. Artifact tensors are captured project
resources and should be verified against the checksum manifest before use.
