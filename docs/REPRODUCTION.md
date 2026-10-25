# Execution and implementation notes

The core scoring, retrieval, and AAKV functions are preserved from the confirmed
paper implementations. The source map is in RESULT_MAP.md.

Run python -m pytest tests -q with the inference/test dependencies installed.
Tests cover empty-evidence fallback, all-class voting, consensus-support priority,
and first-occurrence tail deduplication. Run python code/check_data.py to check
sample manifests, class labels, and stored seeds.

Prepare resources following DATA.md and run code/predict.py for new audio
predictions. Archived result files are not required. The fixed paper settings
are in configs/paper.json. A full clean-environment GPU rerun of the portable
adapter has not been completed; the dataset-specific source programs produced
the paper results.

Original source-run paths are configured through SAKI_RUN_ROOT, SAKI_DATA_ROOT,
SAKI_MODEL_ROOT, SAKI_KGE_ROOT, and SAKI_CLAP_SOURCE. These identify the experiment
workspace, dataset/audio root, CLAP/Qwen resources, RotatE resources, and CLAP
tokenizer support resources. Inspect each program's inputs before running it.

Regeneration of graph or AAKV caches may change evidence. Raw audio, pretrained
weights, frozen caches, logs, and result tables are not included in this repository.
