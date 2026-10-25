# Paper components and source code

| Component | Implementation |
|---|---|
| Relation selection and fusion | saki/scoring.py; saki/inference.py |
| Retrieval and filtering | saki/retrieval.py |
| AAKV | saki/aakv.py; code/generate_aakv.py |
| ESC-50 / UrbanSound8K comparison | experiments/source_runs/main_esc_urban.py |
| Corrected FSD50K / AudioSet comparison | experiments/source_runs/main_fsd_audioset_corrected.py |
| Complete TUT2017 comparison | experiments/source_runs/main_tut6300.py |
| Relation-selection controls | experiments/source_runs/selector_*.py |
| Raw-triple controls | experiments/source_runs/raw_triple_*.py |
| Corrected AAKV cache | experiments/source_runs/aakv_corrected_*.py |
| Development selection | experiments/source_runs/development_corrected.py; development_core.py |
| Serial efficiency measurement | experiments/source_runs/efficiency_serial_corrected.py |
| Statistical analysis | experiments/source_runs/statistics.py |
| Dataset preprocessing | experiments/dataset_loaders/ |

FSD50K and AudioSet use the corrected Task-31 sources; TUT2017 uses the
complete Task-28 version. Portable source-run paths use environment variables.
Their historical input layouts are retained. Generated results and plotting
data are excluded. Source hashes are retained in provenance/source_registry.json.
