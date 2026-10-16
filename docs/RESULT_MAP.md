# Paper item to data and implementation

| Current paper item | Frozen outputs in SAKI_artifacts | Code |
|---|---|---|
| Table 1: overall results | results/factorial; rebuilt/table1_main.csv | main_esc_urban.py, main_fsd_audioset_corrected.py, main_tut6300.py |
| Table 2 and Table D.1: component ablations | results/factorial; rebuilt/table2_ablation.csv | the same nine-branch main programs |
| Table 3: relation selection | results/selector; rebuilt/table3_selector_all_seeds.csv | selector_esc_urban.py, selector_fsd_audioset_corrected.py, selector_tut6300.py |
| Table 4: text construction | factorial FS/TFS_Final; results/raw_triple | raw_triple_esc_urban_fsd_audioset.py, raw_triple_tut6300.py and main programs |
| Table 5: computational cost | results/efficiency; corrected serial run, attempt_1 | efficiency_serial_corrected.py |
| Figure 2 and Table F.1 | rebuilt/paired_statistics.csv | code/verify_release.py --bootstrap |
| Figure 3: prediction transitions | same paired-statistics output and original plotting CSV | code/verify_release.py; figures/plot_prediction_transitions.py |
| Figure 4 / relation-exchange case | results/paper_cases | exact case records; selected-relation scoring in saki/scoring.py |
| Footsteps and air-conditioner cases | results/paper_cases | fixed-text/fixed-relation comparison records |
| Figure 5: parameter sensitivity | results/development; rebuilt/development_grid.csv | development_corrected.py and development_core.py |
| Table D.2: fixed relation sets | selector protocol.json and resource frozen_inputs/config.json | FrozenRq / Frequency Top-5 controls |
| Table E.1: selector behavior | rebuilt/selector_behavior.csv; relation_frequency.csv | code/verify_release.py; selector_behavior.py |

Source-run names in this table refer to `experiments/source_runs`.
Historical filenames and table numbers are not authoritative. In particular,
the old source `Table3_AAKV_verbalization.csv` panel A and the old Task-26
package are not substituted for the corrected paper results.

`reference_results` contains results recomputed from the unrounded sample-level
records. Means use equal dataset weights. Performance differences are computed
before rounding. The currently approved Raw-triple MRR values round to 95.76
for ESC-50 and 70.99 for FSD50K.
