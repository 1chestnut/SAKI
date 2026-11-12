# Final manuscript implementation map

| Manuscript component | Public implementation |
|---|---|
| Class-to-entity mapping, final retained graph evidence | inputs/<dataset>/class_to_entity.json and graph.json |
| RotatE top-M retrieval/filtering | code/prepare_graph.py |
| AAKV prompts/checks/repair/fallback | saki/aakv.py, code/generate_aakv.py |
| Final AAKV sound descriptions | inputs/<dataset>/aakv.json |
| Fixed controlled-baseline evidence | inputs/<dataset>/fixed_evidence.json |
| Offline text encoding | code/prepare_inputs.py |
| Online audio and scoring | saki/runtime.py, saki/scoring.py, saki/inference.py |
| Main results / component ablations | experiments.run --suite main / components |
| Relation-selection comparisons | experiments.run --suite relations |
| Text-representation comparisons | experiments.run --suite text |
| Development grid / test sensitivity | experiments.run --suite grid |
| 47/30/20 candidate-pool comparison | experiments.run --suite pools |
| Final serial online timing | experiments/benchmark.py and _timing_core.py |
| Ranking, Macro-F1, mAP, per-class metrics | saki/metrics.py, code/evaluate.py |
| Paired bootstrap, transitions, McNemar/Holm | saki/statistics.py, experiments/statistics.py |

Historical result-producing source files are preserved in the author's private
backup. Public entry points consume method inputs and raw audio, without reading
reported result tables or archived model predictions.
