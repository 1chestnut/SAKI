# Data sources and final protocols

| Resource | Official source | Use in this work |
|---|---|---|
| ESC-50 | https://github.com/karolpiczak/ESC-50 | 2,000 clips; 50 classes; single-label |
| UrbanSound8K | https://urbansounddataset.weebly.com/urbansound8k.html | 8,732 clips; 10 classes; single-label |
| FSD50K | https://zenodo.org/records/4060432 | 10,231 evaluation clips; 200 classes; multilabel |
| AudioSet | https://research.google.com/audioset/download.html | 17,233 accessible evaluation clips; 527 classes; multilabel |
| TUT2017 | https://dcase.community/challenge2017/download | 4,680 development + 1,620 evaluation clips; 15 classes; single-label |
| DCASE17-T4 | https://dcase.community/challenge2017/task-large-scale-sound-event-detection | 416 development clips; 17 classes; multilabel |
| MS-CLAP | https://github.com/microsoft/CLAP | MS-CLAP 2023 checkpoint |
| AKG / pretrained RotatE | https://github.com/michelolzam/iknow-audio | Graph, entity mapping, and trained graph model |
| Qwen2.5-7B-Instruct | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct | Offline AAKV generation |

Raw audio, the full third-party graph and pretrained checkpoints are obtained
from the providers under their respective terms. AudioSet availability can
change; the supplied manifest defines the exact accessible evaluation subset
used in the paper. Do not silently replace missing clips with other recordings.

## Final manifests

`manifests/<dataset>/samples.json` supplies sample identity/order, all recorded
ground-truth label indices, the accepted crop seed, and the random-selection RNG
keys for seeds 42/43/44. `classes.json` and `relations.json` fix candidate order.
Keep these orders and seeds when relocating data. Test labels are never inputs
to graph retrieval, relation selection or fusion.

Three DCASE17-T4 development clips overlap AudioSet evaluation clips and are
excluded from the final 416-clip development manifest:

| Source ID | Segment |
|---|---|
| -6cTEqIcics | 30.000--40.000 s |
| 1oJAVJPX0YY | 20.000--30.000 s |
| A-GNszKtjJc | 93.000--103.000 s |

The official DCASE list is linked by the task page:
https://raw.githubusercontent.com/ankitshah009/Task-4-Large-scale-weakly-supervised-sound-event-detection-for-smart-cars/master/testing_set.csv
The AudioSet evaluation metadata are at:
https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv
416 is the project-specific accessible subset after excluding cross-set overlap;
it is not a claim that the official release contains only 416 clips. Retained
sample indices and crop seeds are unchanged and need not be contiguous.

## Audio directory layout

Each `audio_path` is relative to `${DATA_ROOT}`. Download/extract the official
audio, then arrange it so that replacing this prefix with `--audio-root` resolves
every file. For example, `${DATA_ROOT}/ESC-50/audio/1-100032-A-0.wav` resolves to
`/data/ESC-50/audio/1-100032-A-0.wav` when `--audio-root /data` is supplied.
The manifests specify the corresponding layout for all other datasets.

```bash
python code/check_data.py --audio-root /data
```

MS-CLAP 2023 performs audio decoding/resampling to 44.1 kHz and uses seven-second
inputs. Short audio is repeated; long audio is cropped under the stored seed.
The original CLAP text inputs are class names, without added acoustic descriptions.

## Published method inputs

Each `inputs/<dataset>/` contains:

- `class_to_entity.json`: the fixed class-to-AKG entity mapping; null means unmapped.
- `graph.json`: retained class--relation--tail evidence after top-M retrieval,
  candidate-class filtering and normalized-tail deduplication.
- `aakv.json`: final AAKV sound descriptions keyed by the exact class name,
  relation and normalized tail, separated by tabs.
- `fixed_evidence.json`: fixed relation set and exact enriched prompts / AAKV
  sound descriptions used in the controlled baseline and component comparisons.

These are model inputs, not prediction results, metric tables or generation logs.
The full AKG, source audio and model weights are not redistributed. Graph-derived
entity/relation strings retain the upstream resource's applicable terms.

Run `code/prepare_inputs.py` to encode these inputs locally into a trusted
`model_inputs.pt` file. This offline file is excluded from Git. The prepared
schema contains normalized class-name embeddings, direct/Raw triple/AAKV text
embeddings, fixed-baseline embeddings and their evidence indices.

## Regenerating graph evidence

Use the pretrained RotatE checkpoint with its exact entity/relation ID maps.
Changing the maps changes the meaning of scores and is not a compatible substitution.

```bash
python code/prepare_graph.py --classes manifests/01_ESC50/classes.json --relations manifests/01_ESC50/relations.json --class-to-entity inputs/01_ESC50/class_to_entity.json --entity-to-id /akg/entity_to_id.json --relation-to-id /akg/relation_to_id.json --kge-model /akg/trained_model.pkl --output outputs/regenerated_graph.json
python code/export_triples.py --graph outputs/regenerated_graph.json --output outputs/triples.json
```

Only trusted local checkpoints should be loaded. Retrieval takes three tails
first, removes candidate-class tails and duplicates after normalization, and
does not replace removed tails. Missing class mappings provide no graph evidence.
Newly retrieved/generated inputs may differ from the published final inputs.
