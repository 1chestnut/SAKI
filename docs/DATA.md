# Data and pretrained model sources

| Resource | Source | Paper use |
|---|---|---|
| ESC-50 | https://github.com/karolpiczak/ESC-50 | 2,000 clips; 50 classes |
| UrbanSound8K | https://urbansounddataset.weebly.com/urbansound8k.html | 8,732 clips; 10 classes |
| FSD50K | https://zenodo.org/records/4060432 | 10,231 evaluation clips; 200 classes |
| AudioSet | https://research.google.com/audioset/download.html | 17,233 accessible evaluation clips; 527 classes |
| TUT2017 | https://dcase.community/challenge2017/download | 4,680 development + 1,620 evaluation clips; 15 classes |
| DCASE17-T4 | https://dcase.community/challenge2017/download | 419 development clips; 17 classes |
| MS-CLAP | https://github.com/microsoft/CLAP | 2023 checkpoint |
| AKG / RotatE | https://github.com/michelolzam/iknow-audio | graph, entity mapping, and trained graph model |
| Qwen2.5-7B-Instruct | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct | offline AAKV generation |

## Manifests and preprocessing

Supplied manifests define sample identity, order, labels, class order, and
stored audio crop seeds. Preserve these seeds when moving audio to a new path.
The dataset loaders resample to 44.1 kHz and use seven-second MS-CLAP segments:
short clips are repeated; long clips use the stored seed for cropping.
Raw audio and third-party model weights are obtained from their providers.

## Offline preparation

Obtain the original AKG, class-to-entity mapping, RotatE checkpoint, and matching
entity/relation ID mappings. The function saki.retrieval.build_static_cache
retrieves M=3 tails per query, filters candidate-class tails, deduplicates
normalized names, and encodes the direct prompts.

For every retained tail, construct a JSON triple with head, relation, and tail.
Use the candidate class name as head. code/generate_aakv.py generates AAKV
sound descriptions with the exact checks, one repair, and deterministic fallback.

Encode original class texts using MS-CLAP 2023 and normalize their embeddings.
Preserve the original class texts and candidate order when comparing runs.
Dataset-specific source implementations retain the original preparation context.

## Input layout for code/predict.py

- manifests/<dataset>/samples.json: the supplied sample manifest.
- resources/<dataset>/static_relation_cache.pt: retrieval output containing
  labels, relations, and tails.
- resources/<dataset>/text_embeddings.pt: a dictionary whose labels tensor
  contains normalized original class-text embeddings in class order.
- resources/aakv/aakv_all.json: a texts dictionary mapping tab-separated
  class-name/relation/tail keys to AAKV sound descriptions.

code/generate_aakv.py writes this texts dictionary format. Merge datasets only
when duplicate triple keys have identical descriptions. The inference adapter
reads prepared resources, re-encodes AAKV texts, and processes audio. It does not
read archived predictions or metric tables; test labels are not selection inputs.

Frozen caches are not included in this code-only release. Regenerated caches
may differ from the paper inputs. The datasets, models, and graph resources
retain their providers' access and usage terms.
