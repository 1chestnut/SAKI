# Data, models and exact experimental samples

Raw audio is not redistributed. Use the original access routes and terms:

| Resource | Original access route | Paper use |
|---|---|---|
| ESC-50 | https://github.com/karolpiczak/ESC-50 | 2,000 clips; all 50 candidate classes |
| UrbanSound8K | https://urbansounddataset.weebly.com/urbansound8k.html | 8,732 clips; 10 classes |
| FSD50K | https://zenodo.org/records/4060432 | 10,231 official evaluation clips; 200 classes |
| AudioSet | https://research.google.com/audioset/download.html | 17,233 accessible, decodable evaluation clips; 527 classes |
| TUT2017 | https://dcase.community/challenge2017/download | 4,680 development + 1,620 evaluation clips; 15 classes |
| DCASE17-T4 | https://dcase.community/challenge2017/download | 419 accessible development clips; 17 classes |
| MS-CLAP | https://github.com/microsoft/CLAP | 2023 weights; fixed parameters |
| AKG / RotatE source project | https://github.com/michelolzam/iknow-audio | exact mapping/retrieval resources captured in the artifact bundle |
| Qwen2.5-7B-Instruct | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct | offline AAKV generation |

A dataset URL does not define our evaluation subset. Use the supplied manifests,
not all files currently downloadable. Do not silently replace unavailable files.
For AudioSet and DCASE17-T4, the selected original clips may become unavailable;
record any missing IDs and do not report a smaller rerun as the same protocol.

## Manifest fields

- `sample_index`: zero-based position in the frozen evaluation order.
- `audio_path`: portable `${DATA_ROOT}/...` resource-relative path.
- `true_indices`: all positive class indices; empty targets are not admitted.
- `seed`: recorded sample-specific seed for audio cropping. Preserve this value.
- `classes.json`: exact class order, including multiword/comma-containing labels.
- `relations.json`: the fixed candidate-relation list.

Historical crop seeds were derived from the original path. The original private
paths have been removed, but the resulting seeds are preserved. Rehashing a new
machine's file path would change long-clip crops and is not an exact reproduction.

Audio is resampled to 44.1 kHz and passed to the paper's seven-second MS-CLAP
preprocessing. Short clips are repeated; long clips use the seeded segment.
Dataset-specific loading logic is preserved in `experiments/dataset_loaders`.
The inference adapter uses the frozen manifest, rather than recreating sample
order by scanning a download directory.

## Frozen graph and text resources

`SAKI_artifacts/resources` contains the exact frozen mapping, retrieval cache,
AAKV texts and class/text embeddings used by the selected runs. Original generation
correction records are retained separately. Final AAKV texts were merged from
verified prior texts and exact-head corrections; regenerating all texts is a new
cache, not automatically the paper cache.

CLAP, Qwen, RotatE and third-party graph resources retain their original terms.
Third-party-derived evidence and embeddings require a redistribution review
before a public deposit. A license for our code does not grant rights to those files.
