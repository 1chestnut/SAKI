# Final graph/text inputs

These files are the inputs used by the method and its controls: retained graph
triples, AAKV sound descriptions, class mappings, and fixed baseline texts.
They contain no per-clip predictions, reported metric tables, latency records
or generation audit histories. The full third-party datasets/models are obtained
from their original providers; see `docs/DATA.md` and `THIRD_PARTY_NOTICES.md`.

Use `code/prepare_inputs.py` to encode these files locally. Prepared binary
embeddings are ignored by Git. Source-input hashes are in
`provenance/final_inputs.json`; current public-file hashes are in SHA256SUMS.csv.
