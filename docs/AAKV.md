# AAKV: offline sound-description construction

For ordinary reproduction, use the final descriptions in `inputs/<dataset>/aakv.json`.
This avoids a new LLM generation changing the experiment's text inputs.
To regenerate descriptions, first export triples from a retained graph:

```bash
python code/export_triples.py --graph inputs/01_ESC50/graph.json --output outputs/esc50_triples.json
python code/generate_aakv.py --triples outputs/esc50_triples.json --model /models/Qwen2.5-7B-Instruct --output outputs/esc50_aakv_regenerated.json
```

The class name is the supplied head text; the AKG entity is used for retrieval.
The exact system/repair prompts, forbidden expressions and relation-term checks
are in `saki/aakv.py`. Generation uses Qwen2.5-7B-Instruct and greedy decoding
with at most 48 new tokens. The automatic checks cover prefix, head/tail content
words, relation terms, length, line/sentence formatting and prohibited expressions.
An output failing checks receives one constrained repair attempt using the
reported failure reasons. A second failure triggers the deterministic template:

`The sound of {head} has the relation {relation} with {tail}.`

This is the implemented fallback, retained for fidelity to the experiments.
Checks constrain text expression; they do not verify the factual accuracy of
retrieved triples. Validation checks are heuristic and do not certify semantic
or factual correctness. The generated audit remains a local output, excluded
from the public release.

Pass a regenerated text file to `code/prepare_inputs.py --aakv ...` only when
intentionally evaluating that input variant. Do not replace final descriptions
silently when comparing with the reported results.
