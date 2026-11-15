"""Exact AAKV prompts, checks and greedy decoding used by the final generation pipeline."""
import re
import torch

SYSTEM_PROMPT = """You are a faithful knowledge-graph verbalizer for an audio-text matching model.
Convert exactly one knowledge-graph triple into one short, natural English sentence about sound.

Rules:
1. Begin the sentence with exactly: "The sound of {head}". Replace {head} with the supplied head text exactly.
2. Use only information explicitly contained in the triple.
3. Preserve the head entity, relation meaning, relation direction, and tail entity.
4. Preserve the original wording of the head and tail. Do not pluralize, abbreviate, replace, or omit either entity.
5. Do not add acoustic properties, causes, locations, purposes, examples, explanations, or external knowledge.
6. Do not correct or reinterpret the factual content, even if it seems unusual.
7. Add only grammatical function words such as articles, prepositions, and forms of "be".
8. Silently verify that the sentence expresses exactly the supplied triple.
9. Output only one final sentence, without reasoning, labels, headers, or quotation marks."""

REPAIR_PROMPT = """Your preceding sentence failed an automatic fidelity check.
Rewrite it as exactly one short English sentence beginning with the exact words "The sound of {head}".
Copy the supplied head and tail text exactly and explicitly express the supplied relation in its original direction.
Do not add, correct, explain, or infer anything. Output only the sentence."""

RELATION_TERMS = {
    "belongs to class": (("belong", "class"),),
    "has parent": (("parent",),),
    "perceived as": (("perceiv",),),
    "event composed of": (("compos",), ("consist",), ("character",)),
    "has children": (("child",), ("subtype",), ("include",)),
    "overlaps with": (("overlap",),),
    "occurs in": (("occur",),),
    "associated with environment": (("associat", "environment"),),
    "localized in": (("localiz",), ("located",)),
    "used for": (("used", "for"),),
    "part of scene": (("part", "scene"),),
    "is sound of": (("sound", "of"),),
    "transcribed as": (("transcrib",),),
    "indicates": (("indicat",),),
    "described by": (("describ",),),
    "is instance of": (("instance",),),
    "is a type of": (("type",),),
    "is variant of": (("variant",),),
    "scene contains": (("scene", "contain"),),
}

FORBIDDEN_CONTENT = {
    "rhythm", "rhythmic", "pitch", "pitched", "timbre", "frequency",
    "frequencies", "transient", "loud", "loudness", "quiet", "high-pitched",
    "low-pitched", "harmonic", "melodic", "resonant", "sharp", "soft",
}

def norm(text):
    text = str(text).lower().replace("_", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())

def content_tokens(text):
    stop = {"a", "an", "the", "of", "to", "and", "or", "in", "on", "with"}
    return [x for x in norm(text).split() if x not in stop]

def entity_covered(entity, sentence):
    tokens = content_tokens(entity)
    sentence_tokens = set(content_tokens(sentence))
    return bool(tokens) and all(t in sentence_tokens for t in tokens)

def relation_covered(relation, sentence):
    s = norm(sentence)
    alternatives = RELATION_TERMS.get(norm(relation), ((norm(relation),),))
    return any(all(term in s for term in terms) for terms in alternatives)

def validate(head, relation, tail, sentence):
    reasons = []
    clean = str(sentence).strip().strip('"').strip("'")
    required_prefix = norm(f"The sound of {head}")
    if not norm(clean).startswith(required_prefix): reasons.append("audio_prefix_not_exact")
    if not entity_covered(head, clean): reasons.append("head_not_covered")
    if not entity_covered(tail, clean): reasons.append("tail_not_covered")
    if not relation_covered(relation, clean): reasons.append("relation_not_explicit")
    if len(clean.split()) > 30: reasons.append("too_long")
    if "\n" in clean: reasons.append("multiple_lines")
    if clean.count(".") + clean.count("!") + clean.count("?") > 1:
        reasons.append("multiple_sentences")
    present_forbidden = sorted(w for w in FORBIDDEN_CONTENT if w in norm(clean).split())
    if present_forbidden: reasons.append("added_acoustic_terms:" + ",".join(present_forbidden))
    if any(x in clean.lower() for x in ("step-by-step", "description:", "head:", "relation:", "tail:")):
        reasons.append("extra_format")
    return clean, reasons

def model_generate(model, tokenizer, messages):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        ids = model.generate(**inputs, max_new_tokens=48, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(ids[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
