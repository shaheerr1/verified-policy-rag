"""Checks that run on every generated answer before a customer sees it."""
import re
from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.glossary import GLOSSARY, policy_uses
from app.privacy import leaks

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
ENTAIL_MIN = 0.5
CONTRA_MIN = 0.8
OVERLAP_MIN = 0.4
SIZES = (1, 2, 3, 4, 6, 8)
UNIT_SPLIT = re.compile(r"\n+|(?<=[a-z]{4}\.)\s+")
ANSWER_WORD = re.compile(r"^(yes|no|however|additionally|also)[,.]?\s+", re.I)
WORD = re.compile(r"[a-z0-9£][a-z0-9£,']*", re.I)
STOP = set("the a an of to and or for in on at by is are be we you your our this that it with as any from not "
           "will can if does his her their my starting".split())
KEY_MAX_USES = 20
GENERIC = set("claim claims cover covered policy insurance pay paid can could would yesterday someone "
              "everything back cost get".split())
NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
NUM_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
             "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fourteen": 14, "fifteen": 15,
             "twenty": 20, "thirty": 30, "sixty": 60, "ninety": 90}
EXCEPTION = re.compile(r"\b(unless|except|you do not need to|you don't need to|does not apply)\b", re.I)
EXC_MIN_OVERLAP = 2
BLOCK = {"uncited", "bad_citation", "unsupported", "contradicted", "off_topic", "number_mismatch"}
FALLBACK = "I couldn't verify an answer from your policy wording. Please speak to an adviser."


@lru_cache(maxsize=1)
def nli():
    tok = AutoTokenizer.from_pretrained(NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).eval()
    labels = {v.lower(): k for k, v in model.config.id2label.items()}
    return tok, model, labels


def nli_batch(premises, hypothesis, batch_size=16):
    tok, model, labels = nli()
    order = sorted(range(len(premises)), key=lambda i: len(premises[i]))
    out = [None] * len(premises)
    for k in range(0, len(order), batch_size):
        idx = order[k:k + batch_size]
        enc = tok([premises[i] for i in idx], [hypothesis] * len(idx), truncation="only_first",
                  max_length=512, padding=True, return_tensors="pt")
        with torch.no_grad():
            probs = model(**enc).logits.softmax(-1)
        for i, p in zip(idx, probs):
            out[i] = {name: float(p[j]) for name, j in labels.items()}
    return out


def units(text):
    return [u.strip() for u in UNIT_SPLIT.split(text) if sum(ch.isalpha() for ch in u) >= 4]


def windows(text):
    us = units(text)
    wins = []
    for n in SIZES:
        for i in range(len(us) - n + 1):
            span = us[i:i + n]
            if span[0].startswith("-"):
                lead = [u for u in us[:i] if u.endswith(":")][-2:]
                span = lead + span
            wins.append(" ".join(span))
    wins.append(text)
    return list(dict.fromkeys(wins))


def content(s):
    return {w.lower().strip(",'") for w in WORD.findall(s)} - STOP


def overlap(window, hypothesis):
    h = content(hypothesis)
    return len(h & content(window)) / max(1, len(h))


def score(ids, hits_by_id, hypothesis):
    best, contra = None, {"contradiction": 0.0}
    for c in ids:
        wins = windows(hits_by_id[c]["text"])
        for w, s in zip(wins, nli_batch(wins, hypothesis)):
            if best is None or s["entailment"] > best["entailment"]:
                best = s | {"id": c, "evidence": w}
            if overlap(w, hypothesis) >= OVERLAP_MIN and s["contradiction"] > contra["contradiction"]:
                contra = s | {"id": c, "evidence": w}
    return best, contra


def check_sentence(sentence, hits_by_id):
    cites = sentence["citations"]
    if not cites:
        return {"status": "uncited"}
    missing = [c for c in cites if c not in hits_by_id]
    if missing:
        return {"status": "bad_citation", "missing": missing}
    hypothesis = ANSWER_WORD.sub("", sentence["text"]).strip()
    best, contra = score(cites, hits_by_id, hypothesis)
    if best["entailment"] >= ENTAIL_MIN:
        status = "supported"
    elif contra["contradiction"] >= CONTRA_MIN:
        status = "contradicted"
    else:
        status = "unsupported"
    result = {"status": status, "best": best["id"], "evidence": best["evidence"],
              "entailment": round(best["entailment"], 2),
              "contradiction": round(contra["contradiction"], 2)}
    if status != "supported":
        others = [c for c in hits_by_id if c not in cites]
        alt, _ = score(others, hits_by_id, hypothesis) if others else (None, None)
        if alt and alt["entailment"] >= ENTAIL_MIN:
            result.update(status="miscited", best=alt["id"], evidence=alt["evidence"],
                          entailment=round(alt["entailment"], 2))
    return result


def norm(word):
    w = word.lower().strip(",?.!")
    if w.endswith("'s"):
        w = w[:-2]
    w = w.strip("'")
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")) else w


def key_terms(question):
    words = {norm(w) for w in WORD.findall(question) if w.lower() not in STOP}
    for pattern, terms in GLOSSARY:
        pat = pattern if hasattr(pattern, "search") else re.compile(pattern, re.I)
        if pat.search(question):
            for term in (terms if isinstance(terms, (list, tuple)) else [terms]):
                words |= {norm(t) for t in term.split()}
    words -= STOP | GENERIC
    return {w for w in words if len(w) > 2 and policy_uses(w) <= KEY_MAX_USES}


def on_topic(evidence, keys):
    return bool(keys & {norm(w) for w in WORD.findall(evidence)})


def numbers(text):
    vals = set()
    for m in NUM.finditer(text):
        v = m.group().replace(",", "")
        if v.replace(".", "").isdigit():
            vals.add(float(v))
    vals |= {float(NUM_WORDS[w]) for w in re.findall(r"[a-z]+", text.lower()) if w in NUM_WORDS}
    return vals


def check_numbers(c, hits_by_id):
    ids = [c["best"]] if c["status"] == "miscited" else c["citations"]
    source = " ".join(hits_by_id[i]["text"] for i in ids if i in hits_by_id)
    return sorted(numbers(c["text"]) - numbers(source))


def check_answer(result):
    hits_by_id = {h["id"]: h for h in result["hits"]}
    keys = key_terms(result["question"])
    a = result["answer"]
    checked = []
    for s in a["sentences"]:
        if a["not_in_policy"] and not s["citations"]:
            checked.append({**s, "status": "refusal"})
            continue
        c = {**s, **check_sentence(s, hits_by_id)}
        if keys and c["status"] in ("supported", "miscited") and not on_topic(c["evidence"], keys):
            c["status"] = "off_topic"
        if c["status"] in ("supported", "miscited"):
            bad = check_numbers(c, hits_by_id)
            if bad:
                c["status"], c["bad_numbers"] = "number_mismatch", bad
        checked.append(c)
    return checked


def verdict(checked):
    for c in checked:
        if c["status"] == "miscited":
            c["citations"], c["status"], c["repaired"] = [c["best"]], "supported", True
    blocked = [c for c in checked if c["status"] in BLOCK]
    if checked and all(c["status"] == "refusal" for c in checked):
        label = "REFUSAL"
    elif blocked:
        label = "HALLUCINATION"
    else:
        label = "SUCCESS"
    deliver = [] if blocked else [c for c in checked if c["status"] not in BLOCK]
    return {"label": label, "blocked": len(blocked), "deliver": deliver}


def matching_exceptions(question, texts):
    q = {norm(w) for w in content(question)} - GENERIC
    found = []
    for t in texts:
        for u in units(t):
            if EXCEPTION.search(u):
                shared = q & {norm(w) for w in content(u)}
                if len(shared) >= EXC_MIN_OVERLAP:
                    found.append({"text": u, "shared": sorted(shared)})
    return found


def final_label(result, checked):
    v = verdict(checked)
    if v["deliver"]:
        hits_by_id = {h["id"]: h for h in result["hits"]}
        cited = {i for c in v["deliver"] for i in c["citations"]}
        exc = matching_exceptions(result["question"], [hits_by_id[i]["text"] for i in cited if i in hits_by_id])
        if exc:
            v.update(label="REVIEW", deliver=[], exceptions=exc)
    found = leaks([c["text"] for c in v["deliver"]], [h["text"] for h in result["hits"]])
    if found:
        v.update(label="UNSAFE", deliver=[], pii=sorted(set(found)))
    return v


if __name__ == "__main__":
    import sys
    from app.generate import generate

    insurer, question = sys.argv[1], " ".join(sys.argv[2:])
    result = generate(question, insurer=insurer)
    checked = check_answer(result)
    print(f"\nQ: {question}")
    print("   key terms:", sorted(key_terms(question)))
    for s in checked:
        flag = "OK" if s["status"] in ("supported", "refusal") else "!!"
        print(f"{flag} {s['status']:<15} {s['text']}")
        if "entailment" in s:
            print(f"     {s['best']}  entail {s['entailment']}  contra {s['contradiction']}")
            print(f"     evidence: {s['evidence'][:200]}")
    v = final_label(result, checked)
    print(f"\n== {v['label']}  ({v['blocked']} sentence(s) blocked)")
    print("CUSTOMER SEES:")
    for c in v["deliver"]:
        print(f"  {c['text']}  [{', '.join(c['citations'])}]" + ("  (citation repaired)" if c.get("repaired") else ""))
    if not v["deliver"]:
        print("  " + FALLBACK)
    if v.get("exceptions"):
        print("  exception may apply:", v["exceptions"][0]["text"][:160])
