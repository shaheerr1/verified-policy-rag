"""Run generation and checks over a question set; print each verdict and save results locally."""
import argparse
import json
import time
from pathlib import Path

from app.checks import FALLBACK, check_answer, final_label
from app.generate import generate

SETS = {"doc": Path("eval/answer_questions.jsonl"),
        "customer": Path("eval/heldout_questions.jsonl")}


def main(only, set_name):
    rows = [json.loads(l) for l in SETS[set_name].read_text().splitlines() if l.strip()]
    results = []
    for q in rows:
        if only and q["id"] not in only:
            continue
        t = time.time()
        result = generate(q["question"], insurer=q["insurer"])
        checked = check_answer(result)
        v = final_label(result, checked)
        print(f"\n[{q['id']}] {q['insurer']} | {v['label']} | {time.time() - t:.0f}s")
        print(f"  Q: {q['question']}")
        print(f"  model saw: {result['question']}")
        print(f"  expected: {q['expected']}")
        for c in checked:
            print(f"    {c['status']:<15} {c['text'][:150]}")
        print("  CUSTOMER SEES:", " ".join(c["text"] for c in v["deliver"]) or FALLBACK)
        if v.get("exceptions"):
            print("  exception may apply:", v["exceptions"][0]["text"][:160])
        results.append({"id": q["id"], "label": v["label"],
                        "not_in_policy": result["answer"]["not_in_policy"],
                        "checked": checked, "deliver": v["deliver"],
                        "exceptions": v.get("exceptions", [])})
    out = Path(f"logs/answer_check_{set_name}.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved {len(results)} results to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--set", default="doc", choices=sorted(SETS))
    p.add_argument("ids", nargs="*")
    a = p.parse_args()
    main(set(a.ids), a.set)
