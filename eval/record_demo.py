"""Record real pipeline runs for the static demo (recorded runs mode)."""
import json
from pathlib import Path

from app.api import Ask, ask

DEMO = [
    ("Verified answer across tiers", "Policy Expert",
     "How much cover do I get for contents in outbuildings and theft from outbuildings across Basic, Bronze, Silver, and Gold?"),
    ("Verified answer with page citations", "Policy Expert",
     "How long is the repair guarantee for contents and buildings?"),
    ("Hallucination caught", "Uinsure",
     "Last night's storm blew down two panels of my garden fence and knocked over my shed. Will my buildings insurance cover the cost of replacing the fence and fixing the shed?"),
    ("Exception sent for review", "Policy Expert",
     "I'm planning to renovate my kitchen next month, just ripping out the old units and fitting new ones without knocking down any walls. Do I have to tell you before the builders start?"),
    ("Personal data masked before the model", "Uinsure",
     "Hi, I'm Sarah Khan at 14 Acacia Avenue, SW1A 1AA, phone 07700 900123. My son is moving into university halls. Are his belongings covered?"),
    ("True but irrelevant answer caught", "Policy Expert",
     "We had a power cut yesterday and everything in our chest freezer defrosted and had to be binned. Can I claim back the cost of the ruined food?"),
]

runs = []
for title, insurer, question in DEMO:
    r = ask(Ask(question=question, insurer=insurer))
    runs.append({"title": title, "insurer": insurer, "question": question, "response": r})
    print(f"{r['label']:<14} {title}")

out = Path("app/static/replay.json")
out.write_text(json.dumps(runs, indent=2, ensure_ascii=False))
print(f"\nSaved {len(runs)} runs to {out}")
