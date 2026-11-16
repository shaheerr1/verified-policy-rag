"""Local API: one endpoint that runs the whole verified pipeline."""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.checks import FALLBACK, check_answer, final_label, nli
from app.generate import generate, page_of


@asynccontextmanager
async def lifespan(app):
    nli()  # load DeBERTa once at startup so the first question is not slow
    yield


app = FastAPI(title="Verified Policy RAG", lifespan=lifespan)


class Ask(BaseModel):
    question: str
    insurer: str


@app.post("/ask")
def ask(req: Ask):
    result = generate(req.question, insurer=req.insurer)
    checked = check_answer(result)
    v = final_label(result, checked)
    hits = {h["id"]: h for h in result["hits"]}

    def pages(ids):
        seen = []
        for i in ids:
            for p in (page_of(hits[i]).split(", ") if i in hits else []):
                if p and p not in seen:
                    seen.append(p)
        return ", ".join(seen)

    return {
        "label": v["label"],
        "model_saw": result["question"],
        "delivered": [{"text": c["text"], "pages": pages(c["citations"])} for c in v["deliver"]],
        "fallback": None if v["deliver"] else FALLBACK,
        "sentences": [{"text": c["text"], "status": c["status"], "evidence": c.get("evidence", ""),
                       "pages": pages(c.get("citations", []))} for c in checked],
        "exceptions": [e["text"] for e in v.get("exceptions", [])],
    }


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/replay.json")
def replay():
    f = Path(__file__).parent / "static" / "replay.json"
    return FileResponse(f) if f.exists() else []
