"""Retrieval: find the chunks most relevant to a question, from the local Chroma index.

Insurer routing: when a question names an insurer, only that insurer's chunks are searched and
the name is removed from the text that gets embedded (otherwise brand-heavy generic pages win).
Comparison questions are searched per insurer and the results interleaved.

    from app.retrieve import retrieve
    hits = retrieve("Is a 17 year old boiler covered by Uinsure?", k=5)
"""
import math
import re
from functools import lru_cache

import chromadb
import ollama
from chromadb.config import Settings

from ingest.parse import ROOT

EMBED_MODEL = "nomic-embed-text"
CHROMA_DIR = ROOT / "chroma"
DEFAULT_COLLECTION = "chunks_400"
QUERY_PREFIX = "search_query: "  # pairs with search_document: used at indexing time

INSURERS = {
    "Policy Expert": re.compile(r"\bpolicy\s*expert('s)?\b", re.I),
    "Uinsure": re.compile(r"\buinsure('s)?\b", re.I),
    "Covea": re.compile(r"\bcov[eé]a('s)?\b", re.I),
}
COMPARISON = re.compile(
    r"\b(which (of the )?(three |3 )?(policies|policy|insurers?)|compare|comparison|"
    r"each (policy|insurer)|all (three|3)|versus|vs\.?)\b", re.I)


@lru_cache(maxsize=4)
def collection(name: str = DEFAULT_COLLECTION):
    db = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
    return db.get_collection(name)


def embed_query(text: str) -> list:
    return ollama.embed(model=EMBED_MODEL, input=[QUERY_PREFIX + text])["embeddings"][0]


def route(question: str) -> tuple:
    """Return (text to embed, insurers to search). An empty list means search everything."""
    named = [name for name, pattern in INSURERS.items() if pattern.search(question)]
    text = question
    for pattern in INSURERS.values():
        text = pattern.sub("the policy", text)  # keep the sentence readable, drop the brand
    if not named and COMPARISON.search(question):
        named = list(INSURERS)  # "which policy...": compare all of them
    return text, named


def _search(vector: list, k: int, insurer: str | None, collection_name: str) -> list:
    res = collection(collection_name).query(
        query_embeddings=[vector], n_results=k,
        where={"insurer": insurer} if insurer else None,
    )
    hits = []
    for cid, text, meta, dist in zip(res["ids"][0], res["documents"][0],
                                     res["metadatas"][0], res["distances"][0]):
        hits.append({
            "id": cid,
            "text": text,
            "insurer": meta["insurer"],
            "section": meta.get("section", ""),
            "headings": meta.get("headings", ""),
            "source": meta["source"],
            "pdf_pages": [int(p) for p in meta["pdf_pages"].split(",") if p],
            "printed_pages": [int(p) for p in meta["printed_pages"].split(",") if p],
            "score": round(1 - dist, 4),  # cosine similarity: higher is closer
        })
    return hits


def retrieve(question: str, k: int = 5, insurer: str | None = None,
             collection_name: str = DEFAULT_COLLECTION, routing: bool = True) -> list:
    """Top-k chunks for a question, best first. routing=False gives plain vector search."""
    if insurer:
        text, insurers = question, [insurer]
    elif routing:
        text, insurers = route(question)
    else:
        text, insurers = question, []

    vector = embed_query(text)
    if not insurers:
        return _search(vector, k, None, collection_name)
    if len(insurers) == 1:
        return _search(vector, k, insurers[0], collection_name)

    # comparison: best results from each insurer, interleaved so each is represented early
    per = math.ceil(k / len(insurers)) + 1
    lists = [_search(vector, per, ins, collection_name) for ins in insurers]
    merged = [h for rank in range(per) for lst in lists if rank < len(lst) for h in [lst[rank]]]
    return merged[:k]