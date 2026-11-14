"""Step 4 of ingestion: embed the chunks and store them in Chroma, on disk.

Each chunk size gets its own collection (chunks_400, chunks_256) so they can be
compared on retrieval recall. Everything runs locally: Ollama for embeddings,
Chroma with telemetry switched off.
Run:  python -m ingest.index            (indexes every data/chunks_*.jsonl)
"""
import json

import chromadb
import ollama
from chromadb.config import Settings

from ingest.parse import ROOT

EMBED_MODEL = "nomic-embed-text"
CHROMA_DIR = ROOT / "chroma"
BATCH = 32
# nomic-embed-text was trained with task prefixes; using them improves retrieval
DOC_PREFIX = "search_document: "


def client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(CHROMA_DIR),
                                     settings=Settings(anonymized_telemetry=False))


def to_metadata(c: dict) -> dict:
    """Chroma metadata must be flat strings or numbers, so lists are joined."""
    return {
        "insurer": c["insurer"],
        "section": c.get("section", ""),
        "source": c["source"],
        "headings": " > ".join(c.get("headings", [])),
        "pdf_pages": ",".join(str(p) for p in c["pdf_pages"]),
        "printed_pages": ",".join(str(p) for p in c["printed_pages"]),
        "tokens": c.get("tokens", 0),
    }


def build(chunks_file) -> None:
    name = chunks_file.stem                                   # e.g. chunks_400
    chunks = [json.loads(line) for line in chunks_file.open()]
    db = client()
    if name in [c.name for c in db.list_collections()]:
        db.delete_collection(name)                            # rebuild from scratch, never mix versions
    col = db.create_collection(name, metadata={"hnsw:space": "cosine"})

    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        vectors = ollama.embed(model=EMBED_MODEL,
                               input=[DOC_PREFIX + c["text"] for c in batch])["embeddings"]
        col.add(ids=[c["id"] for c in batch],
                embeddings=vectors,
                documents=[c["text"] for c in batch],
                metadatas=[to_metadata(c) for c in batch])
        print(f"\r{name}: {min(i + BATCH, len(chunks))}/{len(chunks)}", end="", flush=True)
    print(f"\r{name}: {col.count()} chunks indexed")


def main():
    files = sorted((ROOT / "data").glob("chunks_*.jsonl"))
    if not files:
        print("No chunk files found. Run python -m ingest.chunk first.")
        return
    for f in files:
        build(f)
    print(f"Index stored in {CHROMA_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()