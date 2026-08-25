"""
Ingesta - ruta naive.

Parsea los documentos de data/raw/, los divide en chunks, los embebe con
BGE-M3 (denso, local) y los sube a Qdrant.

Por ahora solo arma el vector denso: la ruta híbrida (BM25/sparse) y el
reranker se agregan sobre esta misma colección más adelante, no requieren
reingestar.
"""
import os
from pathlib import Path

import pymupdf
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "legal_docs")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")

# Chunking simple por caracteres con solapamiento. Para el corpus jurídico
# real (leyes con artículos numerados) esto se puede afinar para cortar en
# límites de artículo, pero el walking skeleton usa algo genérico que
# funcione con cualquier .txt/.pdf.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def read_document(path: Path) -> str:
    """Extrae texto plano de un .txt o .pdf."""
    if path.suffix.lower() == ".pdf":
        with pymupdf.open(path) as doc:
            return "\n".join(page.get_text() for page in doc)
    return path.read_text(encoding="utf-8")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Divide el texto en ventanas solapadas. El solapamiento evita que una
    oración con la respuesta quede cortada exactamente en el borde de dos
    chunks."""
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return [c for c in chunks if c.strip()]


def load_documents() -> list[dict]:
    """Lee todos los documentos de data/raw/ y devuelve chunks con metadata
    de origen (para poder citar la fuente en la respuesta)."""
    records = []
    for path in sorted(DATA_DIR.glob("*")):
        if path.suffix.lower() not in (".txt", ".pdf"):
            continue
        text = read_document(path)
        for i, chunk in enumerate(chunk_text(text)):
            records.append({"source": path.name, "chunk_index": i, "text": chunk})
    return records


def build_collection(client: QdrantClient, vector_size: int) -> None:
    """Recrea la colección desde cero. Para la demo esto es intencional:
    cada corrida de ingest.py parte de un estado limpio y reproducible."""
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def main():
    print(f"Cargando modelo de embeddings: {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Leyendo documentos de {DATA_DIR} ...")
    records = load_documents()
    if not records:
        raise SystemExit(f"No se encontraron documentos .txt/.pdf en {DATA_DIR}")
    print(f"{len(records)} chunks generados a partir de {len(set(r['source'] for r in records))} documentos.")

    print("Generando embeddings ...")
    texts = [r["text"] for r in records]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    client = QdrantClient(url=QDRANT_URL)
    build_collection(client, vector_size=embeddings.shape[1])

    print(f"Subiendo {len(records)} puntos a la colección '{COLLECTION}' ...")
    points = [
        PointStruct(
            id=i,
            vector=embeddings[i].tolist(),
            payload={"source": r["source"], "chunk_index": r["chunk_index"], "text": r["text"]},
        )
        for i, r in enumerate(records)
    ]
    client.upsert(collection_name=COLLECTION, points=points)
    print("Listo.")


if __name__ == "__main__":
    main()
