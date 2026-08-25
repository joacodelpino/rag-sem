"""
Recuperación.

Cada configuración (naive, híbrida, híbrida+reranking) se expone como una
función independiente con la misma firma: retrieve_x(query, top_k) ->
list[Chunk]. Esto es a propósito: Ragas evalúa cada configuración por
separado para armar la tabla de ablación, así que necesitan poder llamarse
en aislamiento sin pasar por el resto del pipeline.

Chunk lleva su id de Qdrant además del texto: las métricas de recuperación
(recall@k, MRR, NDCG) se calculan comparando ids recuperados contra ids
relevantes, no comparando strings de texto.

Por ahora solo está implementada retrieve_naive. Las otras dos se agregan
sobre la misma colección de Qdrant (ya tiene vector denso; el sparse se
suma como un named vector adicional cuando se implemente la ruta híbrida).
"""
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "legal_docs")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")

_client = None
_embedder = None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL)
    return _client


def _get_embedder() -> SentenceTransformer:
    # El modelo se carga una sola vez por proceso: en Streamlit esto evita
    # recargar BGE-M3 (~2GB) en cada rerun de la app.
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


@dataclass
class Chunk:
    id: int
    text: str
    source: str
    score: float


def retrieve_naive(query: str, top_k: int = 5) -> list[Chunk]:
    """Búsqueda vectorial densa pura: embebe la consulta y trae los top_k
    chunks más cercanos por similitud coseno. Sin fusión ni reordenamiento."""
    client = _get_client()
    embedder = _get_embedder()

    query_vector = embedder.encode(query, normalize_embeddings=True).tolist()
    hits = client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=top_k,
    ).points

    return [
        Chunk(
            id=hit.id,
            text=hit.payload["text"],
            source=hit.payload["source"],
            score=hit.score,
        )
        for hit in hits
    ]
