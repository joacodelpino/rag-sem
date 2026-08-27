"""
Ingesta — parsing, chunking e indexado en Qdrant.

Corre una sola vez, offline: no interviene durante la consulta.

Cada chunk se sube con DOS vectores en el mismo punto de la colección:

  - "dense": embedding de BGE-M3, captura significado (encuentra "plazo para
    apelar" aunque el documento diga "término para recurrir").
  - "bm25":  vector sparse léxico, captura coincidencia exacta de términos
    (encuentra "artículo 34" o "UVA", donde el denso se diluye).

Las tres configuraciones de recuperación (naive, híbrida, híbrida+rerank) leen
de esta misma colección. No hay que reingestar para cambiar de configuración:
eso es justamente lo que hace comparable la tabla de ablación.
"""
import os
from pathlib import Path

import pymupdf 
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

import bm25

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

# Cuántos puntos por llamada a Qdrant. Ver el comentario en main(): existe
# para no pasarse del límite de 32 MB por request.
UPSERT_BATCH = 256


def read_document(path: Path) -> str:
    """Extrae texto plano de un .txt o .pdf (PDFs con texto nativo; los
    escaneados necesitarían OCR, todavía no implementado)."""
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
    """Lee todos los documentos de data/raw/ y los devuelve troceados, con
    metadata de origen para poder citar la fuente en la respuesta."""
    records = []
    for path in sorted(DATA_DIR.glob("*")):
        if path.suffix.lower() not in (".txt", ".pdf"):
            continue
        text = read_document(path)
        for i, chunk in enumerate(chunk_text(text)):
            records.append({"source": path.name, "chunk_index": i, "text": chunk})
    return records


def build_collection(client: QdrantClient, vector_size: int) -> None:
    """Recrea la colección con los dos vectores nombrados: denso y sparse.

    Se borra y se rehace en cada corrida a propósito: cada ingesta parte de un
    estado limpio y reproducible, sin puntos huérfanos de corridas anteriores.

    Modifier.IDF es la línea clave de la ruta híbrida: le dice a Qdrant que
    mantenga las frecuencias de documento de cada término y aplique el IDF en
    tiempo de consulta. Sin esto, el vector sparse sería solo TF y BM25 quedaría
    a medias.
    """
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "dense": VectorParams(size=vector_size, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "bm25": SparseVectorParams(modifier=Modifier.IDF),
        },
    )


def build_sparse_vectors(records: list[dict]) -> list[SparseVector]:
    """Calcula el vector BM25 de cada chunk.

    Necesita dos pasadas sobre el corpus: la primera tokeniza y mide la
    longitud promedio de documento, la segunda calcula los pesos. Es que la
    normalización por longitud de BM25 compara cada documento contra ese
    promedio, así que no se puede calcular chunk por chunk de forma aislada.
    """
    tokenized = [bm25.tokenize(r["text"]) for r in records]
    avg_len = sum(len(t) for t in tokenized) / len(tokenized)

    vectors = []
    for tokens in tokenized:
        weights = bm25.document_weights(tokens, avg_len)
        vectors.append(
            SparseVector(indices=list(weights.keys()), values=list(weights.values()))
        )
    return vectors


def main():
    print(f"Cargando modelo de embeddings: {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Leyendo documentos de {DATA_DIR} ...")
    records = load_documents()
    if not records:
        raise SystemExit(f"No se encontraron documentos .txt/.pdf en {DATA_DIR}")
    print(f"{len(records)} chunks generados a partir de {len(set(r['source'] for r in records))} documentos.")

    print("Generando embeddings densos (BGE-M3, local en CPU) ...")
    texts = [r["text"] for r in records]
    dense = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    print("Calculando vectores sparse (BM25) ...")
    sparse = build_sparse_vectors(records)

    client = QdrantClient(url=QDRANT_URL)
    build_collection(client, vector_size=dense.shape[1])

    print(f"Subiendo {len(records)} puntos a la colección '{COLLECTION}' ...")
    points = [
        PointStruct(
            id=i,
            vector={"dense": dense[i].tolist(), "bm25": sparse[i]},
            payload={"source": r["source"], "chunk_index": r["chunk_index"], "text": r["text"]},
        )
        for i, r in enumerate(records)
    ]

    # Subida por lotes, no de una. Qdrant rechaza payloads de más de 32 MB
    # (error 400, "JSON payload is larger than allowed") y cada punto pesa
    # bastante: 1024 floats del vector denso serializados como JSON, más el
    # texto del chunk. Con el corpus real un único upsert daba 56 MB.
    #
    # Que esto falle al final es lo peor posible, porque la colección ya se
    # borró y los embeddings —media hora de CPU— se pierden. Por eso el lote
    # es conservador: 256 puntos son ~7 MB, bien lejos del límite, y el costo
    # de hacer varias llamadas HTTP es despreciable al lado de embeber.
    for inicio in range(0, len(points), UPSERT_BATCH):
        lote = points[inicio:inicio + UPSERT_BATCH]
        client.upsert(collection_name=COLLECTION, points=lote)
        print(f"  {min(inicio + UPSERT_BATCH, len(points))}/{len(points)}")

    print("Listo.")


if __name__ == "__main__":
    main()
