"""
Recuperación — las configuraciones que compara la ablación.

Cada configuración se expone como una función independiente con la MISMA firma:

    retrieve_x(query, top_k) -> list[Chunk]

Esto es a propósito y es el punto central del diseño: Ragas evalúa cada
configuración por separado, así que necesitan poder llamarse en aislamiento,
sin pasar por el resto del pipeline y sin conocerse entre sí.

Las tres configuraciones de la ablación:
    retrieve_naive          — solo vectorial densa
    retrieve_hybrid         — densa + BM25 fusionadas con RRF
    retrieve_hybrid_rerank  — lo anterior, reordenado con un cross-encoder

Más retrieve_sparse (BM25 puro), que no es una configuración de la ablación:
está para poder mostrar en la demo qué aporta cada mitad de la híbrida.
"""
import os
from dataclasses import replace

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector
from sentence_transformers import SentenceTransformer

import bm25
import rerank
from chunk import Chunk

load_dotenv()

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "legal_docs")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")

# Cuántos candidatos trae cada rama antes de fusionar. Más grande que el top_k
# final a propósito: RRF solo puede rescatar un documento si al menos una de
# las dos ramas lo trajo, así que conviene darle margen. 20 es el valor típico
# en la literatura para top_k=5.
CANDIDATES_PER_BRANCH = 20

# Constante de RRF. Amortigua el peso de las primeras posiciones: con k=60 la
# diferencia entre el puesto 1 y el 2 pesa poco más que entre el 10 y el 11.
# Es el valor del paper original (Cormack et al., 2009) y el que usa Qdrant.
RRF_K = 60

# Cuántos candidatos ve el cross-encoder. Es EL parámetro que gobierna el
# trade-off de la arquitectura en dos etapas: más candidatos significa más
# chances de rescatar un documento que la primera etapa dejó en el puesto 15,
# pero el costo crece lineal porque el modelo corre una vez por candidato.
# Con 20 el reranking tarda unos segundos en CPU; con 100 no sería usable en
# vivo durante la exposición.
RERANK_CANDIDATES = 20

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


def _to_chunks(hits) -> list[Chunk]:
    """Traduce la respuesta de Qdrant al tipo que consume el resto del código.

    Los campos de identidad se leen con .get() y no con [] a propósito: un
    índice construido antes de conectar el manifiesto no los tiene, y es mejor
    que la app muestre el nombre de archivo a que reviente con KeyError. Si
    aparecen vacíos, hay que reingestar.
    """
    return [
        Chunk(
            id=hit.id,
            text=hit.payload["text"],
            source=hit.payload["source"],
            score=hit.score,
            titulo=hit.payload.get("titulo", ""),
            numero_ley=hit.payload.get("numero_ley", ""),
            version=hit.payload.get("version", ""),
            seccion=hit.payload.get("seccion", ""),
        )
        for hit in hits
    ]


def retrieve_naive(query: str, top_k: int = 5) -> list[Chunk]:
    """Configuración 1 — búsqueda vectorial densa pura.

    Embebe la consulta con el mismo modelo que se usó en la ingesta y trae los
    top_k chunks más cercanos por similitud coseno. Sin fusión ni reordenamiento.

    Fuerte en paráfrasis (encuentra el concepto aunque el vocabulario no
    coincida), débil en términos exactos y raros: números de artículo, siglas,
    nombres propios se diluyen en el promedio del embedding.
    """
    client = _get_client()
    query_vector = _get_embedder().encode(query, normalize_embeddings=True).tolist()

    hits = client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        using="dense",
        limit=top_k,
    ).points
    return _to_chunks(hits)


def retrieve_sparse(query: str, top_k: int = 5) -> list[Chunk]:
    """BM25 puro. Es el complemento exacto del denso: acierta el término
    literal y no entiende nada de sinónimos.

    No es una de las tres configuraciones de la ablación — está para poder
    mostrar en la demo qué aporta cada rama por separado antes de fusionarlas.
    """
    client = _get_client()
    weights = bm25.query_weights(query)

    hits = client.query_points(
        collection_name=COLLECTION,
        query=SparseVector(indices=list(weights.keys()), values=list(weights.values())),
        using="bm25",
        limit=top_k,
    ).points
    return _to_chunks(hits)


def reciprocal_rank_fusion(rankings: list[list[Chunk]], top_k: int) -> list[Chunk]:
    """Fusiona varias listas rankeadas en una sola, usando solo las POSICIONES.

        score(doc) = suma sobre cada ranking de  1 / (RRF_K + posición)

    Por qué RRF y no un promedio de los scores originales, que es la pregunta
    obvia: los dos scores viven en escalas incomparables. El coseno de BGE-M3
    da valores en un rango angosto y siempre positivo (típicamente 0.4–0.8
    incluso para resultados malos), mientras que BM25 no tiene techo y depende
    del largo de la consulta y del IDF de la colección. Promediarlos, o
    normalizarlos min-max por consulta, deja que la rama con más varianza
    domine la fusión — y peor, el resultado cambia según qué otros documentos
    entraron en el lote.

    RRF tira los scores y se queda solo con el orden, que es lo único
    comparable entre las dos ramas. Un documento que salió 2º en las dos
    listas le gana a uno que salió 1º en una sola: eso es exactamente lo que
    queremos, evidencia de dos señales independientes.

    Recibe una lista de rankings (no solo dos) para poder sumar una tercera
    rama después sin tocar la función.
    """
    scores: dict[int, float] = {}
    chunks: dict[int, Chunk] = {}

    for ranking in rankings:
        for position, chunk in enumerate(ranking, start=1):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (RRF_K + position)
            chunks[chunk.id] = chunk

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    # El score que sale es el de RRF, no el original: es el que explica este
    # orden, y es el que hay que mostrar en la demo para que se entienda por
    # qué un documento subió o bajó respecto de las ramas individuales.
    # replace() y no un Chunk() nuevo: copia todos los campos de identidad sin
    # tener que enumerarlos, así agregar un campo al dataclass no obliga a
    # acordarse de tocar también esta función (que es como se pierden en el
    # camino los metadatos).
    return [
        replace(chunks[chunk_id], score=rrf_score)
        for chunk_id, rrf_score in ordered[:top_k]
    ]


def retrieve_hybrid(query: str, top_k: int = 5) -> list[Chunk]:
    """Configuración 2 — densa + BM25 fusionadas con RRF.

    Corre las dos ramas por separado, cada una trayendo CANDIDATES_PER_BRANCH
    candidatos, y las fusiona por posición. La idea es que las dos fallan en
    casos distintos: donde el denso se pierde con un número de artículo, BM25
    lo clava; donde BM25 no encuentra nada porque el documento usa otro
    vocabulario, el denso sí.

    Nota: Qdrant puede hacer esta fusión del lado del servidor (prefetch +
    FusionQuery). Acá se hace explícita en Python a propósito, para poder
    mostrarla y explicarla en la exposición, y para poder inspeccionar los
    rankings intermedios de cada rama.
    """
    dense_hits = retrieve_naive(query, top_k=CANDIDATES_PER_BRANCH)
    sparse_hits = retrieve_sparse(query, top_k=CANDIDATES_PER_BRANCH)
    return reciprocal_rank_fusion([dense_hits, sparse_hits], top_k=top_k)


def retrieve_hybrid_rerank(query: str, top_k: int = 5) -> list[Chunk]:
    """Configuración 3 — híbrida + reranking con cross-encoder.

    Arquitectura en dos etapas, que es la idea central del "RAG de segunda
    generación":

      1. RECUPERAR barato y amplio: la ruta híbrida trae RERANK_CANDIDATES
         candidatos en vez de top_k. Acá lo que importa es el RECALL — que el
         chunk correcto esté en la bolsa, aunque salga en el puesto 14. El
         orden todavía no importa.
      2. REORDENAR caro y preciso: el cross-encoder mira cada par
         (consulta, chunk) y los reordena. Acá lo que importa es la PRECISIÓN
         en las primeras posiciones, que es lo que efectivamente le llega al
         LLM.

    Por qué no usar el cross-encoder para todo: no se puede precomputar nada,
    así que correrlo sobre el corpus entero significaría una pasada del modelo
    por cada chunk en cada consulta. Ver rerank.py para el detalle.

    Fijate que el top_k que se le pide a la primera etapa NO es el top_k final:
    el reranker no puede rescatar lo que la recuperación no trajo, así que
    darle una bolsa más grande es lo que le da margen para mejorar.
    """
    candidatos = retrieve_hybrid(query, top_k=RERANK_CANDIDATES)
    return rerank.rerank(query, candidatos, top_k=top_k)


def config_snapshot() -> dict:
    """Devuelve la configuración con la que corre AHORA el pipeline.

    Existe para que el módulo de evaluación pueda etiquetar cada corrida sin
    adivinar: una fila de la tabla de ablación no significa nada si no se sabe
    con qué reranker, cuántos candidatos y qué chunking se produjo. Como esto
    lee las constantes y variables de entorno reales, no puede desincronizarse
    de lo que efectivamente se ejecutó (que es justo lo que pasa cuando la
    configuración se anota a mano en el CSV).

    El chunking se importa de ingest.py adentro de la función, no arriba, por
    dos razones: evita pagar el import de PyMuPDF en cada proceso que solo
    quiere consultar, y evita duplicar las constantes acá, que es como
    terminan divergiendo del valor con el que realmente se indexó.
    """
    import ingest

    return {
        "collection": COLLECTION,
        "embedding_model": EMBEDDING_MODEL,
        "reranker_model": rerank.RERANKER_MODEL,
        "candidates_per_branch": CANDIDATES_PER_BRANCH,
        "rrf_k": RRF_K,
        "rerank_candidates": RERANK_CANDIDATES,
        "chunk_strategy": ingest.CHUNK_STRATEGY,
        "chunk_size": ingest.CHUNK_SIZE,
        "chunk_overlap": ingest.CHUNK_OVERLAP,
        "chunk_max_chars": ingest.CHUNK_MAX_CHARS,
        "chunk_min_chars": ingest.CHUNK_MIN_CHARS,
    }


def warmup() -> None:
    """Carga por adelantado los dos modelos locales y despierta la conexión.

    Ver rerank.warmup(): la idea es que el costo de leer los pesos desde disco
    lo pague el arranque de la app y no la primera consulta de la exposición.
    Se cargan los dos porque la demo muestra las cuatro configuraciones a la
    vez, así que la primera consulta usa el embedder Y el reranker.
    """
    # encode() y no solo cargar el modelo: igual que en el reranker, la
    # primera inferencia inicializa el grafo de cómputo y cuesta de más.
    _get_embedder().encode("calentamiento", normalize_embeddings=True)
    rerank.warmup()
    # Abre la conexión HTTP a Qdrant para que tampoco la pague la 1ª consulta.
    _get_client().get_collections()
