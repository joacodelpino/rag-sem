"""
Reranking con cross-encoder — la última etapa de la ruta de segunda generación.

La diferencia con el modelo de embeddings es el punto conceptual del bloque, y
conviene tenerla clara para poder explicarla:

  BI-ENCODER (BGE-M3, el de la ingesta)
      Embebe la consulta y el documento POR SEPARADO, y compara los dos
      vectores con un coseno. Como el documento se puede embeber de antemano,
      escala a millones: en tiempo de consulta solo hay que embeber la query
      y hacer una búsqueda de vecinos. El precio es que el modelo nunca ve
      la consulta y el documento juntos — comprime cada documento en un solo
      vector "por las dudas", sin saber qué le van a preguntar.

  CROSS-ENCODER (BGE-reranker-v2-m3, este)
      Recibe el par (consulta, documento) CONCATENADO en una sola pasada, con
      atención cruzada entre los dos. Puede razonar sobre la relación
      específica: si la consulta pide un plazo y el documento menciona un
      plazo pero de otro recurso, lo nota. Es bastante más preciso.
      El precio es que NO escala: hay que correr el modelo una vez por
      documento candidato, en tiempo de consulta. No se puede precomputar
      nada, porque el resultado depende de la consulta.

De ahí sale la arquitectura en dos etapas, que es la idea central:
recuperación barata y amplia (bi-encoder + BM25) para bajar de miles de chunks
a unas decenas, y reranking caro y preciso solo sobre esas decenas.

Corre local en CPU, igual que los embeddings: nada del corpus sale de la
máquina.
"""
import os

import torch
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

from chunk import Chunk

load_dotenv()

RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# Por defecto torch usa la mitad de los núcleos (asume hyper-threading y que
# el proceso comparte la máquina). Acá el reranking es lo único que corre y es
# lo más lento del pipeline, así que le damos todos: medido, ~17% más rápido.
torch.set_num_threads(os.cpu_count())

_reranker = None


def _get_reranker() -> CrossEncoder:
    """Carga perezosa y una sola vez por proceso.

    Importa además que sea perezosa y no al importar el módulo: las
    configuraciones sin reranking no deberían pagar la carga de ~2.3 GB solo
    porque este archivo está en el proyecto.
    """
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank(query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
    """Reordena los chunks candidatos por relevancia real a la consulta.

    Recibe la lista que armó la etapa de recuperación y devuelve las mismas
    piezas reordenadas y recortadas a top_k, con el score del cross-encoder en
    lugar del score de la etapa anterior: ese es el que explica el orden nuevo.

    El score es una probabilidad de relevancia 0–1, y a diferencia del coseno
    usa todo el rango: un chunk irrelevante saca 0.00001, no 0.4. Esa es una
    ventaja práctica del cross-encoder que conviene mostrar — permite poner un
    umbral y decir "ninguno de estos documentos responde la pregunta", algo
    que con similitud coseno es mucho más difícil.

    El tipo de retorno es el mismo list[Chunk] que consume el resto del
    pipeline, así que esta función se puede intercalar sin que generate.py ni
    app.py se enteren.
    """
    if not chunks:
        return []

    # Una sola llamada con todos los pares, no una por chunk: el modelo los
    # procesa en batch y la diferencia de tiempo es enorme.
    #
    # predict() ya devuelve valores 0–1: sentence-transformers aplica una
    # sigmoide por defecto sobre el logit crudo (m.activation_fn es Sigmoid()
    # para modelos de una sola etiqueta como este). Aplicarle otra sigmoide
    # encima comprime todo el rango entre 0.5 y 0.73 y hace que los scores de
    # la demo no signifiquen nada — no cambia el orden, pero no se puede leer.
    pairs = [(query, chunk.text) for chunk in chunks]
    scores = _get_reranker().predict(pairs)

    rescored = [
        Chunk(
            id=chunk.id,
            text=chunk.text,
            source=chunk.source,
            score=float(score),
        )
        for chunk, score in zip(chunks, scores)
    ]
    rescored.sort(key=lambda c: c.score, reverse=True)
    return rescored[:top_k]
