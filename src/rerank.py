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

MODELOS SOPORTADOS (via RERANKER_MODEL, ver .env.example)

  cross-encoder/mmarco-mMiniLMv2-L12-H384-v1   ~118M params
  BAAI/bge-reranker-v2-m3                      ~568M params

Existen para poder poner el trade-off calidad/latencia en la tabla de
ablación: el chico para la demo en vivo, el grande para la corrida de
evaluación. Cambiar la variable no requiere reingestar, porque el reranker
actúa sobre candidatos ya recuperados y no sobre el índice.

OJO CON LA ESCALA DE LOS SCORES: los dos modelos NO devuelven lo mismo de
fábrica. bge-reranker-v2-m3 trae activation_fn = Sigmoid() y predict() da
probabilidades 0-1; mmarco-mMiniLMv2 trae Identity() y predict() da el logit
crudo (medido: +7.18 para un chunk relevante, -7.52 para uno que no lo es).
El ORDEN es el mismo en los dos casos —la sigmoide es monótona— pero los
números no serían comparables entre corridas, y un umbral fijo se rompería al
cambiar de modelo. Por eso abajo se fija la activación explícitamente en vez
de confiar en el default de cada modelo.
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
        # activation_fn explícita: ver la nota sobre escalas en el docstring
        # del módulo. Para bge esto coincide con su default; para mmarco
        # convierte el logit crudo en la misma probabilidad 0-1, y así el
        # score significa lo mismo sin importar qué modelo esté configurado.
        _reranker = CrossEncoder(RERANKER_MODEL, activation_fn=torch.nn.Sigmoid())
    return _reranker


def warmup() -> None:
    """Fuerza la carga de los pesos Y una primera inferencia de descarte.

    La inferencia falsa no sobra: medido, la PRIMERA llamada a predict() de
    cada modelo cuesta bastante más que las siguientes aunque los pesos ya
    estén en memoria (13.6s vs 2.0s en el modelo chico, 27.8s vs 18.7s en el
    grande, sobre 20 pares). Es el costo de inicializar los kernels y el grafo
    de cómputo de torch, y se paga una sola vez por proceso.

    O sea que precargar solo los pesos NO alcanza para sacarle el costo a la
    primera consulta: hay que hacerla correr de verdad una vez. Llamando a
    warmup() al arrancar la app, ese costo cae en el startup de Streamlit y no
    sobre quien pregunte primero en la exposición.
    """
    _get_reranker().predict([("calentamiento", "texto de descarte")])


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
    # predict() ya devuelve la probabilidad 0–1: la sigmoide la aplica
    # sentence-transformers usando la activation_fn que fijamos en el
    # constructor. NO aplicarle otra encima: una segunda sigmoide comprime
    # todo el rango entre 0.5 y 0.73 y hace que los scores de la demo no
    # signifiquen nada (no cambia el orden, pero deja de poder leerse).
    # Fue un bug real de este archivo, no una hipótesis.
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
