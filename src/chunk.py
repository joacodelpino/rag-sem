"""
El tipo que viaja por todo el pipeline.

Vive en su propio módulo, y no dentro de retrieval.py, para que rerank.py
pueda construir Chunks sin importar a retrieval.py (que a su vez importa a
rerank.py, y sería un import circular).

El campo `id` no es decorativo: las métricas de recuperación (recall@k, MRR,
NDCG) se calculan comparando ids recuperados contra ids relevantes del set
dorado. Comparar strings de texto sería frágil.

`score` NO significa lo mismo en todas las configuraciones — es siempre el
score de la última etapa que tocó el chunk, que es la que explica el orden en
que salió:
    naive    -> similitud coseno (0–1, pero en la práctica 0.4–0.8)
    sparse   -> BM25 (sin techo)
    híbrida  -> score de RRF (valores chicos, ~0.03)
    rerank   -> probabilidad del cross-encoder (0–1, bien repartido)
Por eso los scores se pueden comparar DENTRO de una columna de la demo, pero
nunca entre columnas.
"""
from dataclasses import dataclass


@dataclass
class Chunk:
    id: int
    text: str
    source: str
    score: float
