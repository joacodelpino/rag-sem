"""
Benchmark de latencia de los rerankers.

Mide, para cada modelo soportado, cuánto cuesta la etapa de reranking con
RERANK_CANDIDATES=20 sobre las mismas consultas. El objetivo es tener el
número concreto del trade-off calidad/latencia para la tabla de ablación, en
vez de estimarlo.

    python evals/bench_rerank.py

Escribe evals/results/rerank_latency.csv.

ADVERTENCIA SOBRE EL CORPUS ACTUAL
El corpus de ejemplo tiene 8 chunks, así que la recuperación no puede
devolver los 20 candidatos que pide RERANK_CANDIDATES. Para medir igual el
costo real de 20 pares, la lista de candidatos se COMPLETA repitiendo los
chunks recuperados de forma cíclica. Eso no altera la latencia —el modelo
paga lo mismo por un par repetido que por uno nuevo, no hay caché— pero sí
invalida cualquier lectura de CALIDAD sobre estas corridas. Este script mide
tiempo, nada más. Cuando esté el corpus real, el relleno deja de aplicarse
solo y las dos columnas de pares coinciden.
"""
import csv
import os
import sys
import time
from itertools import islice, cycle
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

import rerank                      # noqa: E402
from retrieval import RERANK_CANDIDATES, retrieve_hybrid  # noqa: E402

MODELOS = [
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    "BAAI/bge-reranker-v2-m3",
]

CONSULTAS = [
    "¿Cuántos días hábiles hay para presentar el recurso jerárquico?",
    "¿Qué pasa si el inquilino se va antes de tiempo?",
    "¿Qué interés se aplica por el incumplimiento?",
]

SALIDA = RAIZ / "evals" / "results" / "rerank_latency.csv"


def cargar_modelo(nombre: str) -> float:
    """Cambia el reranker activo y devuelve cuánto tardó en cargar los pesos.

    Resetea el singleton a mano: es el precio de tener el modelo cacheado por
    proceso, que es lo correcto para la app pero estorba para un benchmark que
    necesita medir varios modelos en la misma corrida.
    """
    rerank.RERANKER_MODEL = nombre
    rerank._reranker = None
    inicio = time.perf_counter()
    rerank.warmup()
    return time.perf_counter() - inicio


def main() -> None:
    filas = []

    for modelo in MODELOS:
        print(f"\n=== {modelo}")
        carga = cargar_modelo(modelo)
        params = sum(p.numel() for p in rerank._reranker.model.parameters()) / 1e6
        print(f"  carga de pesos: {carga:.2f}s   ({params:.0f}M params)")

        for consulta in CONSULTAS:
            candidatos = retrieve_hybrid(consulta, top_k=RERANK_CANDIDATES)
            reales = len(candidatos)
            if reales and reales < RERANK_CANDIDATES:
                candidatos = list(islice(cycle(candidatos), RERANK_CANDIDATES))

            inicio = time.perf_counter()
            rerank.rerank(consulta, candidatos, top_k=5)
            transcurrido = time.perf_counter() - inicio

            print(
                f"  {transcurrido:6.2f}s  ({transcurrido / len(candidatos):.3f}s/par)"
                f"  {consulta[:45]}"
            )
            filas.append(
                {
                    "modelo": modelo,
                    "params_millones": round(params),
                    "carga_pesos_s": round(carga, 2),
                    "consulta": consulta,
                    "pares_medidos": len(candidatos),
                    "pares_reales_del_corpus": reales,
                    "latencia_rerank_s": round(transcurrido, 2),
                    "latencia_por_par_s": round(transcurrido / len(candidatos), 3),
                }
            )

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    with SALIDA.open("w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)
    print(f"\nEscrito: {SALIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
