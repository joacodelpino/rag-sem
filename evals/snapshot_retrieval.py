# -*- coding: utf-8 -*-
"""
Congela el resultado de las consultas de prueba en un archivo JSON.

Para qué: la ingesta es destructiva —borra la colección y la rehace— así que
una vez reindexado NO hay forma de volver a ver qué recuperaba el índice
anterior. Este script guarda ese "antes" para poder ponerlo al lado del
"después" en la exposición, que es el argumento entero del trabajo.

Guarda, por cada consulta y cada configuración:
  - los chunks recuperados con id, fuente, score, posición y texto
  - la respuesta del LLM con ese contexto (que es donde se ve la falla real:
    una cita perfecta de la ley equivocada)
  - config_snapshot(), para que dentro de dos semanas se sepa con qué
    reranker, cuántos candidatos y qué chunking se produjo cada fila

Uso:
    python evals/snapshot_retrieval.py --etiqueta antes-chunking-fijo
    python evals/snapshot_retrieval.py --etiqueta antes --sin-llm
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from generate import generate_answer  # noqa: E402
import retrieval  # noqa: E402

SALIDA_DIR = RAIZ / "evals" / "results"

# El set de referencia. Son pocas y elegidas, no aleatorias: cada una existe
# para exponer un modo de falla concreto que ya vimos en el índice actual.
CONSULTAS = [
    {
        "id": "cp-art-72",
        "texto": "Que dice el articulo 72 de la ley 11179?",
        "por_que": (
            "Número de ley escrito sin punto. El tokenizador parte '11.179' en "
            "'11'/'179' y la consulta dice '11179', así que BM25 no puede "
            "matchear nunca. Además ningún chunk lleva el número de ley: vive "
            "en el encabezado del documento, en otro chunk. Resultado medido: "
            "híbrida+rerank cita el art. 72 de la ley 11.723 como si fuera el "
            "de la 11.179."
        ),
    },
    {
        "id": "lct-art-208",
        "texto": "¿Qué dice el artículo 208 de la Ley de Contrato de Trabajo?",
        "por_que": (
            "El corpus tiene la LCT en su texto ORIGINAL de 1974, no el texto "
            "ordenado vigente. El art. 208 indexado habla de menores; el 208 "
            "que espera cualquier abogado (enfermedades inculpables) está en "
            "el art. 225 de este PDF. La recuperación no falla: falla el "
            "corpus, y sin metadato de versión nadie se entera."
        ),
    },
    {
        "id": "lct-art-57",
        "texto": "¿Qué dice el artículo 57 de la Ley de Contrato de Trabajo?",
        "por_que": (
            "Mismo problema, caso extremo: el contenido del art. 57 vigente "
            "(presunción en contra del empleador por su silencio) directamente "
            "NO existe en el texto de 1974. Cualquier respuesta afirmativa es "
            "falsa."
        ),
    },
    {
        "id": "consumidor-revocacion",
        "texto": "¿Qué plazo tiene el consumidor para revocar la aceptación en una venta domiciliaria?",
        "por_que": (
            "Consulta sana, sin trampa: la Ley 24.240 está en el corpus en su "
            "texto actualizado. Sirve de control — si esta también se rompe "
            "después de reindexar, el problema es el chunking nuevo y no el "
            "corpus."
        ),
    },
    {
        "id": "siri-amparo",
        "texto": "¿Qué resolvió la Corte en el caso Siri sobre la acción de amparo?",
        "por_que": (
            "Fallo, no ley: no tiene artículos numerados. Es el caso que "
            "justifica el fallback del chunking por estructura, y el control "
            "de que cortar por artículo no rompa los documentos que no los "
            "tienen."
        ),
    },
]

CONFIGS = {
    "naive": retrieval.retrieve_naive,
    "sparse": retrieval.retrieve_sparse,
    "hibrida": retrieval.retrieve_hybrid,
    "hibrida_rerank": retrieval.retrieve_hybrid_rerank,
}


def serializar(chunk, posicion: int) -> dict:
    # Se guarda el texto completo, no un recorte: el punto de la comparación es
    # poder leer QUÉ decía el chunk que se recuperó.
    fila = {
        "posicion": posicion,
        "id": chunk.id,
        "source": chunk.source,
        "score": round(float(chunk.score), 6),
        "text": chunk.text,
    }
    # Campos que solo existen después de conectar el manifiesto. Se leen con
    # getattr para que el mismo script sirva para el "antes" y el "después".
    for extra in ("titulo", "version"):
        valor = getattr(chunk, extra, None)
        if valor:
            fila[extra] = valor
    return fila


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--etiqueta", required=True, help="Nombre de la corrida, va en el archivo de salida.")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--sin-llm", action="store_true", help="Solo recuperación, sin gastar llamadas a la API.")
    # La advertencia de versión es lo ÚNICO que se toca entre las dos ramas de
    # la comparación del prompt. Para el snapshot "antes" hay que apagarla:
    # con la advertencia prendida el modelo ya avisa que el texto es de 1974, y
    # entonces el antes/después no muestra nada.
    ap.add_argument("--sin-advertencia-version", action="store_true",
                    help="Apaga la advertencia de versión del system prompt (para el snapshot 'antes').")
    args = ap.parse_args()

    print("Precargando modelos ...")
    retrieval.warmup()

    corrida = {
        "etiqueta": args.etiqueta,
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "top_k": args.top_k,
        "genero_respuestas": not args.sin_llm,
        "advertencia_version": not args.sin_advertencia_version,
        "config": retrieval.config_snapshot(),
        "consultas": [],
    }

    for consulta in CONSULTAS:
        print(f"\n[{consulta['id']}] {consulta['texto']}")
        entrada = dict(consulta, resultados={})
        for nombre, fn in CONFIGS.items():
            inicio = time.perf_counter()
            chunks = fn(consulta["texto"], top_k=args.top_k)
            latencia = time.perf_counter() - inicio

            respuesta = None
            if not args.sin_llm:
                respuesta = generate_answer(
                    consulta["texto"], chunks,
                    advertir_version=not args.sin_advertencia_version,
                )

            entrada["resultados"][nombre] = {
                "latencia_recuperacion_s": round(latencia, 3),
                "respuesta_llm": respuesta,
                "chunks": [serializar(c, i) for i, c in enumerate(chunks, start=1)],
            }
            print(f"  {nombre:16s} {latencia:6.2f}s  " +
                  ", ".join(f"{c.source[:22]}({c.score:.3f})" for c in chunks[:3]))
        corrida["consultas"].append(entrada)

    SALIDA_DIR.mkdir(parents=True, exist_ok=True)
    salida = SALIDA_DIR / f"snapshot_{args.etiqueta}.json"
    salida.write_text(json.dumps(corrida, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nEscrito: {salida}")


if __name__ == "__main__":
    main()
