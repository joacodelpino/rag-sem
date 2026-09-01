# -*- coding: utf-8 -*-
"""
Recorte de UNA pregunta del set dorado contra todas las configuraciones.

No reemplaza a run_ragas.py, lo complementa. La diferencia no es de tamaño
sino de propósito:

  - run_ragas.py responde "¿qué configuración es mejor?" y para eso promedia
    sobre el set completo. Es la tabla de ablación.
  - este script responde "¿qué pasó exactamente con ESTA pregunta?" y para eso
    muestra los ids recuperados y la respuesta generada al lado de las
    métricas. Es material para explicar un caso.

OJO AL LEER LOS NÚMEROS: con n=1 las métricas de Ragas no tienen significancia
estadística. El juez es un LLM y tiene varianza medida entre corridas (la misma
pregunta dio 1.000 y 0.750 en dos corridas del set completo). Acá esa varianza
no se promedia contra nada, así que una diferencia entre dos filas puede ser
ruido puro. Sirve para ILUSTRAR un caso, nunca para ordenar configuraciones.

La razón por la que igual vale la pena: las métricas por id (recall@k, MRR) sí
son determinísticas, y ponerlas al lado de las de Ragas es lo que deja ver
cuándo las dos familias se contradicen. Ese desacuerdo es el hallazgo, no el
valor absoluto de ninguna de las dos. Medido sobre s01 (caso Siri): el naive
sacó recall@5 = 0.000 —no trajo el fallo— y aun así context_recall 1.000 y una
respuesta correcta, porque el corpus MENCIONA a Siri en otros documentos y
porque gpt-4o-mini además lo sabe de memoria.

A DIFERENCIA de run_ragas.py, acá el reranking aparece DOS veces —con el
modelo chico y con el grande— porque el tamaño del cross-encoder es el eje que
más cambia el resultado sin tocar el índice. Son la misma función con distinto
modelo: se intercambia invalidando la carga perezosa de rerank.py.

Uso:
    python evals/ragas_una_pregunta.py --id s01
    python evals/ragas_una_pregunta.py --id i08 --sin-ragas    # gratis, sin juez
    python evals/ragas_una_pregunta.py --id f01 --configs "naive,sparse (BM25)"
"""
import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

# Mismo motivo que en run_ragas.py: el aviso de deprecación de ragas 0.4.3 se
# repite por métrica y por corrida y tapa la única tabla que hay que leer.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas.*")

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "evals"))

import rerank  # noqa: E402
import retrieval  # noqa: E402
import run_ragas as R  # noqa: E402

SALIDA_DIR = RAIZ / "evals" / "results"

RERANKER_CHICO = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
RERANKER_GRANDE = "BAAI/bge-reranker-v2-m3"

# (etiqueta en la tabla, configuración de run_ragas, modelo de reranking).
# El orden va de menos a más sofisticada, igual que en la tabla de ablación,
# para que la progresión se lea sola.
CORRIDAS = [
    ("naive",                   "naive",         None),
    ("sparse (BM25)",           "sparse",        None),
    ("hibrida (RRF)",           "hybrid",        None),
    ("hibrida + rerank chico",  "hybrid_rerank", RERANKER_CHICO),
    ("hibrida + rerank grande", "hybrid_rerank", RERANKER_GRANDE),
]


def cambiar_reranker(modelo: str) -> None:
    """Intercambia el cross-encoder en caliente, dentro del mismo proceso.

    rerank.py carga el modelo una sola vez por proceso y lo guarda en el global
    `_reranker` (carga perezosa: ver _get_reranker). Para correr las dos
    variantes en una sola corrida hay que invalidar ese cache a mano; si no, la
    segunda fila saldría con los pesos de la primera y las dos darían idéntico
    — un error que no rompe nada y por eso es difícil de notar.

    El warmup() no sobra: la PRIMERA inferencia de cada modelo cuesta bastante
    más que las siguientes aunque los pesos ya estén cargados. Sin esto, el
    tiempo de la primera fila con reranking incluiría esa penalidad de arranque
    y no sería comparable con la otra.
    """
    rerank.RERANKER_MODEL = modelo
    rerank._reranker = None
    rerank.warmup()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--id", required=True,
                        help="id de la pregunta en data/golden_set.csv (ej: s01)")
    parser.add_argument("--configs", default=",".join(e for e, _, _ in CORRIDAS),
                        help="etiquetas a correr, separadas por coma")
    parser.add_argument("--top-k", type=int, default=R.TOP_K)
    parser.add_argument("--sin-ragas", action="store_true",
                        help="solo métricas por id: no llama al juez LLM (gratis)")
    parser.add_argument("--modelo-juez", default="gpt-4o-mini")
    args = parser.parse_args()

    todas = R.cargar_set_dorado()
    preguntas = [p for p in todas if p["id"] == args.id]
    if not preguntas:
        raise SystemExit(
            f"No existe la pregunta {args.id!r}.\n"
            f"Ids disponibles: {', '.join(p['id'] for p in todas)}"
        )
    pregunta = preguntas[0]

    etiquetas_validas = {e for e, _, _ in CORRIDAS}
    seleccion = [c.strip() for c in args.configs.split(",") if c.strip()]
    desconocidas = [s for s in seleccion if s not in etiquetas_validas]
    if desconocidas:
        raise SystemExit(
            f"Etiqueta desconocida: {', '.join(desconocidas)}.\n"
            f"Opciones: {', '.join(e for e, _, _ in CORRIDAS)}"
        )
    corridas = [c for c in CORRIDAS if c[0] in seleccion]

    # Las negativas no tienen chunk relevante ni respuesta verificable, así que
    # Ragas las descarta y esas columnas saldrían vacías. Se avisa acá en vez
    # de dejar cinco filas de guiones sin explicación.
    if not pregunta["ids_relevantes"] and not args.sin_ragas:
        print(f"Aviso: {args.id} es una pregunta negativa (sin chunks relevantes).\n"
              f"Ragas la descarta: lo único medible acá es si el sistema se abstiene.\n")

    print(f"pregunta : {pregunta['pregunta']}")
    print(f"esperada : {pregunta['respuesta_esperada']}")
    print(f"chunk ok : {pregunta['ids_relevantes'] or '(negativa)'}")
    print(f"juez     : {'no (--sin-ragas)' if args.sin_ragas else args.modelo_juez}\n")

    retrieval.warmup()

    tabla, detalle = [], {}
    for etiqueta, config, modelo in corridas:
        if modelo:
            print(f"  cargando reranker {modelo} …")
            cambiar_reranker(modelo)

        t0 = time.perf_counter()
        filas = R.correr_configuracion(config, [pregunta], args.top_k,
                                       con_llm=True, advertir_version=None)
        ragas = None if args.sin_ragas else R.metricas_ragas(filas, args.modelo_juez)

        fila = R.resumir(etiqueta, filas, ragas)
        fila["reranker"] = modelo or "-"
        fila["seg_total"] = round(time.perf_counter() - t0, 2)
        tabla.append(fila)
        detalle[etiqueta] = {"preguntas": filas, "ragas": ragas}

    imprimir(tabla, detalle)
    guardar(args, pregunta, tabla, detalle)


def imprimir(tabla: list[dict], detalle: dict) -> None:
    """Tabla de métricas y, debajo, lo que cada configuración recuperó y contestó.

    Las dos partes van juntas a propósito: con una sola pregunta el número
    suelto no explica nada, y la fila de ids es lo que deja ver POR QUÉ una
    configuración sacó lo que sacó.
    """
    def fmt(valor) -> str:
        # NaN != NaN: así se detecta sin importar math. Aparece cuando la
        # métrica no aplica (una negativa no tiene recall).
        return f"{valor:.3f}" if isinstance(valor, float) and valor == valor else "  -  "

    print("\n" + "=" * 100)
    print(f"{'config':<26}{'faithful.':>11}{'ans_corr.':>11}{'ctx_prec.':>11}"
          f"{'ctx_recall':>11}{'recall@k':>10}{'mrr':>8}{'seg':>9}")
    print("-" * 100)
    for f in tabla:
        print(f"{f['config']:<26}{fmt(f.get('faithfulness')):>11}"
              f"{fmt(f.get('answer_correctness')):>11}"
              f"{fmt(f.get('context_precision')):>11}"
              f"{fmt(f.get('context_recall')):>11}"
              f"{fmt(f.get('recall@k')):>10}{fmt(f.get('mrr')):>8}"
              f"{f['seg_total']:>9.1f}")

    print("\n--- qué recuperó y qué contestó cada una")
    for etiqueta, d in detalle.items():
        fila = d["preguntas"][0]
        # El chunk correcto va marcado con >< para poder leer de un vistazo en
        # qué puesto entró, que es lo que las métricas promedian y esconden.
        ids = ", ".join(
            f">{i}<" if i in fila["ids_relevantes"] else str(i)
            for i in fila["ids_recuperados"]
        )
        print(f"\n[{etiqueta}]  ids: {ids}")
        print(f"  {' '.join(fila['respuesta'].split())[:300]}")

    print("\nn=1: las metricas de Ragas aca ilustran un caso, no ordenan configuraciones.")


def guardar(args, pregunta: dict, tabla: list[dict], detalle: dict) -> None:
    """Guarda el recorte completo, con el snapshot de configuración.

    Mismo criterio que run_ragas.py: una fila de la tabla no significa nada si
    no se sabe con qué chunking y cuántos candidatos se produjo. Se guardan
    también los textos recuperados, así se puede releer el caso —o recalcular
    las métricas— sin volver a levantar Qdrant.
    """
    SALIDA_DIR.mkdir(parents=True, exist_ok=True)
    salida = SALIDA_DIR / f"ragas_pregunta_{args.id}.json"
    salida.write_text(json.dumps({
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "pregunta": pregunta,
        "top_k": args.top_k,
        "modelo_juez": None if args.sin_ragas else args.modelo_juez,
        "config": retrieval.config_snapshot(),
        "tabla": tabla,
        "detalle": detalle,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nGuardado: {salida.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
