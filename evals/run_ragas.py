# -*- coding: utf-8 -*-
"""
Evaluación del set dorado — produce la tabla de ablación.

Corre las 29 preguntas de data/golden_set.csv contra cada configuración de
recuperación y mide dos cosas MUY distintas, que conviene no mezclar:

1. MÉTRICAS POR ID (determinísticas, gratis, sin LLM)
   recall@k, precision@k y MRR, comparando los ids que devolvió la
   recuperación contra los ids de `chunks_relevantes` del set dorado. Miden
   SOLO la recuperación, no la respuesta. Son reproducibles al bit: dos
   corridas sobre el mismo índice dan exactamente el mismo número.

2. MÉTRICAS DE RAGAS (juzgadas por un LLM, cuestan plata, tienen varianza)
   faithfulness, answer_correctness, answer_relevancy, context_precision y
   context_recall. Miden la calidad de la RESPUESTA y la utilidad del
   contexto. Un juez LLM no es determinístico: correr esto dos veces da
   números parecidos, no iguales. Medido sobre una misma pregunta y el mismo
   juez: answer_correctness dio 0.619 y 0.801 en dos corridas seguidas,
   mientras recall@k y MRR salieron idénticos al bit.

   Ojo con answer_correctness y answer_relevancy, que suenan igual y no lo
   son: la primera compara contra la respuesta esperada, la segunda ni la
   mira —solo pregunta si la respuesta contesta la pregunta—. Ver el
   comentario de cada una en metricas_ragas().

Por qué las dos y no solo Ragas: si la tabla de ablación solo tuviera métricas
juzgadas por un LLM, no habría forma de saber si una configuración bajó porque
recupera peor o porque el juez tuvo un mal día. Las métricas por id son el
ancla dura; las de Ragas son las que responden la pregunta que le importa al
usuario final.

Las 4 preguntas NEGATIVAS se miden aparte, con su propia métrica —tasa de
abstención—, porque no tienen chunks relevantes ni respuesta que verificar:
lo único correcto es que el sistema diga que no sabe. Incluirlas en el
promedio de recall daría cero y ensuciaría la tabla sin significar nada.

Uso:
    python evals/run_ragas.py                        # todo, las 4 configuraciones
    python evals/run_ragas.py --sin-ragas            # solo métricas por id (gratis)
    python evals/run_ragas.py --configs naive,hybrid_rerank
    python evals/run_ragas.py --limite 3             # prueba de humo, 3 preguntas
    python evals/run_ragas.py --etiqueta con-rerank-grande
"""
import argparse
import csv
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

# ragas 0.4.3 avisa que `ragas.metrics` se va en 1.0, pero es la única API que
# su propio evaluate() acepta (ver el comentario en metricas_ragas). El aviso
# se repite una vez por métrica y por corrida y tapa la tabla, que es lo único
# que hay que leer acá.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas.*")

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from chunk import Chunk  # noqa: E402
from generate import generate_answer  # noqa: E402
import retrieval  # noqa: E402

SET_DORADO = RAIZ / "data" / "golden_set.csv"
SALIDA_DIR = RAIZ / "evals" / "results"

# Las configuraciones de la ablación. El orden importa: es el orden en que
# salen en la tabla, y va de menos a más sofisticada para que la progresión se
# lea sola. `sparse` no es una configuración de la propuesta —es BM25 puro—
# pero está porque sin ella no se puede argumentar qué aporta cada mitad de la
# híbrida.
CONFIGS = {
    "naive": retrieval.retrieve_naive,
    "sparse": retrieval.retrieve_sparse,
    "hybrid": retrieval.retrieve_hybrid,
    "hybrid_rerank": retrieval.retrieve_hybrid_rerank,
}

TOP_K = 5

# Marcas de abstención para las preguntas negativas. Es una HEURÍSTICA léxica,
# no un juez: busca las formas en que el modelo dice "no sé" cuando el system
# prompt le pide que lo diga en vez de inventar. Puede equivocarse en los dos
# sentidos, así que el JSON guarda la respuesta completa de cada negativa para
# poder auditarla a mano. No se usa un LLM acá a propósito: son 4 preguntas por
# configuración, se leen en un minuto, y una métrica auditable a ojo vale más
# que una automática que nadie revisa.
MARCAS_ABSTENCION = (
    "no está en el corpus",
    "no se encuentra en el corpus",
    "no figura",
    "no contiene información",
    "no hay información",
    "no se menciona",
    "no menciona",
    "no proporciona",
    "no incluye",
    "no especifica",
    "no permite responder",
    "no alcanza",
    "no puedo responder",
    "no es posible responder",
    "el contexto no",
    "no se encuentra información",
    "no se proporciona",
)


# ---------------------------------------------------------------------------
# Set dorado
# ---------------------------------------------------------------------------

def cargar_set_dorado(limite: int | None = None) -> list[dict]:
    """Lee golden_set.csv y parsea chunks_relevantes a una lista de ints.

    `chunks_relevantes` viene como "39;40;41" (o vacío en las negativas). Se
    parsea acá y no en cada métrica para que el resto del archivo trabaje con
    listas de ids y no con strings.
    """
    filas = []
    with open(SET_DORADO, encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            crudo = (fila["chunks_relevantes"] or "").strip()
            fila["ids_relevantes"] = [
                int(x) for x in crudo.split(";") if x.strip()
            ]
            filas.append(fila)

    if limite is not None:
        filas = filas[:limite]

    if not filas:
        raise SystemExit(f"El set dorado está vacío: {SET_DORADO}")
    return filas


# ---------------------------------------------------------------------------
# Métricas por id — determinísticas, sin LLM
# ---------------------------------------------------------------------------

def recall_at_k(recuperados: list[int], relevantes: list[int]) -> float:
    """Qué fracción de los chunks relevantes entró en el top_k.

    Es LA métrica de la primera etapa: el reranker no puede rescatar lo que la
    recuperación no trajo. Si el recall es bajo, agregar un cross-encoder más
    grande no arregla nada.
    """
    if not relevantes:
        return float("nan")
    return len(set(recuperados) & set(relevantes)) / len(relevantes)


def precision_at_k(recuperados: list[int], relevantes: list[int]) -> float:
    """Qué fracción de lo recuperado era efectivamente relevante.

    Importa porque cada chunk irrelevante que entra al contexto es ruido que
    el LLM tiene que descartar — y a veces no lo descarta.

    OJO al leerla: con top_k=5 y una sola respuesta correcta, el máximo posible
    es 0.2, no 1.0. Un 0.2 en las preguntas factuales es el techo, no una nota
    mala. Por eso esta columna solo sirve para comparar configuraciones ENTRE
    sí, nunca como valor absoluto.
    """
    if not relevantes or not recuperados:
        return float("nan")
    return len(set(recuperados) & set(relevantes)) / len(recuperados)


def mrr(recuperados: list[int], relevantes: list[int]) -> float:
    """1 / posición del PRIMER chunk relevante. 0 si no apareció ninguno.

    Es la métrica sensible al orden, y por eso es donde se ve el efecto del
    reranking: mover el chunk correcto del puesto 4 al 1 no cambia el recall@5
    en nada, pero duplica el MRR. Sin esta métrica el reranking parecería no
    hacer nada.
    """
    if not relevantes:
        return float("nan")
    relevantes_set = set(relevantes)
    for posicion, chunk_id in enumerate(recuperados, start=1):
        if chunk_id in relevantes_set:
            return 1.0 / posicion
    return 0.0


def se_abstuvo(respuesta: str) -> bool:
    """True si la respuesta admite que el corpus no tiene el dato."""
    bajo = respuesta.lower()
    return any(marca in bajo for marca in MARCAS_ABSTENCION)


# ---------------------------------------------------------------------------
# Recuperación + generación
# ---------------------------------------------------------------------------

def correr_configuracion(
    nombre: str,
    preguntas: list[dict],
    top_k: int,
    con_llm: bool,
    advertir_version: bool | None,
) -> list[dict]:
    """Corre las N preguntas contra UNA configuración.

    Devuelve una fila por pregunta con todo lo necesario para calcular después
    cualquier métrica: los ids recuperados, el texto de los chunks (que es lo
    que Ragas necesita como `retrieved_contexts`) y la respuesta generada.

    Guardar el texto y no solo los ids es a propósito: permite recalcular las
    métricas de Ragas sobre un JSON viejo sin volver a consultar Qdrant, que es
    lo que salva la exposición si el día de la presentación algo no levanta.
    """
    recuperar = CONFIGS[nombre]
    filas = []

    for i, pregunta in enumerate(preguntas, start=1):
        t0 = time.perf_counter()
        chunks: list[Chunk] = recuperar(pregunta["pregunta"], top_k=top_k)
        t_recuperacion = time.perf_counter() - t0

        t0 = time.perf_counter()
        respuesta = (
            generate_answer(pregunta["pregunta"], chunks, advertir_version)
            if con_llm
            else ""
        )
        t_generacion = time.perf_counter() - t0

        ids = [c.id for c in chunks]
        relevantes = pregunta["ids_relevantes"]

        filas.append({
            "id": pregunta["id"],
            "categoria": pregunta["categoria"],
            "pregunta": pregunta["pregunta"],
            "respuesta_esperada": pregunta["respuesta_esperada"],
            "ids_relevantes": relevantes,
            "ids_recuperados": ids,
            "etiquetas_recuperadas": [c.etiqueta() for c in chunks],
            "textos_recuperados": [c.text for c in chunks],
            "scores": [round(float(c.score), 5) for c in chunks],
            "respuesta": respuesta,
            "recall": recall_at_k(ids, relevantes),
            "precision": precision_at_k(ids, relevantes),
            "mrr": mrr(ids, relevantes),
            "abstuvo": se_abstuvo(respuesta) if (con_llm and not relevantes) else None,
            "segundos_recuperacion": round(t_recuperacion, 3),
            "segundos_generacion": round(t_generacion, 3),
        })

        print(f"  [{nombre}] {i}/{len(preguntas)} {pregunta['id']}", end="\r", flush=True)

    print(f"  [{nombre}] {len(preguntas)}/{len(preguntas)} listo." + " " * 20)
    return filas


# ---------------------------------------------------------------------------
# Ragas
# ---------------------------------------------------------------------------

def _embeddings_locales():
    """Adapta BGE-M3 (el mismo modelo de la ingesta) a la interfaz de Ragas.

    Ragas necesita embeddings para answer_correctness. Por defecto usaría los
    de OpenAI, que cuestan plata y —más importante— serían un modelo DISTINTO
    del que construyó el índice: la evaluación estaría midiendo con una regla
    que el sistema evaluado no usa. Reusar el embedder local mantiene la
    coherencia y sale gratis; el costo es CPU, que en 29 preguntas es
    despreciable.

    Se implementa BaseRagasEmbedding a mano en vez de usar un wrapper de
    LangChain porque son cuatro líneas y evita meter otra capa de la que
    depender. aembed_text no es async de verdad —SentenceTransformer es
    bloqueante— pero la interfaz la exige; Ragas la llama dentro de su propio
    executor y no le molesta que devuelva enseguida.

    HAY QUE IMPLEMENTAR DOS INTERFACES, no una, y esto no se deduce leyendo
    BaseRagasEmbedding: ragas 0.4.3 no usa la misma en todas sus métricas.
    answer_correctness llama embed_text (la de BaseRagasEmbedding), pero
    answer_relevancy llama embed_query y embed_documents (la vieja, estilo
    LangChain) — en su propio código fuente esas dos líneas van con
    `# type: ignore[attr-defined]`, o sea que ni ragas cree que existan.

    Sin los dos métodos de abajo, answer_relevancy no falla: sale `n/a` en la
    tabla, con un `AttributeError` sepultado entre los logs del executor. Se
    descubrió en un piloto de 3 preguntas, que es exactamente para lo que
    sirve correr el piloto antes que el set completo.
    """
    from ragas.embeddings import BaseRagasEmbedding

    modelo = retrieval._get_embedder()

    class BGEM3(BaseRagasEmbedding):
        def embed_text(self, text: str, **kwargs) -> list[float]:
            return modelo.encode(text, normalize_embeddings=True).tolist()

        async def aembed_text(self, text: str, **kwargs) -> list[float]:
            return self.embed_text(text)

        # Interfaz vieja, la que espera answer_relevancy. Ver el docstring.
        def embed_query(self, text: str) -> list[float]:
            return self.embed_text(text)

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            # De a lote y no uno por uno: answer_relevancy embebe las N
            # preguntas artificiales que generó el juez, y SentenceTransformer
            # las hace todas juntas mucho más rápido que en N llamadas.
            return modelo.encode(texts, normalize_embeddings=True).tolist()

    return BGEM3()


def metricas_ragas(filas: list[dict], modelo_juez: str) -> dict:
    """Corre Ragas sobre las filas que tienen respuesta de referencia.

    Las negativas quedan afuera: Ragas necesita un `reference` verificable y
    un contexto que efectivamente contenga la respuesta. "No está en el
    corpus" no es ninguna de las dos cosas — medirlas acá daría cero en las
    cuatro métricas y arrastraría el promedio hacia abajo por una razón que no
    tiene que ver con la calidad del sistema.

    Devuelve el promedio por métrica más el detalle por pregunta.
    """
    # Se importan las métricas de `ragas.metrics` y no de `ragas.metrics.
    # collections`, aunque las primeras avisen que están deprecadas: en
    # ragas 0.4.3 evaluate() valida `isinstance(m, Metric)` y las de
    # collections son de la API v2, así que rechaza la lista entera con
    # "All metrics must be initialised metric objects". Es decir, la API
    # nueva todavía no está enchufada al runner viejo. Cuando ragas llegue a
    # 1.0 hay que migrar las dos cosas juntas, no solo los imports.
    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        AnswerCorrectness,
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    evaluables = [f for f in filas if f["ids_relevantes"]]
    if not evaluables:
        return {"promedios": {}, "detalle": [], "nota": "sin filas evaluables"}

    muestras = [
        SingleTurnSample(
            user_input=f["pregunta"],
            retrieved_contexts=f["textos_recuperados"],
            response=f["respuesta"],
            reference=f["respuesta_esperada"],
        )
        for f in evaluables
    ]

    # temperature=0 en el juez por la misma razón que en la generación: no
    # elimina la varianza (el modelo sigue siendo estocástico) pero la reduce
    # lo suficiente como para que dos corridas sean comparables.
    juez = LangchainLLMWrapper(ChatOpenAI(model=modelo_juez, temperature=0))
    embeddings = _embeddings_locales()

    metricas = [
        # ¿La respuesta se sostiene SOLO con el contexto recuperado? Es la
        # métrica de alucinación: baja si el modelo agregó algo de su
        # conocimiento previo. No necesita ground truth.
        Faithfulness(llm=juez),
        # ¿La respuesta dice lo mismo que la respuesta esperada? Es la métrica
        # que le importa al usuario final, y la única que puede estar alta con
        # un contexto malo (el modelo acertó de memoria) — por eso se lee
        # junto con faithfulness, no sola.
        AnswerCorrectness(llm=juez, embeddings=embeddings),
        # ¿La respuesta CONTESTA la pregunta? Ojo que no es lo mismo que
        # answer_correctness y son fáciles de confundir: correctness compara
        # contra la respuesta esperada, esta ni la mira. Ragas la calcula al
        # revés de lo que uno esperaría —le pide al juez que genere preguntas
        # artificiales A PARTIR de la respuesta y mide cuánto se parecen a la
        # original— así que mide PERTINENCIA, no verdad.
        #
        # Consecuencia práctica al leer la tabla: una respuesta rotundamente
        # falsa pero bien encarada saca answer_relevancy alto. Solo sirve al
        # lado de faithfulness y correctness, nunca sola. Lo que sí detecta
        # bien es la evasiva: las respuestas no comprometidas ("no puedo
        # responder") las manda cerca de cero, que es justamente lo que hace
        # la híbrida cuando el contexto no le alcanza.
        #
        # La clase se llama ResponseRelevancy desde ragas 0.4 (AnswerRelevancy
        # quedó como alias deprecado), pero su .name sigue siendo
        # "answer_relevancy" y ese es el nombre de la columna.
        ResponseRelevancy(llm=juez, embeddings=embeddings),
        # De los chunks recuperados, ¿los útiles salieron primero? Es el
        # equivalente juzgado por LLM del MRR, y donde debería verse el
        # reranking. Se le acorta el nombre para que entre en la tabla.
        LLMContextPrecisionWithReference(llm=juez, name="context_precision"),
        # ¿El contexto alcanza para reconstruir la respuesta esperada? Es el
        # techo de todo: ninguna generación puede superar su context_recall.
        LLMContextRecall(llm=juez),
    ]

    resultado = evaluate(
        dataset=EvaluationDataset(samples=muestras),
        metrics=metricas,
        show_progress=True,
    )

    df = resultado.to_pandas()
    columnas = [m.name for m in metricas]
    promedios = {c: float(df[c].mean()) for c in columnas if c in df}
    detalle = [
        {"id": f["id"], **{c: float(df[c][i]) for c in columnas if c in df}}
        for i, f in enumerate(evaluables)
    ]
    return {"promedios": promedios, "detalle": detalle}


# ---------------------------------------------------------------------------
# Tabla de ablación
# ---------------------------------------------------------------------------

def promedio(valores: list[float]) -> float:
    limpios = [v for v in valores if v == v]  # descarta NaN
    return sum(limpios) / len(limpios) if limpios else float("nan")


def percentil(valores: list[float], p: float) -> float:
    """Percentil por el método del vecino más cercano, sin interpolar.

    Se implementa a mano en vez de usar statistics.quantiles porque acá el
    valor tiene que ser UNA consulta que realmente ocurrió: "el p95 son 5.481 s"
    significa que hubo una consulta de 5.481 s, no un promedio ponderado entre
    dos. Con 29 preguntas la diferencia entre métodos es de milisegundos, pero
    la interpretación cambia y es la que se cuenta en la exposición.
    """
    limpios = sorted(v for v in valores if v == v)
    if not limpios:
        return float("nan")
    return limpios[min(int(len(limpios) * p), len(limpios) - 1)]


def resumir(nombre: str, filas: list[dict], ragas: dict | None) -> dict:
    """Arma UNA fila de la tabla de ablación."""
    con_relevantes = [f for f in filas if f["ids_relevantes"]]
    negativas = [f for f in filas if not f["ids_relevantes"]]
    abstenciones = [f["abstuvo"] for f in negativas if f["abstuvo"] is not None]

    # Latencia punta a punta = recuperación + generación, por pregunta. Es la
    # que percibe quien pregunta, y la única honesta para comparar
    # configuraciones: seg_recuperacion sola exagera la diferencia porque deja
    # afuera la llamada al LLM, que es un piso de ~1.7 s que TODAS pagan igual.
    # Medido sobre el set completo: por recuperación el reranking cuesta 9x lo
    # que la híbrida (2.699 vs 0.296), punta a punta poco más del doble
    # (4.374 vs 2.018).
    #
    # p50 y no promedio porque un solo outlier corre la media y el p50 no. Y el
    # p95 al lado porque es donde se ve el peor caso realista: si la demo se
    # cuelga en vivo va a ser en una consulta del p95, no en la mediana.
    e2e = [f["segundos_recuperacion"] + f["segundos_generacion"] for f in filas]

    fila = {
        "config": nombre,
        "preguntas": len(filas),
        "recall@k": promedio([f["recall"] for f in con_relevantes]),
        "precision@k": promedio([f["precision"] for f in con_relevantes]),
        "mrr": promedio([f["mrr"] for f in con_relevantes]),
        "abstencion_negativas": (
            sum(abstenciones) / len(abstenciones) if abstenciones else float("nan")
        ),
        "seg_recuperacion": promedio([f["segundos_recuperacion"] for f in filas]),
        "seg_p50_e2e": percentil(e2e, 0.50),
        "seg_p95_e2e": percentil(e2e, 0.95),
    }
    if ragas:
        fila.update({k: v for k, v in ragas["promedios"].items()})
    return fila


def imprimir_tabla(tabla: list[dict]) -> None:
    if not tabla:
        return
    columnas = list(tabla[0].keys())
    anchos = {
        c: max(len(c), *(len(_fmt(f.get(c))) for f in tabla)) for c in columnas
    }
    print()
    print("  ".join(c.ljust(anchos[c]) for c in columnas))
    print("  ".join("-" * anchos[c] for c in columnas))
    for f in tabla:
        print("  ".join(_fmt(f.get(c)).ljust(anchos[c]) for c in columnas))
    print()


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return "n/a" if v != v else f"{v:.3f}"
    return str(v)


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--etiqueta", default="",
                        help="sufijo del archivo de salida (default: timestamp)")
    parser.add_argument("--configs", default=",".join(CONFIGS),
                        help=f"configuraciones a correr, separadas por coma. Opciones: {', '.join(CONFIGS)}")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--limite", type=int, default=None,
                        help="usar solo las primeras N preguntas (prueba de humo)")
    parser.add_argument("--sin-ragas", action="store_true",
                        help="solo métricas por id: no llama al juez LLM (gratis)")
    parser.add_argument("--sin-llm", action="store_true",
                        help="no genera respuestas. Implica --sin-ragas: sin respuesta no hay nada que juzgar")
    parser.add_argument("--sin-advertencia-version", action="store_true",
                        help="apaga la advertencia de versión del system prompt (el 'antes')")
    parser.add_argument("--modelo-juez", default=os.environ.get("RAGAS_JUDGE_MODEL", "gpt-4o-mini"),
                        help="modelo que juzga las métricas de Ragas")
    args = parser.parse_args()

    nombres = [c.strip() for c in args.configs.split(",") if c.strip()]
    desconocidas = [c for c in nombres if c not in CONFIGS]
    if desconocidas:
        raise SystemExit(
            f"Configuración desconocida: {', '.join(desconocidas)}.\n"
            f"Opciones válidas: {', '.join(CONFIGS)}")

    con_llm = not args.sin_llm
    con_ragas = con_llm and not args.sin_ragas
    advertir = False if args.sin_advertencia_version else None

    preguntas = cargar_set_dorado(args.limite)
    negativas = sum(1 for p in preguntas if not p["ids_relevantes"])

    print(f"Set dorado: {len(preguntas)} preguntas ({negativas} negativas)")
    print(f"Configuraciones: {', '.join(nombres)}  ·  top_k={args.top_k}")
    print(f"Generación: {'sí' if con_llm else 'no'}  ·  Ragas: "
          f"{args.modelo_juez if con_ragas else 'no'}")
    print()

    retrieval.warmup()

    tabla, detalle = [], {}
    for nombre in nombres:
        filas = correr_configuracion(nombre, preguntas, args.top_k, con_llm, advertir)
        ragas = None
        if con_ragas:
            print(f"  [{nombre}] Ragas ({args.modelo_juez})…")
            ragas = metricas_ragas(filas, args.modelo_juez)
        tabla.append(resumir(nombre, filas, ragas))
        detalle[nombre] = {"preguntas": filas, "ragas": ragas}

    imprimir_tabla(tabla)

    etiqueta = args.etiqueta or datetime.now().strftime("%Y%m%d-%H%M")
    SALIDA_DIR.mkdir(parents=True, exist_ok=True)

    # El snapshot de configuración va en el JSON y no en un comentario: una
    # fila de la tabla no significa nada si no se sabe con qué reranker, qué
    # chunking y cuántos candidatos se produjo.
    salida_json = SALIDA_DIR / f"ragas_{etiqueta}.json"
    salida_json.write_text(json.dumps({
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "top_k": args.top_k,
        "advertir_version": advertir is not False,
        "modelo_juez": args.modelo_juez if con_ragas else None,
        "config": retrieval.config_snapshot(),
        "tabla": tabla,
        "detalle": detalle,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    salida_csv = SALIDA_DIR / f"ragas_{etiqueta}.csv"
    with open(salida_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(tabla[0].keys()))
        writer.writeheader()
        writer.writerows(tabla)

    print(f"Tabla:   {salida_csv.relative_to(RAIZ)}")
    print(f"Detalle: {salida_json.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
