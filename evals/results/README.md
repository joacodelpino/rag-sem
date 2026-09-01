# `evals/results/` — qué es cada archivo

Salidas de los scripts de `evals/`. Ninguna se genera a mano: cada una la
escribe un comando que está anotado abajo, así que **todas se pueden
regenerar**. Si una contradice al código, gana el código — volvé a correrla.

## Convención de nombres

El prefijo dice qué script la produjo; lo que sigue dice qué variaba en esa
corrida:

| Prefijo | Lo escribe | Con qué |
|---|---|---|
| `ragas_<etiqueta>` | `run_ragas.py` | `--etiqueta <etiqueta>` |
| `ragas_pregunta_<id>` | `ragas_una_pregunta.py` | `--id <id>` |
| `snapshot_<etiqueta>` | `snapshot_retrieval.py` | `--etiqueta <etiqueta>` |
| `rerank_latency` | `bench_rerank.py` | (nombre fijo) |

Los `.csv` son la tabla resumida —una fila por configuración—, para leer de un
vistazo o pegar en la presentación. Los `.json` traen el detalle por pregunta:
chunks recuperados, respuesta generada, tiempos y el `config_snapshot()` de la
corrida. **Ese snapshot es lo que hace interpretable a la tabla**: una fila no
significa nada sin saber con qué reranker, qué chunking y cuántos candidatos se
produjo.

`historico/` guarda corridas superadas por otra más nueva. No se borran porque
comparar dos corridas de la misma configuración es lo que muestra cuánta
varianza tiene el juez LLM.

---

## Archivos vigentes

### `ragas_rerank-chico.{csv,json}` — la tabla de ablación
Las 4 configuraciones × 29 preguntas del set dorado, con el reranker chico
(`mmarco-mMiniLMv2-L12-H384-v1`). Juez `gpt-4o-mini`. **Es la tabla principal.**

```bash
python evals/run_ragas.py --configs naive,sparse,hybrid,hybrid_rerank --etiqueta rerank-chico
```

### `ragas_rerank-grande.{csv,json}` — el mismo set con el reranker grande
Idéntica a la anterior salvo por `RERANKER_MODEL=BAAI/bge-reranker-v2-m3`.
Existe para responder cuánta calidad compra el modelo de 568M frente al de
118M, y a qué costo de latencia. Cambiar de reranker **no requiere reingestar**:
actúa sobre los candidatos ya recuperados, no sobre el índice.

⚠️ **No es comparable columna a columna con la de arriba.** Se corrió antes de
que existieran `answer_relevancy`, `seg_p50_e2e` y `seg_p95_e2e`, así que esas
tres columnas no están. Para compararlas de verdad hay que volver a correrla:

```bash
# con RERANKER_MODEL=BAAI/bge-reranker-v2-m3 en .env  (~15 min, el grande tarda ~30 s/pregunta)
python evals/run_ragas.py --configs hybrid_rerank --etiqueta rerank-grande
```

### `ragas_pregunta_s01.json` — el recorte del caso Siri
Una sola pregunta (*"¿En qué fallo la Corte Suprema creó la acción de amparo?"*)
contra las 5 configuraciones, con los dos rerankers lado a lado.

Documenta el desacuerdo entre las dos familias de métricas: el naive saca
`recall@5 = 0.000` —no trae el fallo— y aun así `context_recall 1.000` y una
respuesta correcta, porque el corpus **menciona** a Siri en otros documentos y
porque `gpt-4o-mini` además lo sabe de memoria. Cuando las métricas juzgadas
por LLM y las métricas por id se contradicen, la de id es la que tiene razón.

**n=1: ilustra un caso, no ordena configuraciones.**

```bash
python evals/ragas_una_pregunta.py --id s01
```

### `rerank_latency.csv` — calidad/latencia de cada reranker
Cuánto tarda cada cross-encoder por consulta, medido aparte de la ablación
para poder discutir el trade-off sin mezclarlo con la calidad. Citado en
`README.md` y en `ARQUITECTURA.md`.

```bash
python evals/bench_rerank.py
```

### `snapshot_antes-chunking-fijo.json` y `snapshot_despues-chunking-articulo.json`
El par que documenta el cambio de estrategia de chunking (ventanas de 800
caracteres → corte por artículo). Son consultas de prueba congeladas **antes**
de reingestar, para poder comparar contra el **después** sin depender de la
memoria de nadie.

No son de Ragas y no llevan métricas: son los chunks crudos que devolvía cada
consulta. Sirven para responder "¿esto mejoró o lo estoy imaginando?".

```bash
python evals/snapshot_retrieval.py --etiqueta despues-chunking-articulo
```

---

## `historico/`

### `ragas_rerank-chico-2026-08-28.{csv,json}`
La ablación con reranker chico anterior a `ragas_rerank-chico.{csv,json}`.
Misma configuración, mismo juez, mismo índice — **superada** solo porque la
nueva agrega `answer_relevancy` y las columnas de latencia punta a punta.

Se guarda porque comparar las dos es la mejor evidencia de qué métricas
aguantan una pregunta incómoda y cuáles no:

| | 2026-08-28 → 2026-09-01 |
|---|---|
| `recall@k`, `precision@k`, `mrr`, `context_recall` | delta **0.000** en las 4 configuraciones |
| `faithfulness` (hybrid) | 0.966 → 0.927 (**−0.039**) |
| `answer_correctness` (sparse) | 0.539 → 0.577 (**+0.038**) |

Nada del sistema cambió entre las dos corridas. Las métricas por id son
reproducibles al bit; las juzgadas por LLM se mueven solas hasta ~0.04. **En
las columnas de Ragas, una diferencia menor a ~0.04 no es un hallazgo.**
