# RAG jurídico — demo (Seminario de Actualización Tecnológica)

Demo académica de RAG vectorial y RAG de segunda generación (híbrido +
reranking) sobre un corpus de documentación jurídica, con **Qdrant** como
base vectorial y **Ragas** como framework de evaluación.

Estado actual: **pipeline completo** — las tres configuraciones de
recuperación (naive densa, híbrida densa+BM25 fusionadas con RRF, e híbrida +
reranking con cross-encoder) comparables lado a lado en la app sobre el corpus
real de 36 PDFs, más el módulo de evaluación que corre el set dorado de 29
preguntas y produce la tabla de ablación.

## Cómo levantarlo

Requisitos: Python 3.11+, Docker, y una `OPENAI_API_KEY`.

```bash
cp .env.example .env          # y completá OPENAI_API_KEY
docker compose up -d          # levanta Qdrant

python -m venv .venv && source .venv/Scripts/activate    # Linux/Mac: .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

> `torch` se instala aparte y desde el índice CPU-only de PyTorch: los wheels
> por defecto traen CUDA (~2 GB extra) que no sirve sin GPU. BGE-M3 y el
> reranker corren en CPU, así que la versión liviana alcanza.

Después, indexá los documentos (la primera corrida descarga BGE-M3, ~2 GB;
la ingesta completa del corpus tarda **~30 minutos** en CPU):

```bash
python src/ingest.py
```

Y levantá la demo:

```bash
streamlit run src/app.py
```

Abrí http://localhost:8501 y probá, por ejemplo:
*"¿Cuántos días hábiles hay para presentar el recurso jerárquico?"*

## Operación día a día

> **Todos los comandos `python` de acá abajo asumen el venv del repo.** Si
> `python` te dice `ModuleNotFoundError: No module named 'pymupdf'` (o
> `streamlit`, o `torch`), estás usando el Python del sistema: las
> dependencias no están ahí. Activá el venv primero —
> `.venv\Scripts\Activate.ps1` en PowerShell, `source .venv/Scripts/activate`
> en Git Bash — o llamá al intérprete por ruta, que no depende de la terminal
> ni de la política de ejecución de scripts:
>
> ```powershell
> .venv\Scripts\python.exe src\ingest.py
> .venv\Scripts\python.exe -m streamlit run src\app.py
> ```


Las tres piezas son independientes y se levantan en este orden.

**1. Qdrant** (una vez por sesión de trabajo; sobrevive reinicios de la app)

```bash
docker compose up -d                  # levantar
curl http://localhost:6333/readyz     # verificar: debe responder 200
docker compose down                   # bajar (los datos persisten en el volumen)
```

Requiere Docker Desktop corriendo. Si `docker compose up` dice
`failed to connect to the docker API`, abrí Docker Desktop y esperá a que
termine de arrancar.

**2. Ingesta** (cada vez que cambian los documentos de `data/raw/`)

```bash
python src/ingest.py
```

> **Reindexa todo, no incrementalmente.** `ingest.py` borra la colección y la
> vuelve a crear en cada corrida, a propósito: cada ingesta parte de un estado
> limpio y reproducible. Agregar un PDF nuevo significa volver a correr esto
> entero, no hay un "agregar solo el nuevo". Con el corpus de 36 PDFs tarda
> **~30 minutos** en CPU, casi todo en generar los 2126 embeddings. Es
> intencional para la demo — el objetivo es que el índice sea reproducible, no
> que la ingesta sea rápida.

Cuándo hay que correrla: se agregan/quitan/editan documentos de `data/raw/`,
o se cambia `EMBEDDING_MODEL`, `CHUNK_STRATEGY`, `CHUNK_SIZE` o
`CHUNK_OVERLAP`, o se edita `data/manifest.csv` (los títulos y las versiones
van dentro de cada chunk, no solo en el payload).
Cuándo **no** hace falta: al cambiar `RERANKER_MODEL`, `RERANK_CANDIDATES`,
`OPENAI_MODEL` o cualquier cosa de `retrieval.py` — todo eso actúa sobre
candidatos ya indexados.

**3. La app**

```bash
streamlit run src/app.py              # http://localhost:8501
```

Tarda ~18s en arrancar porque precarga los dos modelos (ver `warmup()`); es a
propósito, para que ese costo no lo pague la primera consulta.

Para bajarla: `Ctrl+C` en su terminal. Si quedó corriendo en background y no
tenés la terminal, en Windows:

```bash
netstat -ano | findstr :8501
taskkill /F /PID <pid>
```

Bajar la app NO baja Qdrant: son procesos separados (`docker compose down`
para eso).

## Evaluación

```bash
python evals/run_ragas.py                 # todo: 4 configuraciones x 29 preguntas
python evals/run_ragas.py --sin-ragas     # solo métricas por id (no gasta API)
python evals/run_ragas.py --limite 3 --configs naive    # prueba de humo
```

Para mirar **una** pregunta en detalle en vez del promedio —qué recuperó y qué
contestó cada configuración, con el reranker chico y el grande lado a lado—:

```bash
python evals/ragas_una_pregunta.py --id s01               # el caso Siri
python evals/ragas_una_pregunta.py --id f01 --sin-ragas   # sin juez
```

Escribe dos archivos en `evals/results/`: un CSV con la tabla de ablación (una
fila por configuración) y un JSON con el detalle por pregunta —chunks
recuperados, respuesta generada, tiempos y `config_snapshot()`—.

Mide **dos familias de métricas que no hay que mezclar**:

| | Cómo se calcula | Qué mide | Reproducible |
|---|---|---|---|
| `recall@k`, `precision@k`, `mrr` | comparando ids recuperados contra `chunks_relevantes` | solo la recuperación | sí, al bit |
| `faithfulness`, `answer_correctness`, `context_precision`, `context_recall` | Ragas, juzgado por un LLM | la calidad de la respuesta | no, tiene varianza |

Si la tabla solo tuviera métricas de Ragas no habría forma de saber si una
configuración bajó porque recupera peor o porque el juez tuvo un mal día. Las
métricas por id son el ancla dura.

Detalles que importan al leer la tabla:

- **`precision@k` tiene techo 0.2** con `top_k=5` y una sola respuesta
  correcta. Sirve para comparar configuraciones entre sí, no como valor
  absoluto.
- **Las 4 preguntas negativas se miden aparte**, con `abstencion_negativas`:
  no tienen chunks relevantes ni respuesta que verificar, y lo único correcto
  es que el sistema diga que no sabe. Meterlas en el promedio de recall daría
  cero por una razón que no tiene que ver con la calidad del sistema.
- **La abstención se detecta con una heurística léxica**, no con un juez. Son
  4 respuestas por configuración: el JSON las guarda enteras y se auditan a
  ojo en un minuto.
- **Ragas usa BGE-M3 local para sus embeddings**, no los de OpenAI: medir con
  un modelo distinto del que construyó el índice sería evaluar con una regla
  que el sistema evaluado no usa. Además sale gratis.

## Cómo funciona

Ver [ARQUITECTURA.md](ARQUITECTURA.md) — recorrido archivo por archivo del
pipeline, pensado para poder explicarlo en la exposición.

## Estructura

```
rag-legal-demo/
├── docker-compose.yml        # solo Qdrant
├── .env.example
├── requirements.txt
├── data/
│   ├── raw/                  # documentos del corpus (36 PDFs)
│   ├── manifest.csv          # identidad de cada PDF: título, ley, versión
│   └── golden_set.csv        # set dorado: 29 preguntas verificadas contra el índice
├── src/
│   ├── ingest.py             # manifiesto + chunking + indexado (denso + sparse)
│   ├── chunk.py              # el tipo que viaja por todo el pipeline
│   ├── bm25.py               # BM25 a mano: la mitad léxica de la híbrida
│   ├── rerank.py             # cross-encoder: la segunda etapa
│   ├── retrieval.py          # las 3 configuraciones, misma firma
│   ├── generate.py           # prompt + llamada a OpenAI (único punto de contacto con el LLM)
│   └── app.py                # Streamlit, configuraciones en columnas
├── evals/
│   ├── build_manifest.py     # genera data/manifest.csv a partir de data/raw/
│   ├── snapshot_retrieval.py # congela las consultas de prueba antes de reingestar
│   ├── build_golden_set.py   # genera data/golden_set.csv verificando contra el índice
│   ├── verify_golden_set.py  # audita el set dorado (independiente del generador)
│   ├── run_ragas.py          # evaluación + tabla de ablación
│   ├── ragas_una_pregunta.py # recorte de UNA pregunta contra las 5 configuraciones
│   ├── bench_rerank.py       # latencia de cada reranker (calidad/latencia)
│   └── results/              # CSV y JSON de cada corrida
├── tests/
│   └── test_bm25.py          # tokenizador (se corre sin pytest)
└── notebooks/                # exploración, no se presenta
```

## Cuando agregás PDFs al corpus

`ingest.py` **falla a propósito** si un archivo de `data/raw/` no está en el
manifiesto. El orden es:

```powershell
.venv\Scripts\python.exe evals\build_manifest.py    # regenera data/manifest.csv
# revisá a mano el título y la versión del documento nuevo
.venv\Scripts\python.exe src\ingest.py              # reindexa TODO (~30 min)
```

Los títulos del manifiesto están curados a mano: el nombre de archivo no
identifica nada (cinco documentos se llaman `0N_argentinagobar.pdf`, y
`08_ley-15.pdf` no es la ley 15 sino el Decreto-Ley 15.348). Si dejás un título
en `indeterminado`, la ingesta se niega a correr.

Antes de reindexar, si querés conservar el comportamiento actual para
compararlo después:

```powershell
.venv\Scripts\python.exe evals\snapshot_retrieval.py --etiqueta antes
```

## Notas

- **Qdrant en Docker, todo lo demás local.** Así el modelo de embeddings se
  cachea una vez en el host y no hay que rebuildear una imagen por cada cambio
  de código. El dashboard de Qdrant queda en http://localhost:6333/dashboard.
- **El tab Graph del dashboard tira `Bad Request`** si no le decís qué vector
  usar: la colección tiene DOS vectores nombrados por punto y la petición por
  defecto no especifica ninguno. Agregale `"using": "dense"` al JSON y anda:
  `{ "limit": 5, "sample": 100, "using": "dense" }`. No es un problema de
  límite.
- **`ingest.py` recrea la colección en cada corrida**, a propósito: cada
  ingesta parte de un estado limpio y reproducible.
- **Cada chunk lleva un encabezado con la identidad de su documento** —
  `[Ley 11.179 - Código Penal de la Nación · texto original · Art. 72]`—
  dentro del texto que se embebe y que tokeniza BM25. Sin eso, el chunk del
  art. 72 no contiene en ningún lado la cadena "11.179" (vive en la carátula,
  que es otro chunk) y ninguna rama puede conectarlo con la consulta.
- **Chunking por límites de artículo** (`CHUNK_STRATEGY=articulo`, default);
  en los fallos corta por voto, y en los documentos sin estructura numerada
  cae a ventanas fijas. `CHUNK_STRATEGY=fijo` recupera el chunking anterior
  para poder comparar las dos estrategias en la ablación. **Cambiar esta
  variable obliga a reingestar**, a diferencia de `RERANKER_MODEL`.
- **Dos rerankers configurables** por `RERANKER_MODEL`, sin reingestar:
  `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (~118M) para la demo en vivo y
  `BAAI/bge-reranker-v2-m3` (~568M) para la corrida de evaluación. El
  reranking corre una vez por par (consulta, chunk), así que el tamaño del
  modelo es el factor de latencia dominante del pipeline. Números medidos:
  `python evals/bench_rerank.py` → `evals/results/rerank_latency.csv`.
- **Los modelos se precargan al arrancar Streamlit**, no en la primera
  consulta. Es por la exposición: la carga de pesos desde disco tardaba ~46s
  y se la comía quien preguntara primero.
- **Corpus real**: 36 PDFs en `data/raw/`, ~1.48M caracteres. Verificado que
  **los 36 tienen texto nativo**: no hace falta OCR, que estaba fuera de
  alcance de todos modos.
- **La advertencia de versión del system prompt se puede apagar**
  (`ADVERTIR_VERSION=0`, o el checkbox "Advertir sobre la versión" en la app).
  Está para poder mostrar el antes y el después de la misma consulta en vivo:
  con la advertencia apagada, el modelo responde el art. 208 de la LCT del
  texto de 1974 sin pestañear. Apagarla **no** desarma los metadatos —la
  versión sigue en el encabezado de cada chunk y en la cita—, así que aísla el
  efecto del prompt del efecto del manifiesto.
- **El corpus tiene la Ley de Contrato de Trabajo en su texto original de
  1974**, no en el texto ordenado vigente, y la numeración está corrida ~17
  artículos. No es un bug del pipeline y no se arregla recuperando mejor; está
  documentado en ARQUITECTURA.md porque es el caso que justifica todo el
  manifiesto.
