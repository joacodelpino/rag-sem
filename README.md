# RAG jurídico — demo (Seminario de Actualización Tecnológica)

Demo académica de RAG vectorial y RAG de segunda generación (híbrido +
reranking) sobre un corpus de documentación jurídica, con **Qdrant** como
base vectorial y **Ragas** como framework de evaluación.

Estado actual: **las tres configuraciones de recuperación implementadas** —
naive (densa), híbrida (densa + BM25 fusionadas con RRF) e híbrida + reranking
con cross-encoder, comparables lado a lado en la app. Falta el módulo de
evaluación con Ragas y el corpus real.

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

Después, indexá los documentos (la primera corrida descarga BGE-M3, ~2 GB,
tarda unos minutos):

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
> entero, no hay un "agregar solo el nuevo". Con este corpus tarda unos
> minutos en CPU. Es intencional para la demo — el objetivo es que el índice
> sea reproducible, no que la ingesta sea rápida.

Cuándo hay que correrla: se agregan/quitan/editan documentos de `data/raw/`,
o se cambia `EMBEDDING_MODEL`, `CHUNK_SIZE` o `CHUNK_OVERLAP`.
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
│   ├── raw/                  # documentos del corpus (4 .txt de ejemplo por ahora)
│   └── golden_set.csv        # 30 preguntas del set dorado (a completar)
├── src/
│   ├── ingest.py             # parsing + chunking + indexado (denso + sparse)
│   ├── chunk.py              # el tipo que viaja por todo el pipeline
│   ├── bm25.py               # BM25 a mano: la mitad léxica de la híbrida
│   ├── rerank.py             # cross-encoder: la segunda etapa
│   ├── retrieval.py          # las 3 configuraciones, misma firma
│   ├── generate.py           # prompt + llamada a OpenAI (único punto de contacto con el LLM)
│   └── app.py                # Streamlit, configuraciones en columnas
├── evals/
│   ├── bench_rerank.py       # latencia de cada reranker (calidad/latencia)
│   └── results/              # CSV de cada corrida (Ragas todavía sin implementar)
└── notebooks/                # exploración, no se presenta
```

## Notas

- **Qdrant en Docker, todo lo demás local.** Así el modelo de embeddings se
  cachea una vez en el host y no hay que rebuildear una imagen por cada cambio
  de código. El dashboard de Qdrant queda en http://localhost:6333/dashboard.
- **`ingest.py` recrea la colección en cada corrida**, a propósito: cada
  ingesta parte de un estado limpio y reproducible.
- **Dos rerankers configurables** por `RERANKER_MODEL`, sin reingestar:
  `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (~118M) para la demo en vivo y
  `BAAI/bge-reranker-v2-m3` (~568M) para la corrida de evaluación. El
  reranking corre una vez por par (consulta, chunk), así que el tamaño del
  modelo es el factor de latencia dominante del pipeline. Números medidos:
  `python evals/bench_rerank.py` → `evals/results/rerank_latency.csv`.
- **Los modelos se precargan al arrancar Streamlit**, no en la primera
  consulta. Es por la exposición: la carga de pesos desde disco tardaba ~46s
  y se la comía quien preguntara primero.
- **Corpus real**: los ~35 documentos van en `data/raw/`. `ingest.py` ya
  soporta `.pdf` con texto nativo (PyMuPDF); los 4 escaneados necesitan
  OCR con Tesseract, que todavía no está implementado.
