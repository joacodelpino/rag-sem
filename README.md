# RAG jurídico — demo (Seminario de Actualización Tecnológica)

Demo académica de RAG vectorial y RAG de segunda generación (híbrido +
reranking) sobre un corpus de documentación jurídica, con **Qdrant** como
base vectorial y **Ragas** como framework de evaluación.

Estado actual: **walking skeleton** — solo la ruta naive (búsqueda vectorial
densa) está implementada, de punta a punta, con 4 documentos de ejemplo.
Las rutas híbrida y híbrida+reranking, y el módulo de evaluación con Ragas,
se agregan en los próximos pasos.

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
│   ├── ingest.py             # parsing + chunking + indexado (ruta naive)
│   ├── retrieval.py          # retrieve_naive() — híbrido y rerank van acá
│   ├── generate.py           # prompt + llamada a OpenAI (único punto de contacto con el LLM)
│   └── app.py                # Streamlit (1 columna por ahora, 3 cuando existan las 3 rutas)
├── evals/
│   └── results/              # CSV de cada corrida de Ragas (a implementar)
└── notebooks/                # exploración, no se presenta
```

## Notas

- **Qdrant en Docker, todo lo demás local.** Así el modelo de embeddings se
  cachea una vez en el host y no hay que rebuildear una imagen por cada cambio
  de código. El dashboard de Qdrant queda en http://localhost:6333/dashboard.
- **`ingest.py` recrea la colección en cada corrida**, a propósito: cada
  ingesta parte de un estado limpio y reproducible.
- **Corpus real**: los ~35 documentos van en `data/raw/`. `ingest.py` ya
  soporta `.pdf` con texto nativo (PyMuPDF); los 4 escaneados necesitan
  OCR con Tesseract, que todavía no está implementado.
