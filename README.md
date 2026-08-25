# RAG jurídico - demo (Seminario de Actualización Tecnológica)

Demo académica de RAG vectorial y RAG de segunda generación (híbrido +
reranking) sobre un corpus de documentación jurídica, con Qdrant como base
vectorial y Ragas como framework de evaluación.

Estado actual: **walking skeleton** — solo la ruta naive (búsqueda vectorial
densa) está implementada, de punta a punta, con 4 documentos de ejemplo.
Las rutas híbrida y híbrida+reranking, y el módulo de evaluación con Ragas,
se agregan en los próximos pasos.

## Cómo levantarlo (3 comandos)

1. Copiá `.env.example` a `.env` y completá `LLM_API_KEY` con tu clave de
   OpenAI (o la de cualquier proveedor compatible con su API, cambiando
   también `LLM_BASE_URL` y `LLM_MODEL`).

   ```
   cp .env.example .env
   ```

2. Levantá Qdrant y la app:

   ```
   docker compose up -d --build
   ```

3. Indexá los documentos de ejemplo (primera vez que corrés esto va a
   descargar el modelo de embeddings BGE-M3, tarda unos minutos):

   ```
   docker compose exec app python src/ingest.py
   ```

Abrí http://localhost:8501 y hacé una consulta, por ejemplo:
*"¿Cuántos días hábiles hay para presentar el recurso jerárquico?"*

## Estructura

```
rag-legal-demo/
├── docker-compose.yml
├── .env.example
├── requirements.txt
├── data/
│   ├── raw/                # documentos de ejemplo (4 .txt jurídicos)
│   └── golden_set.csv       # set dorado de preguntas para Ragas (a completar)
├── src/
│   ├── ingest.py            # parsing + chunking + indexado (ruta naive)
│   ├── retrieval.py         # retrieve_naive() — híbrido y rerank van acá después
│   ├── generate.py          # prompt + llamada al LLM (proveedor intercambiable)
│   └── app.py                # Streamlit
├── evals/
│   └── results/              # CSV de cada corrida de Ragas (a implementar)
└── notebooks/                 # exploración, no se presenta
```

## Corpus real

`data/raw/` tiene 4 documentos de ejemplo cortos para probar la cadena
completa. El corpus real (~35 documentos, incluidos los escaneados que
necesitan OCR) se agrega en el mismo directorio antes de la exposición —
`ingest.py` ya soporta `.pdf` además de `.txt`.
