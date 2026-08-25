"""
Generación.

Todo el contacto con el LLM vive en este archivo. Si en algún momento hay que
cambiar de proveedor, se cambia solo generate_answer() — retrieval.py, app.py
y los evals no se enteran.
"""
import os

from dotenv import load_dotenv
from openai import OpenAI

from retrieval import Chunk

load_dotenv()

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = (
    "Sos un asistente que responde consultas sobre documentación jurídica "
    "(leyes, fallos, contratos, reglamentos). Respondé únicamente en base al "
    "contexto provisto. Si el contexto no alcanza para responder, decilo "
    "explícitamente en vez de inventar. Citá la fuente de cada afirmación "
    "usando el nombre de documento entre corchetes, por ejemplo [ley_x.txt]."
)

_client = None


def _get_client() -> OpenAI:
    # OpenAI() toma OPENAI_API_KEY del entorno por sí solo.
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def _build_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{c.source}]\n{c.text}" for c in chunks)


def generate_answer(query: str, chunks: list[Chunk]) -> str:
    """Genera una respuesta citando las fuentes recuperadas. Si no hay
    contexto, no llama al LLM: no tiene sentido pedirle que responda sin
    nada, y evita gastar una llamada de API en vano."""
    if not chunks:
        return "No se encontró contexto relevante en el corpus para responder esta consulta."

    context = _build_context(chunks)
    client = _get_client()

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Contexto:\n{context}\n\nConsulta: {query}"},
        ],
        temperature=0,
    )
    return response.choices[0].message.content
