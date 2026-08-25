"""
Generación.

La llamada al LLM está detrás de una interfaz mínima (generate_answer) para
poder cambiar de proveedor cambiando solo variables de entorno, sin tocar
retrieval.py ni app.py. Se usa el cliente de OpenAI porque su API la exponen
también Groq, Together, vLLM y Ollama entre otros — no ata la demo a un
proveedor específico.
"""
import os

from dotenv import load_dotenv
from openai import OpenAI

from retrieval import Chunk

load_dotenv()

LLM_API_KEY = os.environ.get("LLM_API_KEY")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = (
    "Sos un asistente que responde consultas sobre documentación jurídica "
    "(leyes, fallos, contratos, reglamentos). Respondé únicamente en base al "
    "contexto provisto. Si el contexto no alcanza para responder, decilo "
    "explícitamente en vez de inventar. Citá la fuente de cada afirmación "
    "usando el nombre de documento entre corchetes, por ejemplo [ley_x.txt]."
)

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
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
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Contexto:\n{context}\n\nConsulta: {query}"},
        ],
        temperature=0,
    )
    return response.choices[0].message.content
