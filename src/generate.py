"""
Generación.

Todo el contacto con el LLM vive en este archivo. Si en algún momento hay que
cambiar de proveedor, se cambia solo generate_answer() — retrieval.py, app.py
y los evals no se enteran.
"""
import os

from dotenv import load_dotenv
from openai import OpenAI

from chunk import Chunk

load_dotenv()

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# Prompt base. Es el mismo en las dos ramas de la comparación: lo único que
# cambia entre el "antes" y el "después" es si se le suma o no la advertencia
# de versión de abajo.
SYSTEM_PROMPT_BASE = (
    "Sos un asistente que responde consultas sobre documentación jurídica "
    "(leyes, fallos, contratos, reglamentos). Respondé únicamente en base al "
    "contexto provisto. Si el contexto no alcanza para responder, decilo "
    "explícitamente en vez de inventar. Citá la fuente de cada afirmación "
    "usando entre corchetes el nombre del documento tal como aparece en el "
    "contexto, con su número de ley y su versión."
)

# Esta instrucción existe por un caso real del corpus: la Ley de Contrato de
# Trabajo está indexada en su TEXTO ORIGINAL de 1974, cuya numeración está
# corrida respecto del texto ordenado vigente. El chunk recuperado es correcto
# y la cita también; lo que engaña es que el lector asume que está leyendo el
# régimen actual.
ADVERTENCIA_VERSION = (
    " Si el contexto identifica al documento como 'texto original' y la "
    "consulta parece referirse al régimen vigente, advertilo explícitamente."
)

# Interruptor de la advertencia. Está para poder mostrar el ANTES y el DESPUÉS
# de la misma consulta: con la advertencia apagada, el modelo responde el
# art. 208 de la LCT del texto de 1974 sin pestañear, que es exactamente el
# comportamiento que queremos exhibir como problema antes de mostrar el
# arreglo. Apagarlo NO desarma los otros dos cambios —la versión sigue estando
# en el encabezado de cada chunk y en la cita—, así que aísla el efecto del
# prompt del efecto de los metadatos, que es lo que hace honesta la
# comparación.
ADVERTIR_VERSION = os.environ.get("ADVERTIR_VERSION", "1").lower() not in ("0", "false", "no")


def system_prompt(advertir_version: bool | None = None) -> str:
    """Arma el system prompt. None = usar el valor de ADVERTIR_VERSION."""
    if advertir_version is None:
        advertir_version = ADVERTIR_VERSION
    return SYSTEM_PROMPT_BASE + (ADVERTENCIA_VERSION if advertir_version else "")

_client = None


def _get_client() -> OpenAI:
    # OpenAI() toma OPENAI_API_KEY del entorno por sí solo.
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def _build_context(chunks: list[Chunk]) -> str:
    """Arma el contexto citando cada chunk por su nombre legible.

    Antes citaba el nombre de archivo, y las respuestas salían con fuentes del
    estilo [07_infoleg-ministerio-de-economia-y-finanzas-publicas.pdf], que no
    le dice nada a nadie. etiqueta() devuelve "Ley 11.179 - Código Penal de la
    Nación · texto original · Art. 72": además de ser legible, le pasa al LLM
    el número de ley y la versión, que es justo el dato que le faltaba para no
    presentar el art. 72 de la ley 11.723 como si fuera el de la 11.179.
    """
    return "\n\n".join(f"[{c.etiqueta()}]\n{c.text}" for c in chunks)


def generate_answer(
    query: str,
    chunks: list[Chunk],
    advertir_version: bool | None = None,
) -> str:
    """Genera una respuesta citando las fuentes recuperadas. Si no hay
    contexto, no llama al LLM: no tiene sentido pedirle que responda sin
    nada, y evita gastar una llamada de API en vano.

    advertir_version=False apaga la advertencia de versión del system prompt,
    para poder mostrar el antes y el después de la misma consulta sin tocar
    nada más del pipeline. None toma el valor de ADVERTIR_VERSION.
    """
    if not chunks:
        return "No se encontró contexto relevante en el corpus para responder esta consulta."

    context = _build_context(chunks)
    client = _get_client()

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt(advertir_version)},
            {"role": "user", "content": f"Contexto:\n{context}\n\nConsulta: {query}"},
        ],
        temperature=0,
    )
    return response.choices[0].message.content
