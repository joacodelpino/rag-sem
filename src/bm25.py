"""
BM25 sparse — la mitad léxica de la búsqueda híbrida.

Por qué implementarlo a mano en vez de usar una librería: el objetivo del
trabajo es poder explicar cada pieza del pipeline, y BM25 son treinta líneas.
Además así queda claro el reparto de tareas con Qdrant, que es lo que la hace
funcionar bien:

  - ACÁ (cliente) calculamos el peso TF de cada término en cada documento —
    la parte de la fórmula que solo depende del documento.
  - QDRANT calcula el IDF, porque es la parte que depende de TODA la colección
    (en cuántos documentos aparece el término). El cliente no tiene esa
    estadística sin recorrer el índice entero; el servidor sí, y la mantiene
    actualizada. Se activa con Modifier.IDF al crear la colección.

Score final que arma Qdrant, por término en común entre consulta y documento:

    idf(t) * peso_consulta(t) * peso_documento(t)

que es exactamente BM25 cuando peso_documento es la saturación de TF de abajo.
"""
import re
import unicodedata
import zlib

# Parámetros estándar de BM25.
#   k1 controla la saturación: que un término aparezca 20 veces en vez de 10 no
#      debe duplicar el score (la segunda mención aporta menos que la primera).
#   b  controla la normalización por longitud: sin esto, los documentos largos
#      ganan siempre por acumulación de menciones.
K1 = 1.5
B = 0.75

# Stopwords mínimas del español. No es una lista exhaustiva a propósito: solo
# saca las palabras que aparecen en casi todos los documentos jurídicos y por
# lo tanto no discriminan nada. Ojo que el IDF ya castiga fuerte a estas
# palabras por sí solo — la lista es más que nada para achicar los vectores.
STOPWORDS = {
    "a", "al", "ante", "con", "contra", "de", "del", "desde", "e", "el", "en",
    "entre", "es", "esta", "este", "ha", "hasta", "la", "las", "lo", "los",
    "mas", "me", "mi", "no", "o", "para", "pero", "por", "que", "se", "sin",
    "sobre", "son", "su", "sus", "tras", "un", "una", "uno", "y", "ya",
}


# Separador de miles adentro de un número. Los documentos escriben "Ley N°
# 11.179" y la gente consulta "ley 11179": sin esto el tokenizador produce
# ["11", "179"] contra ["11179"], que son term_ids distintos, y BM25 no puede
# matchear NUNCA un número de ley — justo el caso donde la rama léxica tendría
# que ser imbatible. Fue un bug real, medido sobre la consulta del art. 72.
#
# El lookahead exige exactamente tres dígitos que no sigan con más dígitos,
# que es la forma del separador de miles en castellano. Así "11.179" colapsa a
# "11179" pero "art. 39" no se toca (hay un espacio) ni "2.5" (no son tres
# dígitos). No incluye el espacio común como separador a propósito: colapsaría
# números vecinos que no tienen nada que ver.
SEPARADOR_MILES = re.compile(r"(?<=\d)[.  ](?=\d{3}(?!\d))")


def tokenize(text: str) -> list[str]:
    """Texto -> lista de términos normalizados.

    Saca acentos y mayúsculas para que "jerárquico" y "JERARQUICO" sean el
    mismo término: en las consultas de la demo la gente casi nunca acentúa, y
    sin normalizar el match léxico fallaría justo donde debería brillar.

    Y normaliza los separadores de miles, para que el número de una ley sea un
    solo término se escriba como se escriba (ver SEPARADOR_MILES).
    """
    # Antes del NFKD: esa normalización convierte el espacio duro en un espacio
    # común y el patrón dejaría de reconocerlo.
    text = SEPARADOR_MILES.sub("", text)
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    tokens = re.findall(r"[a-z0-9]+", text)
    # Términos de un solo carácter no aportan y ensucian el vector.
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]


def _term_id(term: str) -> int:
    """Término -> entero, que es lo que Qdrant usa como índice del vector sparse.

    Usamos CRC32 y no hash() de Python porque hash() de strings está
    randomizado por proceso: los ids de la ingesta no coincidirían con los de
    la consulta y no matchearía nada.
    """
    return zlib.crc32(term.encode("utf-8")) & 0x7FFFFFFF


def document_weights(tokens: list[str], avg_doc_len: float) -> dict[int, float]:
    """Vector sparse de un documento: {term_id: peso BM25 de TF}.

    Falta el IDF a propósito — lo aplica Qdrant en tiempo de consulta.
    """
    doc_len = len(tokens)
    if doc_len == 0:
        return {}

    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1

    # Saturación de TF normalizada por longitud del documento.
    norm = K1 * (1 - B + B * doc_len / avg_doc_len)
    return {
        _term_id(term): tf * (K1 + 1) / (tf + norm)
        for term, tf in counts.items()
    }


def query_weights(query: str) -> dict[int, float]:
    """Vector sparse de una consulta: {term_id: 1.0}.

    Peso plano porque las consultas son cortas y casi nunca repiten términos;
    la discriminación entre términos la pone el IDF del lado de Qdrant.
    """
    return {_term_id(term): 1.0 for term in set(tokenize(query))}
