# -*- coding: utf-8 -*-
"""
Tests del tokenizador de BM25.

Sin pytest a propósito: el proyecto es una demo de once días y no quiero sumar
una dependencia para cinco asserts. Se corre directo:

    .venv/Scripts/python.exe tests/test_bm25.py

El test que importa es el primero. Los demás están para que el arreglo del
separador de miles no se lleve puesto nada por el camino.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import bm25  # noqa: E402

fallos = []


def check(descripcion: str, condicion: bool, detalle: str = "") -> None:
    estado = "OK  " if condicion else "FALLA"
    print(f"  [{estado}] {descripcion}" + (f"  -> {detalle}" if detalle and not condicion else ""))
    if not condicion:
        fallos.append(descripcion)


print("\nSeparador de miles — el caso que motivó el cambio")

# El corpus escribe "Ley N° 11.179"; la gente consulta "ley 11179".
tokens_doc = bm25.tokenize("LEY N° 11.179 CODIGO PENAL DE LA NACION")
tokens_consulta = bm25.tokenize("Que dice el articulo 72 de la ley 11179?")
check(
    "la consulta 'ley 11179' y el texto 'Ley N° 11.179' comparten término",
    set(tokens_doc) & set(tokens_consulta) >= {"ley", "11179"},
    f"doc={tokens_doc} consulta={tokens_consulta}",
)

# Lo mismo pero en term_ids, que es lo que efectivamente viaja a Qdrant:
# compartir el string no alcanza si el hash difiere.
ids_doc = set(bm25.document_weights(tokens_doc, avg_doc_len=len(tokens_doc)))
ids_consulta = set(bm25.query_weights("ley 11179"))
check(
    "los term_ids se cruzan (es lo que compara Qdrant, no los strings)",
    ids_doc & ids_consulta,
    f"interseccion={ids_doc & ids_consulta}",
)

check("'11.179' -> '11179'", bm25.tokenize("11.179") == ["11179"])
check("'24.240' -> '24240'", bm25.tokenize("Ley 24.240") == ["ley", "24240"])
check("espacio duro tambien colapsa", bm25.tokenize("11\u00a0179") == ["11179"])

print("\nLo que NO se debe tocar")

# "Art. 39" tiene un espacio: no es separador de miles.
check("'Art. 39' sigue siendo dos términos", bm25.tokenize("Art. 39") == ["art", "39"])
# Dos dígitos después del punto: no es separador de miles.
check("'2.75' no colapsa", bm25.tokenize("2.75") == ["75"], str(bm25.tokenize("2.75")))
# Cuatro dígitos después del punto: tampoco.
check("'1.2345' no colapsa", "12345" not in bm25.tokenize("1.2345"))
# Fin de oración seguido de un año al principio de la próxima.
check(
    "'... del año. 1994 fue' no fusiona el punto final con el año",
    # "del" cae por stopword, no por el separador.
    bm25.tokenize("del ano. 1994 fue") == ["ano", "1994", "fue"],
    str(bm25.tokenize("del ano. 1994 fue")),
)

print("\nNormalización que ya existía (regresión)")
check("saca acentos", bm25.tokenize("jerárquico") == bm25.tokenize("JERARQUICO"))
check("saca stopwords", "de" not in bm25.tokenize("el plazo de la ley"))
check("descarta términos de un carácter", bm25.tokenize("a b ab") == ["ab"])

print()
if fallos:
    print(f"{len(fallos)} test(s) fallaron:")
    for f in fallos:
        print("  -", f)
    sys.exit(1)
print("Todo OK.")
