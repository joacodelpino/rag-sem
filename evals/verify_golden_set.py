# -*- coding: utf-8 -*-
"""
Audita data/golden_set.csv contra el índice de Qdrant.

Es DELIBERADAMENTE independiente de build_golden_set.py: lee el CSV ya escrito
y los chunks del índice, y no importa nada del script que generó el set. Si
usara las mismas frases de `verificar` que usó el generador, estaría
comprobando que el generador hace lo que dice, no que el ground truth sea
cierto.

Qué chequea, por categoría:

  todas        la respuesta_esperada está sostenida por el texto de los chunks
               citados (números y palabras clave, con tolerancia a que el
               documento escriba "DIEZ (10)" y la respuesta "10").
  multihop     además, ningún chunk por separado alcanza para responder.
  fuente       los chunks apuntan SOLO al documento fuente, y se listan los
               otros documentos del corpus que mencionan el mismo caso — que
               son justamente los que NO deben estar.
  negativa     los términos de la PREGUNTA no aparecen en ningún chunk del
               corpus. Se buscan varias variantes por pregunta, no una sola.

El veredicto automático es una ayuda, no la prueba: para cada fila se imprime
además el fragmento literal del documento, que es lo que hay que leer.

Uso:
    .venv/Scripts/python.exe evals/verify_golden_set.py
    .venv/Scripts/python.exe evals/verify_golden_set.py --detalle
"""
import csv
import re
import sys
import unicodedata
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from qdrant_client import QdrantClient  # noqa: E402

CSV_PATH = RAIZ / "data" / "golden_set.csv"
QDRANT_URL = "http://localhost:6333"
COLLECTION = "legal_docs"

# Los documentos escriben los números con letras y con dígitos, y no siempre
# los dos: "durante treinta años más" no tiene ningún dígito. Sin este mapa la
# verificación rechazaría respuestas correctas.
NUMEROS = {
    "un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
    "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10, "once": 11,
    "doce": 12, "trece": 13, "catorce": 14, "quince": 15, "dieciseis": 16,
    "diecisiete": 17, "dieciocho": 18, "diecinueve": 19, "veinte": 20,
    "veintiun": 21, "veintiuno": 21, "veinticinco": 25, "veintiocho": 28,
    "treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60,
    "setenta": 70, "ochenta": 80, "noventa": 90, "cien": 100,
}

VACIAS = {
    "para", "por", "con", "los", "las", "del", "que", "una", "uno", "sus",
    "este", "esta", "esos", "como", "mas", "pero", "sin", "sobre", "entre",
    "cuando", "donde", "segun", "hay", "son", "ser", "esta", "estan", "tiene",
    "tienen", "puede", "pueden", "debe", "deben", "y", "o", "el", "la", "de",
    "en", "a", "al", "se", "no", "es", "un", "su", "lo", "le", "corresponde",
    "corresponden", "otros", "demas", "que", "cada", "todo", "toda",
}

# Términos a buscar en TODO el corpus para cada negativa. Salen de la pregunta,
# no del título de ningún documento: incluyen el número de ley, el nombre
# corriente de la norma y el concepto que la pregunta menciona.
TERMINOS_NEGATIVAS = {
    "n01": [r"27\.?551", r"ley de alquiler", r"locaciones urbanas",
            r"plazo m[ií]nimo de.{0,20}locaci[oó]n", r"alquileres"],
    "n02": [r"teletrabajo", r"tele-?trabajo", r"27\.?555", r"desconexi[oó]n",
            r"trabajo remoto", r"home ?office"],
    "n03": [r"g[oó]ndola", r"27\.?545", r"supermercado", r"g[oó]ndolas"],
    "n04": [r"\b500\b", r"art[íi]?c?u?l?o?\.? ?500\b"],
}

# Para las de categoría fuente: qué documento DEBE ser la fuente, y con qué
# patrón se buscan los documentos que solo lo mencionan.
FUENTES = {
    "s01": ("12_siri-angel-1957.pdf", r"\bSiri\b"),
    "s02": ("13_samuel-kot-1958.pdf", r"\bKot\b"),
    "s03": ("07_infoleg-ministerio-de-economía-y-finanzas-públicas.pdf", r"C[oó]digo Penal"),
    "s04": ("01_argentinagobar.pdf", r"24\.240"),
}


def normalizar(t: str) -> str:
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", t)


def claves(respuesta: str) -> tuple[set[int], set[str]]:
    """Saca de la respuesta esperada los números y las palabras de contenido."""
    n = normalizar(respuesta)
    nums = {int(x) for x in re.findall(r"\d+", n)}
    palabras = set()
    for w in re.findall(r"[a-z]{4,}", n):
        if w in NUMEROS:
            nums.add(NUMEROS[w])
        elif w not in VACIAS:
            palabras.add(w)
    return nums, palabras


def numero_presente(n: int, texto_norm: str) -> bool:
    """El número puede estar como dígito o escrito con letras."""
    if re.search(rf"\b{n}\b", texto_norm):
        return True
    return any(re.search(rf"\b{p}\b", texto_norm) for p, v in NUMEROS.items() if v == n)


def palabra_presente(w: str, texto_norm: str) -> bool:
    """Match por raíz: tolera plural y género ('asignatura'/'asignaturas')."""
    return re.search(rf"\b{re.escape(w[:6])}", texto_norm) is not None


def evaluar(respuesta: str, texto: str) -> tuple[bool, list[str], float]:
    # La marca de continuación mete dígitos en el chunk que no son del
    # documento: el chunk 61 se llama "Art. 45 (cont. 4/5)" y ese "5" hacía que
    # la verificación creyera que el chunk contenía el plazo de cinco días.
    # Falso positivo del formato de encabezado, no del set dorado.
    tn = normalizar(re.sub(r"\(cont\. \d+/\d+\)", "", texto))
    nums, palabras = claves(respuesta)
    faltan = [str(n) for n in sorted(nums) if not numero_presente(n, tn)]
    hallada = [w for w in palabras if palabra_presente(w, tn)]
    cobertura = len(hallada) / len(palabras) if palabras else 1.0
    faltan += sorted(w for w in palabras if w not in hallada)
    ok = not [f for f in faltan if f.isdigit()] and cobertura >= 0.6
    return ok, faltan, cobertura


def evidencia(respuesta: str, texto: str, ancho: int = 150) -> str:
    """Fragmento del documento alrededor del primer número de la respuesta."""
    tn = re.sub(r"\s+", " ", texto)
    nums, palabras = claves(respuesta)
    anclas = [str(n) for n in sorted(nums)] + sorted(palabras, key=len, reverse=True)[:2]
    for a in anclas:
        m = re.search(rf"\b{re.escape(a)}", normalizar(tn))
        if m:
            i = m.start()
            return "..." + tn[max(0, i - 60):i + ancho] + "..."
    return tn[:ancho] + "..."


def main() -> None:
    detalle = "--detalle" in sys.argv

    c = QdrantClient(url=QDRANT_URL)
    chunks, offset = {}, None
    while True:
        pts, offset = c.scroll(collection_name=COLLECTION, limit=512,
                               offset=offset, with_payload=True, with_vectors=False)
        for p in pts:
            chunks[p.id] = p.payload
        if offset is None:
            break

    filas = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
    print(f"CSV: {len(filas)} filas   |   índice: {len(chunks)} chunks\n")

    problemas = []

    # ---------------- tabla principal ----------------
    print("=" * 100)
    print("VERIFICACIÓN CHUNK POR CHUNK")
    print("=" * 100)
    print(f"{'preg':<5} {'cat':<14} {'chunk':>6}  {'documento':<26} {'sección':<22} {'¿respuesta?':<12}")
    print("-" * 100)

    for f in filas:
        ids = [int(x) for x in f["chunks_relevantes"].split(";") if x]
        if not ids:
            print(f"{f['id']:<5} {f['categoria']:<14} {'—':>6}  {'(sin chunks: es negativa)':<50}")
            continue

        juntos = " ".join(chunks[i]["text"] for i in ids if i in chunks)
        ok_total, faltan_total, _ = evaluar(f["respuesta_esperada"], juntos)

        for n, i in enumerate(ids):
            if i not in chunks:
                print(f"{f['id']:<5} {f['categoria']:<14} {i:>6}  *** ID INEXISTENTE ***")
                problemas.append(f"{f['id']}: el chunk {i} no existe en el índice")
                continue
            p = chunks[i]
            ok_solo, _, cob = evaluar(f["respuesta_esperada"], p["text"])
            if len(ids) == 1:
                marca = "SI" if ok_total else "NO  <-- REVISAR"
            else:
                marca = f"parcial {cob:.0%}" + ("  (alcanza solo)" if ok_solo else "")
            print(f"{f['id'] if n == 0 else '':<5} {f['categoria'] if n == 0 else '':<14} "
                  f"{i:>6}  {p['source'][:25]:<26} {p['seccion'][:21]:<22} {marca:<12}")

        if len(ids) > 1:
            print(f"{'':<5} {'':<14} {'juntos':>6}  {'':<26} {'':<22} "
                  f"{'SI' if ok_total else 'NO  <-- REVISAR'}")
        if not ok_total:
            problemas.append(f"{f['id']}: falta en el texto -> {faltan_total}")
        if detalle:
            print(f"      esperado : {f['respuesta_esperada']}")
            print(f"      documento: {evidencia(f['respuesta_esperada'], juntos)}\n")
        print("-" * 100)

    # ---------------- multihop ----------------
    print("\n" + "=" * 100)
    print("MULTIHOP — ¿alcanza algún chunk por separado?")
    print("=" * 100)
    for f in filas:
        if f["categoria"] != "multihop":
            continue
        ids = [int(x) for x in f["chunks_relevantes"].split(";") if x]
        print(f"\n{f['id']}  {f['pregunta']}")
        solos = []
        for i in ids:
            _, faltan, cob = evaluar(f["respuesta_esperada"], chunks[i]["text"])
            # Umbral más exigente que el de la tabla principal (0.6). Decir "este
            # chunk solo ya responde" es una afirmación fuerte y descarta una
            # fila del set: con 0.6 daba falso positivo en el art. 34, que fija
            # el plazo pero remite a los arts. 32 y 33 sin repetir qué ventas
            # son, así que no responde la pregunta completa.
            ok = not [x for x in faltan if x.isdigit()] and cob >= 0.9
            print(f"     chunk {i:<5} [{chunks[i]['seccion'][:20]:<20}] cobertura {cob:>4.0%}"
                  f"  {'ALCANZA SOLO <-- NO ES MULTIHOP' if ok else 'incompleto (falta ' + ', '.join(faltan[:3]) + ')'}")
            if ok:
                solos.append(i)
        if solos:
            problemas.append(f"{f['id']}: no es multihop, los chunks {solos} alcanzan solos")
        if len(ids) < 2:
            problemas.append(f"{f['id']}: multihop con un solo chunk")

    # ---------------- fuente ----------------
    print("\n" + "=" * 100)
    print("FUENTE — ¿los chunks apuntan solo al documento original?")
    print("=" * 100)
    for f in filas:
        if f["categoria"] != "fuente":
            continue
        ids = [int(x) for x in f["chunks_relevantes"].split(";") if x]
        esperado, patron = FUENTES[f["id"]]
        pat = re.compile(patron, re.I)
        citantes = {}
        for i, p in chunks.items():
            if pat.search(p["text"]):
                citantes[p["source"]] = citantes.get(p["source"], 0) + 1
        apuntados = {chunks[i]["source"] for i in ids}
        bien = apuntados == {esperado}
        print(f"\n{f['id']}  {f['pregunta']}")
        print(f"     fuente esperada : {esperado}")
        print(f"     chunks apuntan a: {', '.join(sorted(apuntados))}   {'OK' if bien else '<-- MAL'}")
        print(f"     otros documentos que mencionan el tema (NO deben estar en chunks_relevantes):")
        for s, n in sorted(citantes.items(), key=lambda x: -x[1]):
            marca = "  <-- ES LA FUENTE" if s == esperado else ""
            print(f"         {n:>4} menciones  {s[:56]}{marca}")
        if not bien:
            problemas.append(f"{f['id']}: apunta a {apuntados}, debería apuntar solo a {esperado}")
        if len(citantes) < 2:
            problemas.append(
                f"{f['id']}: solo {len(citantes)} documento menciona el tema — no hay competencia, "
                f"la categoría 'fuente' no prueba nada acá")

    # ---------------- negativas ----------------
    print("\n" + "=" * 100)
    print("NEGATIVAS — ¿el tema aparece en algún lado del corpus?")
    print("=" * 100)
    for f in filas:
        if f["categoria"] != "negativa":
            continue
        print(f"\n{f['id']}  {f['pregunta']}")
        for patron in TERMINOS_NEGATIVAS[f["id"]]:
            pat = re.compile(patron, re.I)
            hits = [(i, p) for i, p in chunks.items() if pat.search(normalizar(p["text"]))]
            if not hits:
                print(f"     {patron:<36} 0 chunks   ok")
            else:
                docs = sorted({p['source'][:34] for _, p in hits})
                print(f"     {patron:<36} {len(hits)} chunks   <-- APARECE en {', '.join(docs[:3])}")
                i, p = hits[0]
                m = pat.search(normalizar(p["text"]))
                frag = re.sub(r"\s+", " ", p["text"])[max(0, m.start() - 70):m.start() + 130]
                print(f"         ej. id={i}: ...{frag}...")
                problemas.append(f"{f['id']}: '{patron}' aparece en {len(hits)} chunks")

    # ---------------- resumen ----------------
    print("\n" + "=" * 100)
    if problemas:
        print(f"{len(problemas)} PROBLEMA(S) — ninguno corregido, hay que decidir a mano:")
        for p in problemas:
            print("   -", p)
    else:
        print("Todas las filas con chunks verifican, y las negativas no aparecen en el corpus.")
    print("=" * 100)


if __name__ == "__main__":
    main()
