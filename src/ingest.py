# -*- coding: utf-8 -*-
"""
Ingesta — parsing, chunking e indexado en Qdrant.

Corre una sola vez, offline: no interviene durante la consulta.

Cada chunk se sube con DOS vectores en el mismo punto de la colección:

  - "dense": embedding de BGE-M3, captura significado (encuentra "plazo para
    apelar" aunque el documento diga "término para recurrir").
  - "bm25":  vector sparse léxico, captura coincidencia exacta de términos
    (encuentra "artículo 34" o "UVA", donde el denso se diluye).

Las tres configuraciones de recuperación (naive, híbrida, híbrida+rerank) leen
de esta misma colección. No hay que reingestar para cambiar de configuración:
eso es justamente lo que hace comparable la tabla de ablación.

DOS DECISIONES QUE SÍ REQUIEREN REINGESTAR, y por qué existen:

1. ENCABEZADO DE IDENTIDAD. Cada chunk arranca con una línea que dice de qué
   documento y de qué versión sale:

       [Ley 11.179 - Código Penal de la Nación · texto original · Art. 72]
       Art. 72. - Son acciones dependientes de instancia privada...

   Sin esto, el chunk del art. 72 del Código Penal no contiene en ningún lado
   la cadena "11.179" —vive en la carátula, en otro chunk— así que ni el denso
   ni BM25 pueden conectarlo con la consulta "artículo 72 de la ley 11179".
   Medido: el pipeline devolvía el art. 72 de la ley 11.723 y el LLM lo citaba
   como si fuera el de la 11.179.

   La versión va en el encabezado y no solo en el payload a propósito: es el
   dato que evita que alguien lea una respuesta sacada del texto de la LCT de
   1974 creyendo que es el régimen vigente.

2. CHUNKING POR ESTRUCTURA. Cortar cada 800 caracteres parte artículos por la
   mitad y mezcla el final de uno con el principio del siguiente. Ver
   chunk_by_structure().

El chunking anterior queda disponible con CHUNK_STRATEGY=fijo, para poder
correr la fila "chunking fijo vs. por artículo" de la ablación sin volver a
tocar el código.
"""
import csv
import os
import re
from pathlib import Path

import pymupdf
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

import bm25

load_dotenv()

RAIZ = Path(__file__).resolve().parent.parent
DATA_DIR = RAIZ / "data" / "raw"
MANIFEST_PATH = RAIZ / "data" / "manifest.csv"
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "legal_docs")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")

# "articulo" corta por límites de artículo / voto; "fijo" es el chunking por
# ventanas de caracteres del walking skeleton. Se deja accesible para poder
# comparar las dos estrategias en la tabla de ablación.
CHUNK_STRATEGY = os.environ.get("CHUNK_STRATEGY", "articulo")

# Ventana del chunking fijo.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Topes del chunking por estructura. Los artículos varían muchísimo de largo:
# hay artículos de una línea ("Art. 5 — Derogado") y artículos con quince
# incisos.
#   MAX: por encima de esto el artículo se parte en ventanas, como antes, pero
#        cada pedazo conserva el encabezado y una marca de continuación.
#   MIN: por debajo de esto el pedazo se pega al anterior. Sirve para dos cosas
#        distintas: evitar miles de chunks de una línea, y absorber los cortes
#        falsos (el regex también matchea una REFERENCIA a un artículo que
#        arranca renglón, no solo el artículo en sí — medido: el Código Penal
#        da 306 cortes para 285 artículos reales).
CHUNK_MAX_CHARS = 1200
CHUNK_MIN_CHARS = 250

# Cuántos puntos por llamada a Qdrant. Ver el comentario en main(): existe
# para no pasarse del límite de 32 MB por request.
UPSERT_BATCH = 256

# Encabezado de artículo al principio de un renglón. Cubre las variantes que
# aparecen en el corpus: "ARTICULO 1°", "ARTICULO 1º", "Artículo 1.-",
# "Art. 2.—", "Art 3", con o sin "bis"/"ter".
#
# El cierre es (?!\d) y NO \b, que fue un bug real: en "ARTICULO 1º" el
# indicador ordinal º (U+00BA) es LETRA para Unicode, así que entre "1" y "º"
# no hay borde de palabra y el match fallaba. Se perdían justo los artículos
# 1 a 9 de cada ley —los que más se consultan— mientras que "ARTICULO 10"
# matcheaba sin problema, con lo cual el error pasaba desapercibido.
PATRON_ARTICULO = re.compile(
    r"^[ \t]*(?:ART[IÍ]CULO|Art[íi]culo|Art)\s*\.?\s*N?[°ºo]?\s*(\d+)\s*(bis|ter|quater)?(?!\d)",
    re.MULTILINE,
)

# Secciones de un fallo, que no tiene artículos. Cada voto es una unidad de
# sentido: mezclar el final del dictamen con el principio de la disidencia
# produce un chunk que dice dos cosas opuestas.
PATRON_VOTO = re.compile(
    r"^[ \t]*((?:Dictamen del Procurador|Voto del|Voto de los|Voto en disidencia|"
    r"Disidencia|Considerando|Y VISTOS|Por ello)[^\n]{0,70})",
    re.MULTILINE | re.IGNORECASE,
)

VERSION_LEGIBLE = {
    "texto_original": "texto original",
    "texto_actualizado": "texto actualizado",
    "texto_ordenado": "texto ordenado",
}


# --------------------------------------------------------------------------
# Manifiesto
# --------------------------------------------------------------------------

def load_manifest() -> dict[str, dict]:
    """Lee data/manifest.csv indexado por nombre de archivo.

    Lo genera evals/build_manifest.py. Se lee acá y no se re-deriva de los PDFs
    porque los títulos están curados a mano: el nombre de archivo es inservible
    (cinco documentos se llaman "0N_argentinagobar.pdf" y "08_ley-15.pdf" no es
    la ley 15 sino el Decreto-Ley 15.348).
    """
    if not MANIFEST_PATH.exists():
        raise SystemExit(
            f"Falta {MANIFEST_PATH}.\n"
            f"Generalo con:  python evals/build_manifest.py"
        )
    with MANIFEST_PATH.open(encoding="utf-8") as f:
        return {fila["archivo"]: fila for fila in csv.DictReader(f)}


def validar_cobertura(paths: list[Path], manifiesto: dict[str, dict]) -> None:
    """Falla ANTES de indexar nada si algún documento no está identificado.

    Es a propósito que reviente en vez de indexar con el título vacío: un chunk
    sin identidad es exactamente el bug que estamos arreglando, y descubrirlo
    después de media hora de embeddings —con la colección ya borrada— es el
    peor momento posible.
    """
    faltantes = [p.name for p in paths if p.name not in manifiesto]
    sin_titulo = [
        p.name for p in paths
        if p.name in manifiesto and not manifiesto[p.name].get("titulo", "").strip()
    ]
    indeterminados = [
        p.name for p in paths
        if manifiesto.get(p.name, {}).get("titulo", "").strip() == "indeterminado"
    ]

    problemas = []
    if faltantes:
        problemas.append("No están en el manifiesto:\n  - " + "\n  - ".join(faltantes))
    if sin_titulo:
        problemas.append("Sin título en el manifiesto:\n  - " + "\n  - ".join(sin_titulo))
    if indeterminados:
        problemas.append("Título 'indeterminado':\n  - " + "\n  - ".join(indeterminados))

    if problemas:
        raise SystemExit(
            "\n".join(problemas)
            + f"\n\nActualizá {MANIFEST_PATH.name} (evals/build_manifest.py) y volvé a correr."
        )


def encabezado(meta: dict, seccion: str = "") -> str:
    """Línea de identidad que se prepende a cada chunk antes de embeberlo."""
    partes = [meta["titulo"]]
    legible = VERSION_LEGIBLE.get(meta.get("version", ""))
    if legible:
        partes.append(legible)
    if seccion:
        partes.append(seccion)
    return "[" + " · ".join(partes) + "]"


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

def read_document(path: Path) -> str:
    """Extrae texto plano de un .txt o .pdf (los 36 PDFs del corpus tienen
    texto nativo; los escaneados necesitarían OCR, fuera de alcance)."""
    if path.suffix.lower() == ".pdf":
        with pymupdf.open(path) as doc:
            return "\n".join(page.get_text() for page in doc)
    return path.read_text(encoding="utf-8")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Divide el texto en ventanas solapadas. El solapamiento evita que una
    oración con la respuesta quede cortada exactamente en el borde de dos
    chunks."""
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return [c for c in chunks if c.strip()]


def _cortar_por_patron(text: str, patron: re.Pattern, etiquetar) -> list[tuple[str, str]]:
    """Parte el texto en los matches del patrón. Devuelve (etiqueta, cuerpo).

    El preámbulo anterior al primer match (carátula, visto, considerandos) se
    conserva como su propia pieza: ahí es donde vive el número de ley y la
    fecha, así que tirarlo sería tirar justo el contexto que estamos tratando
    de rescatar.
    """
    matches = list(patron.finditer(text))
    if not matches:
        return []

    piezas = []
    preambulo = text[: matches[0].start()].strip()
    if preambulo:
        # Se etiqueta en vez de dejarla vacía: si esta pieza hay que partirla,
        # el encabezado quedaría diciendo solo "(cont. 2/8)", que no le dice
        # nada a nadie. "Carátula" es de dónde salen el número de ley, la
        # fecha de sanción y las partes del juicio.
        piezas.append(("Carátula", preambulo))

    for i, m in enumerate(matches):
        fin = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        cuerpo = text[m.start():fin].strip()
        if cuerpo:
            piezas.append((etiquetar(m), cuerpo))
    return piezas


def _fusionar_cortos(piezas: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Pega al anterior todo pedazo más corto que CHUNK_MIN_CHARS.

    Ver el comentario de CHUNK_MIN_CHARS: absorbe tanto los artículos de una
    línea como los cortes falsos del regex. La etiqueta que sobrevive es la del
    pedazo que abrió el grupo, que es la que corresponde al contenido principal.
    """
    fusionadas: list[list] = []
    for etiqueta, cuerpo in piezas:
        if fusionadas and len(fusionadas[-1][1]) < CHUNK_MIN_CHARS:
            fusionadas[-1][1] += "\n" + cuerpo
            if not fusionadas[-1][0]:
                fusionadas[-1][0] = etiqueta
        else:
            fusionadas.append([etiqueta, cuerpo])
    return [(e, c) for e, c in fusionadas]


def _partir_largos(piezas: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Aplica el tope de tamaño: lo que pase de CHUNK_MAX_CHARS vuelve a las
    ventanas solapadas de siempre, pero marcado como continuación para que el
    encabezado diga que el artículo sigue en otro chunk."""
    salida = []
    for etiqueta, cuerpo in piezas:
        if len(cuerpo) <= CHUNK_MAX_CHARS:
            salida.append((etiqueta, cuerpo))
            continue
        ventanas = chunk_text(cuerpo, chunk_size=CHUNK_MAX_CHARS, overlap=CHUNK_OVERLAP)
        for n, ventana in enumerate(ventanas, start=1):
            marca = f"cont. {n}/{len(ventanas)}"
            salida.append((f"{etiqueta} ({marca})" if etiqueta else f"({marca})", ventana))
    return salida


def chunk_by_structure(text: str) -> list[tuple[str, str]]:
    """Corta por unidades de sentido en vez de cada N caracteres.

    Prioridad: artículos, y si el documento no tiene (los fallos), votos. Si no
    tiene ninguno de los dos (contratos modelo, el folleto de la SRT) cae al
    chunking fijo, que para textos sin estructura numerada es lo correcto.

    Por qué importa para la recuperación: un artículo ES la unidad de respuesta
    de una consulta jurídica. Cortando cada 800 caracteres, la mitad de los
    chunks empiezan a mitad de un inciso y terminan a mitad del siguiente
    artículo, así que el chunk que "contiene la respuesta" también contiene
    media respuesta a otra pregunta — y el reranker tiene que elegir con esa
    mezcla.
    """
    piezas = _cortar_por_patron(
        text,
        PATRON_ARTICULO,
        lambda m: "Art. " + m.group(1) + (f" {m.group(2)}" if m.group(2) else ""),
    )
    if not piezas:
        piezas = _cortar_por_patron(
            text, PATRON_VOTO, lambda m: " ".join(m.group(1).split())
        )
    if not piezas:
        return [("", c) for c in chunk_text(text)]

    return _partir_largos(_fusionar_cortos(piezas))


def chunk_document(text: str) -> list[tuple[str, str]]:
    """Despacha según CHUNK_STRATEGY. Devuelve (seccion, texto)."""
    if CHUNK_STRATEGY == "fijo":
        return [("", c) for c in chunk_text(text)]
    if CHUNK_STRATEGY == "articulo":
        return chunk_by_structure(text)
    raise SystemExit(
        f"CHUNK_STRATEGY desconocida: {CHUNK_STRATEGY!r} (los valores validos son 'articulo' y 'fijo')"
    )


def load_documents() -> list[dict]:
    """Lee data/raw/, trocea, y prepende a cada chunk su línea de identidad."""
    manifiesto = load_manifest()
    paths = [p for p in sorted(DATA_DIR.glob("*")) if p.suffix.lower() in (".txt", ".pdf")]
    if not paths:
        raise SystemExit(f"No se encontraron documentos .txt/.pdf en {DATA_DIR}")
    validar_cobertura(paths, manifiesto)

    records = []
    for path in paths:
        meta = manifiesto[path.name]
        text = read_document(path)
        for i, (seccion, cuerpo) in enumerate(chunk_document(text)):
            records.append({
                "source": path.name,
                "chunk_index": i,
                "seccion": seccion,
                "titulo": meta["titulo"],
                "numero_ley": meta.get("numero_ley", ""),
                "version": meta.get("version", ""),
                # El encabezado va DENTRO del texto, no solo en el payload:
                # tiene que estar en lo que se embebe y en lo que tokeniza
                # BM25, que es donde hace falta para que matchee.
                "text": encabezado(meta, seccion) + "\n" + cuerpo,
            })
    return records


# --------------------------------------------------------------------------
# Indexado
# --------------------------------------------------------------------------

def build_collection(client: QdrantClient, vector_size: int) -> None:
    """Recrea la colección con los dos vectores nombrados: denso y sparse.

    Se borra y se rehace en cada corrida a propósito: cada ingesta parte de un
    estado limpio y reproducible, sin puntos huérfanos de corridas anteriores.

    Modifier.IDF es la línea clave de la ruta híbrida: le dice a Qdrant que
    mantenga las frecuencias de documento de cada término y aplique el IDF en
    tiempo de consulta. Sin esto, el vector sparse sería solo TF y BM25 quedaría
    a medias.
    """
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "dense": VectorParams(size=vector_size, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "bm25": SparseVectorParams(modifier=Modifier.IDF),
        },
    )


def build_sparse_vectors(records: list[dict]) -> list[SparseVector]:
    """Calcula el vector BM25 de cada chunk.

    Necesita dos pasadas sobre el corpus: la primera tokeniza y mide la
    longitud promedio de documento, la segunda calcula los pesos. Es que la
    normalización por longitud de BM25 compara cada documento contra ese
    promedio, así que no se puede calcular chunk por chunk de forma aislada.
    """
    tokenized = [bm25.tokenize(r["text"]) for r in records]
    avg_len = sum(len(t) for t in tokenized) / len(tokenized)

    vectors = []
    for tokens in tokenized:
        weights = bm25.document_weights(tokens, avg_len)
        vectors.append(
            SparseVector(indices=list(weights.keys()), values=list(weights.values()))
        )
    return vectors


def resumen_chunks(records: list[dict]) -> None:
    """Imprime el recuento ANTES de embeber.

    Existe porque los embeddings son ~30 minutos de CPU: si la estrategia de
    chunking produjo un número disparatado de chunks, conviene enterarse ahora
    y no después de pagarlos.
    """
    largos = [len(r["text"]) for r in records]
    con_seccion = sum(1 for r in records if r["seccion"])
    print(f"\n  estrategia         {CHUNK_STRATEGY}")
    print(f"  chunks             {len(records)}")
    print(f"  documentos         {len(set(r['source'] for r in records))}")
    print(f"  con seccion        {con_seccion} ({100 * con_seccion // max(len(records), 1)}%)")
    print(f"  largo min/prom/max {min(largos)} / {sum(largos) // len(largos)} / {max(largos)}\n")


def verificar_qdrant() -> QdrantClient:
    """Comprueba que Qdrant responde ANTES de embeber nada.

    Los embeddings son ~30 minutos de CPU y hasta ahora el primer contacto con
    Qdrant ocurría recién al terminarlos: si el contenedor no estaba levantado
    —o se caía mientras tanto— la corrida entera se tiraba a la basura por una
    conexión rechazada. Pasó de verdad: el contenedor recibió un SIGTERM
    durante una ingesta y el script murió después de embeber los 2126 chunks.

    Mover la verificación acá arriba convierte esos 30 minutos perdidos en un
    error inmediato.
    """
    client = QdrantClient(url=QDRANT_URL)
    try:
        client.get_collections()
    except Exception as e:
        raise SystemExit(
            f"Qdrant no responde en {QDRANT_URL} ({type(e).__name__}).\n"
            f"Levantalo antes de ingestar:\n"
            f"    docker compose up -d\n"
            f"    curl {QDRANT_URL}/readyz\n"
            f"Si Docker Desktop no está corriendo, abrilo y esperá a que arranque."
        )
    return client


def main():
    print(f"Leyendo documentos de {DATA_DIR} ...")
    records = load_documents()
    resumen_chunks(records)

    # Antes del modelo y antes de los embeddings: ver verificar_qdrant().
    print(f"Verificando Qdrant en {QDRANT_URL} ...")
    client = verificar_qdrant()

    print(f"Cargando modelo de embeddings: {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Generando embeddings densos (BGE-M3, local en CPU) ...")
    texts = [r["text"] for r in records]
    dense = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    print("Calculando vectores sparse (BM25) ...")
    sparse = build_sparse_vectors(records)

    build_collection(client, vector_size=dense.shape[1])

    print(f"Subiendo {len(records)} puntos a la colección '{COLLECTION}' ...")
    points = [
        PointStruct(
            id=i,
            vector={"dense": dense[i].tolist(), "bm25": sparse[i]},
            payload={
                "source": r["source"],
                "chunk_index": r["chunk_index"],
                "text": r["text"],
                # Identidad documental: sirve para mostrarla en la app y para
                # poder filtrar por versión o por ley desde Qdrant.
                "titulo": r["titulo"],
                "numero_ley": r["numero_ley"],
                "version": r["version"],
                "seccion": r["seccion"],
            },
        )
        for i, r in enumerate(records)
    ]

    # Subida por lotes, no de una. Qdrant rechaza payloads de más de 32 MB
    # (error 400, "JSON payload is larger than allowed") y cada punto pesa
    # bastante: 1024 floats del vector denso serializados como JSON, más el
    # texto del chunk. Con el corpus real un único upsert daba 56 MB.
    #
    # Que esto falle al final es lo peor posible, porque la colección ya se
    # borró y los embeddings —media hora de CPU— se pierden. Por eso el lote
    # es conservador: 256 puntos son ~7 MB, bien lejos del límite, y el costo
    # de hacer varias llamadas HTTP es despreciable al lado de embeber.
    for inicio in range(0, len(points), UPSERT_BATCH):
        lote = points[inicio:inicio + UPSERT_BATCH]
        client.upsert(collection_name=COLLECTION, points=lote)
        print(f"  {min(inicio + UPSERT_BATCH, len(points))}/{len(points)}")

    print("Listo.")


if __name__ == "__main__":
    main()
