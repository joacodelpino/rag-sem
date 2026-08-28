# -*- coding: utf-8 -*-
"""
Genera data/golden_set.csv — las preguntas de la evaluación con Ragas.

Son 29: el set se diseñó con 30 y una (m04) se descartó al auditarlo porque no
era multihop de verdad. Ver la nota en PREGUNTAS.

NINGUNA pregunta está escrita de memoria. Cada una salió de leer un chunk real
del índice, y este script vuelve a verificarlo antes de escribir el CSV:

  - positivas: cada frase de `verificar` tiene que aparecer LITERAL en alguno
    de los chunks de `chunks_relevantes`.
  - multihop:  además, ningún chunk por separado puede contener todas las
    frases. Si uno solo alcanza, la pregunta no es multihop y el script avisa.
  - negativas: el patrón de `verificar` NO puede aparecer en NINGÚN chunk del
    corpus. Así una negativa no se vuelve positiva sin que nos enteremos.

Si algo no verifica, el script FALLA y no escribe nada. Es a propósito: un set
dorado con ground truth mal es peor que no tener set dorado, porque las
métricas salen igual y no hay forma de darse cuenta.

Los ids salen del índice actual (2126 chunks, CHUNK_STRATEGY=articulo). OJO:
los ids son el orden de ingesta, así que CUALQUIER reingesta que cambie el
corpus o el chunking los invalida. Volver a correr este script después de
reingestar y revisar que siga verificando.

Uso:
    .venv/Scripts/python.exe evals/build_golden_set.py
"""
import csv
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from qdrant_client import QdrantClient  # noqa: E402

SALIDA = RAIZ / "data" / "golden_set.csv"
QDRANT_URL = "http://localhost:6333"
COLLECTION = "legal_docs"

# --------------------------------------------------------------------------
# Las 29 preguntas.
#
# `verificar` son las frases que tienen que estar literalmente en los chunks
# citados (en las negativas, el regex que NO tiene que estar en ningún lado).
# `duda` marca las preguntas sobre las que tengo una reserva; se escriben igual
# pero el script las lista aparte al final para que se revisen a mano.
#
# Sobre las tres reservas de abajo (i02, i03, i04): son documentos del corpus en
# su texto original, con numeración o plazos distintos a los vigentes. Se dejan
# tal cual POR DECISIÓN: el trabajo demuestra cómo funciona un RAG, no la
# vigencia del corpus, y el ground truth de Ragas es lo que dice el documento
# recuperado. La nota queda para que nadie las lea después creyendo que son
# afirmaciones sobre el derecho actual.
# --------------------------------------------------------------------------
PREGUNTAS = [
    # ---------------- factual (8) ----------------
    dict(id="f01", categoria="factual",
         pregunta="¿Cuántos días tiene el consumidor para arrepentirse de una compra hecha fuera del local?",
         respuesta_esperada="10 días corridos",
         chunks=[41], verificar=["DIEZ (10) días corridos"],
         por_que="Control básico: el dato está en un solo artículo y con palabras parecidas a las de la pregunta. Si esta falla, algo del pipeline está roto."),
    dict(id="f02", categoria="factual",
         pregunta="¿Cuánto dura la garantía legal de un producto usado?",
         respuesta_esperada="3 meses (6 meses para los productos que no son usados)",
         chunks=[16], verificar=["TRES (3) meses cuando se trate de bienes muebles usados"],
         por_que="El chunk contiene los dos plazos juntos: mide si el sistema devuelve el que corresponde a 'usado' y no el otro."),
    dict(id="f03", categoria="factual",
         pregunta="¿Cuántos días de vacaciones le corresponden a un trabajador con menos de 5 años de antigüedad?",
         respuesta_esperada="14 días corridos",
         chunks=[328], verificar=["catorce (14) días corridos cuando la antigüedad"],
         por_que="La pregunta no usa ninguna palabra del texto salvo 'antigüedad': mide recuperación semántica, no coincidencia léxica."),
    dict(id="f04", categoria="factual",
         pregunta="¿Cuántas materias por año tiene que aprobar un alumno de la UNLAR para seguir siendo regular?",
         respuesta_esperada="2 asignaturas por año académico",
         chunks=[2027], verificar=["dos (2) asignaturas durante cada año académico"],
         por_que="Documento chico y poco citado del corpus: comprueba que la recuperación no se va siempre a los documentos grandes."),
    dict(id="f05", categoria="factual",
         pregunta="¿Cuánto tiempo dura la regularidad de una materia en la UNLAR?",
         respuesta_esperada="12 épocas o turnos de examen",
         chunks=[2033], verificar=["doce (12) épocas o turnos de exámenes"],
         por_que="Vocabulario propio de la institución ('épocas o turnos'): el denso solo tiende a confundirlo con otros plazos del mismo reglamento."),
    dict(id="f06", categoria="factual",
         pregunta="¿En qué banco se deposita el dinero secuestrado en una causa penal?",
         respuesta_esperada="Banco de la Ciudad de Buenos Aires, o la sucursal del Banco de la Nación Argentina que corresponda",
         chunks=[94], verificar=["Banco de la Ciudad de Buenos Aires"],
         por_que="Respuesta que es un nombre propio, no un número: es donde BM25 suele ganarle al denso."),
    dict(id="f07", categoria="factual",
         pregunta="¿Cuántos años duran en su cargo los consiliarios de la UNLAR?",
         respuesta_esperada="3 años, y pueden ser reelectos una vez",
         chunks=[1779], verificar=["duran tres (3) años en el ejercicio de sus mandatos"],
         por_que="El Estatuto tiene decenas de plazos parecidos: mide precisión, no solo recall."),
    dict(id="f08", categoria="factual",
         pregunta="¿En cuánto tiempo hay que avisarle al Registro Nacional de Armas que se secuestró un arma?",
         respuesta_esperada="10 días hábiles",
         chunks=[167], verificar=["dentro de los diez (10) días hábiles"],
         por_que="'10 días hábiles' aparece en varias leyes del corpus: mide si el sistema trae el de la ley correcta."),

    # ---------------- identificador (8) ----------------
    dict(id="i01", categoria="identificador",
         pregunta="¿Qué dice el artículo 72 de la ley 11179?",
         respuesta_esperada="Enumera las acciones dependientes de instancia privada",
         chunks=[510], verificar=["Son acciones dependientes de instancia privada"],
         por_que="EL caso del trabajo. El número va sin punto y el artículo 72 existe en varias leyes del corpus; antes del encabezado de identidad el sistema devolvía el art. 72 de la ley 11.723."),
    dict(id="i02", categoria="identificador",
         pregunta="¿Qué dice el artículo 208 de la Ley de Contrato de Trabajo?",
         respuesta_esperada="Que a los menores de 18 años que trabajan de mañana y de tarde se les aplican los artículos 190, 191 y 192",
         chunks=[367], verificar=["Con relación a los menores de dieciocho (18) años"],
         duda="El corpus tiene la LCT de 1974: este NO es el art. 208 que espera un abogado (enfermedades inculpables), que en este PDF es el art. 225. La respuesta esperada es correcta SEGÚN EL CORPUS, no según el derecho vigente.",
         por_que="Pide un artículo por su número dentro de una ley de 297 artículos. El número no está en el cuerpo del texto: sin el encabezado de identidad, el sistema tiene que adivinar cuál de los 297 es."),
    dict(id="i03", categoria="identificador",
         pregunta="¿Qué dice el artículo 57 de la Ley de Contrato de Trabajo?",
         respuesta_esperada="Que los libros sin las formalidades del artículo 56 no tienen valor en juicio a favor del empleador",
         chunks=[228], verificar=["no tendrán valor en juicio en favor del empleador"],
         duda="Mismo problema que i02: el art. 57 vigente (presunción por el silencio del empleador) no existe en el texto de 1974.",
         por_que="Igual que i02 con un número de artículo bajo. Los artículos de dos cifras son más difíciles: '57' aparece suelto en decenas de chunks como parte de otros números."),
    dict(id="i04", categoria="identificador",
         pregunta="Según el artículo 5 de la ley 11.723, ¿cuántos años después de la muerte del autor sigue vigente la propiedad intelectual?",
         respuesta_esperada="30 años",
         chunks=[761], verificar=["durante treinta años más"],
         duda="El corpus tiene el texto original de 1933. La ley vigente dice 70 años (reforma de la ley 24.870, de 1997). La respuesta esperada es la del corpus.",
         por_que="La pregunta nombra la ley Y el artículo, y la respuesta es un número escrito con letras ('treinta años'), que BM25 no puede matchear contra un '30' en la consulta."),
    dict(id="i05", categoria="identificador",
         pregunta="¿Qué ley creó el Registro Nacional de Armas de Fuego?",
         respuesta_esperada="La ley 25.938",
         chunks=[165], verificar=["Ley 25.938"],
         por_que="La respuesta es el número de ley, no un dato del articulado: el sistema tiene que traer la carátula del documento."),
    dict(id="i06", categoria="identificador",
         pregunta="¿Qué artículo de la ley 24.240 regula la garantía legal?",
         respuesta_esperada="El artículo 11",
         chunks=[16], verificar=["Art. 11"],
         por_que="La pregunta pide el NÚMERO de artículo, que solo está en el encabezado del chunk. Sin encabezado de identidad es irrespondible."),
    dict(id="i07", categoria="identificador",
         pregunta="¿Qué leyes regulan el sistema de riesgos del trabajo?",
         respuesta_esperada="La ley 24.557 y su modificatoria, la ley 26.773",
         chunks=[1925], verificar=["24.557", "26.773"],
         por_que="Dos números de ley en la misma respuesta, dentro de un folleto de divulgación y no de una norma: prueba que el corpus no es homogéneo."),
    dict(id="i08", categoria="identificador",
         pregunta="¿Qué pena fija el artículo 204 del Código Penal según la ley 23.737?",
         respuesta_esperada="Prisión de 6 meses a 3 años",
         chunks=[118], verificar=["prisión de seis meses a tres años"],
         por_que="Menciona dos normas a la vez (un artículo del Código Penal reformado por otra ley) y las dos están en el corpus como documentos distintos."),

    # ---------------- multihop (6) ----------------
    dict(id="m01", categoria="multihop",
         pregunta="¿En qué tipo de ventas puede arrepentirse el consumidor y cuántos días tiene para hacerlo?",
         respuesta_esperada="En la venta domiciliaria y en la venta por correspondencia; tiene 10 días corridos",
         chunks=[39, 40, 41], verificar=["Venta domiciliaria", "Venta por Correspondencia", "DIEZ (10) días corridos"],
         por_que="El plazo está en el art. 34 y los casos en los arts. 32 y 33: el art. 34 dice 'en los casos previstos en los artículos 32 y 33' sin repetirlos."),
    dict(id="m02", categoria="multihop",
         pregunta="¿Qué tiene que hacer un alumno de la UNLAR para ser regular y cuánto le dura la regularidad de una materia?",
         respuesta_esperada="Aprobar 2 asignaturas por año académico; la regularidad de la materia dura 12 turnos de examen",
         chunks=[2027, 2033], verificar=["dos (2) asignaturas durante cada año académico", "doce (12) épocas o turnos de exámenes"],
         por_que="Dos artículos del mismo reglamento que hablan de 'regular' con sentidos distintos: condición del alumno y vigencia de la materia."),
    dict(id="m03", categoria="multihop",
         pregunta="¿Cuánto tiempo hay para informar un arma secuestrada y cuánto para informar un cambio de lugar de depósito?",
         respuesta_esperada="10 días hábiles para informar el secuestro y 48 horas para informar el cambio de depósito",
         chunks=[167, 168], verificar=["dentro de los diez (10) días hábiles", "cuarenta y ocho (48) horas"],
         por_que="Dos plazos parecidos en artículos consecutivos: si el sistema trae uno solo, contesta con seguridad y a medias."),
    # DESCARTADA — m04, "¿Qué estableció la Corte en Siri y qué agregó en Kot?"
    # (chunks 1026;1061). No era multihop: la carátula de Kot sola ya contiene
    # las dos mitades de la respuesta, porque el resumen del fallo explica a Kot
    # EN RELACIÓN a Siri — lo nombra, dice que la restricción anterior venía de
    # la "autoridad pública" y que esta viene de "actos de particulares". Un
    # sistema que recupere ese único chunk contesta perfecto, así que la fila
    # habría medido recuperación de un chunk disfrazada de multihop.
    # La relación Siri/Kot se sigue evaluando en s01 y s02, que sí funcionan.
    dict(id="m05", categoria="multihop",
         pregunta="¿Cuántos días de vacaciones corresponden con menos de 5 años de antigüedad y qué hay que haber trabajado para tener derecho?",
         respuesta_esperada="14 días corridos; hay que haber trabajado como mínimo la mitad de los días hábiles del año",
         chunks=[328, 329], verificar=["catorce (14) días corridos cuando la antigüedad", "la mitad, como mínimo, de los días hábiles"],
         por_que="Artículos consecutivos, uno con la cantidad y otro con el requisito. Con chunking fijo los dos caían en el mismo chunk por casualidad; con chunking por artículo el multihop es real."),
    dict(id="m06", categoria="multihop",
         pregunta="En un sumario por defensa del consumidor, ¿cuántos días hay para presentar el descargo y cuántos para apelar la sanción?",
         respuesta_esperada="5 días hábiles para el descargo y 10 días hábiles para el recurso",
         chunks=[58, 61], verificar=["plazo de cinco (5) días hábiles", "dentro de los diez (10) días hábiles de notificada la resolución"],
         por_que="Los dos plazos están en el mismo artículo 45, pero tan largo que quedó partido en cinco chunks: multihop provocado por el tope de tamaño."),

    # ---------------- fuente (4) ----------------
    dict(id="s01", categoria="fuente",
         pregunta="¿En qué fallo la Corte Suprema creó la acción de amparo?",
         respuesta_esperada="En el fallo Siri, de 1957",
         chunks=[1026], verificar=["Siri, Angel (1957)"],
         por_que="Cuatro documentos del corpus mencionan a Siri, pero uno solo ES el fallo. Mide si el sistema distingue la fuente de las citas."),
    dict(id="s02", categoria="fuente",
         pregunta="¿Qué fallo extendió el amparo a los actos de particulares?",
         respuesta_esperada="El fallo Samuel Kot, de 1958",
         chunks=[1061], verificar=["Samuel Kot (1958)"],
         por_que="Los cinco fallos del corpus hablan de amparo. Kot es el único que trata el amparo contra particulares, y Siri —que es el más citado— es la respuesta incorrecta más tentadora."),
    dict(id="s03", categoria="fuente",
         pregunta="¿Qué ley es el Código Penal de la Nación?",
         respuesta_esperada="La ley 11.179",
         chunks=[457], verificar=["LEY N° 11.179"],
         por_que="Seis documentos del corpus mencionan el Código Penal (lo reforman, lo reglamentan o lo citan) y uno solo es el Código Penal."),
    dict(id="s04", categoria="fuente",
         pregunta="¿Qué ley establece las normas de protección y defensa de los consumidores?",
         respuesta_esperada="La ley 24.240",
         chunks=[0], verificar=["Ley Nº 24.240"],
         por_que="Dos resoluciones de la Secretaría de Comercio citan la ley 24.240 decenas de veces; la fuente es un tercer documento."),

    # ---------------- negativa (4) ----------------
    dict(id="n01", categoria="negativa",
         pregunta="¿Cuál es el plazo mínimo de un contrato de alquiler según la ley 27.551?",
         respuesta_esperada="No está en el corpus",
         chunks=[], verificar=[r"27\.?551"],
         por_que="Negativa tentadora: el corpus tiene dos contratos de alquiler, así que la recuperación va a traer documentos que PARECEN responder."),
    dict(id="n02", categoria="negativa",
         pregunta="¿Qué dice la ley de teletrabajo sobre el derecho a la desconexión?",
         respuesta_esperada="No está en el corpus",
         chunks=[], verificar=[r"teletrabajo|27\.?555|desconexi[oó]n"],
         por_que="Negativa tentadora: el corpus tiene la Ley de Contrato de Trabajo entera, y el reranker puede darle un score alto a cualquier artículo sobre jornada."),
    dict(id="n03", categoria="negativa",
         pregunta="¿Qué obliga la ley de góndolas a los supermercados?",
         respuesta_esperada="No está en el corpus",
         chunks=[], verificar=[r"g[oó]ndola|27\.?545"],
         por_que="Negativa tentadora: es una ley de defensa del consumidor y el corpus tiene la 24.240 completa."),
    dict(id="n04", categoria="negativa",
         pregunta="¿Qué dice el artículo 500 del Código Penal?",
         respuesta_esperada="No está en el corpus",
         chunks=[], verificar=[r"[Aa]rt[íi]?c?u?l?o?\.? ?500\b"],
         por_que="Negativa con forma de identificador: el Código Penal del corpus llega al artículo 316. El sistema tiene que decir que no existe, no traer el artículo 50 o el 300."),
]


def plano(texto: str) -> str:
    """Colapsa los espacios en blanco antes de comparar.

    Hace falta porque PyMuPDF corta los renglones donde los cortaba el PDF, y
    entonces la frase "DIEZ (10) días corridos" aparece en el texto extraído
    como "DIEZ (10) días\\ncorridos". Sin esto la verificación rechazaría
    respuestas que sí están, que es peor que no verificar: invita a relajar el
    chequeo hasta que deje de servir.
    """
    return re.sub(r"\s+", " ", texto)


def cargar_chunks() -> dict[int, dict]:
    c = QdrantClient(url=QDRANT_URL)
    todos, offset = {}, None
    while True:
        pts, offset = c.scroll(collection_name=COLLECTION, limit=512,
                               offset=offset, with_payload=True, with_vectors=False)
        for p in pts:
            todos[p.id] = p.payload
        if offset is None:
            return todos


def main() -> None:
    print(f"Leyendo la colección '{COLLECTION}' ...")
    chunks = cargar_chunks()
    print(f"{len(chunks)} chunks indexados.\n")

    errores, dudas = [], []

    for q in PREGUNTAS:
        ids = q["chunks"]

        if q["categoria"] == "negativa":
            # La negativa solo vale si el tema NO está en NINGÚN chunk.
            for patron in q["verificar"]:
                pat = re.compile(patron)
                encontrados = [i for i, p in chunks.items() if pat.search(plano(p["text"]))]
                if encontrados:
                    errores.append(
                        f"{q['id']}: la negativa NO es negativa — '{patron}' aparece en "
                        f"{len(encontrados)} chunks (ej. id={encontrados[0]})")
            if ids:
                errores.append(f"{q['id']}: una negativa no puede tener chunks_relevantes")
        else:
            faltantes = [i for i in ids if i not in chunks]
            if faltantes:
                errores.append(f"{q['id']}: ids inexistentes {faltantes}")
                continue
            juntos = plano(" ".join(chunks[i]["text"] for i in ids))
            for frase in q["verificar"]:
                if plano(frase) not in juntos:
                    errores.append(f"{q['id']}: no encontré literal {frase!r} en los chunks {ids}")

            if q["categoria"] == "multihop":
                if len(ids) < 2:
                    errores.append(f"{q['id']}: multihop con un solo chunk")
                # Ningún chunk por separado puede alcanzar para responder.
                for i in ids:
                    if all(plano(f) in plano(chunks[i]["text"]) for f in q["verificar"]):
                        errores.append(
                            f"{q['id']}: NO es multihop — el chunk {i} solo ya contiene "
                            f"todas las frases de la respuesta")

        if q.get("duda"):
            dudas.append((q["id"], q["pregunta"], q["duda"]))

    reparto = {}
    for q in PREGUNTAS:
        reparto[q["categoria"]] = reparto.get(q["categoria"], 0) + 1
    # 29 y no 30: m04 se descartó al auditar el set porque no era multihop.
    # Ver la nota en PREGUNTAS. Se prefirió quedarse con 5 multihop reales a
    # completar el cupo con una fila que mide otra cosa.
    esperado = {"factual": 8, "identificador": 8, "multihop": 5, "fuente": 4, "negativa": 4}
    if reparto != esperado:
        errores.append(f"reparto por categoría {reparto} != {esperado}")

    if errores:
        print("NO se escribió nada. Problemas:\n")
        for e in errores:
            print("  -", e)
        raise SystemExit(1)

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    with SALIDA.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "categoria", "pregunta", "respuesta_esperada",
                    "chunks_relevantes", "por_que"])
        for q in PREGUNTAS:
            w.writerow([q["id"], q["categoria"], q["pregunta"], q["respuesta_esperada"],
                        ";".join(str(i) for i in q["chunks"]), q["por_que"]])

    print(f"Verificadas las {len(PREGUNTAS)} preguntas. Reparto: {reparto}")
    print(f"Escrito: {SALIDA}")
    if dudas:
        print(f"\n{len(dudas)} pregunta(s) MARCADAS PARA REVISAR A MANO:")
        for pid, preg, motivo in dudas:
            print(f"\n  [{pid}] {preg}")
            print(f"        {motivo}")


if __name__ == "__main__":
    main()
