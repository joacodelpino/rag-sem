"""
Streamlit — la demo.

Muestra la misma consulta resuelta por cada configuración de recuperación, en
columnas paralelas, para que se vea la diferencia en vivo: qué chunks trajo
cada una, en qué orden, y qué respondió el LLM con ese contexto.

Qué columnas se muestran se elige en la barra lateral, pero la consulta se
resuelve SIEMPRE en las cuatro configuraciones: ver el comentario de
`resolver_todas()`.
"""
import streamlit as st

import generate
from generate import generate_answer
from retrieval import (
    config_snapshot,
    retrieve_hybrid,
    retrieve_hybrid_rerank,
    retrieve_naive,
    retrieve_sparse,
    warmup,
)

# Las configuraciones que se muestran en columnas. Agregar una acá es lo único
# que hace falta para sumarla a la demo: todas comparten la misma firma.
CONFIGS = {
    "Naive (densa)": retrieve_naive,
    "BM25 (sparse)": retrieve_sparse,
    "Híbrida (RRF)": retrieve_hybrid,
    "Híbrida + rerank": retrieve_hybrid_rerank,
}


@st.cache_resource(show_spinner="Cargando modelos (solo la primera vez)...")
def precargar_modelos():
    """Carga los pesos de BGE-M3 y del reranker al arrancar la app.

    Sin esto, la carga desde disco la paga la PRIMERA consulta: medido, ~46s
    de espera para quien pregunte primero. En una exposición en vivo eso es
    inaceptable, así que se mueve al startup — cuando la página termina de
    cargar, los modelos ya están en memoria y la primera consulta cuesta lo
    mismo que las siguientes.

    cache_resource y no cache_data porque lo que se cachea es un efecto
    (modelos cargados en los singletons de retrieval/rerank), no un valor
    serializable. Streamlit lo ejecuta una vez por proceso, no por sesión.
    """
    warmup()
    return config_snapshot()


CONFIG = precargar_modelos()


@st.cache_data(show_spinner=False)
def recuperar(config: str, query: str, top_k: int):
    """Cachea la recuperación por (configuración, consulta, top_k).

    Importa para la exposición: el reranking tarda varios segundos en CPU, y
    sin caché repetir una consulta —para volver atrás, o porque se tocó otro
    control de la página y Streamlit re-ejecutó el script— la vuelve a pagar
    entera. Con caché, la segunda vez es instantánea.

    Se cachea por el NOMBRE de la configuración y no por la función, porque
    Streamlit necesita poder hashear los argumentos.
    """
    return CONFIGS[config](query, top_k=top_k)


@st.cache_data(show_spinner=False)
def responder(config: str, query: str, top_k: int, advertir_version: bool) -> str:
    """Cachea la respuesta del LLM por (configuración, consulta, top_k, aviso).

    Hace falta desde que las columnas se pueden mostrar y ocultar: cada vez que
    se toca un control, Streamlit re-ejecuta el script entero de arriba a abajo.
    Sin caché, prender y apagar una columna dispararía una llamada a la API por
    cada columna visible, cada vez. Con caché, tocar los checkboxes no gasta un
    centavo: solo la primera vez que aparece una combinación se llama al LLM.

    `advertir_version` entra en la clave a propósito: es justamente el
    interruptor que la demo prende y apaga sobre la misma consulta, y las dos
    ramas tienen que poder convivir en el caché para poder ir y volver.
    """
    return generate_answer(query, recuperar(config, query, top_k), advertir_version)


def resolver_todas(query: str, top_k: int) -> dict[str, list]:
    """Corre la consulta por las CUATRO configuraciones, se muestren o no.

    Es deliberado que no se limite a las visibles. La demo se apoya en poder
    prender una columna a mitad de una explicación —"miren qué pasa si le sumo
    BM25"— y si esa columna hubiera que recuperarla recién en ese momento, la
    espera caería justo en el peor momento posible. Recuperando todo de una,
    tocar los checkboxes es instantáneo porque sale del caché de `recuperar()`.

    El costo de traer las cuatro es el de la más cara: son secuenciales, pero
    naive, sparse e híbrida juntas suman menos de un segundo contra los ~3s del
    reranking. En la práctica no se nota la diferencia.

    La generación NO se hace acá: esa sí se paga por columna visible, porque
    cuesta plata y una columna oculta no la necesita.
    """
    return {nombre: recuperar(nombre, query, top_k) for nombre in CONFIGS}


st.set_page_config(page_title="RAG jurídico — demo", layout="wide")
st.title("Asistente de consulta sobre documentación jurídica")
st.caption(
    "Comparación de configuraciones de recuperación sobre el mismo corpus. "
    "El score significa algo distinto en cada columna (coseno, BM25, RRF, "
    "cross-encoder): comparalos dentro de una columna, no entre columnas."
)

with st.sidebar:
    st.subheader("Configuración de la corrida")
    st.caption(
        "Es la misma que devuelve `config_snapshot()` y con la que se etiquetan "
        "las corridas de evaluación."
    )
    st.json(CONFIG)
    st.caption(
        "La advertencia de versión no figura acá porque no es configuración de "
        "recuperación: se prende y se apaga por consulta, con el control que "
        "está sobre las columnas."
    )

    st.divider()
    st.subheader("Qué configuraciones mostrar")
    seleccion = st.multiselect(
        "Columnas visibles",
        options=list(CONFIGS),
        default=list(CONFIGS),
        label_visibility="collapsed",
        help=(
            "Solo cambia qué se muestra. La consulta se resuelve igual en las "
            "cuatro configuraciones, así que prender una columna después de "
            "haber consultado es instantáneo."
        ),
    )
    # Se reordena según CONFIGS y no según el orden en que se clickearon: las
    # columnas cuentan una progresión (densa -> léxica -> fusión -> reranking) y
    # esa progresión es la mitad de la explicación. Dejar que el orden dependa
    # de en qué orden tocó los chips quien maneja la demo la rompe.
    mostrar = [nombre for nombre in CONFIGS if nombre in seleccion]

query = st.text_input(
    "Consulta",
    placeholder="¿Cuántos días hábiles hay para presentar el recurso jerárquico?",
)

col_a, col_b, col_c = st.columns(3)
top_k = col_a.slider("Chunks recuperados (top_k)", 1, 10, 5)
generar = col_b.checkbox(
    "Generar respuesta con el LLM", value=True,
    help="Desactivalo para comparar solo la recuperación, sin gastar llamadas a la API.",
)
# Interruptor para la exposición: permite mostrar la MISMA consulta con y sin
# la advertencia, en vivo, sin reiniciar nada. Apagarlo solo saca esa frase del
# system prompt — la versión sigue estando en el encabezado de cada chunk y en
# la cita, así que lo que se ve es el efecto del prompt aislado del efecto de
# los metadatos.
advertir_version = col_c.checkbox(
    "Advertir sobre la versión", value=generate.ADVERTIR_VERSION,
    help=(
        "Agrega al system prompt la instrucción de avisar cuando el contexto es "
        "un 'texto original' y la consulta parece referirse al régimen vigente. "
        "Apagalo para reproducir el comportamiento anterior (probá con el "
        "artículo 208 de la LCT)."
    ),
)

# La consulta ejecutada se guarda en session_state en vez de renderizar dentro
# del `if` del botón. Si no, cualquier interacción posterior —tocar el selector
# de columnas, mover top_k— re-ejecuta el script con el botón ya en False y la
# pantalla queda en blanco. Guardarla hace que los resultados sobrevivan a los
# reruns, que es lo que vuelve usable al selector.
if st.button("Consultar", type="primary") and query:
    st.session_state.corrida = (query, top_k)

if "corrida" in st.session_state:
    query_corrida, top_k_corrido = st.session_state.corrida

    with st.spinner("Recuperando en las cuatro configuraciones..."):
        resultados = resolver_todas(query_corrida, top_k_corrido)

    if not mostrar:
        st.warning(
            "No hay ninguna configuración seleccionada. Elegí al menos una en "
            "la barra lateral."
        )

    columns = st.columns(len(mostrar)) if mostrar else []

    for column, nombre in zip(columns, mostrar):
        with column:
            st.subheader(nombre)

            chunks = resultados[nombre]

            if generar:
                with st.spinner("Generando..."):
                    st.write(
                        responder(
                            nombre, query_corrida, top_k_corrido, advertir_version
                        )
                    )

            st.markdown("**Chunks recuperados**")
            if not chunks:
                st.info("Ningún chunk recuperado.")
            # La posición importa tanto como el contenido: es lo que cambia
            # entre configuraciones y lo que mide MRR/NDCG en la evaluación.
            for posicion, chunk in enumerate(chunks, start=1):
                # Se muestra el título del manifiesto y no el nombre de
                # archivo: cinco documentos del corpus se llaman
                # "0N_argentinagobar.pdf" y no hay forma de saber cuál es cuál.
                # La versión va al lado del título porque es lo que evita leer
                # el texto de la LCT de 1974 creyendo que es el vigente.
                with st.expander(f"{posicion}. {chunk.etiqueta()} — {chunk.score:.3f}"):
                    st.caption(f"archivo: {chunk.source}  ·  chunk id: {chunk.id}")
                    st.text(chunk.text)
