"""
Streamlit — la demo.

Muestra la misma consulta resuelta por cada configuración de recuperación, en
columnas paralelas, para que se vea la diferencia en vivo: qué chunks trajo
cada una, en qué orden, y qué respondió el LLM con ese contexto.

La tercera columna (híbrida + reranking) se suma cuando exista esa ruta.
"""
import streamlit as st

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

query = st.text_input(
    "Consulta",
    placeholder="¿Cuántos días hábiles hay para presentar el recurso jerárquico?",
)

col_a, col_b = st.columns(2)
top_k = col_a.slider("Chunks recuperados (top_k)", 1, 10, 5)
generar = col_b.checkbox(
    "Generar respuesta con el LLM", value=True,
    help="Desactivalo para comparar solo la recuperación, sin gastar llamadas a la API.",
)

if st.button("Consultar", type="primary") and query:
    columns = st.columns(len(CONFIGS))

    for column, nombre in zip(columns, CONFIGS):
        with column:
            st.subheader(nombre)

            with st.spinner("Recuperando..."):
                chunks = recuperar(nombre, query, top_k)

            if generar:
                with st.spinner("Generando..."):
                    st.write(generate_answer(query, chunks))

            st.markdown("**Chunks recuperados**")
            if not chunks:
                st.info("Ningún chunk recuperado.")
            # La posición importa tanto como el contenido: es lo que cambia
            # entre configuraciones y lo que mide MRR/NDCG en la evaluación.
            for posicion, chunk in enumerate(chunks, start=1):
                with st.expander(f"{posicion}. {chunk.source} — {chunk.score:.3f}"):
                    st.caption(f"chunk id: {chunk.id}")
                    st.text(chunk.text)
