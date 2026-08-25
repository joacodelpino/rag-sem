"""
Streamlit - walking skeleton.

Una consulta, una configuración (naive), una respuesta con las fuentes
citadas. El layout de 3 columnas (naive / híbrida / híbrida+reranking) se
agrega cuando existan las otras dos rutas de retrieval.
"""
import streamlit as st

from generate import generate_answer
from retrieval import retrieve_naive

st.set_page_config(page_title="RAG jurídico - demo", layout="centered")
st.title("Asistente de consulta sobre documentación jurídica")
st.caption("Walking skeleton — ruta naive (búsqueda vectorial densa)")

query = st.text_input("Consulta", placeholder="¿Cuántos días hábiles hay para presentar el recurso jerárquico?")

if st.button("Consultar") and query:
    with st.spinner("Buscando en el corpus..."):
        chunks = retrieve_naive(query, top_k=5)
    with st.spinner("Generando respuesta..."):
        answer = generate_answer(query, chunks)

    st.subheader("Respuesta")
    st.write(answer)

    st.subheader("Fuentes recuperadas")
    for chunk in chunks:
        with st.expander(f"{chunk.source} (score: {chunk.score:.3f})"):
            st.text(chunk.text)
