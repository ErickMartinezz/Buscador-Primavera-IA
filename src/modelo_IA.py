import streamlit as st
import tensorflow_hub as hub
import numpy as np
import tensorflow as tf

# Cargar modelo de embeddings (solo una vez)

@st.cache_resource
def cargar_modelo_USE():
    try:
        modelo = hub.KerasLayer('https://tfhub.dev/google/universal-sentence-encoder/4')
        return modelo
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None
def generar_embeddings(textos, modelo):
    """
    Genera embeddings (vectores de significado) para una lista de textos.
    """
    embeddings = modelo(textos)
    return np.array(embeddings)

def buscar_semanticamente(paginas, consulta, modelo):
    """
    Busca la página más similar semánticamente a la consulta.
    Retorna lista ordenada de (nro_pagina, similitud).
    """
    # Generar embeddings para todas las páginas y la consulta
    embeddings_paginas = generar_embeddings(paginas, modelo)
    embedding_consulta = generar_embeddings([consulta], modelo)[0]

    # Calcular similitud de coseno entre la consulta y cada página
    similitudes = np.inner(embeddings_paginas, embedding_consulta)
    
    # Ordenar por similitud descendente
    ranking = sorted(
        list(enumerate(similitudes, start=1)),
        key=lambda x: x[1],
        reverse=True
    )

    return ranking
