# app.py
import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.extractor import extraer_texto_pdf
from src.buscador import buscar_palabras
from src.buscador_integrado import buscar_integrado
from src.modelo_IA import cargar_modelo_USE  # Modelo IA
from src.clasificador import ClasificadorTexto
from collections import defaultdict

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="🌸 Buscador Primavera + IA 🤖",
    page_icon="🌷",
    layout="centered"
)

# --- TÍTULO PRINCIPAL ---
st.markdown("""
<div style="
    background: linear-gradient(to right, #fff8e7, #d7f7e7);
    padding: 20px;
    border-radius: 15px;
    border: 2px solid #a2d5c6;
    text-align: center;
">
<h1 style="color:#3c8d2f;">🌸 Buscador Primavera + IA 🤖</h1>
<p style="color:#207561; font-size:16px;">
Subí tu archivo PDF y elegí el tipo de búsqueda para explorar su contenido. 🌼  
Podés usar una búsqueda simple, informada o basada en inteligencia artificial.
</p>
</div>
""", unsafe_allow_html=True)

# --- PANEL LATERAL DE MODO DE BÚSQUEDA ---
modo_busqueda = st.sidebar.selectbox(
    "Seleccioná el tipo de búsqueda 🌸",
    ["Búsqueda clásica", "Búsqueda informada (heurística)", "Búsqueda semántica (IA)", "Búsqueda integral"]
)

# --- CARGA DEL PDF ---
archivo_pdf = st.file_uploader("📂 Subí tu archivo PDF", type=["pdf"])
paginas = None

if archivo_pdf:
    st.success(f"Archivo cargado: {archivo_pdf.name}")
    paginas = extraer_texto_pdf(archivo_pdf)

    # --- INICIALIZAR CLASIFICADOR ---
    clasificador = ClasificadorTexto()
    # Entrenar con ejemplos mínimos (puedes reemplazar con dataset real)
    textos_ejemplo = [
        "Informe académico sobre física",
        "Factura de venta producto comercial",
        "Plan de viaje recreativo",
        "Noticia sobre política"
    ]
    etiquetas_ejemplo = [0, 1, 2, 3]  # índices de categorías
    clasificador.entrenar(textos_ejemplo, etiquetas_ejemplo, epochs=10)

    # --- BÚSQUEDA CLÁSICA ---
    if modo_busqueda == "Búsqueda clásica":
        st.markdown("### 🔍 Búsqueda clásica por palabras")
        palabras_input = st.text_input("Escribí las palabras a buscar (separadas por espacio):")

        if st.button("🌱 Buscar", key="buscar_clasica"):
            if not palabras_input.strip():
                st.error("Por favor, ingresá al menos una palabra para buscar.")
            else:
                palabras = palabras_input.split()
                resultados = buscar_palabras(paginas, palabras, modo="clasica")
                if not resultados:
                    st.warning("No se encontraron coincidencias.")
                else:
                    filas = []
                    for palabra, paginas_dict in resultados.items():
                        for pagina, ocurrencias in paginas_dict.items():
                            filas.append({
                                "Palabra": palabra,
                                "Página": pagina,
                                "Ocurrencias": ocurrencias
                            })
                    st.markdown("### 🌷 Resultados de búsqueda")
                    st.dataframe(filas, use_container_width=True)

    # --- BÚSQUEDA INFORMADA (HEURÍSTICA) ---
    elif modo_busqueda == "Búsqueda informada (heurística)":
        st.markdown("### 💡 Búsqueda informada con heurística de relevancia")
        palabras_input = st.text_input("Escribí las palabras a buscar (separadas por espacio):")

        if st.button("🔥 Buscar con heurística", key="buscar_heuristica"):
            if not palabras_input.strip():
                st.error("Por favor, ingresá al menos una palabra para buscar.")
            else:
                palabras = palabras_input.split()
                resultados = buscar_palabras(paginas, palabras, modo="heuristica")
                if not resultados:
                    st.warning("No se encontraron coincidencias.")
                else:
                    st.markdown("### 🔥 Ranking de relevancia")
                    st.dataframe(resultados, use_container_width=True)

    # --- BÚSQUEDA SEMÁNTICA (IA) ---
    elif modo_busqueda == "Búsqueda semántica (IA)":
        st.markdown("### 🧠 Buscador Semántico (IA)")
        modelo_USE = cargar_modelo_USE()
        if modelo_USE is None:
            st.error("Error al cargar el modelo Universal Sentence Encoder.")
        else:
            st.info("✅ Modelo IA cargado correctamente.")
            consulta = st.text_input("💬 Escribí tu búsqueda por significado:")

            if st.button("🔮 Buscar significado similar", key="buscar_semantica"):
                if not consulta.strip():
                    st.error("Por favor, escribí una consulta.")
                else:
                    resultados = buscar_palabras(paginas, [consulta], modo="semantica", modelo_USE=modelo_USE)
                    if not resultados:
                        st.warning("No se encontraron similitudes significativas.")
                    else:
                        st.success("✅ Análisis semántico completado. Mostrando las páginas más relevantes:")
                        st.dataframe(resultados, use_container_width=True)

    # --- BÚSQUEDA INTEGRAL ---
    elif modo_busqueda == "Búsqueda integral":
        st.markdown("### 🌟 Búsqueda Integral")
        palabras_input = st.text_input("🔍 Escribí palabras o consulta para búsqueda integral:")
        usar_heuristica = st.checkbox("✨ Aplicar heurística")
        usar_semantica = st.checkbox("🤖 Aplicar búsqueda semántica (IA)")

        if st.button("🔄 Buscar integral"):
            if palabras_input.strip():
                palabras = palabras_input.split()
                resultados = buscar_integrado(paginas, palabras, usar_heuristica, usar_semantica)
                st.dataframe(resultados, use_container_width=True)
            else:
                st.error("Por favor, ingresá al menos una palabra o consulta.")

    # --- CLASIFICACIÓN DE TEXTO ---
    st.markdown("### 🏷️ Clasificación de las páginas")
    categorias_detectadas = clasificador.predecir(paginas)
    st.dataframe([{"Página": i+1, "Categoría": cat} for i, cat in enumerate(categorias_detectadas)])

else:
    st.info("Esperando que subas un archivo PDF 📄")

