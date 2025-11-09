#  Buscador Primavera + IA 

## Descripción
Sistema integral de búsqueda de información en archivos PDF, combinando tres motores:
- **Búsqueda clásica**: coincidencia exacta de palabras.
- **Búsqueda heurística**: ranking de relevancia basado en frecuencia y contexto.
- **Búsqueda semántica (IA)**: similaridad de significado usando Universal Sentence Encoder.

Incluye además un **clasificador de texto** que analiza el tipo de contenido encontrado.

## Objetivo
Facilitar la exploración de PDFs de manera inteligente, integrando técnicas de búsqueda, razonamiento lógico y aprendizaje automático.

## Funcionalidades
1. Subida de archivos PDF.
2. Búsqueda por palabras (clásica).
3. Búsqueda con heurística de relevancia.
4. Búsqueda semántica usando modelo IA.
5. Clasificación del tipo de texto encontrado.

## Arquitectura
- **app.py**: interfaz Streamlit.
- **src/buscador.py**: motores de búsqueda (clásico y heurístico).
- **src/buscador_integrado.py**: motor integral.
- **src/modelo_IA.py**: carga del modelo Universal Sentence Encoder.
- **src/clasificador.py**: clasificación de textos.

### Flujo de datos
PDF → Extracción de texto → Selección de motor de búsqueda → Resultados → Clasificación de texto.

## Instalación


pip install -r requirements.txt

## Uso

1- Ejecutar la app: streamlit run app.py

2- Subir archivo PDF.

3- Elegir el tipo de búsqueda.

4- Visualizar resultados y clasificación del texto.

## Tecnologías

Python 3.11

Streamlit

TensorFlow / TensorFlow Hub

Scikit-learn

PDFPlumber / PyPDFium2

## Autor

# Erick Martínez

```bash