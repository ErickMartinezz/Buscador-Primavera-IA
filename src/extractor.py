# extractor.py
import pdfplumber
import io

def extraer_texto_pdf(archivo_pdf):
    """
    Extrae el texto de un PDF.
    Puede recibir una ruta (str) o un archivo subido (Streamlit uploader).
    Devuelve una lista de textos, uno por cada página.
    """
    texto_paginas = []

    # Si el archivo viene como bytes (por ejemplo, desde Streamlit)
    if not isinstance(archivo_pdf, str):
        archivo_pdf = io.BytesIO(archivo_pdf.read())

    with pdfplumber.open(archivo_pdf) as pdf:
        for pagina in pdf.pages:
            texto = pagina.extract_text()
            if texto:
                texto = texto.replace("\n", " ")
                texto_paginas.append(texto)

    return texto_paginas


if __name__ == "__main__":
    # Prueba local (usando una ruta de archivo)
    archivo = "ejemplo.pdf"
    paginas = extraer_texto_pdf(archivo)
    print(f"✅ Se extrajeron {len(paginas)} páginas.")
    print(paginas[0][:200])
    print("✅ Texto de la primera página:")