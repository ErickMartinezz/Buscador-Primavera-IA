# src/buscador.py
from collections import defaultdict
from src.normalizador import normalizar_texto
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def buscar_palabras(paginas, palabras, modo="clasica", modelo_USE=None):
    """
    Busca palabras o frases en las páginas del PDF.

    Parámetros:
    -----------
    paginas : list[str]
        Texto de cada página del PDF.
    palabras : list[str]
        Palabras o frases a buscar.
    modo : str
        Puede ser "clasica", "heuristica" o "semantica".
    modelo_USE : modelo Universal Sentence Encoder (solo si modo='semantica')

    Retorna:
    --------
    dict o list :
        - Modo clásica → dict con ocurrencias por palabra y página.
        - Modo heurística → lista ordenada por relevancia.
        - Modo semántica → lista ordenada por similitud.
    """
    resultados = defaultdict(lambda: defaultdict(int))

    # --- 🔍 BÚSQUEDA CLÁSICA ---
    if modo == "clasica":
        for i, texto in enumerate(paginas):
            texto_norm = normalizar_texto(texto)
            for palabra in palabras:
                palabra_norm = normalizar_texto(palabra)
                ocurrencias = texto_norm.count(palabra_norm)
                if ocurrencias > 0:
                    resultados[palabra][i + 1] += ocurrencias  # Página i+1
        return dict(resultados)

    # --- 🧭 BÚSQUEDA HEURÍSTICA ---
    elif modo == "heuristica":
        for i, texto in enumerate(paginas):
            texto_norm = normalizar_texto(texto)
            for palabra in palabras:
                palabra_norm = normalizar_texto(palabra)
                ocurrencias = texto_norm.count(palabra_norm)
                if ocurrencias > 0:
                    resultados[palabra][i + 1] += ocurrencias

        ranking = []
        for palabra, paginas_dict in resultados.items():
            total = sum(paginas_dict.values())
            promedio_pagina = sum(paginas_dict.keys()) / len(paginas_dict)
            relevancia = total * (1 / promedio_pagina)
            ranking.append({
                "Palabra": palabra,
                "Frecuencia total": total,
                "Páginas": ", ".join(map(str, paginas_dict.keys())),
                "Heurística (relevancia)": round(relevancia, 3)
            })

        ranking.sort(key=lambda x: x["Heurística (relevancia)"], reverse=True)
        return ranking

 # --- 🤖 BÚSQUEDA SEMÁNTICA ---
    elif modo == "semantica" and modelo_USE is not None:
        consulta = " ".join(palabras)
        emb_consulta = modelo_USE([consulta]).numpy()

        similitudes = []
        for i, texto in enumerate(paginas):
            # Embedding general de la página
            emb_pagina = modelo_USE([texto]).numpy()
            sim_pagina = cosine_similarity(emb_consulta, emb_pagina)[0][0]

            # --- buscar palabra más similar dentro de la página ---
            palabras_doc = normalizar_texto(texto).split()
            sims_palabras = []
            for palabra in palabras_doc:
                emb_pal = modelo_USE([palabra]).numpy()
                sim_pal = cosine_similarity(emb_consulta, emb_pal)[0][0]
                sims_palabras.append((palabra, sim_pal))

            palabra_mas_similar, max_sim_pal = max(sims_palabras, key=lambda x: x[1])

            similitudes.append({
                "Página": i + 1,
                "Similitud (página)": round(float(sim_pagina), 4),
                "Palabra más similar": palabra_mas_similar,
                "Similitud (palabra)": round(float(max_sim_pal), 4),
                "Consulta": consulta
            })

        similitudes.sort(key=lambda x: x["Similitud (página)"], reverse=True)
        return similitudes

    else:
        raise ValueError("Modo de búsqueda no reconocido o modelo no cargado.")
