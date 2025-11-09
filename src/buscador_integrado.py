from collections import defaultdict
from src.normalizador import normalizar_texto
from src.buscador import buscar_palabras
from src.modelo_IA import cargar_modelo_USE
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def buscar_integrado(paginas, palabras, usar_heuristica=False, usar_semantica=False):
    """
    Combina búsqueda clásica, heurística y semántica en un solo resultado.
    
    Parámetros:
    -----------
    paginas : list[str] - Lista de textos de las páginas del PDF.
    palabras : list[str] - Lista de palabras o consulta para buscar.
    usar_heuristica : bool - Activar ranking de relevancia.
    usar_semantica : bool - Activar búsqueda semántica.
    
    Retorna:
    --------
    lista de dicts con los resultados integrados.
    """
    resultados = []

    # --- Búsqueda clásica / heurística ---
    if not usar_semantica:
        modo = "heuristica" if usar_heuristica else "clasica"
        res = buscar_palabras(paginas, palabras, modo=modo)

        if modo == "clasica":
            # dict de dicts a lista de filas
            for palabra, paginas_dict in res.items():
                for p, c in sorted(paginas_dict.items()):
                    resultados.append({
                        "Palabra": palabra,
                        "Página": p,
                        "Ocurrencias": c,
                        "Tipo": "Clásica"
                    })
        else:
            # heurística devuelve lista de dicts
            for r in res:
                r["Tipo"] = "Heurística"
                resultados.append(r)

    # --- Búsqueda semántica ---
    if usar_semantica:
        modelo_USE = cargar_modelo_USE()
        if modelo_USE is None:
            raise ValueError("No se pudo cargar el modelo USE para búsqueda semántica.")
        
        consulta = " ".join(palabras)
        emb_consulta = modelo_USE([consulta]).numpy()
        for i, texto in enumerate(paginas):
            emb_pagina = modelo_USE([texto]).numpy()
            sim = cosine_similarity(emb_consulta, emb_pagina)[0][0]
            resultados.append({
                "Página": i + 1,
                "Similitud": round(float(sim), 4),
                "Consulta": consulta,
                "Tipo": "Semántica"
            })

    # Ordenar resultados por relevancia si hay semántica o heurística
    resultados.sort(key=lambda x: x.get("Ocurrencias", x.get("Similitud", 0)), reverse=True)
    return resultados
