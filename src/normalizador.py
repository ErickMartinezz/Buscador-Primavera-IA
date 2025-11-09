# normalizador.py
import unicodedata

def normalizar_texto(texto: str) -> str:
    """
    Normaliza el texto:
    - Pasa todo a minúsculas.
    - Elimina tildes y acentos.
    - Quita saltos de línea y espacios redundantes.
    """
    texto = texto.lower().strip()
    texto = ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )
    texto = ' '.join(texto.split())  # Quita espacios dobles o saltos de línea
    return texto


if __name__ == "__main__":
    ejemplo = "¡Hola! ¿Cómo estás? Árbol, Niño, Perú."
    print("Texto original:", ejemplo)
    print("Texto normalizado:", normalizar_texto(ejemplo))
