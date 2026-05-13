import re

def clean_text(text):
    """
    Limpia y normaliza un texto, eliminando ruido y mejorando el formato.

    Realiza las siguientes transformaciones:
    - Elimina contenido entre corchetes (ej. referencias tipo [1], [nota]).
    - Normaliza saltos de línea múltiples a uno solo.
    - Corrige saltos de línea dentro de palabras, uniéndolas.
    - Elimina espacios innecesarios antes de signos de puntuación.
    - Reemplaza secuencias de puntos suspensivos largas por un espacio.

    Args:
        text (str): texto de entrada a limpiar.

    Returns:
        str: texto limpio y normalizado.
    """

    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\n+", "\n", text.strip())
    text = re.sub(r"(\w)\n(\w)", r"\1 \2", text)
    text = re.sub(r"\s+([.,;!?])", r"\1", text)
    text = re.sub(r"\.{3,}", " ", text)

    return text