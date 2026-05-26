import re
import os

def clean_text(text) -> str:
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

def format_filename(base_name: str):
    formatted = re.sub(r'[\\/*?:"<>|]', "", base_name)
    formatted = re.sub(r"\s+", "_", formatted).strip()

    if len(formatted) > 100:
        formatted = formatted[:100]
    if not formatted:
        formatted = f"video_descargado_{os.urandom(4).hex()}"

    return formatted