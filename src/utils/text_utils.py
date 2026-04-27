import re

def limpiar_texto(texto):
    texto = re.sub(r"\[.*?\]", "", texto)
    texto = re.sub(r"\n+", "\n", texto.strip())
    texto = re.sub(r"(\w)\n(\w)", r"\1 \2", texto)
    texto = re.sub(r"\s+([.,;!?])", r"\1", texto)
    texto = re.sub(r"\.{3,}", " ", texto)

    return texto