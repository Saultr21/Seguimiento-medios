from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from pysentimiento import create_analyzer

from utils.file_utils import read_json_file

# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────
# Convierte probabilidades decimales a porcentajes con 2 decimales.
to_percentage = lambda p: f"{p * 100:.2f}%"  

# Diccionarios de traducción
SENTIMENT_DICT = {"POS": "Positivo", "NEG": "Negativo", "NEU": "Neutral"}
EMOTION_DICT = {
    "joy": "alegría",
    "anger": "ira",
    "fear": "miedo",
    "surprise": "sorpresa",
    "sadness": "tristeza",
    "disgust": "asco",
    "others": "otros",
}
HATE_DICT = {"hateful": "odio", "targeted": "dirigido", "aggressive": "agresivo"}

# ────────────────────────────────────────────────────────────────────────────────
# Lectura y preparación de datos
# ────────────────────────────────────────────────────────────────────────────────
def _load_fragments(path_json: Path, debug: bool = False) -> List[Dict[str, Any]]:
    """
    Carga un archivo JSON con fragmentos de texto y los transforma en una lista de diccionarios, cada uno representando un fragmento identificado por el título de vídeo y un índice.

    Args:
        path_json (Path):
            Ruta al archivo JSON de entrada.
        
        debug (bool, optional):
            Permite añadir información adicional de debug para el método.

    Returns:
        list[dict]: 
            Lista de fragmentos con la estructura:
                {
                    "title": str,   # Título del vídeo del fragmento.
                    "index": int,   # Número de fragmento en el vídeo.
                    "text": str     # Contenido del fragmento.
                }
    """

    data = read_json_file(path_json)
    if not data:
        print(f"Error leyendo el archivo en {path_json}.", flush=True)

    fragments = []
    for title, text_list in data.items():
        for index, text in enumerate(text_list, start=1):
            if text.strip():
                fragment = {
                    "title": title,
                    "index": index,
                    "text": text,
                }
                fragments.append(fragment)

    if debug:
        print(f"› Se han generado { len(fragments) } fragmentos a partir de { len(data) } claves", flush=True)
    
    return fragments

# ────────────────────────────────────────────────────────────────────────────────
# Analizadores
# ────────────────────────────────────────────────────────────────────────────────
def _load_analyzers(language: str):
    """
    Inicializa y devuelve los analizadores de NLP de PySentimiento para diferentes tareas, incluyendo:
        - Análisis de sentimiento (positivo / negativo / neutro)
        - Emociones (alegría, tristeza, asco, otros...)
        - Hate speech (SÍ / NO y probabilidad de odio)
        - Contextual hate speech
        - Named Entity Recognition (NER, recoger palabras "clave" del texto)
        - Sentimiento dirigido (positivo / negativo / neutro)

    Returns:
        dict:
            Diccionario con los analizadores cargados, clave = nombre de tarea, valor = analizador.
    """

    print("Cargando analizadores…", end=" ", flush=True)
    analyzers = {
        "sentiment": create_analyzer(task="sentiment", lang="es"),
        "emotion": create_analyzer(task="emotion", lang="es"),
        "hate": create_analyzer(task="hate_speech", lang="es"),
        "ner": create_analyzer(task="ner", lang="es"),
        "context_hate": create_analyzer(task="context_hate_speech", lang="es"),
        "targeted_sentiment": create_analyzer(task="targeted_sentiment", lang="es"),
    }
    print("Analizadores cargados.")

    return analyzers

# ────────────────────────────────────────────────────────────────────────────────
# Procesamiento de cada fragmento
# ────────────────────────────────────────────────────────────────────────────────
def _analyze_fragments(frag: dict, analyzers, debug=False) -> dict:
    """
    Aplica los analizadores de NLP a cada fragmento de texto y devuelve un diccionario con resultados estandarizados y legibles.

    Args:
        frag (dict):
            Fragmento de texto con claves "title", "index" y "text" (método `_load_fragments()`).

        az (dict):
            Diccionario de analizadores cargados (método `_load_analyzers()`).
        
        debug (bool, optional):
            Permite añadir información adicional de debug para el método.

    Returns:
        dict[str, Any]:
            Diccionario con resultados del análisis, incluyendo:
            - sentimiento y probabilidades
            - emociones y probabilidades
            - detección de odio y sus probabilidades
            - entidades reconocidas
            - hate speech contextual
            - sentimiento dirigido y sus probabilidades
    """

    txt = frag["text"]

    sentiment = analyzers["sentiment"].predict(txt)
    emotion = analyzers["emotion"].predict(txt)
    hate = analyzers["hate"].predict(txt)
    ner = analyzers["ner"].predict(txt)
    context_hate = analyzers["context_hate"].predict(txt)
    targeted_sentiment = analyzers["targeted_sentiment"].predict(txt)

    entities = []
    for ent in getattr(ner, "entities", []):
        if isinstance(ent, dict):
            entities.append(f"{ent.get('text','')} ({ent.get('tag','')})")
        else:
            entities.append(str(ent))


    out = {
        "titulo": frag["title"],
        "fragmento": frag["index"],
        "texto": txt,
        "sentimiento": SENTIMENT_DICT.get(sentiment.output, sentiment.output),
        "prob_pos": to_percentage(sentiment.probas["POS"]),
        "prob_neg": to_percentage(sentiment.probas["NEG"]),
        "prob_neu": to_percentage(sentiment.probas["NEU"]),
        "emocion": EMOTION_DICT.get(emotion.output, emotion.output),
        "odio_detectado": ", ".join(HATE_DICT.get(t, t) for t in hate.output) or "No",
        "prob_odio": to_percentage(hate.probas["hateful"]),
        "prob_dirigido": to_percentage(hate.probas["targeted"]),
        "prob_agresivo": to_percentage(hate.probas["aggressive"]),
        "entidades": ", ".join(entities) if entities else "No hay entidades",
        "odio_contextual": "Sí" if context_hate.output else "No",
        "prob_odio_contextual": to_percentage(context_hate.probas.get("HATE", 0)),
        "sentimiento_dirigido": str(targeted_sentiment.output),
    }

    # Probabilidades de emociones
    for k, v in emotion.probas.items():
        out[f"emo_{EMOTION_DICT.get(k, k)}"] = to_percentage(v)

    # Probabilidades de sentimiento dirigido 
    if hasattr(targeted_sentiment, "probas"):
        for k, v in targeted_sentiment.probas.items():
            out[f"prob_sent_dirigido_{k}"] = to_percentage(v)

    if debug:
        print(f"  -> [{frag['titulo']} #{frag['fragmento']}] listo", flush=True)

    return out

# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────
def analyze_texts(input_file: str, output_file: str, language: str, debug: bool = False) -> None:
    """
    Función principal para analizar el archivo JSON de fragmentos de texto, aplicando los modelos de PySentimiento y guardando los resultados en CSV.

    Args:
        input_file (str):
            Ruta al archivo JSON de entrada.

        output_file (str):
            Ruta del archivo CSV de salida.

        debug (bool, optional):
            Permite añadir información adicional de debug para el método.
    """
    frags = _load_fragments(Path(input_file), debug=debug)
    analyzers = _load_analyzers(language)

    results = [ _analyze_fragments(frag, analyzers, debug=debug) for frag in frags ]
    pd.DataFrame(results).to_csv(output_file, index=False)

    print(f"[analizar_textos] {len(results)} fragmentos -> {output_file}", flush=True)