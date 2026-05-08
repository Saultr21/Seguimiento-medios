import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import logging
import pandas as pd
from pysentimiento import create_analyzer

from config.cargar_config import cargar_config
from utils.file_utils import read_json_file

logging.getLogger("transformers").setLevel(logging.ERROR)
config = cargar_config()

# ────────────────────────────────────────────────────────────────────────────────
# Utilidades CLI
# ────────────────────────────────────────────────────────────────────────────────
def parse_arguments() -> argparse.Namespace:
    """Procesa los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Analiza textos con pysentimiento."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=config["json_output_path"],
        help="Archivo JSON de entrada",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=config["csv_output_path"],
        help="Archivo CSV de salida",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activa mensajes de depuración detallados",
    )
    return parser.parse_args()

# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────

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
    """Convierte el JSON (fecha.edición > [frags]) en lista uniforme."""

    data = read_json_file(path_json)
    if not data:
        print(f"Error leyendo el archivo en {path_json}.")

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

    if len(fragments) == 0:
        raise SystemExit(f"¡Error! No se encontraron fragmentos en {path_json}")

    if debug:
        print(f"› Se han generado { len(fragments) } fragmentos a partir de {len(data)} llaves")
    return fragments

# ────────────────────────────────────────────────────────────────────────────────
# Analizadores
# ────────────────────────────────────────────────────────────────────────────────
def _load_analyzers():
    print("Cargando analizadores…", end=" ")
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
def _analyze_fragments(frag: Dict[str, Any], az, debug=False) -> Dict[str, Any]:
    txt = frag["text"]

    sentiment = az["sentiment"].predict(txt)
    emotion = az["emotion"].predict(txt)
    hate = az["hate"].predict(txt)
    ner = az["ner"].predict(txt)
    context_hate = az["context_hate"].predict(txt)
    targeted_sentiment = az["targeted_sentiment"].predict(txt)

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
        print(f"  -> [{frag['titulo']} #{frag['fragmento']}] listo")

    return out

# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_arguments()
    print("*" * 8 + " Iniciando análisis " + "*" * 8)
    in_path = Path(args.input)
    out_path = Path(args.output)
    frags = _load_fragments(in_path, debug=args.debug)
    analyzers = _load_analyzers()
    resultados = [_analyze_fragments(f, analyzers, debug=args.debug) for f in frags]
    df = pd.DataFrame(resultados)
    df.to_csv(out_path, index=False)
    print(f"\nAnálisis completado: {len(df)} fragmentos -> {out_path}\n")
    if args.debug:
        print(df.head())

def analyze_texts(input_file: str, output_file: str, debug: bool = False) -> None:
    frags = _load_fragments(Path(input_file), debug=debug)
    analyzers = _load_analyzers()

    results = [ _analyze_fragments(frag, analyzers, debug=debug) for frag in frags ]
    pd.DataFrame(results).to_csv(output_file, index=False)

    print(f"[analizar_textos] {len(results)} fragmentos -> {output_file}")

if __name__ == "__main__":
    main()
