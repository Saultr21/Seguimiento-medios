import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import logging
import pandas as pd
from pysentimiento import create_analyzer
from config.cargar_config import cargar_config

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

a_porcentaje = lambda p: f"{p * 100:.2f}%"  

# Diccionarios de traducción
SENTIMENT_TR = {"POS": "Positivo", "NEG": "Negativo", "NEU": "Neutral"}
EMOTION_TR = {
    "joy": "alegría",
    "anger": "ira",
    "fear": "miedo",
    "surprise": "sorpresa",
    "sadness": "tristeza",
    "disgust": "asco",
    "others": "otros",
}
HATE_TR = {"hateful": "odio", "targeted": "dirigido", "aggressive": "agresivo"}

# ────────────────────────────────────────────────────────────────────────────────
# Lectura y preparación de datos
# ────────────────────────────────────────────────────────────────────────────────

def cargar_fragmentos(path_json: Path, debug: bool = False) -> List[Dict[str, Any]]:
    """Convierte el JSON (fecha.edición > [frags]) en lista uniforme."""
    try:
        data = json.loads(path_json.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"¡Error! No se encontró {path_json}")
    except json.JSONDecodeError:
        raise SystemExit(f"¡Error! {path_json} no contiene JSON válido")

    fragmentos = [
        {"titulo": clave, "fragmento": i, "texto": txt}
        for clave, lista in data.items()
        for i, txt in enumerate(lista, 1)
        if txt.strip()
    ]

    if not fragmentos:
        raise SystemExit(f"¡Error! No se encontraron fragmentos en {path_json}")

    if debug:
        print(f"› Se han generado {len(fragmentos)} fragmentos a partir de {len(data)} llaves")
    return fragmentos

# ────────────────────────────────────────────────────────────────────────────────
# Analizadores
# ────────────────────────────────────────────────────────────────────────────────

def cargar_analizadores():
    print("Cargando analizadores…", end=" ")
    analizadores = {
        "sentiment": create_analyzer(task="sentiment", lang="es"),
        "emotion": create_analyzer(task="emotion", lang="es"),
        "hate": create_analyzer(task="hate_speech", lang="es"),
        "ner": create_analyzer(task="ner", lang="es"),
        "context_hate": create_analyzer(task="context_hate_speech", lang="es"),
        "targeted_sentiment": create_analyzer(task="targeted_sentiment", lang="es"),
    }
    print("Cargados.")
    return analizadores

# ────────────────────────────────────────────────────────────────────────────────
# Procesamiento de cada fragmento
# ────────────────────────────────────────────────────────────────────────────────

def analizar_fragmento(frag: Dict[str, Any], az, debug=False) -> Dict[str, Any]:
    txt = frag["texto"]

    snt = az["sentiment"].predict(txt)
    emo = az["emotion"].predict(txt)
    h8 = az["hate"].predict(txt)
    ner_pred = az["ner"].predict(txt)
    ctx_h8 = az["context_hate"].predict(txt)
    tsent = az["targeted_sentiment"].predict(txt)

    entidades = []
    for ent in getattr(ner_pred, "entities", []):
        if isinstance(ent, dict):
            entidades.append(f"{ent.get('text','')} ({ent.get('tag','')})")
        else:
            entidades.append(str(ent))

    out = {
        "titulo": frag["titulo"],
        "fragmento": frag["fragmento"],
        "texto": txt,
        "sentimiento": SENTIMENT_TR.get(snt.output, snt.output),
        "prob_pos": a_porcentaje(snt.probas["POS"]),
        "prob_neg": a_porcentaje(snt.probas["NEG"]),
        "prob_neu": a_porcentaje(snt.probas["NEU"]),
        "emocion": EMOTION_TR.get(emo.output, emo.output),
        "odio_detectado": ", ".join(HATE_TR.get(t, t) for t in h8.output) or "No",
        "prob_odio": a_porcentaje(h8.probas["hateful"]),
        "prob_dirigido": a_porcentaje(h8.probas["targeted"]),
        "prob_agresivo": a_porcentaje(h8.probas["aggressive"]),
        "entidades": ", ".join(entidades) if entidades else "No hay entidades",
        "odio_contextual": "Sí" if ctx_h8.output else "No",
        "prob_odio_contextual": a_porcentaje(ctx_h8.probas.get("HATE", 0)),
        "sentimiento_dirigido": str(tsent.output),
    }

    # Probabilidades de emociones
    for k, v in emo.probas.items():
        out[f"emo_{EMOTION_TR.get(k, k)}"] = a_porcentaje(v)

    # Probabilidades de sentimiento dirigido 
    if hasattr(tsent, "probas"):
        for k, v in tsent.probas.items():
            out[f"prob_sent_dirigido_{k}"] = a_porcentaje(v)

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
    frags = cargar_fragmentos(in_path, debug=args.debug)
    az = cargar_analizadores()
    resultados = [analizar_fragmento(f, az, debug=args.debug) for f in frags]
    df = pd.DataFrame(resultados)
    df.to_csv(out_path, index=False)
    print(f"\nAnálisis completado: {len(df)} fragmentos -> {out_path}\n")
    if args.debug:
        print(df.head())

def analizar_textos(input_file: str, output_file: str, debug: bool = False) -> None:
    frags = cargar_fragmentos(Path(input_file), debug=debug)
    az = cargar_analizadores()
    resultados = [analizar_fragmento(f, az, debug=debug) for f in frags]
    pd.DataFrame(resultados).to_csv(output_file, index=False)
    print(f"[analizar_textos] {len(resultados)} fragmentos -> {output_file}")

if __name__ == "__main__":
    main()
