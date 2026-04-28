from pathlib import Path

def guardar_transcripcion(transcription_dir: str, nombre: str, texto: str) -> 0:
    path = Path(transcription_dir + f"/{nombre}.txt")
    path.write_text(texto, encoding="utf-8")
    print(f"Transcripción guardada en: {path}", flush=True)