from pathlib import Path

def guardar_transcripcion(transcription_dir: str, nombre: str, texto: str) -> 0:
    path = Path(transcription_dir + f"/{nombre}.txt")
    path.write_text(texto, encoding="utf-8")
    print(f"Transcripción guardada en: {path}", flush=True)

def clean_temp_files(AUDIO_DIR: Path) -> None:
    for f in AUDIO_DIR.glob("tmp_*.mp3"):
        try:
            f.unlink()
        except Exception as e:
            print(f"No se pudo borrar el fichero {f}: {e}")