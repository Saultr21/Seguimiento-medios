from __future__ import annotations
import os
import re
import warnings
from pathlib import Path
from typing import List, Dict
import logging
from src.asr.asr_factory import ASRFactory
from pytubefix import Channel, YouTube

from src.config.cargar_config import cargar_config

warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)

# ════════════════════════════════════════════════
# Configuración (env vars + defaults)
# ════════════════════════════════════════════════
config = cargar_config()

AUDIO_DIR = Path(config["audio_dir"])
TRANSCRIPCIONES_DIR = Path(config["transcripciones_dir"])
WHISPER_MODEL_ID = config["whisper_model_url"]

# ════════════════════════════════════════════════
# Carga del modelo (Whisper o cualquiera).
# ════════════════════════════════════════════════
transcription_model = ASRFactory.load_whisper_asr(config)

# ════════════════════════════════════════════════
# Utilidades
# ════════════════════════════════════════════════
def limpiar_texto(texto: str) -> str:
    """Normaliza transcripciones eliminando saltos y marcas."""
    texto = re.sub(r"\[.*?\]", "", texto)
    texto = re.sub(r"\n+", "\n", texto.strip())
    texto = re.sub(r"(\w)\n(\w)", r"\1 \2", texto)
    texto = re.sub(r"\s+([.,;!?])", r"\1", texto)
    texto = re.sub(r"\.{3,}", " ", texto)
    return texto

def guarda_transcripcion(nombre: str, texto: str) -> None:
    path = TRANSCRIPCIONES_DIR / f"{nombre}.txt"
    path.write_text(texto, encoding="utf-8")
    print(f"Transcripción guardada en: {path}", flush=True)

def limpiar_temporales() -> None:
    for f in AUDIO_DIR.glob("tmp_*.mp3"):
        try:
            f.unlink()
        except Exception as e:  # noqa: BLE001
            print(f"No se pudo borrar {f}: {e}")

def generar_nombre_base_para_video(titulo_video: str) -> str:
    """Genera un nombre base limpio para archivos a partir del título de un vídeo."""
    m = re.search(r"Telenoticias\s+(\d+)\s+[|]??\s*(\d{2})/(\d{2})/(\d{2})", titulo_video)
    if m:
        num, dd, mm, yy = m.groups()
        base = f"{dd}-{mm}-{yy}.{num}"
    else:
        base = re.sub(r'[\\/*?:"<>|]', "", titulo_video)
        base = re.sub(r"\s+", "_", base).strip()
        if len(base) > 100:
            base = base[:100]
        if not base:
            base = f"video_descargado_{os.urandom(4).hex()}"
    return base

# ════════════════════════════════════════════════
# YouTube helpers
# ════════════════════════════════════════════════
def filtrar_videos(channel_url: str, keyword: str, limite: int) -> List[Dict]:
    if not keyword or not keyword.strip():
        print(f"No se proporcionó palabra clave; tomando los últimos {limite} vídeos del canal...", flush=True)
        videos = []
        for i, vid in enumerate(Channel(channel_url).videos):
            videos.append({"titulo": vid.title, "yt": vid})
            if len(videos) >= limite:
                break
        print(f"Se seleccionaron {len(videos)} vídeos (ultimos del canal).", flush=True)
        return videos

    print(f"Buscando hasta {limite} vídeos con ‘{keyword}’ en el título…", flush=True)
    videos = []
    for vid in Channel(channel_url).videos:
        if keyword.lower() in vid.title.lower():
            videos.append({"titulo": vid.title, "yt": vid})
            if len(videos) >= limite:
                break
    print(f"Se encontraron {len(videos)} vídeos.", flush=True)
    return videos

def descargar_audio(stream, base_name: str) -> Path | None:
    if stream is None:
        return None
    tmp_name = f"tmp_{base_name}.mp3"
    out_path = AUDIO_DIR / tmp_name
    try:
        stream.download(output_path=AUDIO_DIR, filename=tmp_name)
        return out_path
    except Exception as e:  
        print(f"Error al descargar audio: {e}")
        return None

# ════════════════════════════════════════════════
# Flujo principal
# ════════════════════════════════════════════════
def _procesar_video_individual(yt_video: YouTube, base_name: str | None = None, forced_language: str | None = None) -> None:
    """
    Procesa un único vídeo: descarga audio, transcribe, limpia y guarda.
    Si base_name no se proporciona, se genera a partir del título del vídeo.
    """
    titulo = yt_video.title
    print(f"  Procesando vídeo: {titulo}", flush=True)

    base = base_name or generar_nombre_base_para_video(titulo)
    print(f"  Nombre base para archivos: {base}", flush=True)

    path_transcripcion_existente = TRANSCRIPCIONES_DIR / f"{base}.txt"
    if path_transcripcion_existente.exists():
        print(f"  La transcripción para '{base}' ya existe. Omitiendo.", flush=True)
        return

    audio_stream = yt_video.streams.filter(only_audio=True).first()
    if not audio_stream:
        print(f"  No se encontró stream de audio para el vídeo: {titulo}. Se omite.", flush=True)
        return

    mp3_path = descargar_audio(audio_stream, base)

    if mp3_path is None:
        print(f"  Vídeo '{titulo}' sin audio o error de descarga; se omite.", flush=True)
        return

    print(f"  Iniciando transcripción para: {base} (esto puede tardar)...", flush=True)
    try:
        texto_transcrito = transcription_model.transcribe(mp3_path, audio_language=forced_language)
        print(f"  Transcripción completada para: {base}.", flush=True)
        
        texto_limpio = limpiar_texto(texto_transcrito)
        guarda_transcripcion(base, texto_limpio)
    except Exception as e:
        print(f"  Error durante la transcripción del vídeo {titulo}: {e}", flush=True)
    finally:
        if mp3_path and mp3_path.exists():
            try:
                mp3_path.unlink()
                print(f"  Archivo de audio temporal '{mp3_path.name}' eliminado.", flush=True)
            except Exception as e:
                print(f"  No se pudo borrar el archivo de audio temporal {mp3_path.name}: {e}", flush=True)

def descargar_video_unico(video_url: str, forced_language: str | None = None) -> None:
    """Descarga, transcribe y guarda un único vídeo de YouTube.

    Parámetro opcional `forced_language` (p.ej. 'english'|'spanish') para forzar el idioma
    durante la transcripción.
    """
    if not video_url:
        print("No se proporcionó URL para vídeo único. Omitiendo.", flush=True)
        return None

    print(f"\nProcesando vídeo único desde URL: {video_url}", flush=True)
    try:
        yt_video = YouTube(video_url)
        _procesar_video_individual(yt_video, forced_language=forced_language)
    except Exception as e:
        print(f"Error al obtener información del vídeo desde {video_url}: {e}", flush=True)
        return None


def subs_whisper(channel_url: str, keyword: str, limite_videos: int = 3, forced_language: str | None = None) -> None:
    vids = filtrar_videos(channel_url, keyword, limite_videos)
    print(f"\nProcesando {len(vids)} vídeo(s) del canal…", flush=True)

    for i, info in enumerate(vids):
        titulo = info["titulo"]
        yt_video = info["yt"]
        print(f"\n--- Vídeo {i+1}/{len(vids)} ---")
        try:
            _procesar_video_individual(yt_video, forced_language=forced_language)
        except Exception as e: 
            print(f"Error general al procesar el vídeo {titulo} del canal: {e}", flush=True)
            continue 

# ════════════════════════════════════════════════
# Renombrado y formateo de .txt existentes 
# ════════════════════════════════════════════════

def renombrar_txt_antiguo(nombre_antiguo: str, nombre_nuevo: str):
    orig = TRANSCRIPCIONES_DIR / f"{nombre_antiguo}.txt"
    dest = TRANSCRIPCIONES_DIR / f"{nombre_nuevo}.txt"
    if orig.exists() and not dest.exists():
        orig.rename(dest)
        print(f"Renombrado: {orig} -> {dest}", flush=True)

def formatear_transcripciones(dry_run: bool = False):
    patron = re.compile(
        r"^Telenoticias\s+(?P<num>\d{1,3})\s+(?P<fecha>\d{6})\.txt$",
        re.IGNORECASE,
    )

    cambios = 0
    for archivo in TRANSCRIPCIONES_DIR.glob("*.txt"):
        m = patron.match(archivo.name)
        if not m:
            continue

        num, fecha = m.group("num"), m.group("fecha")
        nuevo = TRANSCRIPCIONES_DIR / f"Telenoticias{num}.{fecha[4:6]}-{fecha[2:4]}-{fecha[0:2]}.txt"
        if nuevo.exists():
            print(f"Ya existe {nuevo.name}, omitiendo.", flush=True)
            continue

        print(f"{archivo.name} -> {nuevo.name}", flush=True)
        if not dry_run:
            archivo.rename(nuevo)
            cambios += 1
        
    if not dry_run:
        print("Renombrados", cambios, "archivo(s).", flush=True)

# ════════════════════════════════════════════════
# Ejecución standalone
# ════════════════════════════════════════════════
if __name__ == "__main__":
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPCIONES_DIR.mkdir(parents=True, exist_ok=True)
    URL_CANAL = config["youtube_channel_url"]
    PALABRA = config["youtube_keyword"]
    LIMITE = config["video_limit"]
    subs_whisper(URL_CANAL, PALABRA, LIMITE)
    formatear_transcripciones()
    limpiar_temporales()