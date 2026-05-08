from __future__ import annotations
import os
import re
import warnings
from pathlib import Path
from typing import List, Dict
import logging
from pytubefix import Channel, YouTube

from config.cargar_config import cargar_config
from utils.file_utils import clean_temp_files

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
# Utilidades
# ════════════════════════════════════════════════

def generate_filename(title: str) -> str:
    """Genera un nombre base limpio para archivos a partir del título de un vídeo."""
    m = re.search(r"Telenoticias\s+(\d+)\s+[|]??\s*(\d{2})/(\d{2})/(\d{2})", title)
    if m:
        num, dd, mm, yy = m.groups()
        base = f"{dd}-{mm}-{yy}.{num}"
    else:
        base = re.sub(r'[\\/*?:"<>|]', "", title)
        base = re.sub(r"\s+", "_", base).strip()
        if len(base) > 100:
            base = base[:100]
        if not base:
            base = f"video_descargado_{os.urandom(4).hex()}"
    return base

# ════════════════════════════════════════════════
# YouTube helpers
# ════════════════════════════════════════════════
def filter_channel_videos(channel_url: str, keyword: str, limit: int) -> List[Dict]:
    if not keyword or len(keyword.strip()) == 0:
        print(f"No se proporcionó palabra clave; tomando los últimos {limit} vídeos del canal...", flush=True)
        selected_videos = []
        channel = Channel(channel_url)
        for i, vid in enumerate(channel.videos):
            selected_videos.append({"titulo": vid.title, "video_url": vid.watch_url})
            if len(selected_videos) >= limit:
                break
        print(f"Se seleccionaron {len(selected_videos)} vídeos (ultimos del canal).", flush=True)
        return selected_videos

    print(f"Buscando hasta {limit} vídeos con ‘{keyword}’ en el título…", flush=True)
    
    selected_videos = []
    channel = Channel(channel_url)
    for i, vid in enumerate(channel.videos):
        if keyword.lower() in vid.title.lower():
            selected_videos.append({"titulo": vid.title, "video_url": vid.watch_url})
            if len(selected_videos) >= limit:
                break
        
        if i >= 99: # Máximo de 100 vídeos revisados.   
            break
    
    print(f"Se encontraron {len(selected_videos)} vídeos.", flush=True)
    return selected_videos

def download_audio(stream, base_name: str) -> Path | None:
    if stream is None:
        return
    
    tmp_name = f"tmp_{base_name}.mp3"
    out_path = AUDIO_DIR / f"{tmp_name}"

    try:
        stream.download(output_path=AUDIO_DIR, filename=tmp_name)
        return out_path
    except Exception as e:  
        print(f"Error al descargar audio: {e}")
        return

# ════════════════════════════════════════════════
# Flujo principal
# ════════════════════════════════════════════════
def download_yt_video(video_url: str) -> None:
    """
    Descarga un único vídeo de YouTube, especificado en el parámetro `video_url`.
    """

    if not video_url:
        print("No se proporcionó URL para vídeo único. Omitiendo.", flush=True)
        return None

    print(f"\nProcesando vídeo único desde URL: {video_url}", flush=True)
    try:
        yt_video = YouTube(video_url)
        titulo = yt_video.title
        
        base_name = generate_filename(titulo)
        print(f"  Procesando vídeo: {titulo}", flush=True)
        print(f"  Nombre base para archivos: {base_name}", flush=True)

        path_transcripcion_existente = TRANSCRIPCIONES_DIR / f"{base_name}.txt"
        if path_transcripcion_existente.exists():
            print(f"  La transcripción para '{base_name}' ya existe. Omitiendo.", flush=True)
            return base_name, mp3_path

        audio_stream = yt_video.streams.filter(only_audio=True).first()
        if not audio_stream:
            print(f"  No se encontró stream de audio para el vídeo: {titulo}. Se omite.", flush=True)
            return

        mp3_path = download_audio(audio_stream, base_name)

        if mp3_path is None:
            print(f"  Vídeo '{titulo}' sin audio o error de descarga; se omite.", flush=True)
            return

        return base_name, mp3_path

    except Exception as e:
        print(f"Error al obtener información del vídeo desde {mp3_path}: {e}", flush=True)
        return


def download_videos_from_channel(channel_url: str, keyword: str, limite_videos: int = 3):
    vids = filter_channel_videos(channel_url, keyword, limite_videos)
    print(f"\nProcesando {len(vids)} vídeo(s) del canal…", flush=True)

    downloaded_videos = []
    for i, vid in enumerate(vids):
        titulo = vid["titulo"]
        video_url = vid["video_url"]
        
        print(f"\n--- Vídeo {i+1}/{len(vids)} ---")
        try:
            base_name, mp3_path = download_yt_video(video_url)
            downloaded_videos.append({"name": base_name, "path": mp3_path})
        except Exception as e: 
            print(f"Error general al procesar el vídeo {titulo} del canal: {e}", flush=True)
            continue
    
    return downloaded_videos

# ════════════════════════════════════════════════
# Ejecución standalone
# ════════════════════════════════════════════════
if __name__ == "__main__":
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPCIONES_DIR.mkdir(parents=True, exist_ok=True)
    URL_CANAL = config["youtube_channel_url"]
    PALABRA = config["youtube_keyword"]
    LIMITE = config["video_limit"]
    download_videos_from_channel(URL_CANAL, PALABRA, LIMITE)
    clean_temp_files()