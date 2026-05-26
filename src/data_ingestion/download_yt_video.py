import os
import re
import warnings
from pathlib import Path

from utils.text_utils import format_filename
from pytubefix import Channel, YouTube

warnings.filterwarnings("ignore", category=FutureWarning)

# ════════════════════════════════════════════════
# YouTube helpers
# ════════════════════════════════════════════════
def _generate_filename(title: str) -> str:
    """
    Genera un nombre base limpio para guardar archivos.

    Si el título coincide con el formato "Telenoticias <num> | dd/mm/yy" (Telenoticias), se transforma en: "dd-mm-yy.<num>"

    En cualquier otro caso:
    - elimina caracteres inválidos para archivos,
    - reemplaza espacios por "_",
    - corta títulos demasiado largos,
    - genera un nombre aleatorio si queda vacío.

    Args:
        title:
            Título original del vídeo.

    Returns:
        str:
            nombre base listo para usarse como archivo.
    """

    m = re.search( 
        r"Telenoticias\s+(\d+)\s+[|]??\s*(\d{2})/(\d{2})/(\d{2})", # Un patrón como "Telenoticias 123 | 01/02/24"
        title
    )

    if m:
        num, dd, mm, yy = m.groups()
        base = f"{dd}-{mm}-{yy}.{num}"
    else:
        base = format_filename(title)
    
    return base

# KNOWN ISSUE:
# pytubefix.Channel.videos puede devolver una lista vacía debido a un bug conocido de la librería:
# https://github.com/JuanBindez/pytubefix/issues/625
#
# Documentado en docs/known-issues.md -> pytubefix
def _filter_channel_videos(channel_url: str, keyword: str, limit: int) -> list[dict]:
    """
    Obtiene una cantidad de vídeos `limit` de un canal de YouTube, filtrando por una palabra clave `keyword`.

    Si se proporciona la palabra clave, filtra vídeos cuyo título contenga dicha palabra. En caso de que no, devuelve simplemente los últimos vídeos del canal.

    Args:
        channel_url:
            URL del canal de YouTube.
        
        keyword:
            Palabra clave a buscar en títulos. Puede ser `None`.

        limit:
            Número máximo de vídeos a devolver.

    Returns:
        list[dict]:
            Lista de diccionarios con estilo:
            {
                "titulo": str,
                "video_url": str
            }
    """

    if not keyword or len(keyword.strip()) == 0: # En caso de que no haya palabra clave, recogemos útlimos vídeos.
        print(f"No se proporcionó palabra clave; tomando los últimos {limit} vídeos del canal...", flush=True)
        selected_videos = []
        channel = Channel(channel_url)
        for i, vid in enumerate(channel.videos):
            selected_videos.append({"title": vid.title, "video_url": vid.watch_url})
            if len(selected_videos) >= limit:
                break
        print(f"Se seleccionaron {len(selected_videos)} vídeos (ultimos del canal).", flush=True)
        return selected_videos

    print(f"Buscando hasta {limit} vídeos con ‘{keyword}’ en el título…", flush=True)
    
    selected_videos = []
    channel = Channel(channel_url)
    for i, vid in enumerate(channel.videos):
        if keyword.lower() in vid.title.lower(): # Buscamos la palabra clave en el título.
            selected_videos.append({"title": vid.title, "video_url": vid.watch_url})
            if len(selected_videos) >= limit:
                break
        
        if i >= 99: # Máximo de 100 vídeos revisados.  
            break
    
    print(f"Se encontraron {len(selected_videos)} vídeos.", flush=True)
    return selected_videos

def _download_audio(yt_video: YouTube, base_name: str, audio_folder: Path) -> Path | None:
    """
    Busca y descarga un stream de audio de YouTube, guardándolo en un archivo.

    Args:
        yt_video:
            Objeto de `pytubefix` para representar un vídeo y sus detalles.

        base_name:
            Nombre base del archivo.

        audio_folder:
            Carpeta donde guardar el audio.

    Returns:
        Path | None:
            Ruta del archivo descargado o None si falla.
    """

    audio_stream = yt_video.streams.filter(only_audio=True).first()
    if not audio_stream:
        print(f"  No se encontró stream de audio para el vídeo: {yt_video.title}. Se omite.", flush=True)
        return
    
    tmp_name = f"tmp_{base_name}.mp3"
    out_path = audio_folder / f"{tmp_name}"

    try:
        audio_stream.download(output_path=audio_folder, filename=tmp_name) # Descarga y guarda el stream de audio.
        return out_path
    except Exception as e:  
        print(f"Error al descargar audio: {e}", flush=True)
        return

# ════════════════════════════════════════════════
# Flujo principal
# ════════════════════════════════════════════════
def download_yt_video(video_url: str, audio_folder: Path) -> tuple[str, Path] | None:
    """
    Descarga el audio de un único vídeo de YouTube, devolviendo la ruta del archivo local donde se almacena.

    Args:
        video_url:
            URL completa del vídeo.

        audio_folder:
            Carpeta de destino para el audio.

    Returns:
        tuple[str, Path] | None:
            Tupla con el `nombre_base` y `ruta_archivo`, o None si ocurre algún error.
    """

    if not video_url:
        print("No se proporcionó URL para vídeo único. Omitiendo.", flush=True)
        return None

    print(f"\nProcesando vídeo único desde URL: {video_url}", flush=True)
    try:
        yt_video = YouTube(video_url)
        title = yt_video.title
        
        base_name = _generate_filename(title)
        print(f"  Procesando vídeo: {title}", flush=True)
        print(f"  Nombre base para archivos: {base_name}", flush=True)

        mp3_path = _download_audio(yt_video, base_name, audio_folder)

        if mp3_path is None:
            print(f"  Vídeo '{title}' sin audio o error de descarga; se omite.", flush=True)
            return

        return base_name, mp3_path

    except Exception as e:
        print(f"Error al obtener información del vídeo desde {mp3_path}: {e}", flush=True)
        return

def download_videos_from_channel(channel_url: str, keyword: str, audio_folder: Path, limit: int = 3) -> list[dict]:
    """
    Descarga múltiples vídeos desde un canal hasta `limit`, filtrando por una palabra clave `keyword`, guardándolos en local.

    Args:
        channel_url:
            URL del canal.
        
        keyword:
            Palabra clave para filtrar títulos de vídeos. Puede ser nulo.
        
        audio_folder:
            Carpeta destino para los audios.
        
        limit:
            Número máximo de vídeos a descargar. Por defecto, 3.

    Returns:
        list[dict]:
            Lista con diccionarios para identificar vídeos descargados (contienen variables `name` y `path`).
    """

    vids = _filter_channel_videos(channel_url, keyword, limit)
    print(f"\nProcesando {len(vids)} vídeo(s) del canal…", flush=True)

    downloaded_videos = []
    for i, vid in enumerate(vids):
        title = vid["title"]
        video_url = vid["video_url"]
        
        print(f"\n--- Vídeo {i+1}/{len(vids)} ---", flush=True)
        try:
            base_name, mp3_path = download_yt_video(video_url, audio_folder)
            downloaded_videos.append({"name": base_name, "path": mp3_path})
        except Exception as e: 
            print(f"Error general al procesar el vídeo {title} del canal: {e}", flush=True)
            continue
    
    return downloaded_videos

# ════════════════════════════════════════════════
# Ejecución standalone
# ════════════════════════════════════════════════
if __name__ == "__main__":
    download_yt_video(
        "https://www.youtube.com/watch?v=Bedcrn0BaZk",
        Path("tmp/audios/")
    )