from __future__ import annotations
import re
import warnings
from pathlib import Path
from typing import List
from config.load_config import load_config
import requests
from bs4 import BeautifulSoup
import yt_dlp

warnings.filterwarnings("ignore", category=FutureWarning)

# ════════════════════════════════════════════════
# Configuración (env vars + defaults)
# ════════════════════════════════════════════════
config = load_config()

AUDIO_DIR = Path(config["audio_dir"])
_DEFAULT_PODCAST_LIMIT = 3  
PODCAST_LIMIT_CONFIG = config.get("podcast_limit", _DEFAULT_PODCAST_LIMIT)

if not isinstance(PODCAST_LIMIT_CONFIG, int) or PODCAST_LIMIT_CONFIG < 0:
    print(f"Advertencia: El valor 'podcast_limit' de la configuración ('{PODCAST_LIMIT_CONFIG}') no es válido. Usando por defecto: {_DEFAULT_PODCAST_LIMIT}", flush=True)
    PODCAST_LIMIT_CONFIG = _DEFAULT_PODCAST_LIMIT

# ════════════════════════════════════════════════
# Descarga de programas de El Espejo Canario
# ════════════════════════════════════════════════
def download_espejocanario_podcasts(quantity: int = 0) -> List[Path]:
    if quantity == 0:
        print("Límite de podcasts establecido en 0. Omitiendo transcripción de podcasts.", flush=True)
        return

    print(f"Guardando hasta {quantity} programa(s) de podcast. Buscando programas en El Espejo Canario...", flush=True)
    base_url = "https://www.elespejocanario.es/programas/"
    downloaded_files = []

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(base_url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('article')
        
        if not articles:
            return []
        
        print(f"Se encontraron {len(articles)} artículos. Procesando hasta {quantity or 'todos'}.", flush=True)
        
        ivoox_links = []
        articulos_a_procesar = articles[:quantity] if quantity is not None else articles

        for i, article in enumerate(articulos_a_procesar):
            title_tag = article.find('h2', class_='entry-title')
            program_title = title_tag.a.text.strip() if title_tag and title_tag.a else f"Programa Desconocido {i+1}"
            
            iframe = article.find('iframe', {'src': lambda x: x and 'ivoox.com' in x})
            if iframe:
                src = iframe['src']
                match = re.search(r'player_ej_(\d+)', src)
                if match:
                    audio_id = match.group(1)
                    ivoox_url = f"https://www.ivoox.com/audios-mp3_rf_{audio_id}_1.html"
                    ivoox_links.append({"url": ivoox_url, "program_title": program_title})
                    print(f"  Encontrado enlace iVoox para '{program_title}'", flush=True)
                else:
                    print(f"  No se pudo extraer ID de iVoox del iframe para '{program_title}'", flush=True)
            else:
                print(f"  No se encontró iframe de iVoox para '{program_title}'.", flush=True)
        
        if len(ivoox_links) == 0:
            print("No se encontraron enlaces válidos de iVoox para descargar.", flush=True)
            return []
        
        print(f"\nIniciando descarga de { len(ivoox_links) } audio", flush=True)
        
        for item in ivoox_links:
            url = item["url"]
            clean_title = re.sub(r'[\\/*?:"<>|]', '_', item["program_title"])
            clean_title = re.sub(r'\s+', '_', clean_title).strip('_')
            if len(clean_title) > 150:
                clean_title = clean_title[:150]

            output_template = AUDIO_DIR / f"{clean_title}.%(ext)s"

            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': str(output_template),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'restrictfilenames': True,
                'quiet': False,
                'verbose': False,
                'noprogress': True,
                'noplaylist': True,
            }
            
            print(f"  Intentando descargar: '{item['program_title']}'", flush=True)
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    nombre_archivo_esperado_mp3 = AUDIO_DIR / f"{clean_title}.mp3"

                    if nombre_archivo_esperado_mp3.exists():
                        downloaded_files.append({ "name": clean_title, "path": nombre_archivo_esperado_mp3 })
                        print(f"    Audio descargado y convertido a MP3: {nombre_archivo_esperado_mp3.name}", flush=True)
                    else:
                        print(f"    Descarga completada para '{item['program_title']}', pero el archivo MP3 esperado ({nombre_archivo_esperado_mp3.name}) no se encontró directamente. Verificar manualmente.", flush=True)

            except yt_dlp.utils.DownloadError as e_dl:
                print(f"    Error de descarga de yt-dlp para '{item['program_title']}': {e_dl}", flush=True)
            
            except Exception as e:
                print(f"    Error inesperado durante la descarga de '{item['program_title']}': {e}", flush=True)
        
        if len(downloaded_files) == 0:
            print("\nNo se descargó ningún audio de podcast.", flush=True)
        else:
            print(f"\nDescarga de podcasts completada. Se obtuvieron { len(downloaded_files) } audios.", flush=True)
        
        return downloaded_files
    
    except requests.exceptions.RequestException as e_req:
        print(f"Error de conexión al buscar programas: {e_req}", flush=True)
        return []
    except Exception as e_main:
        print(f"Error general al obtener programas de podcast: {e_main}", flush=True)
        return []

# ════════════════════════════════════════════════
# Ejecución standalone
# ════════════════════════════════════════════════
if __name__ == "__main__":
    podcast_quantity = 1    
    print(f"Ejecutando script de descarga de podcasts directamente. Cantidad final a procesar: {podcast_quantity}", flush=True)
    download_espejocanario_podcasts(podcast_quantity)