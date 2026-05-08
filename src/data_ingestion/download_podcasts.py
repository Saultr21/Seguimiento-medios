from __future__ import annotations
import os
import re
import warnings
from pathlib import Path
from typing import List, Optional
from config.cargar_config import cargar_config
from asr.asr_factory import ASRFactory
import requests
from bs4 import BeautifulSoup
import yt_dlp

warnings.filterwarnings("ignore", category=FutureWarning)

# ════════════════════════════════════════════════
# Configuración (env vars + defaults)
# ════════════════════════════════════════════════
config = cargar_config()

AUDIO_DIR = Path(config["audio_dir"])
TRANSCRIPCIONES_DIR = Path(config["transcripciones_dir"])
_DEFAULT_PODCAST_LIMIT = 3  
PODCAST_LIMIT_CONFIG = config.get("podcast_limit", _DEFAULT_PODCAST_LIMIT)

if not isinstance(PODCAST_LIMIT_CONFIG, int) or PODCAST_LIMIT_CONFIG < 0:
    print(f"Advertencia: El valor 'podcast_limit' de la configuración ('{PODCAST_LIMIT_CONFIG}') no es válido. Usando por defecto: {_DEFAULT_PODCAST_LIMIT}", flush=True)
    PODCAST_LIMIT_CONFIG = _DEFAULT_PODCAST_LIMIT

for dir_path in [AUDIO_DIR, TRANSCRIPCIONES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════
# Descarga de programas de El Espejo Canario
# ════════════════════════════════════════════════
def download_espejocanario_podcasts(cantidad: int = 0) -> List[Path]:
    if cantidad == 0:
        print("Límite de podcasts establecido en 0. Omitiendo transcripción de podcasts.", flush=True)
        return

    print(f"Guardando hasta {cantidad} programa(s) de podcast. Buscando programas en El Espejo Canario...", flush=True)
    base_url = "https://www.elespejocanario.es/programas/"
    archivos_descargados_final = []

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
        
        print(f"Se encontraron {len(articles)} artículos. Procesando hasta {cantidad or 'todos'}.", flush=True)
        
        ivoox_links_con_titulos = []
        articulos_a_procesar = articles[:cantidad] if cantidad is not None else articles

        for i, article in enumerate(articulos_a_procesar):
            titulo_tag = article.find('h2', class_='entry-title')
            titulo_programa = titulo_tag.a.text.strip() if titulo_tag and titulo_tag.a else f"Programa Desconocido {i+1}"
            
            iframe = article.find('iframe', {'src': lambda x: x and 'ivoox.com' in x})
            if iframe:
                src = iframe['src']
                match = re.search(r'player_ej_(\d+)', src)
                if match:
                    audio_id = match.group(1)
                    ivoox_url = f"https://www.ivoox.com/audios-mp3_rf_{audio_id}_1.html"
                    ivoox_links_con_titulos.append({"url": ivoox_url, "titulo_original": titulo_programa})
                    print(f"  Encontrado enlace iVoox para '{titulo_programa}'", flush=True)
                else:
                    print(f"  No se pudo extraer ID de iVoox del iframe para '{titulo_programa}'", flush=True)
            else:
                print(f"  No se encontró iframe de iVoox para '{titulo_programa}'.", flush=True)
        
        if not ivoox_links_con_titulos:
            print("No se encontraron enlaces válidos de iVoox para descargar.", flush=True)
            return []
        
        print(f"\nIniciando descarga de { len(ivoox_links_con_titulos) } audio", flush=True)
        
        for item in ivoox_links_con_titulos:
            url = item["url"]
            titulo_limpio = re.sub(r'[\\/*?:"<>|]', '_', item["titulo_original"])
            titulo_limpio = re.sub(r'\s+', '_', titulo_limpio).strip('_')
            if len(titulo_limpio) > 150:
                titulo_limpio = titulo_limpio[:150]

            output_template = AUDIO_DIR / f"{titulo_limpio}.%(ext)s"

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
            
            print(f"  Intentando descargar: '{item['titulo_original']}'", flush=True)
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    nombre_archivo_esperado_mp3 = AUDIO_DIR / f"{titulo_limpio}.mp3"

                    if nombre_archivo_esperado_mp3.exists():
                        archivos_descargados_final.append({ "name": titulo_limpio, "path": nombre_archivo_esperado_mp3 })
                        print(f"    Audio descargado y convertido a MP3: {nombre_archivo_esperado_mp3.name}", flush=True)
                    else:
                        print(f"    Descarga completada para '{item['titulo_original']}', pero el archivo MP3 esperado ({nombre_archivo_esperado_mp3.name}) no se encontró directamente. Verificar manualmente.", flush=True)

            except yt_dlp.utils.DownloadError as e_dl:
                print(f"    Error de descarga de yt-dlp para '{item['titulo_original']}': {e_dl}", flush=True)
            
            except Exception as e:
                print(f"    Error inesperado durante la descarga de '{item['titulo_original']}': {e}", flush=True)
        
        if archivos_descargados_final:
            print(f"\nDescarga de podcasts completada. Se obtuvieron { len(archivos_descargados_final) } audios.", flush=True)
        else:
            print("\nNo se descargó ningún audio de podcast.", flush=True)
        
        return archivos_descargados_final
    
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