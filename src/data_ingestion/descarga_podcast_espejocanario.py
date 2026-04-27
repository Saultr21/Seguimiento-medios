from __future__ import annotations
import os
import re
import warnings
from pathlib import Path
from typing import List, Optional
from src.config.cargar_config import cargar_config
from src.utils.text_utils import limpiar_texto
from src.asr.asr_factory import ASRFactory
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
# Carga única del modelo Whisper
# ════════════════════════════════════════════════
transcription_model = ASRFactory.load_whisper_asr(config)

# ════════════════════════════════════════════════
# Utilidades
# ════════════════════════════════════════════════
def guarda_transcripcion_podcast(nombre_base: str, texto: str) -> None:
    path = TRANSCRIPCIONES_DIR / f"podcast_{nombre_base}.txt"
    path.write_text(texto, encoding="utf-8")
    print(f"  Transcripción de podcast guardada en: {path}", flush=True)


def limpiar_audios_descargados_podcast(archivos_descargados: List[Path]) -> None:
    for f_path in archivos_descargados:
        try:
            if f_path.exists():
                f_path.unlink()
        except Exception as e:
            print(f"  No se pudo borrar {f_path.name}: {e}", flush=True)


def _obtener_cantidad_valida(env_var_key: str, config_limit: int, fallback_default: int) -> int:
    """
    Obtiene y valida la cantidad de programas a procesar.
    Prioridad: Variable de entorno > Configuración > Fallback default.
    """
    cantidad_str = os.getenv(env_var_key)
    cantidad_final = config_limit  

    if cantidad_str is not None:
        try:
            cantidad_env = int(cantidad_str)
            if cantidad_env >= 0:
                print(f"Se usará la cantidad de la variable de entorno {env_var_key}: {cantidad_env}", flush=True)
                return cantidad_env
            else:
                print(f"Advertencia: {env_var_key} ('{cantidad_str}') es negativo. Se usará el límite de la configuración: {config_limit}", flush=True)
        except ValueError:
            print(f"Advertencia: {env_var_key} ('{cantidad_str}') no es un número válido. Se usará el límite de la configuración: {config_limit}", flush=True)
    elif config_limit != fallback_default:  
        print(f"No se encontró la variable de entorno {env_var_key}. Se usará el límite de la configuración: {config_limit}", flush=True)

    if not isinstance(cantidad_final, int) or cantidad_final < 0:
        print(f"Advertencia: La cantidad de configuración ('{cantidad_final}') no es válida. Usando el valor por defecto global: {fallback_default}", flush=True)
        return fallback_default

    return cantidad_final

# ════════════════════════════════════════════════
# Descarga de programas de El Espejo Canario
# ════════════════════════════════════════════════

def descargar_programas_espejo_canario(cantidad: Optional[int] = None) -> List[Path]:
    print("Buscando programas en El Espejo Canario...", flush=True)
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
        
        print(f"\nIniciando descarga de {len(ivoox_links_con_titulos)} audio", flush=True)
        
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
                        archivos_descargados_final.append(nombre_archivo_esperado_mp3)
                        print(f"    Audio descargado y convertido a MP3: {nombre_archivo_esperado_mp3.name}", flush=True)
                    else:
                        print(f"    Descarga completada para '{item['titulo_original']}', pero el archivo MP3 esperado ({nombre_archivo_esperado_mp3.name}) no se encontró directamente. Verificar manualmente.", flush=True)

            except yt_dlp.utils.DownloadError as e_dl:
                print(f"    Error de descarga de yt-dlp para '{item['titulo_original']}': {e_dl}", flush=True)
            except Exception as e:
                print(f"    Error inesperado durante la descarga de '{item['titulo_original']}': {e}", flush=True)
        
        if archivos_descargados_final:
            print(f"\nDescarga de podcasts completada. Se obtuvieron {len(archivos_descargados_final)} audios.", flush=True)
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
# Flujo principal
# ════════════════════════════════════════════════

def procesar_programas(cantidad: int) -> None:
    """Procesa una cantidad determinada de programas de podcast."""
    if cantidad == 0:
        print("Límite de podcasts establecido en 0. Omitiendo procesamiento de podcasts.", flush=True)
        return

    print(f"Procesando hasta {cantidad} programa(s) de podcast.", flush=True)
    archivos_audio_descargados = descargar_programas_espejo_canario(cantidad)
    
    if not archivos_audio_descargados:
        print("No se descargaron audios de podcast para procesar.", flush=True)
        return

    print(f"\nIniciando procesamiento de {len(archivos_audio_descargados)} audios de podcast descargados...", flush=True)
    for i, archivo_path in enumerate(archivos_audio_descargados):
        print(f"\nProcesando audio {i+1}/{len(archivos_audio_descargados)}: {archivo_path.name}", flush=True)
        
        nombre_base_transcripcion = archivo_path.stem
        
        print(f"  Iniciando transcripción para: {archivo_path.name} (esto puede tardar)...", flush=True)
        try:
            texto_transcrito = transcription_model.transcribe(archivo_path, "spanish")
        except Exception as e:
            print(f"    Error inesperado durante la transcripción de {archivo_path.name}: {e}", flush=True)
            texto_transcrito = None
        
        if texto_transcrito:
            print(f"  Transcripción completada para: {archivo_path.name}.", flush=True)
            texto_limpio = limpiar_texto(texto_transcrito)
            guarda_transcripcion_podcast(nombre_base_transcripcion, texto_limpio)
        else:
            print(f"  No se generó transcripción para {archivo_path.name} o la transcripción está vacía.", flush=True)
    
    print("\nProceso de podcasts completado.", flush=True)

# ════════════════════════════════════════════════
# Ejecución standalone
# ════════════════════════════════════════════════
if __name__ == "__main__":
    cantidad_a_procesar = _obtener_cantidad_valida(
        env_var_key="CANTIDAD_PROGRAMAS",
        config_limit=PODCAST_LIMIT_CONFIG,
        fallback_default=_DEFAULT_PODCAST_LIMIT
    )
        
    print(f"Ejecutando script de descarga de podcasts directamente. Cantidad final a procesar: {cantidad_a_procesar}", flush=True)
    procesar_programas(cantidad_a_procesar)