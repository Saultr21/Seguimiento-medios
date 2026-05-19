import sys
import io  
from pathlib import Path
import shutil  
from typing import List
import argparse

from config.load_config import load_config
from asr import ASRFactory
from llm import LLMClient
from services.llm_service import LLMService
from services.transcription_service import TranscriptionService
from data_ingestion.download_yt_video import download_videos_from_channel, download_yt_video
from data_ingestion.download_podcasts import download_espejocanario_podcasts
from utils.file_utils import clean_temp_files
from nlp.sentiment_analysis import analyze_texts  

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def clean_folder(path: Path):
    """Elimina todo el contenido de una carpeta, pero no la carpeta misma."""
    if path.exists() and path.is_dir():
        for item_path in path.iterdir():
            try:
                if item_path.is_file() or item_path.is_symlink():
                    item_path.unlink()
                elif item_path.is_dir():
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f"Error al eliminar {item_path}: {e}", flush=True)
    elif not path.exists():
        path.mkdir(parents=True, exist_ok=True)

def pipeline(
    channel_url: str,
    channel_keyword: str,
    video_limit: int,
    mention_keywords: List[str],
    podcast_limit: int,
    single_video_urls: List[str],
    only_transcribe: bool = False,
    language: str = "",
    asr_model: str = ""
):
    print("PROGRESS:0:Iniciando flujo de trabajo...", flush=True)
    current_progress = 0

    config = load_config()

    # Recogemos las rutas para los archivos.
    audio_folder = Path(config['audio_folder'])
    transcription_folder = Path(config["transcription_folder"])
    json_output_path = Path(config["json_output_path"])
    csv_output_path = Path(config["csv_output_path"])

    # Eliminar archivos de salida anteriores si existen (carpeta o fichero).
    clean_temp_files(transcription_folder)
    json_output_path.unlink(missing_ok=True)
    csv_output_path.unlink(missing_ok=True)

    # Carga de modelo y servicio ASR.
    match asr_model:
        case "whisper":
            transcription_model = ASRFactory.load_whisper_asr(config)
        case "nemo":
            transcription_model = ASRFactory.load_nemo_asr(config)
        case _:
            print("PROGRESS:100:No se ha especificado un modelo de transcripción. Terminando...", flush=True)
            return
        
    transcription_service = TranscriptionService(str(transcription_folder), transcription_model)

    # Carga de cliente de LLM.
    llm_model = LLMClient(config["llm_url"], config["llm_model"])
    llm_service = LLMService(llm_model)
    
    if not channel_url and video_limit > 0:
        print("Error: No se proporcionó la URL del canal de YouTube y se solicitó procesar videos del canal.", flush=True)
        print("PROGRESS:100:Flujo terminado con error.", flush=True)
        return
    
    # Si no hay palabra clave, aceptamos y en la función de filtrado se tomarán los últimos videos
    # (por compatibilidad con la nueva opción de procesar los últimos N del canal).
    
    if video_limit is None or not isinstance(video_limit, int) or video_limit < 0:
        print("Error: El límite de videos no es válido. Debe ser un entero mayor o igual a 0.", flush=True)
        print("PROGRESS:100:Flujo terminado con error.", flush=True)
        return
    
    if podcast_limit is None or not isinstance(podcast_limit, int) or podcast_limit < 0:
        print("Error: El límite de podcasts no es válido. Debe ser un entero mayor o igual a 0.", flush=True)
        print("PROGRESS:100:Flujo terminado con error.", flush=True)
        return
    
    if not only_transcribe and not single_video_urls and video_limit == 0 and podcast_limit == 0:
        print("Error: No se especificaron URLs de vídeos únicos, ni se configuró la descarga de vídeos de canal o podcasts. Nada que procesar.", flush=True)
        print("PROGRESS:100:Flujo terminado con error.", flush=True)
        return

    current_progress = 5
    print(f"PROGRESS:{current_progress}:Validaciones completadas.", flush=True)

    if not csv_output_path.parent.exists():
        csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"PROGRESS:{current_progress}:=== Flujo de Trabajo: Descargar, Transcribir, Extraer Contextos y Analizar Sentimientos ===", flush=True)

    # Paso Adicional: Descargar y transcribir vídeos únicos (si se proporcionan)
    if single_video_urls:
        print(f"\nPROGRESS:{current_progress}:=== Paso Adicional: Descargar y Transcribir Vídeos Únicos ===", flush=True)
        
        for i, video_url in enumerate(single_video_urls):
            if not video_url.strip():
                print(f"  URL de vídeo único vacía omitida (índice {i+1}).", flush=True)
                continue

            try:
                print(f"Descargando y procesando vídeo de YouTube especificado { i+1 }/{ len(single_video_urls) } ({ video_url })...", flush=True)
                base_name, mp3_path = download_yt_video(video_url, audio_folder)
                transcription_service.transcribe_audio(base_name, mp3_path, language)
            except Exception as e:
                print(f"  Error al procesar vídeo único en {video_url}: {e}", flush=True)
        
        current_progress += 5
        print(f"PROGRESS:{current_progress}:Descarga y transcripción de vídeos únicos completada (o intentada).", flush=True)

    else:
        print(f"PROGRESS:{current_progress}:No se proporcionaron URLs de vídeos únicos, omitiendo este paso.", flush=True)

    # Paso 1: Descargar y transcribir vídeos de YouTube
    if video_limit > 0:
        print(f"\nPROGRESS:{current_progress}:=== Paso 1: Descargar y Transcribir Vídeos de YouTube ({channel_keyword}, Límite: {video_limit}) ===", flush=True)
        downloaded_videos = download_videos_from_channel(channel_url, channel_keyword, video_limit)

        for i, video in enumerate(downloaded_videos):
            base_name, video_path = video['name'], video['path']
            print(f"Procesando vídeo del canal {channel_url} { i+1 }/{ len(downloaded_videos) } ({ base_name })...", flush=True)
            transcription_service.transcribe_audio(base_name, video_path, language)

        current_progress += 20
        print(f"PROGRESS:{current_progress}:Descarga y transcripción de YouTube completada.", flush=True)
    elif video_limit == 0:
        print(f"PROGRESS:{current_progress}:Límite de vídeos de canal establecido en 0. Omitiendo descarga de vídeos del canal.", flush=True)
            
    # Paso 1.5: Descargar y transcribir podcasts de El Espejo Canario
    if podcast_limit > 0:
        print(f"\nPROGRESS:{current_progress}:=== Paso 1.5: Descargar y Transcribir Podcasts (Límite: {podcast_limit}) ===", flush=True)
        downloaded_videos = download_espejocanario_podcasts(audio_folder, podcast_limit)

        for i, video in enumerate(downloaded_videos):
            base_name, video_path = video['name'], video['path']
            print(f"Procesando programa de El Espejo Canario { i+1 }/{ len(downloaded_videos) } ({ base_name })...", flush=True)
            
            transcription_service.transcribe_audio(base_name, video_path, "spanish")
        
        current_progress += 15
        print(f"PROGRESS:{current_progress}:Descarga y transcripción de podcasts completada (o intentada).", flush=True)
    else:
        print(f"PROGRESS:{current_progress}:Límite de podcasts establecido en 0. Omitiendo descarga de podcasts.", flush=True)
            
    # Paso de Mantenimiento: Formatear Nombres y Limpiar Temporales
    print(f"\nPROGRESS:{current_progress}:=== Paso de Mantenimiento: Limpiar Temporales ===", flush=True)
    try:
        clean_temp_files(Path(config['audio_folder']))
        print("Limpieza de archivos temporales de audio completada.", flush=True)
    except Exception as e:
        print(f"Error durante la limpieza de temporales: {e}", flush=True)
    current_progress += 5
    print(f"PROGRESS:{current_progress}:Formateo y limpieza completados.", flush=True)


    if only_transcribe:
        # Saltar extracción y análisis
        print(f"PROGRESS:{current_progress}:Modo 'Solo transcripción' activo. Se omiten extracción de contextos y análisis.", flush=True)
        current_progress = 95
        print(f"PROGRESS:{current_progress}:Transcripciones listas.", flush=True)
    else:
        # Paso 2: Extraer contextos con Window-Sliding
        print(f"\nPROGRESS:{current_progress}:=== Paso 2: Extraer Contextos con Window Sliding ===", flush=True)
        archivos_transcripcion = list(transcription_folder.glob("*.txt"))
        total_archivos = len(archivos_transcripcion)
        progreso_ws_base = current_progress
        progreso_ws_rango = 25

        if total_archivos > 0:
            for i, filename in enumerate(archivos_transcripcion):
                progreso_interno_ws = int(((i + 1) / total_archivos) * progreso_ws_rango)
                print(f"PROGRESS:{progreso_ws_base + progreso_interno_ws}:Procesando archivo de transcripción {i+1}/{total_archivos}: {filename.name}", flush=True)
                llm_service.analize_transcription(str(filename), json_output_path, mention_keywords)
        else:
            print("No hay archivos de transcripción para procesar en el Paso 2.", flush=True)
        current_progress += progreso_ws_rango
        print(f"PROGRESS:{current_progress}:Extracción de contextos completada.", flush=True)

        # Paso 3: Analizar sentimientos/emociones
        print(f"\nPROGRESS:{current_progress}:=== Paso 3: Analizar Sentimientos ===", flush=True)
        if Path(json_output_path).exists() and Path(json_output_path).stat().st_size > 0 :
            analyze_texts(input_file=str(json_output_path), output_file=str(csv_output_path), debug=False)
        else:
            print(f"El archivo JSON '{json_output_path}' no existe o está vacío. Omitiendo análisis de sentimientos.", flush=True)
        current_progress = 95
        print(f"PROGRESS:{current_progress}:Análisis de sentimientos completado.", flush=True)

    # Resumen final
    print("\nPROGRESS:100:=== Flujo de Trabajo Completado ===", flush=True)
    if only_transcribe:
        # En modo solo transcripción no generamos JSON/CSV de análisis
        print(f"Resultados: transcripciones guardadas en: {transcription_folder}", flush=True)
    else:
        # Marcar para frontend que el CSV está listo
        if Path(csv_output_path).exists() and Path(csv_output_path).stat().st_size > 0:
            # Solo notificamos CSV disponible cuando realmente se generó
            print(f"Resultados guardados en:\n- Fragmentos JSON: {json_output_path}\n- Análisis de Sentimientos CSV: {csv_output_path}\n", flush=True)
            print("CSV_AVAILABLE:1", flush=True)
        else:
            print("CSV_AVAILABLE:0", flush=True)

def run():
    parser = argparse.ArgumentParser(
        description="Pipeline de ejecución"
    )

    # Añadimos argumentos requeridos para la llamada con "subprocess".
    parser.add_argument("channel_url")
    parser.add_argument("channel_keyword")

    parser.add_argument("video_limit", type=int)

    parser.add_argument("mention_keywords_str", nargs="?", default="")
    parser.add_argument("podcast_limit", type=int)

    parser.add_argument("single_video_urls_str", nargs="?", default="")

    parser.add_argument("only_transcribe", type=int, default=0)
    parser.add_argument("language", default="")
    parser.add_argument("asr_model", default="")

    args = parser.parse_args()

    mention_keywords_list = [
        kw.strip()
        for kw in args.mention_keywords_str.split(",")
        if kw.strip()
    ] if args.mention_keywords_str else []

    single_video_urls_list = [
        url.strip()
        for url in args.single_video_urls_str.split(",")
        if url.strip()
    ] if args.single_video_urls_str else []

    only_transcribe = bool(args.only_transcribe)
    language = args.language or None

    pipeline(
        args.channel_url,
        args.channel_keyword,
        args.video_limit,
        mention_keywords_list,
        args.podcast_limit,
        single_video_urls_list,
        only_transcribe,
        language,
        args.asr_model
    )

if __name__ == "__main__":
    run()