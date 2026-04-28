import sys
import io  
from pathlib import Path
import shutil  
from typing import List

from src.config.cargar_config import cargar_config
from src.data_ingestion.descarga_videos_yt import download_videos_from_channel, download_yt_video, limpiar_temporales, formatear_transcripciones
from src.services.transcription_service import TranscriptionService
from src.asr.asr_factory import ASRFactory
from src.data_ingestion.descarga_podcast_espejocanario import procesar_programas as procesar_podcasts
from src.llm.peticion_window_sliding import main as window_sliding_main
from src.nlp.analisis_pysentimiento_json import analizar_textos  

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def limpiar_carpeta(ruta_carpeta: Path):
    """Elimina todo el contenido de una carpeta, pero no la carpeta misma."""
    if ruta_carpeta.exists() and ruta_carpeta.is_dir():
        for item_path in ruta_carpeta.iterdir():
            try:
                if item_path.is_file() or item_path.is_symlink():
                    item_path.unlink()
                elif item_path.is_dir():
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f"Error al eliminar {item_path}: {e}", flush=True)
    elif not ruta_carpeta.exists():
        ruta_carpeta.mkdir(parents=True, exist_ok=True)

def flujo_completo(
    channel_url: str, 
    channel_keyword: str, 
    video_limit: int, 
    mention_keywords: List[str], 
    podcast_limit: int, 
    single_video_urls: List[str],
    transcripciones_dir_str: str,
    json_output_path_str: str,
    csv_output_path_str: str,
    only_transcribe: bool = False,
    whisper_language: str = "",
):
    # only_transcribe se recibe como parámetro opcional (bool)

    print("PROGRESS:0:Iniciando flujo de trabajo...", flush=True)
    current_progress = 0

    # Usar paths pasados como argumentos
    transcripciones_folder = Path(transcripciones_dir_str)
    json_output_path = Path(json_output_path_str)
    csv_output_path = Path(csv_output_path_str)

    # Eliminar archivos de salida anteriores si existen
    json_output_path.unlink(missing_ok=True)
    csv_output_path.unlink(missing_ok=True)

    limpiar_carpeta(transcripciones_folder)
    transcripciones_folder.mkdir(parents=True, exist_ok=True)

    # ════════════════════════════════════════════════
    # Carga del modelo de transcripción (Whisper o cualquiera).
    # ════════════════════════════════════════════════
    config = cargar_config()
    transcription_model = ASRFactory.load_whisper_asr(config)
    transcription_service = TranscriptionService(str(transcripciones_folder), transcription_model)
    
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

            print(f"  Procesando vídeo único {i+1}/{ len(single_video_urls) }: {video_url}", flush=True)
            try:
                base_name, mp3_path = download_yt_video(video_url)
                transcription_service.transcribe_audio(base_name, mp3_path, whisper_language)    
            except Exception as e:
                print(f"  Error al procesar vídeo único '{base_name}': {e}", flush=True)
        
        current_progress += 5
        print(f"PROGRESS:{current_progress}:Descarga y transcripción de vídeos únicos completada (o intentada).", flush=True)

    else:
        print(f"PROGRESS:{current_progress}:No se proporcionaron URLs de vídeos únicos, omitiendo este paso.", flush=True)

    # Paso 1: Descargar y transcribir vídeos de YouTube
    if video_limit > 0:
        print(f"\nPROGRESS:{current_progress}:=== Paso 1: Descargar y Transcribir Vídeos de YouTube ({channel_keyword}, Límite: {video_limit}) ===", flush=True)
        downloaded_videos = download_videos_from_channel(channel_url, channel_keyword, video_limit)

        for video in downloaded_videos:
            base_name, video_path = video['name'], video['path']
            transcription_service.transcribe_audio(base_name, video_path)

        current_progress += 20
        print(f"PROGRESS:{current_progress}:Descarga y transcripción de YouTube completada.", flush=True)
    elif video_limit == 0:
        print(f"PROGRESS:{current_progress}:Límite de vídeos de canal establecido en 0. Omitiendo descarga de vídeos del canal.", flush=True)
            
    # Paso 1.5: Descargar y transcribir podcasts de El Espejo Canario
    if podcast_limit > 0:
        print(f"\nPROGRESS:{current_progress}:=== Paso 1.5: Descargar y Transcribir Podcasts (Límite: {podcast_limit}) ===", flush=True)
        procesar_podcasts(podcast_limit)
        current_progress += 15
        print(f"PROGRESS:{current_progress}:Descarga y transcripción de podcasts completada (o intentada).", flush=True)
    else:
        print(f"PROGRESS:{current_progress}:Límite de podcasts establecido en 0. Omitiendo descarga de podcasts.", flush=True)
            
    # Paso de Mantenimiento: Formatear Nombres y Limpiar Temporales
    print(f"\nPROGRESS:{current_progress}:=== Paso de Mantenimiento: Formatear Nombres y Limpiar Temporales ===", flush=True)
    try:
        formatear_transcripciones()
        print("Formateo de nombres de transcripciones completado.", flush=True)
    except Exception as e:
        print(f"Error durante el formateo de transcripciones: {e}", flush=True)
    try:
        limpiar_temporales()
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
        archivos_transcripcion = list(transcripciones_folder.glob("*.txt"))
        total_archivos = len(archivos_transcripcion)
        progreso_ws_base = current_progress
        progreso_ws_rango = 25

        if total_archivos > 0:
            for i, filename in enumerate(archivos_transcripcion):
                progreso_interno_ws = int(((i + 1) / total_archivos) * progreso_ws_rango)
                print(f"PROGRESS:{progreso_ws_base + progreso_interno_ws}:Procesando archivo de transcripción {i+1}/{total_archivos}: {filename.name}", flush=True)
                window_sliding_main(
                    input_path=str(filename),
                    json_output_path=str(json_output_path),
                    palabras_clave=mention_keywords
                )
        else:
            print("No hay archivos de transcripción para procesar en el Paso 2.", flush=True)
        current_progress += progreso_ws_rango
        print(f"PROGRESS:{current_progress}:Extracción de contextos completada.", flush=True)

        # Paso 3: Analizar sentimientos/emociones
        print(f"\nPROGRESS:{current_progress}:=== Paso 3: Analizar Sentimientos ===", flush=True)
        if Path(json_output_path).exists() and Path(json_output_path).stat().st_size > 0 :
            analizar_textos(input_file=str(json_output_path), output_file=str(csv_output_path), debug=False)
        else:
            print(f"El archivo JSON '{json_output_path}' no existe o está vacío. Omitiendo análisis de sentimientos.", flush=True)
        current_progress = 95
        print(f"PROGRESS:{current_progress}:Análisis de sentimientos completado.", flush=True)

    # Resumen final
    print("\nPROGRESS:100:=== Flujo de Trabajo Completado ===", flush=True)
    if only_transcribe:
        # En modo solo transcripción no generamos JSON/CSV de análisis
        print(f"Resultados: transcripciones guardadas en: {transcripciones_folder}", flush=True)
    else:
        # Solo notificamos CSV disponible cuando realmente se generó
        print(f"Resultados guardados en:\n- Fragmentos JSON: {json_output_path}\n- Análisis de Sentimientos CSV: {csv_output_path}", flush=True)
        # Marcar para frontend que el CSV está listo
        if Path(csv_output_path).exists() and Path(csv_output_path).stat().st_size > 0:
            print("CSV_AVAILABLE:1", flush=True)
        else:
            print("CSV_AVAILABLE:0", flush=True)

if __name__ == "__main__":
    if len(sys.argv) < 10: # Se esperan 9 argumentos + el nombre del script
        print(f"Error: Faltan argumentos. Se esperaban 9, se recibieron {len(sys.argv)-1}", flush=True)
        print("Uso: ejecucion.py <channel_url> <channel_keyword> <video_limit> <transcripciones_dir> <json_output_path> <csv_output_path> <mention_keywords_str> <podcast_limit> [single_video_urls_str]", flush=True)
        sys.exit(1)
    channel_url_arg = sys.argv[1]
    channel_keyword_arg = sys.argv[2]
    video_limit_arg = int(sys.argv[3])
    # Argumentos de paths
    transcripciones_dir_arg = sys.argv[4]
    json_output_path_arg = sys.argv[5]
    csv_output_path_arg = sys.argv[6]
    mention_keywords_str_arg = sys.argv[7]
    mention_keywords_list = [kw.strip() for kw in mention_keywords_str_arg.split(",") if kw.strip()] if mention_keywords_str_arg else []
    podcast_limit_arg = int(sys.argv[8])
    single_video_urls_str_arg = sys.argv[9] if len(sys.argv) > 9 and sys.argv[9] else ""
    single_video_urls_list_arg = [url.strip() for url in single_video_urls_str_arg.split(",") if url.strip()] if single_video_urls_str_arg else []
    only_transcribe_arg = bool(int(sys.argv[10])) if len(sys.argv) > 10 else False
    whisper_language_arg = sys.argv[11] if len(sys.argv) > 11 else ""
    
    flujo_completo(
        channel_url_arg, 
        channel_keyword_arg, 
        video_limit_arg, 
        mention_keywords_list, 
        podcast_limit_arg, 
        single_video_urls_list_arg,
        transcripciones_dir_arg, 
        json_output_path_arg,    
        csv_output_path_arg      
        , only_transcribe=only_transcribe_arg,
        whisper_language=whisper_language_arg
    )