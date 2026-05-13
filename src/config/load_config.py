from dotenv import load_dotenv
import os

load_dotenv()

def load_config() -> dict:
    """
    Carga y devuelve la configuración de la aplicación desde variables de entorno. Para ello, se obtienen automáticamente desde el archivo `.env` utilizando `python-dotenv`.

    Variables esperadas:
        AUDIOS_DIR:
            Directorio donde se almacenan los audios descargados.

        TRANSCRIPTIONS_DIR:
            Directorio donde se guardan las transcripciones generadas.

        JSON_OUTPUT_PATH:
            Ruta del archivo JSON de salida.

        CSV_OUTPUT_PATH:
            Ruta del archivo CSV de salida.

        WHISPER_MODEL_URL:
            Endpoint o ruta del modelo Whisper.

        NEMO_MODEL_URL:
            Endpoint o ruta del modelo NeMo.

        LLM_URL:
            URL del endpoint del modelo LLM.

        LLM_MODEL:
            Nombre del modelo LLM utilizado.

    Returns:
        dict:
            Diccionario con la configuración cargada desde el entorno.
    """

    return {
        "audio_folder": os.getenv("AUDIOS_DIR"),
        "transcription_folder": os.getenv("TRANSCRIPTIONS_DIR"),
        "json_output_path": os.getenv("JSON_OUTPUT_PATH"),
        "csv_output_path": os.getenv("CSV_OUTPUT_PATH"),
        "whisper_model_url": os.getenv("WHISPER_MODEL_URL"),
        "nemo_model_url": os.getenv("NEMO_MODEL_URL"),
        "llm_url": os.getenv("LLM_URL"),
        "llm_model": os.getenv("LLM_MODEL")
    }