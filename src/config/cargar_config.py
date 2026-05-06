from dotenv import load_dotenv
import os

load_dotenv()

def cargar_config() -> dict:
    """Carga el archivo de configuración en ".env"."""

    return {
        "audio_dir": os.getenv("AUDIOS_DIR"),
        "transcripciones_dir": os.getenv("TRANSCRIPTIONS_DIR"),
        "json_output_path": os.getenv("JSON_OUTPUT_PATH"),
        "csv_output_path": os.getenv("CSV_OUTPUT_PATH"),
        "whisper_model_url": os.getenv("WHISPER_MODEL_URL"),
        "nemo_model_url": os.getenv("NEMO_MODEL_URL"),
        "llm_url": os.getenv("LLM_URL"),
        "llm_model": os.getenv("LLM_MODEL"),
        "podcast_limit": int(os.getenv("PODCAST_LIMIT"))
    }