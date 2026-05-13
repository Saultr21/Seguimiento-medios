from pathlib import Path
from asr.base_asr import BaseASR
from utils.text_utils import clean_text
from utils.file_utils import write_file

class TranscriptionService:
    """
    Servicio encargado de transcribir archivos de audio y gestionar su almacenamiento.

    Para ello:
    - Transcribe audio usando un modelo ASR.
    - Limpia y guarda la transcripción en disco.
    - Elimina archivos temporales de audio.
    """

    def __init__(self, transcription_dir: str, transcription_model: BaseASR):
        """
        Inicializa el servicio de transcripción.

        Args:
            transcription_dir (str): directorio donde se guardarán las transcripciones.
            transcription_model (BaseASR): modelo ASR encargado de transcribir audio. Hijo de clase `BaseASR`.
        """

        self._transcription_dir = transcription_dir
        self._transcription_model = transcription_model

    def _save_transcription(self, transcription_dir: str, basename: str, text: str):
        """
        Guarda una transcripción en un archivo de texto.

        Args:
            transcription_dir (str): directorio de salida.
            basename (str): nombre base del archivo (sin extensión).
            text (str): texto transcrito a guardar.
        """

        path = Path(transcription_dir + f"/{basename}.txt")
        write_file(path, text)
        print(f"Transcripción guardada en: {path}", flush=True)

    def transcribe_audio(self, base_name: str, mp3_path: Path, language: str):
        """
        Transcribe un archivo de audio, lo limpia y lo guarda como texto.

        Para ello:
        - Ejecuta el método `transcribe` del modelo ASR sobre el archivo de audio.
        - Limpia el texto obtenido.
        - Guarda la transcripción en disco.
        - Elimina el archivo de audio temporal si existe.

        Args:
            base_name (str): Nombre base del archivo (sin extensión).
            mp3_path (Path): Ruta al archivo de audio a transcribir.
            language (str): Idioma del audio para mejorar la precisión del ASR.
        """

        print(f"  Iniciando transcripción para: {base_name} (esto puede tardar)...", flush=True)
        try:
            transcribed_text = self._transcription_model.transcribe(str(mp3_path), audio_language=language)
            print(f"  Transcripción completada para: {base_name}.", flush=True)
            
            cleaned_text = clean_text(transcribed_text)
            self._save_transcription(self._transcription_dir, base_name, cleaned_text)

        except Exception as e:
            print(f"  Error durante la transcripción del vídeo {base_name}: {e}", flush=True)
        finally:
            if mp3_path and mp3_path.exists():
                try:
                    mp3_path.unlink()
                    print(f"  Archivo de audio temporal '{mp3_path.name}' eliminado.", flush=True)
                except Exception as e:
                    print(f"  No se pudo borrar el archivo de audio temporal {mp3_path.name}: {e}", flush=True)