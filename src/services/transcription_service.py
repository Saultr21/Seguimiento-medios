from pathlib import Path
from asr.base_asr import BaseASR
from utils.text_utils import clean_text
from utils.file_utils import write_file

class TranscriptionService:
    def __init__(self, transcription_dir: str, transcription_model: BaseASR):
        self._transcription_dir = transcription_dir
        self._transcription_model = transcription_model

    def _save_transcription(self, transcription_dir: str, basename: str, text: str):
        path = Path(transcription_dir + f"/{basename}.txt")
        write_file(path, text)
        print(f"Transcripción guardada en: {path}", flush=True)

    def transcribe_audio(self, base_name: str, mp3_path: Path, language: str):
        """
        Procesa un único vídeo: toma el audio descargado, lo transcribe, limpia y guarda.
        """

        print(f"  Iniciando transcripción para: {base_name} (esto puede tardar)...", flush=True)
        try:
            transcribed_text = self._transcription_model.transcribe(str(mp3_path), audio_language=language)
            print(f"  Transcripción completada para: {base_name}.", flush=True)
            
            clean_text = clean_text(transcribed_text)
            self._save_transcription(self._transcription_dir, base_name, clean_text)

        except Exception as e:
            print(f"  Error durante la transcripción del vídeo {base_name}: {e}", flush=True)
        finally:
            if mp3_path and mp3_path.exists():
                try:
                    mp3_path.unlink()
                    print(f"  Archivo de audio temporal '{mp3_path.name}' eliminado.", flush=True)
                except Exception as e:
                    print(f"  No se pudo borrar el archivo de audio temporal {mp3_path.name}: {e}", flush=True)