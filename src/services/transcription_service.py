from pathlib import Path
from src.asr.base_asr import BaseASR
from src.utils.text_utils import limpiar_texto
from src.utils.file_utils import guarda_transcripcion

class TranscriptionService:
    def __init__(self, transcription_dir: str, transcription_model: BaseASR):
        self._transcription_dir = transcription_dir
        self._transcription_model = transcription_model

    def transcribe_audio(self, base_name: str, mp3_path: Path, language: str):
        """
        Procesa un único vídeo: toma el audio descargado, lo transcribe, limpia y guarda.
        """

        print(f"  Iniciando transcripción para: {base_name} (esto puede tardar)...", flush=True)
        try:
            texto_transcrito = self._transcription_model.transcribe(str(mp3_path), audio_language=language)
            print(f"  Transcripción completada para: {base_name}.", flush=True)
            
            texto_limpio = limpiar_texto(texto_transcrito)
            guarda_transcripcion(self._transcription_dir, base_name, texto_limpio)

        except Exception as e:
            print(f"  Error durante la transcripción del vídeo {base_name}: {e}", flush=True)
        finally:
            if mp3_path and mp3_path.exists():
                try:
                    mp3_path.unlink()
                    print(f"  Archivo de audio temporal '{mp3_path.name}' eliminado.", flush=True)
                except Exception as e:
                    print(f"  No se pudo borrar el archivo de audio temporal {mp3_path.name}: {e}", flush=True)