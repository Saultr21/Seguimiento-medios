import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline
)

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

from .base_asr import BaseASR
from config.torch_config import resolve_device_and_dtype

class WhisperASR(BaseASR):
    """
    Clase que implementa la transcripción utilizando los modelos Whisper de OpenAI, como:
    - `whisper-large-v3`
    - `whisper-large-v3-turbo`
    """

    def __init__(self, config):
        """
        Inicializa el modelo de transcripción de Whisper.

        Args:
            config(dict):
                Diccionario de configuración, obtenido de `load_config`. Debe incluir 'nemo_model_url' para especificar el modelo de ASR.
        """
        self._model, self._device = self._load_model(config)
    
    def _load_model(self, config):
        """
        Inicializa y configura el modelo Whisper.
        
        Args:
            config (dict):
                Diccionario de configuración, obtenido de `load_config`. Debe incluir 'nemo_model_url' para especificar el modelo de ASR.

        Returns:
            tuple:
                A tuple containing:
                    - model: modelo de NeMo ASR cargado
                    - device (str): dispositivo utilizado para la transcripción ("cuda" o "cpu")
        """

        _DEVICE, _DTYPE, _PIPELINE_DEVICE = resolve_device_and_dtype()
        WHISPER_MODEL_ID = config["whisper_model_url"]

        try:
            model_kwargs = {"low_cpu_mem_usage": True, "use_safetensors": True}
            if _DTYPE is not None:
                model_kwargs["dtype"] = _DTYPE
            _MODEL = AutoModelForSpeechSeq2Seq.from_pretrained(WHISPER_MODEL_ID, **model_kwargs)

            # Intentar mover modelo al dispositivo (si falla, pipeline puede manejarlo).
            try:
                _MODEL = _MODEL.to(_DEVICE)
            except Exception:
                pass

            self._PROCESSOR = AutoProcessor.from_pretrained(WHISPER_MODEL_ID) # Guardamos el procesador como atributo porque lo necesitaremos para transcribir.
            _PROCESSOR = self._PROCESSOR

            ASR_PIPE = pipeline(
                "automatic-speech-recognition",
                model=_MODEL,
                tokenizer=_PROCESSOR.tokenizer,
                feature_extractor=_PROCESSOR.feature_extractor,
                device=_PIPELINE_DEVICE,
                dtype=_DTYPE
            )

            print(f"[INFO] torch.version: {torch.__version__}", flush=True)
            print(f"[INFO] cuda_available: {torch.cuda.is_available()}", flush=True)
            print(f"[INFO] dispositivo elegido: {_DEVICE} (pipeline_device={_PIPELINE_DEVICE})", flush=True)

            return ASR_PIPE, _DEVICE
        except Exception as e:
            print(f"[WARN] No se pudo inicializar el pipeline ASR: {e}", flush=True)
            return None, None
    
    def transcribe(self, audio_path: str, audio_language: str):
        """
        Transcribe un archivo de audio utilizando el modelo de Whisper y devuelve el texto.

        Args:
            audio_path (str):
                Ruta al archivo de audio para transcribir.

            audio_language (str):
                Código del idioma hablado en el audio.

        Returns:
            str:
                La transcripción del texto.
        """
        
        if self._model is None:
            raise ValueError
        else:
            _PROCESSOR = self._PROCESSOR
            try:
                try:
                    forced_ids_local = _PROCESSOR.get_decoder_prompt_ids(language=audio_language, task="transcribe")
                    kwargs = {"forced_decoder_ids": forced_ids_local} # Forzamos la transcripción a un idioma determinado.
                except Exception:
                    kwargs = {} # Si no se puede obtener forced ids, seguimos sin forzar el idioma.

                result = self._model(
                    str(audio_path),
                    chunk_length_s=30, # Chunks de 30 segundos.
                    stride_length_s=3, # Con 3 segundos de "solapamiento".
                    batch_size=4, # Batch size 4 es bueno para ejecuciones sin GPU.
                    return_timestamps=False,
                    **kwargs
                )

                return result["text"]
            except ValueError:
                # Reintenta con timestamps si sigue siendo muy largo
                result = self._model(str(audio_path), return_timestamps=True)
                return result["text"]