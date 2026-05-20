from omegaconf import DictConfig
import numpy as np
import subprocess
import torch
import os
import gc

from asr.base_asr import BaseASR
from config.torch_config import resolve_device_and_dtype
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.models.aed_multitask_models import EncDecMultiTaskModel
from nemo.utils import logging as nemo_logging

nemo_logging.set_verbosity(nemo_logging.ERROR) # Solo se hará log de errores.

class NemoASR(BaseASR):
    """
    Clase que implementa la transcripción utilizando los modelos ASR de NeMo Toolkit (NVIDIA), como:
    - `canary-1b-flash`
    - `canary-1b-v2`
    """

    def __init__(self, config):
        """
        Inicializa el modelo de transcripción de NeMo.

        Args:
            config(dict):
                Diccionario de configuración, obtenido de `load_config`. Debe incluir 'nemo_model_url' para especificar el modelo de ASR.
        """
        
        self._model, self._device = self._load_model(config)
        self._warmup()
    
    def _load_model(self, config):
        """
        Inicializa y configura el modelo de ASR de NeMo.
        
        Args:
            config (dict):
                Diccionario de configuración, obtenido de `load_config`. Debe incluir 'nemo_model_url' para especificar el modelo de ASR.

        Returns:
            tuple:
                A tuple containing:
                    - model: modelo de NeMo ASR cargado
                    - device (str): dispositivo utilizado para la transcripción ("cuda" o "cpu")
        """

        model: EncDecMultiTaskModel = ASRModel.from_pretrained(model_name=config['nemo_model_url'])
        
        decoding_cfg = DictConfig({
            "strategy": "beam",
            "beam": {
                "search_type": "default",
                "beam_size": 1,
                "return_best_hypothesis": True,
                "preserve_alignments": False,
                "score_norm": False,
            }
        })
        model.change_decoding_strategy(decoding_cfg)
        
        device, _, _ = resolve_device_and_dtype()

        if "cuda" in device:
            model = model.half()
        else:
            model = model.float()
        
        model = model.to(device)
        model = model.eval()
        
        # Deshabilitamos el "almacenamiento de gradiente", que solo se usa para entrenamiento.
        for param in model.parameters():
            param.requires_grad = False
        
        if "cuda" in device:
            model = torch.compile(model, mode="reduce-overhead")

        return model, device
    
    def _flush_memory(self):
        """
        Libera memoria RAM y VRAM tras operaciones pesadas, llamando al recolector de basura y a limpiar el caché de CUDA.
        """

        gc.collect()  # Forzamos a funcionar al recolector de basura de Python.

        if "cuda" in self._device:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def _warmup(self):
        """
        Realiza una inferencia para "calentamiento" con un audio silencioso.

        Básicamente, se ejecuta tras cargar el modelo como primera inferencia para que las siguientes vayan más rápido.
        """

        dummy = np.zeros(16000, dtype=np.float32)  # 1 segundo de silencio.
        self._model.transcribe(audio=[dummy], source_lang="es", target_lang="es", task="asr", pnc="no")
        self._flush_memory()

    def transcribe(self, audio_path: str, audio_language: str) -> str:
        """
        Transcribe un archivo de audio utilizando un modelo de NeMo y devuelve el texto.

        Args:
            audio_path (str):
                Ruta al archivo de audio para transcribir.

            audio_language (str):
                Código del idioma hablado en el audio. Si está puesto como "automático", se pondrá en español, ya que la transcripción automática no funciona muy bien.

        Returns:
            str:
                La transcripción del texto.
        """
        
        if self._model is None:
            raise ValueError("Modelo no cargado.")
        
        self._to_mono(audio_path)

        is_cuda = "cuda" in self._device
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=is_cuda):
                print("Iniciando transcripción...", flush=True)
                if audio_language is None or audio_language == "spanish":
                    language = "es"
                else:
                    language = "en"

                result = self._model.transcribe(
                    audio=[audio_path],
                    batch_size=1,
                    source_lang=language,  # Idioma de entrada
                    target_lang=language,  # Idioma de salida
                    task="asr",        # Tarea de reconocimiento de voz
                    pnc="yes",          # Incluir puntuación y mayúsculas
                    chunk_len_in_secs=20.0, # Chunks de 20 segundos con 4 segundos de solapamiento.
                    shift_len_in_secs=4.0,
                )

                print("Transcripción completada.", flush=True)
                self._flush_memory()

                return result[0].text            

    def _to_mono(self, audio_path: str):
        """
        Método para transformar un audio a formato "mono".

        Para que el modelo de Canary procese correctamente el audio, necesita que esté en mono.
        Este método usa el comando de ffmpeg para realizar la transformación a un solo canal de audio.

        Args:
            audio_path (str):
                Ruta al archivo de audio para transcribir. Será sobreescrita con el audio en mono.
        """

        try:
            print("Tratando de convertir archivo a mono...", flush=True)

            tmp_audio_path = audio_path + ".tmp.mp3"
            result = subprocess.run(
                [ "ffmpeg", "-y", "-i", audio_path, "-ac", "1", tmp_audio_path ],
                capture_output=True,
                text=True,
                check=True,
                encoding="utf-8"
            )
            os.replace(tmp_audio_path, audio_path)

            print(f"Se ha convertido el archivo a mono: {result.returncode}.", flush=True)
        except subprocess.CalledProcessError as e:
            print(f"FFMPEG ha fallado con el código de error {e.returncode}.", flush=True)
            print(e.stderr, flush=True)
        except Exception as e:
            print(f"Ha habido un fallo cambiando el formato del fichero {audio_path} a mono.", flush=True)