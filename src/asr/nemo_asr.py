from omegaconf import DictConfig
import numpy as np
import subprocess
import torch
import os
import gc

from asr.base_asr import BaseASR
from config.torch_config import _resolve_device_and_dtype
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.models.aed_multitask_models import EncDecMultiTaskModel
from nemo.utils import logging as nemo_logging

nemo_logging.set_verbosity(nemo_logging.ERROR) # Solo se hará log de errores.

class _NemoASR(BaseASR):
    def __init__(self, config):
        self._model, self._device = self._load_model(config)
        self._warmup()
    
    def _load_model(self, config):
        # Carga del modelo.
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
        
        device, _, _ = _resolve_device_and_dtype()

        if "cuda" in device:
            model = model.half()
        else:
            model = model.bfloat16()
        
        model = model.to(device)
        model = model.eval()
        
        # Deshabilitamos el "almacenamiento de gradiente", que solo se usa para entrenamiento.
        for param in model.parameters():
            param.requires_grad = False

        model = torch.compile(model, mode="reduce-overhead")

        return model, device
    
    def _flush_memory(self):
        """Libera memoria RAM y VRAM tras operaciones pesadas."""
        gc.collect()  # Forzamos a funcionar al recolector de basura de Python.

        if "cuda" in self._device:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def _warmup(self):
        dummy = np.zeros(16000, dtype=np.float32)  # 1 segundo de silencio.
        self._model.transcribe(audio=[dummy], source_lang="es", target_lang="es", task="asr", pnc="no")
        self._flush_memory()

    def transcribe(self, audio_path: str, audio_language: str):
        """
        Transcribe audio usando el "pipeline". Si `forced_language` se proporciona (p.ej. 'english'), se calcula `forced_decoder_ids` localmente y se pasa a generate_kwargs para forzar el idioma.
        """
        
        if self._model is None:
            raise ValueError("Modelo no cargado.")
        
        self._to_mono(audio_path)

        is_cuda = "cuda" in self._device
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=is_cuda):
                print("Iniciando transcripción...")
                result = self._model.transcribe(
                    audio=[audio_path],
                    batch_size=1,
                    source_lang="es",  # Idioma de entrada
                    target_lang="es",  # Idioma de salida
                    task="asr",        # Tarea de reconocimiento de voz
                    pnc="yes",          # Incluir puntuación y mayúsculas
                    chunk_len_in_secs=20.0, # Chunks de 40 segundos con 4 segundos de solapamiento.
                    shift_len_in_secs=4.0,
                )

                print("Transcripción completada.")

                if is_cuda:
                    self._flush_memory()

                return result[0].text            

    def _to_mono(self, audio_path: str):
        """
        Para que el modelo de Canary procese correctamente el audio, necesita que esté en mono. Este método usa ffmpeg para realizar esta transformación.
        """

        try:
            print("Tratando de convertir archivo a mono...")

            tmp_audio_path = audio_path + ".tmp.mp3"
            result = subprocess.run(
                [ "ffmpeg", "-y", "-i", audio_path, "-ac", "1", tmp_audio_path ],
                capture_output=True,
                text=True,
                check=True
            )
            os.replace(tmp_audio_path, audio_path)

            print(f"Se ha convertido el archivo a mono: {result.returncode}.")
        except subprocess.CalledProcessError as e:
            print(f"FFMPEG ha fallado con el código de error {e.returncode}.")
            print(e.stderr)
        except Exception as e:
            print(f"Ha habido un fallo cambiando el formato del fichero {audio_path} a mono.")