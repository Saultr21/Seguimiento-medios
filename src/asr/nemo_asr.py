from asr.base_asr import BaseASR
from config.torch_config import _resolve_device_and_dtype

from omegaconf import DictConfig
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.models.aed_multitask_models import EncDecMultiTaskModel
import torch
import numpy as np

class _NemoASR(BaseASR):
    def __init__(self, config):
        self._model, self._device = self._load_model(config)
        self._warmup()
    
    def _load_model(self, config):
        # Carga del modelo.
        model: EncDecMultiTaskModel = ASRModel.from_pretrained(model_name="nvidia/canary-1b-v2")
        
        decoding_cfg = DictConfig({
            "strategy": "beam",
            "beam": {
                "search_type": "default",
                "beam_size": 1,
                "return_best_hypothesis": True
            }
        })
        model.change_decoding_strategy(decoding_cfg)
        
        device, _, _ = _resolve_device_and_dtype()
        model = model.to(device)

        model = model.eval()
        model = model.half()  # Coloca float16, en lugar de float32.
        model = torch.compile(model, mode="max-autotune")    

        return model, device
    
    def _warmup(self):
        dummy = np.zeros(16000, dtype=np.float32)  # 1 segundo de silencio.
        self._model.transcribe(audio=[dummy], source_lang="es", target_lang="es", task="asr", pnc="no")

    def transcribe(self, audio_path: str, audio_language: str):
        """
        Transcribe audio usando el "pipeline". Si `forced_language` se proporciona (p.ej. 'english'), se calcula `forced_decoder_ids` localmente y se pasa a generate_kwargs para forzar el idioma.
        """
        
        if self._model is None:
            raise ValueError("Model is not loaded")
        
        is_cuda = False
        if "cuda" in self._device:
            is_cuda = True
        
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=is_cuda):
                result = self._model.transcribe(
                    audio=["tmp/audios/output.mp3"],
                    source_lang="es",  # Idioma de entrada
                    target_lang="es",  # Idioma de salida (mismo para transcripción)
                    task="asr",        # Tarea de reconocimiento de voz
                    pnc="yes",          # Incluir puntuación y mayúsculas
                    chunk_len_in_secs=40.0,
                    shift_len_in_secs=4.0,
                )
                print(result)
                return result[0].text

if __name__ == "__main__":
    
    model = _NemoASR({})
    model.transcribe("", "")
