import torchaudio

from asr.base_asr import BaseASR
from config.torch_config import _resolve_device_and_dtype

from omegaconf import DictConfig
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.models.aed_multitask_models import EncDecMultiTaskModel

class _NemoASR(BaseASR):
    def __init__(self, config):
        self._model = self._load_model(config)
    
    def _load_model(self, config):
        # Carga del modelo.
        model: EncDecMultiTaskModel = ASRModel.from_pretrained(model_name="nvidia/canary-1b-v2")
        
        decoding_cfg = DictConfig({
            "strategy": "beam",
            "beam": {
                "search_type": "default",
                "width": 10,
                "alpha": 0.5,
                "beta": 1.0,
                "return_best_hypothesis": True
            }
        })
        model.change_decoding_strategy(decoding_cfg)
        
        device, _, _ = _resolve_device_and_dtype()
        model = model.to(device)
        if "cuda" in device:
            model = model.half()  # Coloca float16, en lugar de float32.
        
        return model

    
    def transcribe(self, audio_path: str, audio_language: str):
        """
        Transcribe audio usando el "pipeline". Si `forced_language` se proporciona (p.ej. 'english'), se calcula `forced_decoder_ids` localmente y se pasa a generate_kwargs para forzar el idioma.
        """
        
        if self._model is None:
            raise ValueError
        else:
            
            result = self._model.transcribe(
                audio="tmp/audios/tmp_Las_noticias_del_LUNES_4_de_MAYO_en_10_minutos_RTVE_Noticias copy.mp3",
                source_lang="es",  # Idioma de entrada
                target_lang="es",  # Idioma de salida (mismo para transcripción)
                task="asr",        # Tarea de reconocimiento de voz
                pnc="yes"          # Incluir puntuación y mayúsculas
            )
            print(result)

if __name__ == "__main__":
    model = _NemoASR({})
    model.transcribe("", "")
