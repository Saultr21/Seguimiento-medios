class BaseASR:
    def _load_model(self, config):
        raise NotImplementedError
    
    def transcribe(self, audio_path: str, audio_language: str):
        raise NotImplementedError