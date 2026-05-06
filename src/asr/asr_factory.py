from .whisper_asr import _WhisperASR
from .nemo_asr import _NemoASR

class ASRFactory:
    whisper_asr: None | _WhisperASR = None
    nemo_asr: None | _NemoASR = None
    
    @staticmethod
    def load_whisper_asr(config):
        if ASRFactory.whisper_asr: return ASRFactory.whisper_asr
        
        ASRFactory.whisper_asr = _WhisperASR(config)
        return ASRFactory.whisper_asr

    @staticmethod
    def load_nemo_asr(config):
        if ASRFactory.nemo_asr: return ASRFactory.nemo_asr
        
        ASRFactory.nemo_asr = _NemoASR(config)
        return ASRFactory.nemo_asr