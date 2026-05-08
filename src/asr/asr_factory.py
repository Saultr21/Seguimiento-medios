from .whisper_asr import WhisperASR
from .nemo_asr import NemoASR

class ASRFactory:
    whisper_asr: None | WhisperASR = None
    nemo_asr: None | NemoASR = None
    
    @staticmethod
    def load_whisper_asr(config):
        if ASRFactory.whisper_asr: return ASRFactory.whisper_asr
        
        ASRFactory.whisper_asr = WhisperASR(config)
        return ASRFactory.whisper_asr

    @staticmethod
    def load_nemo_asr(config):
        if ASRFactory.nemo_asr: return ASRFactory.nemo_asr
        
        ASRFactory.nemo_asr = NemoASR(config)
        return ASRFactory.nemo_asr