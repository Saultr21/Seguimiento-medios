from .whisper_asr import _WhisperASR

class ASRFactory:
    whisper_asr: None | _WhisperASR = None
    
    @staticmethod
    def load_whisper_asr(config):
        if ASRFactory.whisper_asr: return ASRFactory.whisper_asr
        
        ASRFactory.whisper_asr = _WhisperASR(config)
        return ASRFactory.whisper_asr