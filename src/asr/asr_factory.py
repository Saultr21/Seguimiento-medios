class ASRFactory:
    """
    Clase de factoría para cargar y reutilizar instancias de modelos ASR. Permite que solo se deban cargar sus métodos una vez por aplicación.

    Motores de transcripción disponibles:
        - WhisperASR
        - NemoASR

    Notas:
        - Utiliza carga "lazy" para evitar que se carguen importaciones pesadas al inicio.
    """

    whisper_asr = None
    nemo_asr = None
    
    @staticmethod
    def load_whisper_asr(config):
        """
        Devuelve la instancia del modelo de Whisper, cargando la instancia estática de ser necesario.

        Args:
            config (dict):
                Diccionario de configuración necesario para inicializar los modelos.

        Returns:
            WhisperASR:
                Instancia inicializada del modelo de Whisper.
        """

        from .whisper_asr import WhisperASR
        if ASRFactory.whisper_asr: return ASRFactory.whisper_asr
        
        ASRFactory.whisper_asr = WhisperASR(config)
        return ASRFactory.whisper_asr

    @staticmethod
    def load_nemo_asr(config):
        """
        Devuelve la instancia del modelo de NeMo, cargando la instancia estática de ser necesario.

        Args:
            config (dict):
                Diccionario de configuración necesario para inicializar los modelos.

        Returns:
            NemoASR:
                Instancia inicializada del modelo de NeMo.
        """
        from .nemo_asr import NemoASR
        if ASRFactory.nemo_asr: return ASRFactory.nemo_asr
        
        ASRFactory.nemo_asr = NemoASR(config)
        return ASRFactory.nemo_asr