from abc import ABC, abstractmethod

class BaseASR(ABC):
    """
    Clase base abstracta para modelos de reconociimento automático de habla (ASR).

    Sus subclases implementarán modelos específicos y funcionalidades de transcripción de audio.
    """

    @abstractmethod
    def _load_model(self, config):
        """
        Inicializa y configura el modelo de ASR.
        
        Este método debería ser implementado por las subclases para manejar lógica específica de carga de cada modelo.

        Args:
            config (dict):
                Diccionario de configuración, obtenido de `load_config`.

        Raises:
            NotImplementedError:
                Devuelve un error si no se ha implementado en su subclase.
        """
        pass
    
    @abstractmethod
    def transcribe(self, audio_path: str, audio_language: str) -> str:
        """
        Transcribe un archivo de audio y devuelve el texto.

        Este método debería ser implementado por las subclases para manejar lógica específica de transcripción de cada modelo.

        Args:
            audio_path (str):
                Ruta al archivo de audio para transcribir.

            audio_language (str):
                Código del idioma hablado en el audio.
        
        Returns:
            str:
                La transcripción del texto.

        Raises:
            NotImplementedError:
                Devuelve un error si no se ha implementado en su subclase.
        """
        pass