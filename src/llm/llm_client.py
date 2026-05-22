import requests
import json

class LLMClient:
    """
    Cliente para interactuar con un modelo LLM vía API HTTP.

    Envía prompts en formato chat (system + user) y devuelve las respuestas filtradas del modelo.
    """

    def __init__(self, url: str, model: str):
        """
        Inicializa el cliente del LLM.

        Args:
            url (str): dirección HTTP (con endpoint) de la API del modelo.
            model (str): nombre del modelo a utilizar para el LLM.
        """

        self.url = url
        self.model = model

    def call(self, user_prompt: str, system_prompt: str, headers) -> list[str] | None:
        """
        Realiza una llamada al modelo LLM y devuelve las respuestas válidas.

        Para ello:
        - Construye el payload en formato chat.
        - Envía la petición HTTP POST.
        - Filtra respuestas vacías o con "NINGUNO".
        - Devuelve una lista de resultados válidos.

        Args:
            user_prompt (str): prompt del usuario.
            system_prompt (str): prompt del sistema (instrucciones del modelo).
            headers (dict): headers HTTP necesarios para la petición.

        Returns:
            list[str] | None: lista de respuestas del modelo o None si hay error HTTP.
        """

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "/nothink\n" + user_prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 4096
        }
            
        res = requests.post(self.url, headers=headers, json=payload, verify=False)
        if res.status_code == 200:
            result = []
            for choice in res.json().get('choices', []):
                content = choice.get('message', {}).get('content', '').strip()
                if content and content.upper() != "NINGUNO":
                    result.append(content)
                else:
                    continue
            return result
        else:
            print(f"Ha habido un problema procesando la solicitud: HTTP {res.status_code}", flush=True)
            return None