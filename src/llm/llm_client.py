import requests
import json

class LLMClient:
    def __init__(self, url: str, model: str):
        self.url = url
        self.model = model

    def call(self, user_prompt, system_prompt, headers):
        # Por el momento, iteramos a través de los fragmentos donde se ha encontrado contenido.
        # La idea sería pasar, con suerte, el texto completo con chunking.
        # ¿Habría que hacer solicitud a "v1/messages/count_tokens" para el tamaño de los chunks?
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt + "\n/no_think"}
            ],
            "temperature": 0.0,
            "max_tokens": 512
        }
            
        res = requests.post(self.url, headers=headers, json=payload, verify=False)
        if res.status_code == 200:
            result = []
            for choice in res.json().get('choices', []):
                content = choice.get('message', {}).get('content', '').strip()
                if content and content.upper() != "NINGUNO":
                    result.append(content)
                else:
                    return None
            return result
        else:
            print(f"Ha habido un problema procesando la solicitud: HTTP {res.status_code}")
            return None