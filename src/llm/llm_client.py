import requests
import json

class LLMClient:
    def __init__(self, url: str, model: str):
        self.url = url
        self.model = model

    def call(self, user_prompt: str, system_prompt: str, headers):
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