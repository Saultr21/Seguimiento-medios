import requests

class LLMClient:
    def __init__(self, url: str, model: str):
        self.url = url
        self.model = model

    def call(self, fragmentos, headers, system_prompt):
        resultados = []
        for idx, frag in enumerate(fragmentos, 1):
            # Por el momento, iteramos a través de los fragmentos donde se ha encontrado contenido.
            # La idea sería pasar, con suerte, el texto completo con chunking.
            # ¿Habría que hacer solicitud a "v1/messages/count_tokens" para el tamaño de los chunks?
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": frag}
                ],
                "temperature": 0.0
            }
            
            res = requests.post(self.url, headers=headers, json=payload, verify=False)
            print(f"[{idx}/{len(fragmentos)}] HTTP {res.status_code}")

            if res.status_code == 200:
                try:
                    for choice in res.json().get('choices', []):
                        content = choice.get('message', {}).get('content', '').strip()
                        if content and content.upper() != "NINGUNO":
                            resultados.append(content)
                except Exception as e:
                    print(f"Error procesando fragmento {idx}: {e}")
                    continue
            
        return resultados