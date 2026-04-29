from llm import LLMClient
from utils.file_utils import read_file, read_json_file, write_json_file

import re
import requests
import os

class LLMService:
    def __init__(self, llm_model: LLMClient):
        self.llm_model = llm_model

    def _search_mentions(self, texto, palabras_clave, contexto=500):
        palabras_regex = r"\b(" + "|".join(re.escape(p) + r"s?" for p in palabras_clave) + r")\b"
        pattern = re.compile(palabras_regex, re.IGNORECASE)
        
        vistos = set()
        resultados = []
        for m in pattern.finditer(texto):
            fragmento = texto[max(0, m.start() - contexto):min(len(texto), m.end() + contexto)].strip()
            if fragmento not in vistos:
                vistos.add(fragmento)
                resultados.append({
                    "mencion": m.group(),
                    "texto": fragmento
                })
        print(resultados)
        return resultados
    
    def _save_results(salida_json, origen_txt, fragmentos):
        clave = os.path.splitext(os.path.basename(origen_txt))[0].lower() 

        data = read_json_file(salida_json) | {}
        data[clave] = list(dict.fromkeys(data.get(clave, []) + fragmentos))

        write_json_file(salida_json, data)
        print(f"Resultados añadidos a '{salida_json}' bajo la clave '{clave}'.")

    def review_fragments(self, fragmentos, palabras_clave, headers):
        if not palabras_clave:
            raise ValueError("El parámetro 'palabras_clave' no puede estar vacío.")
        
        if not fragmentos:
            return []

        palabras_clave_str = ", ".join(palabras_clave)
        prompt = (
            "Tienes una lista de fragmentos de texto."
            f"Devuelve únicamente las oraciones que contengan alguna de las siguientes palabras clave: {palabras_clave_str}, o sus alteraciones inmediatas (plural, indivdual)."
            "No combines ni recortes oraciones."
            "Devuelve únicamente la lista de oraciones, sin explicaciones ni comentarios."
        )

        joined = "\n".join(f"- {f}" for f in fragmentos)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": joined}
            ],
            "temperature": 0.0
        }

        try:
            res = requests.post(self.url, headers=headers, json=payload, verify=False)
            if res.status_code == 200:
                content = res.json()["choices"][0]["message"]["content"]
                oraciones = list(dict.fromkeys(
                    line.strip("-•– ").strip()
                    for line in content.strip().split("\n") if line.strip()
                ))
                oraciones_filtradas = [
                    oracion for oracion in oraciones
                    if any(re.search(rf"\b{re.escape(kw)}\b", oracion, re.IGNORECASE) for kw in palabras_clave)
                ]
                return oraciones_filtradas
            else:
                print(f"Error HTTP {res.status_code} en revisión")
        except Exception as e:
            print(f"Error al llamar al LLM para revisión: {e}")

        return fragmentos
    
    def analize_transcription(self, input_path, palabras_clave, json_output_path, contexto=500):
        headers = { "Content-Type": "application/json" }

        palabras_clave_str = ", ".join(palabras_clave)
        system_prompt = (
            f"Devuelve todas las oraciones o fragmentos que contengan las palabras clave {palabras_clave_str}. Debe empezar en un punto y acabar en un punto."        
            "No recortes dentro de una oración, ni añadas ni quites ni modifiques nada."
            "Incluye todas las menciones, incluso si varias oraciones expresan ideas similares."
            "No filtres ni resumas. Devuelve todo lo relevante sin eliminar nada por parecer repetido."
            "Si no hay ninguna mención, responde exactamente 'NINGUNO'."
            "Devuelve únicamente el resultado sin explicaciones ni comentarios."
        )

        texto = read_file(input_path)
        fragmentos_crudos = self._search_mentions(texto, palabras_clave, contexto)
        print(f"\nProcesando archivo: { os.path.basename(input_path) }")
        print(f"Fragmentos encontrados: { len(fragmentos_crudos) }")

        if not fragmentos_crudos:
            print("No se encontraron palabras clave en el texto.")
            return

        fragmentos_llm = self.llm_model.call(fragmentos_crudos, headers, system_prompt)
        print(f"Contextos relevantes: { len(fragmentos_llm) }")

        fragmentos_finales = self.review_fragments(fragmentos_llm, palabras_clave,  headers)
        print(f"Fragmentos únicos después de revisión: { len(fragmentos_finales) }\n")

        if fragmentos_finales:
            self._save_results(json_output_path, input_path, fragmentos_finales)