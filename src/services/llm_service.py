from llm import LLMClient
from utils.file_utils import read_file, read_json_file, write_json_file

import re
import os

class LLMService:
    def __init__(self, llm_model: LLMClient):
        self.llm_model = llm_model
    
    def _chunking(self, text, chunk_size=850, overlap=150):
        """
        Crea chunks a partir de finales de frases para obtener texto reducido, con algo de overlap.
        """

        sentences = re.split(r'(?<=[.!?])\s+', text) # Busca caracteres de final de línea.
    
        chunks = []
        current = []
        current_len = 0

        for sentence in sentences:
            words = len(sentence.split())
            if current_len + words > chunk_size and current:
                chunks.append(" ".join(current))
                overlap_text = " ".join(current).split()[-overlap:] # Mantener las últimas palabras como "overlap".
                current = [" ".join(overlap_text)]
                current_len = len(overlap_text)
            current.append(sentence)
            current_len += words

        if current:
            chunks.append(" ".join(current))

        return chunks
    
    def _save_results(self, salida_json, origen_txt, fragmentos):
        try:
            key = os.path.splitext(os.path.basename(origen_txt))[0].lower() 

            data = read_json_file(salida_json) or {}
            data[key] = list(dict.fromkeys(data.get(key, []) + fragmentos))

            write_json_file(salida_json, data)
            print(f"Resultados añadidos a '{salida_json}' bajo la clave '{key}'.")
        except Exception as e:
            print(f"Problema encontrado al guardar los resultados: {e}")

    def _search_relevant_fragments(self, fragments, keywords, headers):
        keywords_str = ", ".join(keywords)
        
        system_prompt = """
            Eres un asistente que extrae menciones de entidades en un texto.
            - Devuelve solo los fragmentos donde aparezcan las entidades dadas.
            - Incluye referencias implícitas si son claras.
            - Los fragmentos recogidos deben terminar en un punto.
            - Separa contextos con: \n=====\n.
            - Si no hay menciones, reponde exactamente: NINGUNO.
            - Respuesta breve, sin explicaciones.

            Formato:
            - Frase 1
            =====
            - Frase 2
            ...
        """

        results = set()
        for idx, frag in enumerate(fragments, 1):
            try:
                print(f"Buscando fragmentos relevantes en el chunk [{ idx }/{ len(fragments) }].")
                user_prompt = (
                    f"Entidades a buscar: {keywords_str}\n"
                    f"Fragmento de transcripción: \n{frag}"
                )

                result = self.llm_model.call(user_prompt, system_prompt, headers)
                if result: 
                    for result_content in result:
                        result_split = result_content.split("\n=====\n")
                        results.update(split.strip() for split in result_split)
                        
            except Exception as e:
                print(f"Error procesando fragmento {idx}: {e}")
                continue

        if len(results) < 1:
            print("No se ha encontrado ninguna mención de las palabras clave.")
        
        return list(results)
    
    def analize_transcription(self, input_path, json_output_path, keywords):
        print(f"\nProcesando archivo: { os.path.basename(input_path) }")

        text = read_file(input_path)
        chunks = self._chunking(text)
        print(f"El texto ha sido dividido en { len(chunks) } chunks.")

        headers = { "Content-Type": "application/json" }
        relevant_fragments = self._search_relevant_fragments(chunks, keywords,  headers)
        
        if relevant_fragments:
            print(f"Fragmentos relevantes encontrados: { len(relevant_fragments) }")
            self._save_results(json_output_path, input_path, relevant_fragments)