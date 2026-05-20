from llm import LLMClient
from utils.file_utils import read_file, read_json_file, write_json_file
from pathlib import Path

import re
import os

class LLMService:
    """
    Servicio encargado de procesar transcripciones utilizando un modelo LLM.

    Para ello:
    - Divide el texto en fragmentos (chunks).
    - Busca framentos relevantes con un LLM.
    - Guarda los resultados (fragmentos) en un archivo JSON.
    """

    def __init__(self, llm_model: LLMClient):
        """
        Inicializa el servicio con un cliente LLM.

        Args:
            llm_model (LLMClient): cliente (wrapper) para solicitudes para el LLM utilizado.
        """
        self.llm_model = llm_model
    
    def _chunking(self, text: str, chunk_size=850, overlap=150):
        """
        Divide un texto en fragmentos (chunks), utilizando solo frases completas. Por tanto, no se cortan frases a la mitad, y se incluye algo de solapamiento por palabras para preservar contetxo.

        Args:
            text (str): texto de entrada.
            chunk_size (int): número aproximado de palabras por chunk.
            overlap (int): número de palabras que se solapan entre chunks.

        Returns:
            list[str]: lista de fragmentos de texto.
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
    
    def _save_results(self, json_output_path: Path, input_path: Path, fragments: list[str]):
        """
        Guarda los fragmentos encontrados en un archivo JSON.
        
        Si el archivo JSON ya existe, actualiza su contenido evitando duplicados.

        Args:
            json_output_path (Path): ruta del archivo JSON de salida.
            input_path (str | Path): ruta del archivo de entrada (se usa como clave).
            fragments (list[str]): fragmentos relevantes a guardar.
        """
        
        try:
            key = os.path.splitext(os.path.basename(input_path))[0].lower() 

            data = read_json_file(json_output_path) or {} # Si el archivo no tiene contenido, recogemos un diccionario vacío.
            data[key] = list(dict.fromkeys(data.get(key, []) + fragments))

            write_json_file(json_output_path, data)
            print(f"Resultados añadidos a '{json_output_path}' bajo la clave '{key}'.", flush=True)
        except Exception as e:
            print(f"Problema encontrado al guardar los resultados: {e}", flush=True)

    def _search_relevant_fragments(self, fragments: list[str], keywords: list[str], headers: dict):
        """
        Utiliza un modelo LLM para extraer fragmentos relevantes.
        
        El modelo recibe el texto como chunks y devuelve frases completas que contengan menciones relevantes.

        Args:
            fragments (list[str]): lista de chunks de texto.
            keywords (list[str]): palabras clave o entidades a buscar.
            headers (dict): encabezados para la llamada HTTP al LLM.

        Returns:
            list[str]: lista de fragmentos relevantes únicos.
        """

        keywords_str = ", ".join(keywords)
        
        system_prompt = """
            Eres un asistente que extrae menciones de entidades en un texto.
            - Devuelve solo los fragmentos donde aparezcan las entidades dadas.
            - Incluye referencias implícitas si son claras.
            - Los fragmentos deben ser frases completas, que no sean cortadas a la mitad.
            - Las frases deben empezar tras un punto y terminar en un punto.
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
                print(f"Buscando fragmentos relevantes en el chunk [{ idx }/{ len(fragments) }].", flush=True)
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
                print(f"Error procesando fragmento {idx}: {e}", flush=True)
                continue

        if len(results) < 1:
            print("No se ha encontrado ninguna mención de las palabras clave.", flush=True)
        
        return list(results)
    
    def analize_transcription(self, input_path: Path, json_output_path: Path, keywords: list[str]):
        """
        Método para llamada externa que procesa una transcripción completa:
        - Lee el archivo de texto.
        - Lo divide en chunks.
        - Extrae fragmentos relevantes usando un LLM.
        - Guarda los resultados en JSON.

        Args:
            input_path (Path): ruta del archivo de transcripción.
            json_output_path (Path): ruta del archivo JSON de salida.
            keywords (list[str]): lista de palabras clave o entidades a buscar.
        """

        print(f"\nProcesando archivo: { os.path.basename(input_path) }", flush=True)

        text = read_file(input_path)
        chunks = self._chunking(text)
        print(f"El texto ha sido dividido en { len(chunks) } chunks.", flush=True)

        headers = { "Content-Type": "application/json" }
        relevant_fragments = self._search_relevant_fragments(chunks, keywords,  headers)
        
        if len(relevant_fragments) == 0:
            print("No se ha encontrado ningún fragmento relevante. No se guardarán archivos.", flush=True)
            return
        
        print(f"Fragmentos relevantes encontrados: { len(relevant_fragments) }", flush=True)
        self._save_results(json_output_path, input_path, relevant_fragments)