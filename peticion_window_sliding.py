import os
import re
import json
import requests
import urllib3
from config.cargar_config import cargar_config

# Desactiva advertencias por certificados SSL no verificados
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def leer_texto(path):
    with open(path, encoding='utf-8') as f:
        return f.read()
    
def buscar_menciones(texto, palabras_clave, contexto=500):
    palabras_regex = r"\b(" + "|".join(re.escape(palabra) + r"s?" for palabra in palabras_clave) + r")\b"
    pattern = re.compile(palabras_regex, re.IGNORECASE)
    posiciones = [(m.start(), m.end()) for m in pattern.finditer(texto)]
    return list(dict.fromkeys(
        texto[max(0, s - contexto):min(len(texto), e + contexto)].strip()
        for s, e in posiciones
    ))

def llamar_llm(fragmentos, url, headers, model, system_prompt):
    resultados = []
    for idx, frag in enumerate(fragmentos, 1):
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": frag}
            ],
            "temperature": 0.1
        }
        res = requests.post(url, headers=headers, json=payload, verify=False)
        print(f"[{idx}/{len(fragmentos)}] HTTP {res.status_code}")
        if res.status_code == 200:
            try:
                for choice in res.json().get('choices', []):
                    content = choice.get('message', {}).get('content', '').strip()
                    if content and content.upper() != "NINGUNO":
                        resultados.append(content)
            except Exception:
                continue
    return resultados

def revisar_fragmentos(fragmentos, url, headers, model, palabras_clave):
    if not fragmentos:
        return []

    if not palabras_clave:
        raise ValueError("El parámetro 'palabras_clave' no puede estar vacío.")

    palabras_clave_extendidas = []
    for palabra in palabras_clave:
        palabras_clave_extendidas.append(palabra)
        palabras_clave_extendidas.append(palabra + "s")

    palabras_clave_str = ", ".join(palabras_clave_extendidas)
    prompt = (
        f"Tienes una lista de fragmentos de texto. "
        f"Devuelve únicamente las oraciones que contengan alguna de las siguientes palabras clave: {palabras_clave_str}. "
        "No combines ni recortes oraciones. "
        "Devuelve únicamente la lista de oraciones, sin explicaciones ni comentarios."
    )

    joined = "\n".join(f"- {f}" for f in fragmentos)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": joined}
        ],
        "temperature": 0.0
    }

    try:
        res = requests.post(url, headers=headers, json=payload, verify=False)
        if res.status_code == 200:
            content = res.json()["choices"][0]["message"]["content"]
            oraciones = list(dict.fromkeys(
                line.strip("-•– ").strip()
                for line in content.strip().split("\n") if line.strip()
            ))
            oraciones_filtradas = [
                oracion for oracion in oraciones
                if any(re.search(rf"\b{re.escape(kw)}\b", oracion, re.IGNORECASE) for kw in palabras_clave_extendidas)
            ]
            return oraciones_filtradas
        else:
            print(f"Error HTTP {res.status_code} en revisión")
    except Exception as e:
        print(f"Error al llamar al LLM para revisión: {e}")

    return fragmentos

def guardar_resultados(salida_json, origen_txt, fragmentos):
    try:
        with open(salida_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    clave = os.path.splitext(os.path.basename(origen_txt))[0].lower()
    data[clave] = list(dict.fromkeys(data.get(clave, []) + fragmentos))

    with open(salida_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Resultados añadidos a '{salida_json}' bajo la clave '{clave}'.")

def procesar_archivo(input_path, palabras_clave, json_output_path, contexto=500):
    config = cargar_config()
    url = config["llm_url"]
    headers = {"Content-Type": "application/json"}
    model = "gemma-3-12b-it-qat"

    palabras_clave_str = ", ".join(palabras_clave)
    system_prompt = (
        f"Devuelve todas las oraciones o fragmentos que contengan las palabras clave {palabras_clave_str}. Debe empezar en un punto y acabar en un punto."        
        "No recortes dentro de una oración, ni añadas ni quites ni modifiques nada. "
        "Incluye todas las menciones, incluso si varias oraciones expresan ideas similares. "
        "No filtres ni resumas. Devuelve todo lo relevante sin eliminar nada por parecer repetido. "
        "Si no hay ninguna mención, responde exactamente 'NINGUNO'. "
        "Devuelve únicamente el resultado sin explicaciones ni comentarios."
    )

    texto = leer_texto(input_path)
    fragmentos_crudos = buscar_menciones(texto, palabras_clave, contexto)
    print(f"\nProcesando archivo: {os.path.basename(input_path)}")
    print(f"Fragmentos encontrados: {len(fragmentos_crudos)}")

    if not fragmentos_crudos:
        print("No se encontraron palabras clave en el texto.")
        return

    fragmentos_llm = llamar_llm(fragmentos_crudos, url, headers, model, system_prompt)
    print(f"Contextos relevantes: {len(fragmentos_llm)}")

    fragmentos_finales = revisar_fragmentos(fragmentos_llm, url, headers, model, palabras_clave)
    print(f"Fragmentos únicos después de revisión: {len(fragmentos_finales)}\n")

    if fragmentos_finales:
        guardar_resultados(json_output_path, input_path, fragmentos_finales)

def main(
    json_output_path=None,
    contexto=500,
    palabras_clave=None,
    input_path=None
):
    if not palabras_clave:
        raise ValueError("El parámetro 'palabras_clave' no puede estar vacío.")
    
    config = cargar_config()
    json_output_path = json_output_path or config["json_output_path"]
    input_path = input_path or config["transcripciones_dir"]
    
    if os.path.isfile(input_path):
        procesar_archivo(input_path, palabras_clave, json_output_path, contexto)
    elif os.path.isdir(input_path):
        for filename in os.listdir(input_path):
            if filename.endswith('.txt'):
                filepath = os.path.join(input_path, filename)
                procesar_archivo(filepath, palabras_clave, json_output_path, contexto)
    else:
        raise ValueError(f"La ruta proporcionada no es válida: {input_path}")

if __name__ == "__main__":
    # Ejemplo de uso:
    # Procesar todos los archivos en el directorio de transcripciones
    main(palabras_clave=["Fernando Clavijo"])

