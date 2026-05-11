from pathlib import Path
import json

def clean_temp_files(dir: Path) -> None:
    for f in dir.glob("*"):
        try:
            if f.is_file(): f.unlink()
        except Exception as e:
            print(f"No se pudo borrar el fichero {f}: {e}")

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def write_file(path, text: str):
    with open(path, 'w', encoding='utf-8') as f:
        return f.write(text)

def read_json_file(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def write_json_file(path, content):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(content, f, ensure_ascii=False)