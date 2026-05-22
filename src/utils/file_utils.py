from pathlib import Path
import json

def clean_temp_files(dir: Path) -> None:
    """
    Elimina todos los archivos dentro de un directorio dado.

    Recorre todos los elementos del directorio y elimina únicamente los archivos (de forma no recursiva). Si ocurre un error al borrar un archivo, lo registra por consola.

    Args:
        dir (Path): directorio del cual se eliminarán los archivos.
    """
    for f in dir.glob("*"):
        try:
            if f.is_file(): f.unlink()
        except Exception as e:
            print(f"No se pudo borrar el fichero {f}: {e}", flush=True)

def read_file(path: Path) -> str:
    """
    Lee el contenido completo de un archivo de texto (.txt o similar).

    Args:
        path (Path): ruta del archivo a leer.

    Returns:
        str: contenido del archivo como texto.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def write_file(path, text: str):
    """
    Escribe texto plano en un archivo (.txt o similar), sobrescribiendo su contenido.

    Args:
        path (Path): ruta del archivo a escribir.
        text (str): texto que se escribirá en el archivo.
    """
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)

def read_json_file(path: Path) -> dict | list | None:
    """
    Lee un archivo JSON y devuelve su contenido como un objeto `dict` o `list`. Si el archivo no existe o contiene JSON inválido, devuelve None.

    Args:
        path (Path): Ruta del archivo JSON.

    Returns:
        dict | list | None: contenido del JSON o `None` si hay error.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def write_json_file(path: Path, content: dict | list):
    """
    Escribe un objeto `dict` o `list` en un archivo JSON.

    Args:
        path (Path): ruta del archivo JSON a escribir.
        content (dict | list): datos serializables a JSON.
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(content, f, ensure_ascii=False)