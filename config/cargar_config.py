import json
from pathlib import Path

def cargar_config(ruta_config: str = "./config/config.json") -> dict:
    """Carga el archivo de configuración JSON."""
    config_path = Path(ruta_config)
    if not config_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {ruta_config}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)