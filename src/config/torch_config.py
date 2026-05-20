import os
import torch
import re

def resolve_device_and_dtype() -> tuple[str, torch.dtype, int]:
    """
    Determina automáticamente el dispositivo de ejecución y el tipo de dato óptimo.

    Para ello:
    - Detecta automáticamente disponibilidad de GPU CUDA.
    - Selecciona `float16` en GPU para reducir uso de memoria.
    - Usa `float32` en CPU para mantener compatibilidad.
    - Permite forzar un dispositivo mediante la variable de entorno `FORCE_DEVICE`.

    Variables de entorno soportadas:
        FORCE_DEVICE=cpu
        FORCE_DEVICE=cuda
        FORCE_DEVICE=cuda:0
        FORCE_DEVICE=cuda:1

    Returns:
        tuple[str, torch.dtype, int]:
            Compuesto por:
            - device_str:
                Cadena compatible con PyTorch (ej. "cpu", "cuda:0").

            - dtype:
                Tipo de dato recomendado para inferencia (`torch.float16` o `torch.float32`).

            - pipeline_device_int:
                Índice entero compatible con pipelines de Hugging Face:
                - `-1` para CPU
                - `0`, `1`, etc. para GPUs CUDA
    """

    # Variable de entorno para forzar dispositivo, como "FORCE_DEVICE=cuda:0" o "FORCE_DEVICE=cpu".
    FORCE_DEVICE = os.getenv("FORCE_DEVICE", "").strip()
    if FORCE_DEVICE:
        fd = FORCE_DEVICE.lower()
        if fd in ("cpu", "-1"):
            return "cpu", torch.float32, -1
        m = re.match(r"cuda[:]?([0-9]+)?", fd)
        if m:
            idx = int(m.group(1)) if m.group(1) is not None else 0
            
            # Como deberíamos tener CUDA, usamos float16 por ahorro de memoria
            return f"cuda:{idx}", torch.float16, idx

    # Detección automática de GPU, en caso de que no se haya forzado el dispositivo.
    if torch.cuda.is_available():
        return "cuda:0", torch.float16, 0
    return "cpu", torch.float32, -1