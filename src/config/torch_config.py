import os
import torch
import re

def _resolve_device_and_dtype():
    """
    Devuelve (device_str, dtype_or_none, pipeline_device_int).
    pipeline_device_int es -1 para CPU o índice de CUDA (0,1,...)
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
            
            # Si torch detecta cuda disponible usamos float16 por ahorro de memoria
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            return f"cuda:{idx}", dtype, idx

    # Detección automática de GPU, en caso de que no se haya forzado el dispositivo.
    if torch.cuda.is_available():
        return "cuda:0", torch.float16, 0
    return "cpu", torch.float32, -1