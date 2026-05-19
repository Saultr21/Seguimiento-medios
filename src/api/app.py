# Importaciones de configuración y frontend
from config.load_config import load_config
from .frontend import demo

import os
import asyncio
import subprocess
import sys
import io
import time
import argparse

import gradio as gr
import contextlib
from fastapi import FastAPI, Form
from fastapi.responses import StreamingResponse, FileResponse, RedirectResponse
from fastapi import Request
from fastapi.staticfiles import StaticFiles
from typing import List, Optional
import uvicorn

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True, encoding='utf-8')

# Cargar configuración
config = load_config()
streamlit_process = None # Variable global para el proceso de Streamlit.

def launch_streamlit():
    """
    Ejecuta el frontend de Streamlit en un subproceso, permitiendo su acceso y reinicio junto con este backend.
    Usa el puerto definido en la variable de entorno `VISUALIZATION_PORT`.
    """

    global streamlit_process
    visualization_port = os.getenv("VISUALIZATION_PORT")

    streamlit_process = subprocess.Popen(
        [
            sys.executable, "-m",
            "streamlit", "run", "src/api/visualization_frontend.py",
            "--server.port", visualization_port,
            "--server.headless", "true",
            "--theme.base", "dark"
        ],
        stdout = None
    )
# ------------------------------
# Lifespan de FastAPI
# ------------------------------
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI): # 
    """
    "lifespan" es una variable de FastAPI para añadir lógica de inicio y finalización a la aplicación.

    Utiliza un context manager asíncrono para FastAPI que le permite eejecutar código al iniciar y cerrar la app, separando ambas mitades con un `yield`.
    """

    # Lógica de inicio.
    launch_streamlit()
    yield # Tras el yield, se añade lógica de apagado.

    # Lógica de apagado.
    if not streamlit_process.returncode:
        try:
            streamlit_process.termiate() # Trata de mandar SIGTERM para acabar con el proceso.
        except Exception:
            streamlit_process.kill() # En caso de que dé error, trata de matarlo.
        print("Terminado proceso de Streamlit.")

# ------------------------------
# Creación de FastAPI
# ------------------------------
app = FastAPI(lifespan=lifespan)

# Monta carpeta "static" para los estilos.
app.mount("/static", StaticFiles(directory="static"), name="static")

# Monta la app de Gradio dentro de FastAPI en el endpoint "/gradio".
gr.mount_gradio_app(app, demo.queue(), "/gradio", css_paths=["static/style.css"])

@app.get("/", response_class=RedirectResponse)
async def launch_gradio(request: Request):
    """
    Redirige la raíz '/' hacia el endpoint de Gradio, mostrando la web.

    Nota:
        Colocar Gradio en la raíz daba problemas, así que tuvimos que hacer algo así parece ser necesario.
    """

    # Si tratamos de reemplazar la raíz con Gradio, da errores. Por tanto, redirigimos a su endpoint.
    return RedirectResponse(url="/gradio", status_code=301)

@app.get("/visualization-frontend", response_class=RedirectResponse)
async def redirect_visualization(request: Request):
    """
    Redirige al frontend de Streamlit, utilizando el mismo puerto `VISUALIZATION_PORT` que para iniciarlo.
    """
    return RedirectResponse(url=f"localhost:{ os.getenv('VISUALIZATION_PORT') }", status_code=301)

@app.post("/ejecutar")
async def run_pipeline(
    channel_url: str = Form(...),
    channel_keyword: Optional[str] = Form(""),
    video_limit: int = Form(...),
    # hacer opcional la lista de menciones para soportar 'Solo transcribir'
    mention_keywords: List[str] = Form([]),
    podcast_limit: int = Form(...),
    single_video_urls: List[str] = Form([]),
    only_transcribe: int = Form(0),
    language: str = Form(""),
    asr_model: str = Form("")
):
    mention_keywords_str = ",".join(mention_keywords)
    single_video_urls_str = ",".join(filter(None, single_video_urls)) 
    
    start_time = time.time()
    cmd = [
        "run-pipeline",
        channel_url,
        channel_keyword,
        str(video_limit),
        mention_keywords_str,
        str(podcast_limit), 
        single_video_urls_str,
        str(int(bool(only_transcribe))),
        language,
        asr_model
    ]
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',  
        bufsize=1
    )
    
    async def log_generator():
        try:
            while True:
                line = await asyncio.to_thread(process.stdout.readline)
                if not line: break

                print(line, end="", flush=True)
                yield line
            
            rc = await asyncio.to_thread(process.wait)
            yield f"\nProceso terminado con código: {rc}\n"
            yield f"Tiempo para ejecución: {time.time() - start_time}s\n"
        
        except asyncio.CancelledError:
            print(
                "\n[INFO] Conexión cerrada por el cliente o cancelada. Matando proceso...",
                flush=True,
            )
            raise

        finally:
            if process.returncode is None: # Realiza cierre limpio (SIGTERM) si se sigue ejecutando.
                process.terminate()

    return StreamingResponse(log_generator(), media_type="text/plain-text")

@app.get("/descargar-csv", response_class=FileResponse)
async def download_csv():
    """
    Devuelve el archivo CSV generado por el pipeline.
    """

    csv_path = config["csv_output_path"]
    # Asegúrate de que el nombre del archivo para la descarga sea el deseado.
    # Podrías extraer el nombre del archivo de csv_path si es necesario.
    return FileResponse(path=csv_path, media_type='text/csv', filename="analisis_sentimientos.csv")

def run():
    """
    Permite ejecutar el servidor FastAPI con uvicorn y puerto configurable por línea de comando.
    Se utiliza con el script personalizado `start-server`.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run("api.app:app", port=args.port, reload=True)