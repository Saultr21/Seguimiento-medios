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
import tempfile
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse, RedirectResponse
from fastapi import Request
from fastapi.staticfiles import StaticFiles
import uvicorn

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True, encoding='utf-8')

# Cargar configuración
config = load_config()
streamlit_process = None # Variable global para el proceso de Streamlit.

def launch_streamlit():
    """
    Inicializa la interfaz de Streamlit como un subproceso.

    Este proceso permite compaginar el frontend de visualización con el backend,
    compartiendo su ciclo de vida y permitiendo arrancarlo y finalizarlo con facilidad.
    El proceso se abre con `subprocess.Popen` y se guarda en la variable global `streamlit_process`.

    El puerto donde se sirve se obtiene desde la variable de entorno `VISUALIZATION_PORT`.
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
async def lifespan(app: FastAPI):
    """
    Inicializa recursos y los cierra al final del ciclo de vida de FastAPI.

    Actualmente, carga Streamlit como un subproceso al inicio
    y lo finaliza al terminar el ciclo de vida.

    Args:
        app:
            Instancia principal de FastAPI.
    """

    # Lógica de inicio.
    launch_streamlit()
    yield # Tras el yield, se añade lógica de apagado.

    # Lógica de apagado.
    if streamlit_process and not streamlit_process.returncode:
        try:
            streamlit_process.termiate() # Trata de mandar SIGTERM para acabar con el proceso.
        except Exception:
            streamlit_process.kill() # En caso de que dé error, trata de matarlo.
        print("Terminado proceso de Streamlit.")

# ------------------------------
# Creación de FastAPI
# ------------------------------
app = FastAPI(lifespan=lifespan) # Le damos el lifespan para que lo ejecute al inicio y al final.

# Monta carpeta "static" para los estilos.
app.mount("/static", StaticFiles(directory="static"), name="static")

# Monta la app de Gradio dentro de FastAPI en el endpoint "/gradio".
gr.mount_gradio_app(app, demo.queue(), "/gradio", css_paths=["static/style.css"])

@app.get("/", response_class=RedirectResponse)
async def launch_gradio(request: Request) -> RedirectResponse:
    """
    Redirige la ruta raíz hacia la interfaz web de Gradio.

    Este endpoint actúa como punto de entrada principal para la aplicación
    y redirige hacia el endpoint de Gradio, `/gradio`.

    Args:
        request:
            Request HTTP entrante.

    Returns:
        RedirectResponse:
            Redirección HTTP 301 (permanente) hacia `/gradio`.

    Notes:
        Montar Gradio en la raíz de FastAPI hacía que el servicio devolviera errores,
        así que por ello utilizamos una redirección explícita.
    """

    # Si tratamos de reemplazar la raíz con Gradio, da errores. Por tanto, redirigimos a su endpoint.
    return RedirectResponse(url="/gradio", status_code=301)

@app.get("/visualization-frontend", response_class=RedirectResponse)
async def redirect_visualization(request: Request) -> RedirectResponse:
    """
    Redirige hacia el frontend de visualización de datos servido con Streamlit.

    Este endpoint proporciona una redirección rápida hacia el frontend ejecutado
    como un subproceso con Streamlit.

    Returns:
        RedirectResponse:
            Redirección HTTP 301 (permanente) hacia el puerto configurado
            en la variable de entorno `VISUALIZATION_PORT`.

    Notes:
        El frontend de Streamlit se ejecuta en un servidor separado,
        por lo que el acceso se realiza mediante redirección HTTP.
    """
    return RedirectResponse(url=f"localhost:{ os.getenv('VISUALIZATION_PORT') }", status_code=301)

async def _save_temp_files(uploaded_files: list[UploadFile]) -> list[str]:
    temp_files_path = []
    for uploaded_file in uploaded_files:
        base_name = uploaded_file.filename.split(".")[-2] + "_"
        with tempfile.NamedTemporaryFile(delete=False, prefix=base_name, suffix=".mp3") as temp_file:
            temp_file.write(await uploaded_file.read())
            temp_files_path.append(temp_file.name)

    return temp_files_path

@app.post("/ejecutar")
async def run_pipeline(
    channel_url: str = Form(...),
    channel_keyword: str = Form(""),
    video_limit: int = Form(...),
    mention_keywords: list[str] = Form([]),
    podcast_limit: int = Form(...),
    single_video_urls: list[str] = Form([]),
    only_transcribe: int = Form(0),
    language: str = Form(""),
    asr_model: str = Form(""),

    audio_files: list[UploadFile] = File(...)
) -> StreamingResponse:
    """
    Ejecuta el pipeline principal de procesamiento multimedia.

    Este endpoint lanza el comando `run-pipeline` (definido en `pipeline:run`) en un
    subproceso independiente, transmitiendo los logs generados en tiempo real mediante una
    respuesta HTTP streaming.

    El pipeline permite procesar vídeos de canales, vídeos individuales y tareas de
    transcripción automática utilizando distintos modelos ASR.

    Args:
        channel_url:
            Opcional. URL del canal principal a procesar.

        channel_keyword:
            Opcional. Palabra clave para buscar en los títulos de vídeos de un canal.

        video_limit:
            Número máximo de vídeos a procesar desde el canal.

        mention_keywords:
            Opcional.
            Lista de entidades (frases) o palabras clave utilizadas para detectar menciones relevantes.

        podcast_limit:
            Número máximo de podcasts a procesar (de Espejo Canario).

        single_video_urls:
            Opcional. Lista de URLs individuales de vídeos.

        only_transcribe:
            Indica si únicamente debe ejecutarse la transcripción.

        language:
            Idioma utilizado durante el procesamiento ASR.

        asr_model:
            Modelo ASR utilizado para la transcripción automática.
            Puede ser `nemo` o `whisper`.

    Returns:
        StreamingResponse:
            Flujo de texto en tiempo real con los logs generados durante la ejecución del pipeline.

    Notes:
        - Lanza un subproceso externo mediante `subprocess.Popen`.
        - Consume muchos recuross de CPU/GPU (dependiendo del modelo ASR).
        - Los logs del pipeline se transmiten progresivamente al cliente utilizando streaming
          HTTP para permitir el seguimiento en tiempo real del procesamiento.
    """

    mention_keywords_str = ",".join(mention_keywords)
    single_video_urls_str = ",".join(filter(None, single_video_urls))
    
    audio_files_path = await _save_temp_files(audio_files)
    audio_files_path_str = ",".join(audio_files_path)
    
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
        asr_model,
        audio_files_path_str
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
            if not process.returncode: # Realiza cierre limpio (SIGTERM) si se sigue ejecutando.
                process.terminate()

    return StreamingResponse(log_generator(), media_type="text/plain")

@app.get("/descargar-csv", response_class=FileResponse)
async def download_csv() -> FileResponse:
    """
    Devuelve el archivo CSV generado por el pipeline de análisis.

    El endpoint permite descargar el resultado consolidado del procesamiento en formato CSV.

    Returns:
        FileResponse:
            Archivo CSV generado por el pipeline.
    """

    csv_path = config["csv_output_path"]
    return FileResponse(path=csv_path, media_type='text/csv', filename="analisis_sentimientos.csv")

def run():
    """
    Inicia el servidor FastAPI utilizando Uvicorn.

    Este método actúa como punto de entrada del backend y permite configurar el puerto
    del servidor mediante argumentos de línea de comandos.

    Se utiliza internamente por el script personalizado del proyeccto `start-server`.

    Command Line Arguments:
        --port:
            Puerto HTTP utilizado por el servidor FastAPI.

    Notes:
        - Bloquea el hilo principal hasta detener el servidor.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run("api.app:app", port=args.port, reload=True)