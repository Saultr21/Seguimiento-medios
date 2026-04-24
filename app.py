from fastapi import FastAPI, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi import Request
from fastapi.staticfiles import StaticFiles
from typing import List
import subprocess
import sys
from src.config.cargar_config import cargar_config
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Cargar configuración
config = cargar_config()
# Crear la instancia de la aplicación FastAPI
app = FastAPI()
#Crear carpeta static para los estilos
app.mount("/static", StaticFiles(directory="static"), name="static")
# Ruta para servir el archivo HTML
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Renderizar la plantilla HTML
    return templates.TemplateResponse(request, "index.html", {"request": request})

@app.post("/ejecutar")
async def ejecutar_stream(
    channel_url: str = Form(...),
    channel_keyword: str = Form(...),
    video_limit: int = Form(...),
    # hacer opcional la lista de menciones para soportar 'Solo transcribir'
    mention_keywords: List[str] = Form([]),
    podcast_limit: int = Form(...),
    single_video_urls: List[str] = Form([]),
    only_transcribe: int = Form(0),
    whisper_language: str = Form("")
):
    mention_keywords_str = ",".join(mention_keywords)
    single_video_urls_str = ",".join(filter(None, single_video_urls)) 

    # Usar valores proporcionados por el formulario
    transcripciones_dir = config["transcripciones_dir"]
    json_output_path = config["json_output_path"]
    csv_output_path = config["csv_output_path"]

    cmd = [
        sys.executable,
        "-u",            
        "ejecucion.py",
        channel_url,
        channel_keyword,
        str(video_limit),
        transcripciones_dir,
        json_output_path,
        csv_output_path,
        mention_keywords_str,
        str(podcast_limit),
        single_video_urls_str,
        str(int(bool(only_transcribe))),
        whisper_language
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
        for line in process.stdout:
            print(line, end="", flush=True)  
            yield line                        
        rc = process.wait()
        yield f"\nProceso terminado con código: {rc}\n"

    return StreamingResponse(log_generator(), media_type="text/plain")

@app.get("/descargar-csv", response_class=FileResponse)
async def descargar_csv():
    csv_path = config["csv_output_path"]
    # Asegúrate de que el nombre del archivo para la descarga sea el deseado.
    # Podrías extraer el nombre del archivo de csv_path si es necesario.
    return FileResponse(path=csv_path, media_type='text/csv', filename="analisis_sentimientos.csv")
