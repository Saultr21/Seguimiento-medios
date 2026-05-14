# 📡 Seguimiento de Medios

Herramienta de análisis automatizado de medios de comunicación. Descarga, transcribe y analiza el sentimiento de vídeos de YouTube y podcasts de **El Espejo Canario**, generando un informe en CSV listo para su análisis.

---

## ¿Para qué sirve?

Esta aplicación permite monitorizar medios audiovisuales de forma automatizada. A partir de una URL de canal de YouTube y/o el feed de podcast de El Espejo Canario:

1. **Descarga** los vídeos o episodios más recientes.
2. **Transcribe** el audio a texto mediante el modelo de ASR (en este momento, por defecto, `nvidia/canary-1b-v2`).
3. **Extrae fragmentos relevantes** utilizando chunking por el número de palabras y pasando el texto a un LLM que filtra las frases donde se encuentren unas palabras clave configurables.
4. **Analiza el sentimiento y las emociones** de cada fragmento usando `pysentimiento`.
5. **Exporta los resultados** a un CSV descargable directamente desde la interfaz web.

El flujo completo se ejecuta desde una interfaz web sencilla servida con FastAPI.

---

## Estructura del proyecto

```bash
Seguimiento-medios/
├── .env                              # Variables de entorno. Archivo no creado por defecto
├── .env.example                      # Ejemplo de configuración de entorno
├── .gitignore
├── comandos.txt                      # Referencia de comandos útiles
├── pyproject.toml                    # Configuración del proyecto y dependencias
├── README.md
├── requirements.txt
│
├── src/
│   ├── __init__.py
│   │
│   ├── api/                          # API y servidor web
│   │   ├── app.py                    # Servidor FastAPI + endpoints web
│   │   ├── frontend.py               # Servidor Gradio para frontend. Expuesto mediante FastAPI. 
│   │   └── __init__.py
│   │
│   ├── asr/                          # Sistemas ASR (Speech-to-Text)
│   │   ├── asr_factory.py            # Factory para seleccionar motor ASR
│   │   ├── base_asr.py               # Clase base abstracta ASR
│   │   ├── nemo_asr.py               # Implementación ASR con NVIDIA NeMo
│   │   ├── whisper_asr.py            # Implementación ASR con Whisper
│   │   └── __init__.py
│   │
│   ├── config/                       # Configuración global del proyecto
│   │   ├── load_config.py            # Carga de configuración general (.env)
│   │   ├── torch_config.py           # Configuración de PyTorch/GPU
│   │   └── __init__.py
│   │
│   ├── data_ingestion/               # Descarga e ingestión de contenido
│   │   ├── download_podcasts.py      # Descarga de podcasts
│   │   ├── download_yt_video.py      # Descarga de vídeos de YouTube
│   │   └── __init__.py
│   │
│   ├── llm/                          # Clientes y utilidades LLM
│   │   ├── llm_client.py             # Cliente para interacción con modelos LLM
│   │   └── __init__.py
│   │
│   ├── nlp/                          # Procesamiento de lenguaje natural
│   │   ├── sentiment_analysis.py     # Análisis de sentimientos/emociones
│   │   └── __init__.py
│   │
│   ├── pipeline/                     # Orquestación del flujo principal
│   │   ├── pipeline.py               # Pipeline principal de procesamiento
│   │   └── __init__.py
│   │
│   ├── services/                     # Servicios de alto nivel
│   │   ├── llm_service.py            # Servicio de interacción con LLMs
│   │   ├── transcription_service.py  # Servicio de transcripción
│   │   └── __init__.py
│   │
│   └── utils/                        # Utilidades auxiliares
│       ├── file_utils.py             # Funciones auxiliares para ficheros
│       ├── text_utils.py             # Utilidades de procesamiento de texto
│       └── __init__.py
│
└── static/
    └── style.css                     # Estilos de la interfaz web
```

---

## Requisitos previos

- Python 3.10 o superior
- GPU recomendada (CUDA) para acelerar la transcripción
- LLM local accesible vía API REST (configurable en `config.json`)
- `ffmpeg` instalado y disponible en el PATH (requerido por varias librerías)

---

## Uso

### Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/Saultr21/Seguimiento-medios.git
cd Seguimiento-medios

# 2. Crear y activar entorno virtual
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Instalar proyecto como paquete editable (permite usar scripts personalizados)
pip install -e .

# 5. (Opcional) Instalar PyTorch con soporte CUDA (por ejemplo, 13.0)
pip install --index-url https://download.pytorch.org/whl/cu130 \
    --extra-index-url https://pypi.org/simple torch
```

### Configuración

Crea el archivo `.env`, copiando el contneido de `.env.example`. Edita su contenido antes de arrancar la aplicación para adaptarse a tu entorno:

```bash
# Directorios y paths.
AUDIOS_DIR=./tmp/audios
TRANSCRIPTIONS_DIR=./tmp/transcriptions
JSON_OUTPUT_PATH=./tmp/fragmentos.json
CSV_OUTPUT_PATH=./tmp/analisis-textos-json.csv

# URL backend (para Gradio).
BACKEND_URL=http://localhost:8000

# Configuración de modelos.
WHISPER_MODEL_URL=openai/whisper-large-v3-turbo
NEMO_MODEL_URL=nvidia/canary-1b-v2
LLM_URL=http://192.168.1.60:1234/v1/chat/completions
LLM_MODEL=gemma-4-e2b
```

| Parámetro | Descripción |
| --- | --- |
| `AUDIOS_DIR` | Carpeta temporal para los audios descargados |
| `TRANSCRIPTIONS_DIR` | Carpeta donde se guardan las transcripciones en `.txt` |
| `JSON_OUTPUT_PATH` | Fichero JSON intermedio con los fragmentos extraídos |
| `CSV_OUTPUT_PATH` | Fichero CSV de salida con el análisis de sentimientos |
| `WHISPER_MODEL_URL` | Modelo Whisper a utilizar para la transcripción |
| `BACKEND_URL` | URL del backend. Solo es utilizada por el frontend de Gradio, ya que no permite rutas relativas. |
| `NEMO_MODEL_URL` | Modelo de NeMO utilizado para la transcripción (opción por defecto actual) |
| `LLM_URL` | URL de la API REST del LLM local (compatible con OpenAI). El valor por defecto proviene de LM Studio (aunque otras aplicaciones como Ollama también son compatibles mientras soporten la sintaxis de OpenAI). |
| `LLM_MODEL` | Modelo de LLM que se está utilizando actualmente. |

### Arrancar el servidor

```bash
start-server
```

Accede a la interfaz en [http://localhost:8000](http://localhost:8000).

> **Si el puerto 8000 está ocupado:**
>
> ```bash
> start-server --port 8001
> ```

### Interfaz web

Desde la UI puedes configurar:

| Campo | Descripción |
| --- | --- |
| **URL del canal** | URL del canal de YouTube a analizar |
| **Palabra clave del canal** | Término para filtrar/identificar el canal |
| **Límite de vídeos** | Número de vídeos del canal a procesar (0 = ninguno) |
| **Palabras clave de mención** | Términos a buscar en las transcripciones para extraer fragmentos relevantes |
| **Límite de podcasts** | Número de episodios de El Espejo Canario a procesar (0 = ninguno) |
| **URLs de vídeos individuales** | URLs de vídeos concretos a procesar, independientemente del canal |
| **Solo transcribir** | Activa este modo para omitir el análisis de sentimientos y obtener solo las transcripciones |
| **Idioma Whisper** | Fuerza un idioma concreto en la transcripción (dejar vacío para detección automática) |

### Descarga del CSV

Una vez completado el análisis, aparece un botón en la interfaz para descargar el fichero `analisis_sentimientos.csv` con todos los fragmentos y su clasificación de sentimiento/emoción.

---

## Flujo de trabajo interno

```bash
Inicio
  │
  ├─▶ Paso 1: Descarga y transcripción de YouTube, con URLs individuales o a través de canales (pytubefix + ASR)
  │
  ├─▶ Paso 1.5: Descarga y transcripción de podcasts (El Espejo Canario)
  │
  ├─▶ Mantenimiento: formateo de nombres y limpieza de temporales
  │
  ├─▶ Paso 2: Extracción de contextos con Window Sliding → fragmentos.json
  │
  └─▶ Paso 3: Análisis de sentimientos/emociones (pysentimiento) → CSV
```

| **En modo _"Solo transcribir"_, los pasos 2 y 3 se omiten**.

---

## Investigando: Dashboards

Se está investigando la utilización de un dashboard de código abierto como Metabase.

Para iniciar Metabase, se utiliza:

```bash
docker run -d -p 3000:3000 --name metabase metabase/metabase
```

Con esto, se puede acceder a la interfaz de Metabase usando a la dirección `localhost:3000`.

```bash
usuario: Usuario 1
email: usuario1@cognitiatech.com
contraseña: usuario1@cognitiatech.com
```

---

## Dependencias principales

| Librería | Uso |
| --- | --- |
| `fastapi` + `uvicorn` | Servidor web y API REST |
| `yt-dlp` + `pytubeFix` | Descarga de vídeos de YouTube |
| `transformers` | Modelo Whisper para transcripción de audio |
| `pysentimiento` | Análisis de sentimientos y emociones en español |
| `sentence-transformers` | Embeddings para el procesado de fragmentos |
| `pandas` | Procesado y exportación de datos a CSV |
| `jinja2` | Motor de plantillas para la interfaz web |
| `torch` | Backend de deep learning (GPU recomendada) |

---

## Notas

- Las transcripciones anteriores se eliminan automáticamente al iniciar cada nueva ejecución.
- El LLM local configurado en `LLM_URL` debe estar activo y accesible antes de lanzar el análisis completo.
- Se recomienda disponer de GPU para tiempos de transcripción razonables con los modelos de transcripción. La opción por defecto con NeMo es un poco pesada, así que si el equipo va lento, se puede cambiar al modelo de Whisper, que resulta más ligero.
  - Si el rendimiento sigue siendo malo, lo mejor que se puede hacer es poner vídeos cortos (<10 minutos).
