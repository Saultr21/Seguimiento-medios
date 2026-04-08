# 📡 Seguimiento de Medios

Herramienta de análisis automatizado de medios de comunicación. Descarga, transcribe y analiza el sentimiento de vídeos de YouTube y podcasts de **El Espejo Canario**, generando un informe en CSV listo para su análisis.

---

## ¿Para qué sirve?

Esta aplicación permite monitorizar medios audiovisuales de forma automatizada. A partir de una URL de canal de YouTube y/o el feed de podcast de El Espejo Canario:

1. **Descarga** los vídeos o episodios más recientes.
2. **Transcribe** el audio a texto mediante el modelo Whisper (`openai/whisper-large-v3-turbo`).
3. **Extrae fragmentos relevantes** mediante una técnica de ventana deslizante (*window sliding*), filtrando por palabras clave configurables.
4. **Analiza el sentimiento y las emociones** de cada fragmento usando `pysentimiento`.
5. **Exporta los resultados** a un CSV descargable directamente desde la interfaz web.

El flujo completo se ejecuta desde una interfaz web sencilla servida con FastAPI.

---

## Estructura del proyecto

```
Seguimiento-medios/
├── app.py                            # Servidor FastAPI + endpoints web
├── ejecucion.py                      # Orquestador del flujo completo
├── descarga_videos_yt.py             # Descarga y transcripción de YouTube (yt-dlp + Whisper)
├── descarga_podcast_espejocanario.py # Descarga y transcripción del podcast El Espejo Canario
├── peticion_window_sliding.py        # Extracción de contextos con ventana deslizante
├── analisis_pysentimiento_json.py    # Análisis de sentimientos/emociones
├── config/
│   ├── config.json                   # Configuración de rutas y modelos
│   └── cargar_config.py              # Cargador de configuración
├── templates/
│   └── index.html                    # Interfaz web
├── static/
│   └── style.css                     # Estilos
├── transcripciones/                  # Transcripciones generadas (se limpian en cada ejecución)
├── requirements.txt
└── comandos.txt                      # Referencia de comandos útiles
```

---

## Requisitos previos

- Python 3.10 o superior
- GPU recomendada (CUDA) para acelerar la transcripción con Whisper
- LLM local accesible vía API REST (configurable en `config.json`)
- `ffmpeg` instalado y disponible en el PATH (requerido por `yt-dlp`)

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/Saultr21/Seguimiento-medios.git
cd Seguimiento-medios

# 2. Crear y activar entorno virtual
python -m venv venv

# Windows
venv\Scripts\activate.bat

# Linux/macOS
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. (Opcional) Instalar PyTorch con soporte CUDA 13.0
pip install --index-url https://download.pytorch.org/whl/cu130 \
    --extra-index-url https://pypi.org/simple torch
```

---

## Configuración

Edita el archivo `config/config.json` antes de arrancar la aplicación:

```json
{
    "transcripciones_dir": "./transcripciones",
    "audio_dir": "./audios",
    "json_output_path": "fragmentos.json",
    "csv_output_path": "analisis_textos_json.csv",
    "whisper_model_url": "openai/whisper-large-v3-turbo",
    "llm_url": "http://192.168.1.60:1234/v1/chat/completions",
    "podcast_limit": 1
}
```

| Parámetro | Descripción |
|---|---|
| `transcripciones_dir` | Carpeta donde se guardan las transcripciones en `.txt` |
| `audio_dir` | Carpeta temporal para los audios descargados |
| `json_output_path` | Fichero JSON intermedio con los fragmentos extraídos |
| `csv_output_path` | Fichero CSV de salida con el análisis de sentimientos |
| `whisper_model_url` | Modelo Whisper a utilizar para la transcripción |
| `llm_url` | URL de la API REST del LLM local (compatible con OpenAI) |
| `podcast_limit` | Número máximo de podcasts a procesar por defecto |

---

## Uso

### Arrancar el servidor

```bash
uvicorn app:app --host 0.0.0.0 --reload --log-level debug
```

Accede a la interfaz en [http://localhost:8000](http://localhost:8000).

> Si el puerto 8000 está ocupado:
> ```bash
> python -m uvicorn app:app --host 127.0.0.1 --port 8001 --reload --log-level debug
> ```

### Interfaz web

Desde la UI puedes configurar:

| Campo | Descripción |
|---|---|
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

```
Inicio
  │
  ├─▶ [Opcional] Vídeos individuales (URLs sueltas)
  │
  ├─▶ Paso 1: Descarga y transcripción de YouTube (yt-dlp + Whisper)
  │
  ├─▶ Paso 1.5: Descarga y transcripción de podcasts (El Espejo Canario)
  │
  ├─▶ Mantenimiento: formateo de nombres y limpieza de temporales
  │
  ├─▶ Paso 2: Extracción de contextos con Window Sliding → fragmentos.json
  │
  └─▶ Paso 3: Análisis de sentimientos/emociones (pysentimiento) → CSV
```

> En modo **"Solo transcribir"**, los pasos 2 y 3 se omiten.

---

## Dependencias principales

| Librería | Uso |
|---|---|
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
- El LLM local configurado en `llm_url` se utiliza en el proceso de extracción de contextos (window sliding). Debe estar activo y accesible antes de lanzar el análisis completo.
- Se recomienda disponer de GPU para tiempos de transcripción razonables con `whisper-large-v3-turbo`.
