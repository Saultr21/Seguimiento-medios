"""
Frontend interactivo desarrollado con Gradio.

Este módulo proporciona una interfaz web para ejecutar el pipeline, monitorizar
su progreso en tiempo real y descargar los resultados generados.

Arquitectura:
    Gradio UI
        ↓
    FastAPI backend
        ↓
    Pipeline subprocess
        ↓
    Streaming HTTP logs

Notes:
    - Permite
        - Procesamiento de canales de YouTube.
        - Procesamiento de vídeos individuales.
        - Descarga de podcasts.
        - Transcripción ASR.
        - Filtrado de menciones.
        - Descarga de resultados CSV.
        - Acceso al frontend de visualización.
"""
from urllib.parse import urlparse
import httpx
import os
import mimetypes
import gradio as gr
from gradio.utils import NamedString

 
# ── Configuración ────────────────────────────────────────────────────────────
BACKEND_URL = os.environ['BACKEND_URL']

# ── Validator ─────────────────────────────────────────────────────────────────
def validate(video_limit: int, urls: list[str], podcast_limit: int, keywords: list[str], only_transcribe: bool, file_inputs: list[NamedString]) -> str | None:
    """
    Valida la configuración introducida en la interfaz Gradio.

    Comprueba que:
    - Exista al menos una fuente de medios (vídeos de Youtube, canal de YouTube, podcasts)
    - Esté activada la opción de solo transcripción o se hayan establecido palabras clave
    - Las URLs individuales proporcionadas sean correctas

    Args:
        video_limit:
            Número de vídeos del canal a procesar.

        urls:
            Lista de URLs individuales de vídeos.

        podcast_limit:
            Número de podcasts a descargar.

        keywords:
            Lista de palabras clave para filtrado.

        only_transcribe:
            Indica si solo debe ejecutarse la transcripción.

    Returns:
        str | None:
            Mensaje de error si la validación falla; `None` en caso contrario.
            Por tanto, `None` es el valor de validación correcta.
    """

    has_media_source = video_limit or urls or podcast_limit or file_inputs

    if not has_media_source:
        return "ERROR: Configure al menos una fuente de medios."

    if not only_transcribe and not keywords:
        return "ERROR: Añada al menos una palabra clave para filtrar menciones, o activa 'Solo transcribir'."

    for url in urls:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not bool(parsed.netloc):
            return f"ERROR: URL no válida: {url}"

    return None

def _render_progress_bar(pct: int, error: bool = False, completed: bool = False) -> str:
    """
    Genera el HTML de la barra de progreso.

    La barra utiliza estilos de Bootstrap y adapta su apariencia dependiendo del
    estado de ejecución (`error`, `completed` o ninguno).

    Args:
        pct:
            Porcentaje actual de progreso.

        danger:
            Indica si debe mostrarse el estado de error.

        done:
            Indica si el proceso se finalizó.

    Returns:
        str:
            Fragmento HTML renderizable por `Gradio.HTML()`.
    """

    pct = max(0, min(100, pct))

    if error:
        classes = "bg-danger"
    elif completed:
        classes = "bg-success"
    else:
        classes = "bg-primary progress-bar-striped progress-bar-animated"
 
    return f"""
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
        <div class="progress" style="height:24px;">
        <div class="progress-bar {classes}" role="progressbar"
            style="width:{pct}%; transition: width 0.4s ease;" aria-valuenow="{pct}"
            aria-valuemin="0" aria-valuemax="100"
        >
            {pct}%
        </div>
        </div>
    """

# ── Line processor ────────────────────────────────────────────────────────────
def process_line(line, state):
    """
    Procesa una línea recibida desde el streaming del backend.

    El backend puede emitir líneas especiales con el formato:
        PROGRESS:<percentage>:<message>

    Estas líneas actualizan el progreso mostrado en la interfaz. El resto de líneas se
    interpretan como logs normales del pipeline.

    Args:
        line:
            Línea recibida desde el backend.

        state:
            Estado mutable compartido de la interfaz.

    Notes:
        Modifica el diccionario `state` in-place.
    """

    if line.startswith("PROGRESS:"): # Línea de progreso.
        _, pct, message = line.split(":", 2)
        state['pct'] = int(pct)
        state['output_text'] += (message + "\n")

        return

    state['output_text'] += (line + "\n") # Líneas de "print" normales.

# ── Main generator ────────────────────────────────────────────────────────────
async def run_pipeline(
        channel_url: str,
        channel_keyword: str,
        video_limit: int,
        language: str,
        single_video_urls: str,
        podcast_limit: int,
        mention_keywords: str,
        only_transcribe: bool,
        asr_model: str,
        file_inputs: list[NamedString]
    ):
    """
    Ejecuta el pipeline desde la interfaz Gradio.

    Este callback envía la configuración del usuario al backend y consume progresivamente
    el streaming HTTP de logs generado durante la ejecución.

    La interfaz se actualiza en tiempo real mostrando:
    - Logs del pipeline
    - Barra de progreso
    - Estado de finalización
    - Archivo CSV descargable

    Yields:
        tuple:
            Actualizaciones progresivas para los componentes visuales de Gradio.

                Structure:
                    output_text (str):
                        Logs en streaming, en formato de texto.

                    progress_bar (str):
                        HTML renderizado de la barra de progreso.

                    csv_file (gr.update | gr.skip):
                        Controla la visibilidad y valor del archivo CSV.
                        Añade o elimina la dirección al archivo (gr.update),
                        o no lo modifica (gr.skip).

    Notes:
        El backend transmite eventos especiales con el prefijo `PROGRESS:` para
        actualizar la barra de progreso.
    """

    state = {
        'output_text': "Empezado el streaming...",
        'pct': 0
    }

    yield (
        state['output_text'],
        _render_progress_bar(state['pct']),
        gr.update(value=None, visible=False)
    )

    urls = [u.strip() for u in (single_video_urls or "").splitlines() if u.strip()]
    keywords  = [k.strip() for k in (mention_keywords  or "").splitlines() if k.strip()]

    error = validate(
        int(video_limit or 0),
        urls,
        int(podcast_limit or 0),
        keywords,
        only_transcribe,
        file_inputs
    )

    if error:
        yield (
            error,
            _render_progress_bar(0, error=True), 
            gr.skip()
        )
        return

    state = {
        'output_text': "",
        'pct': 0
    }

    try:

        payload = {
            "channel_url": channel_url or "",
            "channel_keyword": channel_keyword or "",
            "video_limit": video_limit or 0,
            "language": language or "",
            "podcast_limit": podcast_limit or 0,
            "only_transcribe": 1 if only_transcribe else 0,
            "single_video_urls": urls,
            "mention_keywords": keywords,
            "asr_model": asr_model
        }

        files_payload = []
        # Gradio ya crea archivos temporales cuando subimos con "Files",
        # pero nosotros forzaremos a que FastAPI cree unos nuevos.
        for file in file_inputs:
            file_name = file.name.split("/")[-1]
            mime_type, _ = mimetypes.guess_type(file_name)

            if mime_type and mime_type.startswith("audio/"):
                files_payload.append(
                    (
                        "audio_files",
                        (file_name, open(file.name, "rb"), mime_type)
                    )
                )
            elif mime_type and mime_type.startswith("text/"):
                files_payload.append(
                    (
                        "text_files",
                        (file_name, open(file.name, "rb"), mime_type)
                    )
                )

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{BACKEND_URL}/ejecutar",
                files=files_payload,
                data=payload,
            ) as response:
                async for line in response.aiter_lines():
                    if not line: continue
                    process_line(line, state)

                    yield (
                        state['output_text'],
                        _render_progress_bar(state['pct']),
                        gr.skip()
                    )
    
    except httpx.ConnectError as exc:
        state['output_text'] += f"❌ Error de conexión: {exc}"

        yield (
            state['output_text'],
            _render_progress_bar(0, error=True),
            gr.skip()
        )

        return

    success = "Proceso terminado con código: 0" in state['output_text']
    csv_available = success and "CSV_AVAILABLE:1" in state['output_text']
    if csv_available:
        yield (
            state['output_text'],
            _render_progress_bar(100 if success else state['pct'], completed=True),
            gr.update(value=os.environ["CSV_OUTPUT_PATH"], visible=True)
        )
    else:
        yield(
            state['output_text'],
            _render_progress_bar(100 if success else state['pct'], completed=success),
            gr.skip()
        )

# ── Interfaz Gradio ───────────────────────────────────────────────────────────
# Interfaz principal de Gradio para la interacción con el pipeline.
with gr.Blocks(title="Análisis de Medios") as demo:

    gr.Markdown("# 📺 Análisis de Medios")

    with gr.Group():
        # ── Idioma (siempre visible) ──────────────────────────────────────────────
        language = gr.Dropdown(
            choices=[
                ("Automático (detección)", ""),
                ("Inglés", "english"),
                ("Español", "spanish"),
            ],
            value="",
            label="Idioma de transcripción",
            info="Selecciona un idioma para forzar la transcripción; útil si el audio siempre está en un idioma concreto.",
        )

        # ── Canal de YouTube ──────────────────────────────────────────────────────
        with gr.Accordion("📹 Canal de YouTube (Opcional)", elem_id="youtube_channel_accordion", open=False):
            channel_url = gr.Textbox(
                label="URL del canal",
                value="https://www.youtube.com/@InformativosTvc/videos",
                placeholder="https://www.youtube.com/@canal/videos",
            )

            channel_keyword = gr.Textbox(
                label="Palabra clave para filtrar vídeos del canal",
                value="Telenoticias",
                info="Dejar vacío para procesar los últimos vídeos sin filtrar por título.",
            )

            video_limit = gr.Number(
                label="Número de vídeos a procesar del canal",
                value=0, minimum=0, precision=0,
                info="0 = no procesar vídeos del canal.",
                elem_classes=["gr-number"]
            )

        # ── Vídeos únicos ─────────────────────────────────────────────────────────
        with gr.Accordion("🎬 Vídeos Únicos (Opcional)", elem_id="youtube_urls_accordion", open=False):
            single_video_urls = gr.Textbox(
                label="URLs de vídeos de YouTube",
                placeholder="https://www.youtube.com/watch?v=AAA\nhttps://www.youtube.com/watch?v=BBB",
                lines=4,
                info="Una URL por línea. Se puede combinar con la configuración del canal.",
            )

        # ── Podcasts ──────────────────────────────────────────────────────────────
        with gr.Accordion("🎙️ Podcasts — El Espejo Canario (Opcional)", elem_id="podcast_accordion", open=False):
            podcast_limit = gr.Number(
                label="Número de podcasts a descargar",
                value=0, minimum=0, precision=0,
                info="Se descargarán los últimos episodios de 'El Espejo Canario'. 0 = no procesar.",
                elem_classes=["gr-number"]
            )

        # ── Palabras clave ────────────────────────────────────────────────────────
        with gr.Accordion("🔍 Filtrado de Menciones (Opcional si se usa 'Solo transcribir')", elem_id="keyowrds_accordion", open=False):
            mention_keywords = gr.Textbox(
                label="Palabras clave (una por línea)",
                placeholder="palabra1\npalabra2\nfrase clave",
                lines=5,
                info="Se buscarán estas palabras/frases en los textos transcritos.",
            )

        # ── Opciones adicionales de ejecución ─────────────────────────────────────────────────
        with gr.Group():
            asr_model = gr.Radio(
                choices=[
                    ("Whisper", "whisper"),
                    ("NeMo", "nemo")
                ],
                value="whisper",
                interactive=True,
                label="Modelo de transcripción",
                info="""
                    Selecciona el modelo para la transcripción de audio.
                    - Whisper: liviano, menos preciso
                    - NeMo (Canary): pesado, más preciso
                """
            )

        only_transcribe = gr.Checkbox(
            label="Solo transcribir (sin extracción de contextos ni análisis de sentimientos)",
            value=False,
            info="ℹ️ **Nota:** Al activar 'Solo transcribir' no es necesario añadir palabras clave para filtrado de menciones.",
        )

        file_inputs = gr.Files(file_types=[".mp3", ".txt"]) # Permitimos archivos ".mp3" y ".txt".

    submit_btn = gr.Button("▶ Ejecutar Flujo", variant="primary")
    visualization_btn = gr.Button("Visitar página de visualización", variant="secondary")

    # ── Salida ────────────────────────────────────────────────────────────────
    with gr.Group():
        progress_bar = gr.HTML(_render_progress_bar(0))
        csv_file = gr.File(value=None, label="Archivo CSV", visible=False)
        output_box = gr.Textbox(
            label="Resultado",
            lines=18,
            interactive=False,
            elem_id="output-box",
        )

    # ── Evento ───────────────────────────────────────────────────────────────
    submit_btn.click( # Ejecuta el pipeline y actualiza la interfaz para streaming.
        fn=run_pipeline,
        inputs=[
            channel_url,
            channel_keyword,
            video_limit,
            language,
            single_video_urls,
            podcast_limit,
            mention_keywords,
            only_transcribe,
            asr_model,
            file_inputs
        ],
        outputs=[output_box, progress_bar, csv_file],
    )

    visualization_btn.click( # Abre el frontend de visualización.
        lambda: None,
        js = f"window.open('{BACKEND_URL}/visualization-frontend', '_blank')"
    )

if __name__ == "__main__":
    demo.launch()