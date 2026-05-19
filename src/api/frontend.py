import gradio as gr
from urllib.parse import urlparse
import httpx
import os
 
# ── Configuración ────────────────────────────────────────────────────────────
BACKEND_URL = os.environ['BACKEND_URL']

# ── Validator ─────────────────────────────────────────────────────────────────
def validate(video_limit, urls, podcast_limit, keywords, only_transcribe):
    has_media_source = video_limit or urls or podcast_limit

    if not only_transcribe and not has_media_source:
        return "ERROR: Configure al menos una fuente de medios o active 'Solo transcribir'."

    if not only_transcribe and not keywords:
        return "ERROR: Añada al menos una palabra clave para filtrar menciones."

    for url in urls:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not bool(parsed.netloc):
            return f"ERROR: URL no válida: {url}"

    return None

def _progress_html(pct: int, danger: bool = False, done: bool = False) -> str:
    pct = max(0, min(100, pct))
    if danger:
        color = "bg-danger"
    elif done:
        color = "bg-success"
    else:
        color = "bg-primary progress-bar-striped progress-bar-animated"
 
    return f"""
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
        <div class="progress" style="height:24px;">
        <div class="progress-bar {color}" role="progressbar"
            style="width:{pct}%; transition: width 0.4s ease;" aria-valuenow="{pct}"
            aria-valuemin="0" aria-valuemax="100">
            {pct}%
        </div>
        </div>
    """

# ── Line processor ────────────────────────────────────────────────────────────
def process_line(line, state):
    """
    Mutates `state` dict in-place and returns it.
    state keys: output_text, pct, csv_available
    """

    if line.startswith("PROGRESS:"): # Línea de progreso.
        _, pct, message = line.split(":", 2)
        state['pct'] = int(pct)
        state['output_text'] += (message + "\n")

        return

    state['output_text'] += (line + "\n") # Líneas de "print" normales.

# ── Main generator ────────────────────────────────────────────────────────────
async def run_pipeline(
        channel_url, channel_keyword, video_limit, language,
        single_video_urls, podcast_limit, mention_keywords, only_transcribe,
        asr_model
    ):

    state = {
        'output_text': "Empezado el streaming...",
        'pct': 0
    }

    yield (
        state['output_text'],
        _progress_html(state['pct']),
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
    )

    if error:
        yield (
            error,
            _progress_html(0, danger=True), 
            gr.skip()
        )
        return

    state = {
        'output_text': "",
        'pct': 0
    }

    try:
        data = {
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

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{BACKEND_URL}/ejecutar",
                data=data,
            ) as response:
                async for line in response.aiter_lines():
                    if not line: continue
                    process_line(line, state)

                    yield (
                        state['output_text'],
                        _progress_html(state['pct']),
                        gr.skip()
                    )
    
    except httpx.ConnectError as exc:
        state['output_text'] += f"❌ Error de conexión: {exc}"

        yield (
            state['output_text'],
            _progress_html(0, danger=True),
            gr.skip()
        )

        return
    
    success = "Proceso terminado con código: 0" in state['output_text']
    csv_available = success and "CSV_AVAILABLE:1" in state['output_text']
    if csv_available:
        yield (
            state['output_text'],
            _progress_html(100 if success else state['pct'], done=success),
            gr.update(value=os.environ["CSV_OUTPUT_PATH"], visible=True)
        )
    else:
        yield(
            state['output_text'],
            _progress_html(100 if success else state['pct'], done=success),
            gr.skip()
        )

# ── Interfaz Gradio ───────────────────────────────────────────────────────────
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

    submit_btn = gr.Button("▶ Ejecutar Flujo", variant="primary")

    # ── Salida ────────────────────────────────────────────────────────────────
    with gr.Group():
        progress_bar = gr.HTML(_progress_html(0))
        csv_file = gr.File(value=None, label="Archivo CSV", visible=False)
        output_box = gr.Textbox(
            label="Resultado",
            lines=18,
            interactive=False,
            elem_id="output-box",
        )

    # ── Evento ───────────────────────────────────────────────────────────────
    submit_btn.click(
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
            asr_model
        ],
        outputs=[output_box, progress_bar, csv_file],
    )

if __name__ == "__main__":
    demo.launch()