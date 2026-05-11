import gradio as gr
from urllib.parse import urlparse
import requests
 
# ── Configuración ────────────────────────────────────────────────────────────
BACKEND_URL = "http://localhost:8000"

# ── Payload builder ───────────────────────────────────────────────────────────
def build_params(channel_url, channel_keyword, video_limit, whisper_language,
                 single_video_urls, podcast_limit, mention_keywords, only_transcribe):
    
    urls = [u.strip() for u in (single_video_urls or "").splitlines() if u.strip()]
    kws  = [k.strip() for k in (mention_keywords  or "").splitlines() if k.strip()]

    params = [
        ("channel_url", channel_url or ""),
        ("channel_keyword", channel_keyword or ""),
        ("video_limit", str(int(video_limit or 0))),
        ("whisper_language", whisper_language or ""),
        ("podcast_limit", str(int(podcast_limit or 0))),
    ]
    
    if only_transcribe:
        params.append(("only_transcribe", "1"))
    
    params += [("single_video_urls", u) for u in urls]
    params += [("mention_keywords",  k) for k in kws]

    return params, urls, kws

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

# ── Line processor ────────────────────────────────────────────────────────────
def process_line(line, state):
    """
    Mutates `state` dict in-place and returns it.
    state keys: output_text, pct, csv_available
    """

    if line.startswith("PROGRESS:"): # Línea de progreso.
        _, pct, message = line.split(":")
        state['pct'] = int(pct)
        state['output_text'] += (message + "\n")

        return

    if line == "CSV_AVAILABLE:1": # Línea de finalización con archivo CSV.
        state['csv_available'] = True
        return

    if line == "CSV_AVAILABLE:0": # Línea de finalización sin archivo CSV.
        state['csv_available'] = False
        return

    state['output_text'] += (line + "\n") # Líneas de "print" normales.

def update_download_visibility(visible: bool):
    return gr.update(
        elem_id="download_btn",
        visible=visible,
    )


# ── Main generator ────────────────────────────────────────────────────────────
def run_pipeline(
        channel_url, channel_keyword, video_limit, whisper_language,
        single_video_urls, podcast_limit, mention_keywords, only_transcribe
    ):

    params, urls, keywords = build_params(
        channel_url,
        channel_keyword,
        video_limit,
        whisper_language,
        single_video_urls,
        podcast_limit,
        mention_keywords,
        only_transcribe,
    )

    error = validate(
        int(video_limit or 0),
        urls,
        int(podcast_limit or 0),
        keywords,
        only_transcribe,
    )

    if error:
        yield error, 0, update_download_visibility(False)
        return

    state = state = {
        'output_text': [],
        'pct': 0,
        'csv_available': False,
    }

    try:
        with requests.post(
            f"{BACKEND_URL}/ejecutar",
            data=params,
            stream=True,
            timeout=None,
        ) as response:

            response.raise_for_status()

            for line in response.iter_lines(decode_unicode=True):
                stripped_line = line.strip()
                if stripped_line is None: continue

                process_line(stripped_line, state)
                yield (
                    state.output_text,
                    state['pct'],
                    update_download_visibility(state.csv_available),
                )

    except requests.RequestException as exc:
        state['output_text'] += f"❌ Error de conexión: {exc}"

        yield (
            state.output_text,
            0,
            update_download_visibility(state.csv_available),
        )

        return

    success = "Proceso terminado con código: 0" in state.output_text

    yield (
        state.output_text,
        100 if success else state['pct'],
        update_download_visibility(state.csv_available),
    )

    ########

# ── Interfaz Gradio ───────────────────────────────────────────────────────────
with gr.Blocks(title="Análisis de Medios") as demo:

    gr.Markdown("# 📺 Análisis de Medios")

    with gr.Group():
        # ── Idioma (siempre visible) ──────────────────────────────────────────────
        whisper_language = gr.Dropdown(
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
        only_transcribe = gr.Checkbox(
            label="Solo transcribir (sin extracción de contextos ni análisis de sentimientos)",
            value=False,
        )

        only_transcribe_note = gr.Markdown(
            "ℹ️ **Nota:** Al activar 'Solo transcribir' no es necesario añadir palabras clave para filtrado de menciones.",
            visible=False,
        )

        only_transcribe.change(
            fn=lambda v: gr.update(visible=v),
            inputs=only_transcribe,
            outputs=only_transcribe_note,
        )

    submit_btn = gr.Button("▶ Ejecutar Flujo", variant="primary")

    # ── Salida ────────────────────────────────────────────────────────────────
    with gr.Group():
        progress_bar  = gr.Slider(0, 100, 0, step=1, interactive=False, label="Progreso de pipeline")
        download_btn = gr.DownloadButton("Descargar CSV", value="tmp/analisis-textos-json.csv", elem_id="download_btn", variant="secondary", visible=False)
        output_box    = gr.Textbox(
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
            whisper_language,
            single_video_urls,
            podcast_limit,
            mention_keywords,
            only_transcribe,
        ],
        outputs=[output_box, progress_bar, download_btn],
    )

if __name__ == "__main__":
    demo.launch(server_name="localhost", server_port=7860, show_error=True)