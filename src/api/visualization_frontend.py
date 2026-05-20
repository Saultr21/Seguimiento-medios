"""
Dashboard interactivo de análisis de texto construido con Streamlit.

Este módulo carga resultados de análisis de sentimiento, emociones y discurso de odio, y los
presenta mediante visualizaciones interactivas.

Flujo del frontend:
    1. Carga de datos CSV mediante subida(s) o archivo por defecto.
    2. Aplicación de filtros en sidebar.
    3. Cálculo de KPIs.
    4. Visualización de gráficos:
        - Distribución de sentimiento
        - Probabilidades por fragmento
        - Discurso de odio
        - Emociones (radar chart)
    5. Tabla detallada de fragmentos.

El dashboard está diseñado como una exploración analítica interactiva.
"""

import re

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
 
# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Análisis de Textos",
    page_icon="🔍",
    layout="wide",
)
 
# ── CSS personalizado ────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 16px;
        border-left: 4px solid #4f8ef7;
        margin-bottom: 8px;
    }
    .stDataFrame { font-size: 13px; }
    h1 { color: #1a1a2e; }
    h2 { color: #16213e; }
    .sentiment-neg { color: #e74c3c; font-weight: bold; }
    .sentiment-neu { color: #3498db; font-weight: bold; }
    .sentiment-pos { color: #2ecc71; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ── Carga de datos ───────────────────────────────────────────────────────────
@st.cache_data
def _load_data(uploaded_files: str) -> pd.DataFrame:
    """
    Carga y consolida uno o varios archivos CSV en un único DataFrame.

    Este loader se encarga de:
    - Leer múltiples archivos CSV
    - Normalizar columnas de probabilidad (prob_* y emo_*)
    - Convertir porcentajes en valores numéricos
    - Unificar todos los datasets en uno solo

    Args:
        uploaded_files:
            Lista de archivos CSV cargados desde la interfaz o ruta por defecto.

    Returns:
        pd.DataFrame:
            Dataset combinado, listo para análisis.
    """

    data = []

    for file in uploaded_files:
        df = pd.read_csv(file)

        # Convertimos columnas de porcentaje (string con %) a float
        for col in df.columns:
            pattern = re.compile(r'^(prob_|emo_)')
            if pattern.match(col):
                df[col] = df[col].map(lambda s: s[:-1]).astype(float)

        data.append(df)
    
    return pd.concat(data, ignore_index=True)
 
# ── Sidebar: carga de archivo ────────────────────────────────────────────────
def _render_sidebar() -> pd.DataFrame:
    """
    Gestiona la carga de datos desde la barra lateral.

    Permite:
    - Subida de archivo(s) CSV por el usuario
    - Carga automática de dataset por defecto en local, si no hay archivos
    - Manejo de errores si no se encuentra ningún dataset (mensaje de error)

    Returns:
        pd.DataFrame:
            Dataset cargado listo para filtrado y visualización.
    """
    st.sidebar.header("⚙️ Configuración")

    uploaded = st.sidebar.file_uploader(
        "Subir CSV de análisis",
        type=["csv"],
        accept_multiple_files=True
    )

    df = None
    if uploaded:
        df = _load_data(uploaded)
        st.sidebar.success(f"✅ { len(df) } fragmentos cargados en { len(uploaded) } archivo(s)")
    else:
        # Usar el archivo de ejemplo por defecto
        DEFAULT_PATH = "tmp/analisis-textos-json.csv"
        try:
            df = _load_data(DEFAULT_PATH)
            st.sidebar.info("Usando archivo de ejemplo.")
        except FileNotFoundError:
            st.error("No se encontró ningún CSV. Sube uno usando el panel lateral.")
            st.stop()
    
    return df
 
# ── Filtros ──────────────────────────────────────────────────────────────────
def _fetch_keywords(entity: str) -> list[str]: # Función helper.
    """
    Extrae palabras clave desde una cadena de entidades separadas por comas y las
    devuelve como una lista.

    Args:
        entity:
            Entidad con conjunto de 'términos clave'.
            Ejemplo: `"Madrid(), Barcelona(), Valencia()"`
    
    Returns:
        list[str]:
            Lista de keywords limpias.
            Ejemplo: `["Madrid", "Barcelona", "Valencia"]`
    """

    keywords = list()
    splits = entity.split(",")
    for split in splits:
        keyword = split.replace("()", "").strip()
        keywords.append(keyword)
        
    return keywords

def _render_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica filtros interactivos al dataset desde la barra lateral.

    Permite filtrar por:
        - Título
        - Sentimiento
        - Entidades / palabras clave

    Args:
        pd.DataFrame:
            Dataset inicial, sin filtros.

    Returns:
        pd.DataFrame:
            Dataset filtrado.
    """

    titulos = df["titulo"].unique().tolist()
    selected_title = st.sidebar.multiselect("Filtrar por título", titulos, default=titulos)
    df = df[df["titulo"].isin(selected_title)]
    
    sentimientos_disponibles = df["sentimiento"].unique().tolist()
    selected_sentiment = st.sidebar.multiselect("Filtrar por sentimiento", sentimientos_disponibles, default=sentimientos_disponibles)
    df = df[df["sentimiento"].isin(selected_sentiment)]

    keywords = df["entidades"].map(_fetch_keywords)
    # Unimos y ordenamos todas las palabras clave / entidades. Usamos un "set" para evitar repetición.
    keywords = sorted( 
        set().union(*keywords)
    )

    selected_keywords = st.sidebar.multiselect("Filtrar por palabras clave", keywords, default=keywords)
    df = df[
        df["entidades"].apply( # Filtrado por si se encuentra la palabra clave.
            lambda entity: any(keyword in entity for keyword in selected_keywords)
        )
    ]

    return df
 
# ── Encabezado ───────────────────────────────────────────────────────────────
def _render_header():
    st.title("🔍 Análisis de Textos — Dashboard")
    st.markdown("Visualización interactiva de resultados de análisis de sentimiento, emociones y discurso de odio.")
    st.divider()
 
# ── KPIs ─────────────────────────────────────────────────────────────────────
def _render_kpis(df: pd.DataFrame):
    """
    Muestra indicadores principales del dataset.

    Incluye:
    - Número total de fragmentos analizados
    - Distribución de sentimiento (negativo, positivo, neutro)
    - Detección de discurso de odio (número de fragmentos)
    """

    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2, row2_col3 = st.columns(3)
    
    n_total = len(df)
    n_neg   = (df["sentimiento"] == "Negativo").sum()
    n_pos   = (df["sentimiento"] == "Positivo").sum()
    n_neu   = (df["sentimiento"] == "Neutral").sum()
    n_odio  = (df["odio_detectado"].str.lower() == "odio").sum()

    # Calcular porcentajes.
    pct_neg = (n_neg / n_total * 100) if n_total > 0 else 0
    pct_pos = (n_pos / n_total * 100) if n_total > 0 else 0
    pct_neu = (n_neu / n_total * 100) if n_total > 0 else 0
    pct_odio = (n_odio / n_total * 100) if n_total > 0 else 0
    
    row1_col1.metric("Fragmentos analizados", n_total, help="Número total de fragmentos analizados.")
    row1_col2.metric("Odio detectado", f"{n_odio} ({pct_odio:.1f}%)", help="´Número y porcentaje de de fragmentos con odio detectado.")
    
    row2_col1.metric(
        "Sentimiento negativo",
        n_neg,
        delta = f"{pct_neg:.1f}%",
        delta_color = "inverse",
        help = "Número y porcentaje de fragmentos con sentimientos negativos."
    )
    row2_col2.metric(
        "Sentimiento neutral",
        n_neu,
        delta=f"{pct_neu:.1f}%",
        delta_color="gray",
        help="Número y porcentaje de fragmentos con sentimientos neutrales."
    )
    row2_col3.metric(
        "Sentimiento positivo",
        n_pos,
        delta=f"{pct_pos:.1f}%",
        delta_color="normal",
        help="Número y porcentaje de fragmentos con sentimientos positivos."
    )
    
    st.divider()
 
# ── Fila 1: Sentimiento + Emociones ─────────────────────────────────────────
def _render_sentiment_section(df: pd.DataFrame):
    """
    Visualiza la distribución de sentimiento y probabilidades por fragmento.

    Incluye:
    - Gráfico de pastel con distribución global
    - Barras apiladas con probabilidades por fragmento
    """
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Distribución de Sentimiento")
        sent_counts = df["sentimiento"].value_counts().reset_index()
        sent_counts.columns = ["Sentimiento", "Fragmentos"]
        color_map = {"Negativo": "#e74c3c", "Neutral": "#3498db", "Positivo": "#2ecc71"}
        fig_sent = px.pie(
            sent_counts, names="Sentimiento", values="Fragmentos",
            color="Sentimiento", color_discrete_map=color_map,
            hole=0.45,
        )
        fig_sent.update_traces(textposition="inside", textinfo="percent+label")
        fig_sent.update_layout(margin=dict(t=20, b=20, l=20, r=20), showlegend=True)
        st.plotly_chart(fig_sent, width='stretch')
    
    with col_right:
        st.subheader("Probabilidades de Sentimiento por Fragmento")
        if all(c in df.columns for c in ["prob_pos", "prob_neg", "prob_neu"]):
            fig_prob = go.Figure()
            frag_labels = df["fragmento"].astype(str).tolist()
            fig_prob.add_trace(go.Bar(name="Negativo", x=frag_labels, y=df["prob_neg"], marker_color="#e74c3c"))
            fig_prob.add_trace(go.Bar(name="Neutral",  x=frag_labels, y=df["prob_neu"], marker_color="#3498db"))
            fig_prob.add_trace(go.Bar(name="Positivo", x=frag_labels, y=df["prob_pos"], marker_color="#2ecc71"))
            fig_prob.update_layout(
                barmode="stack",
                xaxis_title="Fragmento",
                yaxis_title="Probabilidad (%)",
                legend_title="Sentimiento",
                margin=dict(t=20, b=30),
            )
            st.plotly_chart(fig_prob, width='stretch')
    
    st.divider()
 
# ── Fila 2: Discurso de odio + Sentimiento dirigido ──────────────────────────
def _render_hate_section(df: pd.DataFrame):
    """
    Visualiza el análisis de discurso de odio y sentimiento dirigido.

    Incluye:
    - Distribución de etiquetas de discurso de odio.
    - Comparación de sentimiento dirigido por fragmento.

    Visualizaciones:
    - Gráfico de barras (odio detectado vs no).
    - Gráfico apilado de probabilidades de sentimiento dirigido.
    """

    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Discurso de Odio")
        odio_counts = df["odio_detectado"].value_counts().reset_index()
        odio_counts.columns = ["Etiqueta", "Fragmentos"]
        fig_odio = px.bar(
            odio_counts, x="Etiqueta", y="Fragmentos",
            color="Etiqueta",
            color_discrete_map={"No": "#95a5a6", "odio": "#c0392b"},
            text="Fragmentos",
        )
        fig_odio.update_traces(textposition="outside")
        fig_odio.update_layout(showlegend=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig_odio, width='stretch')
    
    with col_right:
        st.subheader("Sentimiento Dirigido")
        if all(c in df.columns for c in ["prob_sent_dirigido_NEG", "prob_sent_dirigido_NEU", "prob_sent_dirigido_POS"]):
            fig_dir = go.Figure()
            frag_labels = df["fragmento"].astype(str).tolist()
            fig_dir.add_trace(go.Bar(name="NEG", x=frag_labels, y=df["prob_sent_dirigido_NEG"], marker_color="#e74c3c"))
            fig_dir.add_trace(go.Bar(name="NEU", x=frag_labels, y=df["prob_sent_dirigido_NEU"], marker_color="#3498db"))
            fig_dir.add_trace(go.Bar(name="POS", x=frag_labels, y=df["prob_sent_dirigido_POS"], marker_color="#2ecc71"))
            fig_dir.update_layout(
                barmode="stack",
                xaxis_title="Fragmento",
                yaxis_title="Probabilidad (%)",
                legend_title="Sent. Dirigido",
                margin=dict(t=20, b=30),
            )
            st.plotly_chart(fig_dir, width='stretch')
    
    st.divider()
 
# ── Fila 3: Radar de emociones ───────────────────────────────────────────────
def _render_emotion_section(df: pd.DataFrame):
    """
    Representa el perfil emocional agregado del dataset.

    Calcula la media de emociones por tipo de sentimiento y las visualiza en
    un gráfico radar comparativo por cada sentimiento.

    Emociones incluidas:
    - Alegría
    - Tristeza
    - Ira
    - Sorpresa
    - Asco
    - Miedo
    """
    st.subheader("🎭 Perfil Emocional (media por sentimiento)")
    
    emo_cols = ["emo_alegría", "emo_tristeza", "emo_ira", "emo_sorpresa", "emo_asco", "emo_miedo"]
    emo_present = [c for c in emo_cols if c in df.columns]
    
    if emo_present:
        emo_labels = [c.replace("emo_", "").capitalize() for c in emo_present]
        fig_radar = go.Figure()
        colors = {"Negativo": "#e74c3c", "Neutral": "#3498db", "Positivo": "#2ecc71"}
        for sent_val in df["sentimiento"].unique():
            subset = df[df["sentimiento"] == sent_val]
            medias = subset[emo_present].mean().tolist()
            medias.append(medias[0])  # Añadir punto de cierre al polígono.

            fig_radar.add_trace(go.Scatterpolar(
                r = medias,
                theta = emo_labels + [emo_labels[0]],
                fill = "toself",
                name = sent_val,
                line_color = colors.get(sent_val, "#7f8c8d"),
                opacity = 0.6,
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max(medias) * 1.1])),
            margin=dict(t=40, b=40),
            legend_title="Sentimiento",
        )
        st.plotly_chart(fig_radar, width='stretch')
    
    st.divider()
 
# ── Tabla de fragmentos ──────────────────────────────────────────────────────
def _render_data_table(df: pd.DataFrame):
    """
    Muestra la tabla detallada de fragmentos analizados.

    Esta tabla permite inspeccionar los resultados del pipeline, con todas las entidades
    que se han ido nombrando. Incluye formato condicional para resaltar el sentimiento.
    """

    st.subheader("📋 Fragmentos detallados")
    
    cols_tabla = ["fragmento", "texto", "sentimiento", "prob_neg", "prob_neu", "prob_pos",
                "odio_detectado", "prob_odio", "sentimiento_dirigido", "entidades"]
    cols_tabla = [c for c in cols_tabla if c in df.columns]
    
    def colorear_sentimiento(val):
        if val == "Negativo":
            return "background-color: #a4a4a4; color: #c0392b"
        elif val == "Positivo":
            return "background-color: #a4a4a4; color: #27ae60"
        elif val == "Neutral":
            return "background-color: #a4a4a4; color: #2980b9"
        return ""
    
    styled = df[cols_tabla].style.map(colorear_sentimiento, subset=["sentimiento"])
    st.dataframe(styled, width='stretch', height=350)
 
# ── Footer ───────────────────────────────────────────────────────────────────
def _render_footer():
    st.markdown("---")
    st.caption("Dashboard generado con Streamlit · Datos: análisis-textos-json.csv")

def main():
    """
    Orquesta la ejecución completa del dashboard de análisis de textos, siendo expuesto al exterior.

    Este es el punto de entrada principal de la aplicación Streamlit y define el flujo
    general de la interfaz.

    Flujo de ejecución:
        1. Renderizado del encabezado.
        2. Carga de datos (upload o dataset por defecto).
        3. Aplicación de filtros en sidebar.
        4. Cálculo de métricas (KPIs).
        5. Renderizado de visualizaciones (Sentimiento, discurso de odio y emociones).
        6. Renderizado de tabla detallada.
        7. Renderizado del footer.

    Notes:
        Streamlit gestiona el estado de la interfaz de forma declarativa, así que no
        devuelve nada. 
    """
    
    _render_header()
    
    df = _render_sidebar()
    df = _render_sidebar_filters(df)
    _render_kpis(df)

    _render_sentiment_section(df)
    _render_hate_section(df)
    _render_emotion_section(df)

    _render_data_table(df)
    _render_footer()

if __name__ == "__main__":
    main()