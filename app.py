import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import io
import os

# --- Configuration de la page ---
st.set_page_config(
    page_title="Analyse Technique des Marchés",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Styles CSS ---
st.markdown("""
<style>
    .stSelectbox, .stDateInput, .stNumberInput {margin-bottom: 1rem;}
    .metric-card {background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;}
    .positive {color: green;}
    .negative {color: red;}
</style>
""", unsafe_allow_html=True)

# --- Fonctions utilitaires ---
@st.cache_data(ttl=3600)
def fetch_data(symbols, period="5y", interval="1d"):
    """Récupère les données historiques pour une liste de symboles."""
    data = {}
    for symbol in symbols:
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period, interval=interval)
            if not df.empty:
                df.index = df.index.tz_localize(None)  # Supprime le timezone
                data[symbol] = df
        except Exception as e:
            st.error(f"Erreur pour {symbol}: {e}")
    return data

def calculate_ta(df):
    """Calcule les indicateurs techniques."""
    # SMA
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Histogram'] = df['MACD'] - df['Signal']

    # Bollinger Bands
    df['BB_upper'] = df['Close'].rolling(window=20).mean() + 2 * df['Close'].rolling(window=20).std()
    df['BB_lower'] = df['Close'].rolling(window=20).mean() - 2 * df['Close'].rolling(window=20).std()

    return df

def create_candlestick(df, title):
    """Crée un graphique en chandeliers avec Plotly."""
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Prix'
    )])
    fig.update_layout(
        title=title,
        yaxis_title='Prix',
        xaxis_rangeslider_visible=False,
        height=500
    )
    return fig

def create_ta_chart(df, title):
    """Crée un graphique avec SMA et Bollinger Bands."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        name='Prix',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['SMA_50'],
        name='SMA 50',
        line=dict(color='orange', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['SMA_200'],
        name='SMA 200',
        line=dict(color='red', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['BB_upper'],
        name='BB Upper',
        line=dict(color='gray', dash='dot')
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['BB_lower'],
        name='BB Lower',
        line=dict(color='gray', dash='dot')
    ))
    fig.update_layout(
        title=title,
        yaxis_title='Prix',
        height=500
    )
    return fig

def create_rsi_chart(df, title):
    """Crée un graphique RSI."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['RSI'],
        name='RSI',
        line=dict(color='purple')
    ))
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Surachat (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Survente (30)")
    fig.update_layout(
        title=title,
        yaxis_title='RSI',
        yaxis_range=[0, 100],
        height=300
    )
    return fig

def create_macd_chart(df, title):
    """Crée un graphique MACD."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MACD'],
        name='MACD',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Signal'],
        name='Signal',
        line=dict(color='orange')
    ))
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Histogram'],
        name='Histogram',
        marker_color=np.where(df['Histogram'] > 0, 'green', 'red')
    ))
    fig.update_layout(
        title=title,
        yaxis_title='MACD',
        height=300
    )
    return fig

# --- Données des indices et secteurs ---
INDICES = {
    "CAC40": "^FCHI",
    "S&P500": "^GSPC",
    "NASDAQ": "^IXIC",
    "Euro Stoxx 50": "^STOXX50E",
    "DAX": "^GDAXI",
    "Nikkei 225": "^N225"
}

SECTEURS = {
    "Technologie": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"],
    "Santé": ["JNJ", "UNH", "PFE", "ABBV", "MRK"],
    "Énergie": ["XOM", "CVX", "COP", "SLB", "EOG"],
    "Finance": ["JPM", "BAC", "WFC", "GS", "MS"],
    "Consommation": ["PG", "KO", "PEP", "WMT", "COST"],
    "Industrie": ["BA", "CAT", "MMM", "GE", "HON"]
}

# --- Sidebar ---
st.sidebar.title("⚙️ Paramètres")
st.sidebar.markdown("---")

# Sélection de la période
period = st.sidebar.selectbox(
    "Période",
    ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
    index=6  # 5y par défaut
)

# Sélection du type d'analyse
analysis_type = st.sidebar.radio(
    "Type d'analyse",
    ["Indices", "Secteurs"]
)

# Sélection des symboles
if analysis_type == "Indices":
    selected_symbols = st.sidebar.multiselect(
        "Sélectionne les indices",
        list(INDICES.keys()),
        default=["CAC40", "S&P500"]
    )
    symbols = [INDICES[s] for s in selected_symbols]
else:
    selected_sectors = st.sidebar.multiselect(
        "Sélectionne les secteurs",
        list(SECTEURS.keys()),
        default=["Technologie"]
    )
    symbols = []
    for sector in selected_sectors:
        symbols.extend(SECTEURS[sector])

# --- Chargement des données ---
if not symbols:
    st.warning("Veuillez sélectionner au moins un indice ou secteur.")
    st.stop()

with st.spinner("Chargement des données..."):
    data = fetch_data(symbols, period=period)

if not data:
    st.error("Aucune donnée disponible pour les symboles sélectionnés.")
    st.stop()

# --- Calcul des indicateurs techniques ---
for symbol in data:
    data[symbol] = calculate_ta(data[symbol])

# --- Affichage principal ---
st.title("📊 Analyse Technique des Marchés")
st.markdown(f"**Période:** {period} | **Type:** {analysis_type}")

# --- Onglets pour chaque symbole ---
for symbol, df in data.items():
    with st.expander(f"📈 {symbol}", expanded=True):
        col1, col2 = st.columns([2, 1])

        with col1:
            # Graphique principal (Prix + SMA + BB)
            ta_fig = create_ta_chart(df, f"{symbol} - Prix et Indicateurs")
            st.plotly_chart(ta_fig, use_container_width=True)

            # Graphiques RSI et MACD
            rsi_fig = create_rsi_chart(df, f"{symbol} - RSI")
            macd_fig = create_macd_chart(df, f"{symbol} - MACD")
            st.plotly_chart(rsi_fig, use_container_width=True)
            st.plotly_chart(macd_fig, use_container_width=True)

        with col2:
            # Métriques
            st.markdown("### 📊 Métriques Clés")
            last_row = df.iloc[-1]
            current_price = last_row['Close']
            sma_50 = last_row['SMA_50']
            sma_200 = last_row['SMA_200']
            rsi = last_row['RSI']

            # Signal Bullish/Bearish
            if current_price > sma_50 > sma_200:
                signal = "🟢 **Bullish**"
                signal_color = "green"
            elif current_price < sma_50 < sma_200:
                signal = "🔴 **Bearish**"
                signal_color = "red"
            else:
                signal = "🟡 **Neutre**"
                signal_color = "orange"

            st.markdown(f"""
            <div class="metric-card">
                <h4>Prix actuel</h4>
                <p>{current_price:.2f}</p>
                <h4>SMA 50</h4>
                <p>{sma_50:.2f}</p>
                <h4>SMA 200</h4>
                <p>{sma_200:.2f}</p>
                <h4>RSI</h4>
                <p>{rsi:.2f}</p>
                <h4>Signal</h4>
                <p style="color:{signal_color};">{signal}</p>
            </div>
            """, unsafe_allow_html=True)

            # Surachat/Survente
            if rsi > 70:
                st.markdown("<p style='color:red;'>⚠️ **Surachat (RSI > 70)**</p>", unsafe_allow_html=True)
            elif rsi < 30:
                st.markdown("<p style='color:green;'>⚠️ **Survente (RSI < 30)**</p>", unsafe_allow_html=True)

# --- Export Excel ---
st.markdown("---")
st.markdown("### 📥 Export des données")
if st.button("Exporter vers Excel"):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for symbol, df in data.items():
            df.to_excel(writer, sheet_name=symbol[:31])  # Limite de 31 caractères pour Excel
    output.seek(0)
    st.download_button(
        label="Télécharger le fichier Excel",
        data=output,
        file_name=f"analyse_marches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
