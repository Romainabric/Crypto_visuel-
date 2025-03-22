
import streamlit as st
import requests
import matplotlib.pyplot as plt
import datetime
import numpy as np
import matplotlib.dates as mdates

# --- Récupération des cryptos ---
@st.cache_data
def get_top_cryptos():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency": "usd", "order": "market_cap_desc", "per_page": 20, "page": 1}
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            return {coin["id"]: f"{coin['name']} ({coin['symbol'].upper()})" for coin in data}
    except:
        pass
    return {
        "bitcoin": "Bitcoin (BTC)",
        "ethereum": "Ethereum (ETH)",
        "binancecoin": "Binance Coin (BNB)",
        "ripple": "XRP",
        "cardano": "Cardano (ADA)",
        "solana": "Solana (SOL)",
        "dogecoin": "Dogecoin (DOGE)",
        "polkadot": "Polkadot (DOT)"
    }

# --- Fonctions d'analyse ---
def compute_ema(values, period):
    ema = np.zeros_like(values)
    ema[:period] = np.mean(values[:period])
    alpha = 2 / (period + 1)
    for i in range(period, len(values)):
        ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]
    return ema

def compute_rsi(values, period):
    delta = np.diff(values)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.convolve(gain, np.ones(period)/period, mode='valid')
    avg_loss = np.convolve(loss, np.ones(period)/period, mode='valid')
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def compute_macd(values, short_period, long_period, signal_period):
    macd_short = compute_ema(values, short_period)
    macd_long = compute_ema(values, long_period)
    macd_line = macd_short - macd_long
    signal_line = compute_ema(macd_line, signal_period)
    return macd_line, signal_line

def analyze_crypto(crypto_id, display_days, time_interval):
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart?vs_currency=usd&days=30"
    response = requests.get(url)
    if response.status_code != 200:
        st.error("Erreur lors de la récupération des données.")
        return None

    data = response.json()
    prices = data["prices"]
    dates = np.array([datetime.datetime.fromtimestamp(p[0]/1000) for p in prices])
    values = np.array([p[1] for p in prices])

    ema = compute_ema(values, 10)
    rsi = compute_rsi(values, 14)
    macd_line, signal_line = compute_macd(values, 12, 26, 9)

    buy_signals, sell_signals = [], []
    for i in range(1, len(ema)):
        if i < 14:
            continue
        if (rsi[i - 14] < 30 and ema[i] > values[i]) or (macd_line[i] > signal_line[i] and macd_line[i - 1] < signal_line[i - 1]):
            buy_signals.append((dates[i], values[i]))
        if (rsi[i - 14] > 70 and ema[i] < values[i]) or (macd_line[i] < signal_line[i] and macd_line[i - 1] > signal_line[i - 1]):
            sell_signals.append((dates[i], values[i]))

    end_date = dates[-1]
    start_date = end_date - datetime.timedelta(days=display_days)
    mask = dates >= start_date
    dates_display, values_display = dates[mask], values[mask]

    buy_signals_display = [(d, v) for d, v in buy_signals if d >= start_date]
    sell_signals_display = [(d, v) for d, v in sell_signals if d >= start_date]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates_display, values_display, label='Prix', color='blue')

    for d, v in buy_signals_display:
        ax.scatter(d, v, marker="^", color="green", s=100, label="Buy")
    for d, v in sell_signals_display:
        ax.scatter(d, v, marker="v", color="red", s=100, label="Sell")

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=time_interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %Hh'))
    plt.xticks(rotation=45)
    plt.grid()
    plt.xlabel("Date")
    plt.ylabel("Prix (USD)")
    plt.title(f"{crypto_id.upper()} - Signaux de Trading")
    ax.legend(loc="upper left")

    return fig

# --- Interface Streamlit ---
st.set_page_config(layout="wide")
st.title("📊 Signaux de Trading Crypto")

top_cryptos = get_top_cryptos()
selected_crypto = st.selectbox("Sélectionnez une crypto :", options=list(top_cryptos.keys()), format_func=lambda x: top_cryptos[x])
days = st.slider("Nombre de jours à afficher :", 1, 30, 7)
interval = st.slider("Intervalle d'affichage (heures) :", 1, 24, 6)

if st.button("📈 Analyser"):
    with st.spinner("Chargement du graphique..."):
        fig = analyze_crypto(selected_crypto, days, interval)
        if fig:
            st.pyplot(fig)
