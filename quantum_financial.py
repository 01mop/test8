"""
╔══════════════════════════════════════════════════════════════════════╗
║   QUANTUM TERMINAL  v3.0  —  All-in-One Professional Quant Dashboard ║
║   Market Data · Statistical Analysis · Backtesting                   ║
║   Risk Management · Factor Models · ML/AI · Correlation              ║
╚══════════════════════════════════════════════════════════════════════╝
Run:  streamlit run quantum_terminal.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import requests
import warnings
warnings.filterwarnings("ignore")

# ── optional imports ───────────────────────────────────────────────────
try:
    import ccxt; CCXT_OK = True
except ImportError:
    CCXT_OK = False

try:
    from fredapi import Fred; FRED_OK = True
except ImportError:
    FRED_OK = False

try:
    from arch import arch_model; ARCH_OK = True
except ImportError:
    ARCH_OK = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer(); VADER_OK = True
except ImportError:
    VADER_OK = False

try:
    import xgboost as xgb; XGB_OK = True
except ImportError:
    XGB_OK = False

try:
    from statsmodels.tsa.stattools import coint, adfuller
    from statsmodels.tsa.arima.model import ARIMA
    import statsmodels.api as sm; SM_OK = True
except ImportError:
    SM_OK = False

# ══════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="QUANTUM TERMINAL v3",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=IBM+Plex+Sans:wght@300;400;600&family=Orbitron:wght@700;900&display=swap');

html, body, [class*="css"] { background-color:#070b10!important; color:#c8d8e8!important; font-family:'IBM Plex Sans',sans-serif; }
.stApp { background-color:#070b10; }
section[data-testid="stSidebar"] { background:linear-gradient(180deg,#0a1520,#080f18)!important; border-right:1px solid #1a2f45; }
section[data-testid="stSidebar"] * { color:#7aafc8!important; }
section[data-testid="stSidebar"] label { color:#4da8da!important; font-family:'Space Mono',monospace; font-size:10px; letter-spacing:1.5px; text-transform:uppercase; }
.stTextInput>div>div input, .stSelectbox>div>div, .stMultiSelect>div>div { background-color:#0d1e2e!important; border:1px solid #1e3a55!important; border-radius:2px!important; color:#4da8da!important; font-family:'Space Mono',monospace!important; }
.stTabs [data-baseweb="tab-list"] { background:#0a1520; border-bottom:1px solid #1a3a55; gap:0; }
.stTabs [data-baseweb="tab"] { color:#4a7a99; font-family:'Space Mono',monospace; font-size:10px; letter-spacing:1px; padding:10px 16px; border-bottom:2px solid transparent; }
.stTabs [aria-selected="true"] { color:#00d4ff!important; border-bottom:2px solid #00d4ff!important; background:transparent!important; }
[data-testid="metric-container"] { background:linear-gradient(135deg,#0d1f30,#0a1520); border:1px solid #1a3a55; border-left:3px solid #00d4ff; border-radius:2px; padding:10px 14px; }
[data-testid="metric-container"] label { color:#4da8da!important; font-family:'Space Mono',monospace; font-size:9px; letter-spacing:1.5px; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color:#e8f4fd!important; font-family:'Space Mono',monospace; font-size:16px; }
.blabel { font-family:'Space Mono',monospace; font-size:9px; letter-spacing:2px; color:#2a6a8a; text-transform:uppercase; border-bottom:1px solid #1a3a55; padding-bottom:4px; margin-bottom:10px; }
.tag-free  { background:#0d2a1a; border:1px solid #00ff88; color:#00ff88; font-family:'Space Mono',monospace; font-size:9px; padding:2px 7px; border-radius:2px; display:inline-block; margin:2px; }
.tag-key   { background:#2a1a0d; border:1px solid #ffaa00; color:#ffaa00; font-family:'Space Mono',monospace; font-size:9px; padding:2px 7px; border-radius:2px; display:inline-block; margin:2px; }
.tag-limit { background:#2a0d0d; border:1px solid #ff4466; color:#ff4466; font-family:'Space Mono',monospace; font-size:9px; padding:2px 7px; border-radius:2px; display:inline-block; margin:2px; }
.news-card { background:#0d1f30; border:1px solid #1a3a55; border-left:3px solid #00d4ff; border-radius:2px; padding:10px 14px; margin-bottom:8px; }
.news-title { font-family:'IBM Plex Sans',sans-serif; font-size:13px; color:#e8f4fd; font-weight:600; }
.news-meta  { font-family:'Space Mono',monospace; font-size:9px; color:#4a7a99; margin-top:4px; }
.status-dot { display:inline-block; width:6px; height:6px; border-radius:50%; background:#00ff88; box-shadow:0 0 6px #00ff88; margin-right:5px; animation:pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1}50%{opacity:.3} }
hr { border-color:#1a3a55!important; }
::-webkit-scrollbar { width:3px; }
::-webkit-scrollbar-thumb { background:#1e3a55; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  CONSTANTS & HELPERS
# ══════════════════════════════════════════════════════════════════════
COLORS = ["#00d4ff","#00ff88","#ffaa00","#ff4466","#aa88ff","#ff88aa","#44ffdd","#ffdd44"]

PLOTLY_BASE = dict(
    paper_bgcolor="#070b10", plot_bgcolor="#070b10",
    font=dict(family="Space Mono, monospace", color="#8aafc8", size=10),
    xaxis=dict(gridcolor="#0d1e2e", zerolinecolor="#0d1e2e"),
    yaxis=dict(gridcolor="#0d1e2e", zerolinecolor="#0d1e2e"),
    margin=dict(l=50,r=20,t=36,b=36),
    legend=dict(bgcolor="#0a1520", bordercolor="#1a3a55", borderwidth=1),
)

def blabel(txt):
    st.markdown(f"<div class='blabel'>▸ {txt}</div>", unsafe_allow_html=True)

def fmt_large(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "N/A"
    if abs(v)>=1e12: return f"${v/1e12:.2f}T"
    if abs(v)>=1e9:  return f"${v/1e9:.2f}B"
    if abs(v)>=1e6:  return f"${v/1e6:.2f}M"
    return f"${v:,.0f}"

def pct_fmt(v): return f"{v*100:.2f}%" if isinstance(v,(int,float)) and not np.isnan(v) else "N/A"

# ── cached data fetchers ───────────────────────────────────────────────
@st.cache_data(ttl=300)
def get_ohlcv(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df.dropna()
    except: return pd.DataFrame()

@st.cache_data(ttl=600)
def get_info(ticker):
    try: return yf.Ticker(ticker).info or {}
    except: return {}

@st.cache_data(ttl=300)
def get_option_expirations(ticker):
    try:
        exps = yf.Ticker(ticker).options
        return list(exps) if exps else []
    except: return []

def get_options(ticker):
    exps = get_option_expirations(ticker)
    if not exps: return None, None, []
    try: return yf.Ticker(ticker), exps[0], exps
    except: return None, None, []

@st.cache_data(ttl=120)
def get_ccxt_ob(exchange_id, symbol, limit=20):
    if not CCXT_OK: return None, "pip install ccxt"
    try:
        ex = getattr(ccxt, exchange_id)(); ob = ex.fetch_order_book(symbol, limit=limit)
        return ob, None
    except Exception as e: return None, str(e)

@st.cache_data(ttl=120)
def get_ccxt_ohlcv(exchange_id, symbol, timeframe="1h", limit=300):
    if not CCXT_OK: return pd.DataFrame(), "pip install ccxt"
    try:
        ex = getattr(ccxt, exchange_id)()
        raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["ts","Open","High","Low","Close","Volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms"); df.set_index("ts", inplace=True)
        return df, None
    except Exception as e: return pd.DataFrame(), str(e)

@st.cache_data(ttl=1800)
def get_fred(series_id, key):
    if not FRED_OK: return pd.Series(dtype=float)
    try: return Fred(api_key=key).get_series(series_id).dropna()
    except: return pd.Series(dtype=float)

@st.cache_data(ttl=60)
def get_forex_rate(base, target, key):
    try:
        r = requests.get(f"https://v6.exchangerate-api.com/v6/{key}/pair/{base}/{target}", timeout=8).json()
        return (r["conversion_rate"], None) if r.get("result")=="success" else (None, r.get("error-type"))
    except Exception as e: return None, str(e)

@st.cache_data(ttl=600)
def get_news_api(query, key, days=7):
    try:
        params = dict(q=query, from_=(datetime.now()-timedelta(days=days)).strftime("%Y-%m-%d"),
                      sortBy="publishedAt", language="en", pageSize=20, apiKey=key)
        r = requests.get("https://newsapi.org/v2/everything", params=params, timeout=10).json()
        return (r.get("articles",[]), None) if r.get("status")=="ok" else ([], r.get("message","Erreur"))
    except Exception as e: return [], str(e)

# ── technical indicators ───────────────────────────────────────────────
def add_indicators(df, sma1=20, sma2=50, ema=21, bb=20, rsi=14):
    c = df["Close"].squeeze(); df = df.copy()
    df[f"SMA{sma1}"] = c.rolling(sma1).mean()
    df[f"SMA{sma2}"] = c.rolling(sma2).mean()
    df[f"EMA{ema}"]  = c.ewm(span=ema, adjust=False).mean()
    df["BB_mid"] = c.rolling(bb).mean()
    bb_std = c.rolling(bb).std()
    df["BB_up"] = df["BB_mid"] + 2*bb_std
    df["BB_dn"] = df["BB_mid"] - 2*bb_std
    delta = c.diff(); gain = delta.clip(lower=0).rolling(rsi).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi).mean()
    df["RSI"] = 100 - 100/(1 + gain/loss.replace(0,np.nan))
    ema12 = c.ewm(span=12,adjust=False).mean(); ema26 = c.ewm(span=26,adjust=False).mean()
    df["MACD"] = ema12-ema26; df["MACD_sig"] = df["MACD"].ewm(span=9,adjust=False).mean()
    df["MACD_hist"] = df["MACD"]-df["MACD_sig"]
    df["ATR"] = pd.concat([df["High"].squeeze()-df["Low"].squeeze(),
                            (df["High"].squeeze()-c.shift()).abs(),
                            (df["Low"].squeeze()-c.shift()).abs()], axis=1).max(axis=1).rolling(14).mean()
    return df

# ── risk metrics ───────────────────────────────────────────────────────
def compute_var(returns, confidence=0.95, method="historical"):
    r = returns.dropna()
    if method == "historical":
        return np.percentile(r, (1-confidence)*100)
    elif method == "parametric":
        mu, sigma = r.mean(), r.std()
        return stats.norm.ppf(1-confidence, mu, sigma)
    elif method == "cornish_fisher":
        mu, sigma = r.mean(), r.std()
        s = stats.skew(r); k = stats.kurtosis(r)
        z = stats.norm.ppf(1-confidence)
        z_cf = z + (z**2-1)*s/6 + (z**3-3*z)*k/24 - (2*z**3-5*z)*s**2/36
        return mu + z_cf*sigma

def compute_cvar(returns, confidence=0.95):
    r = returns.dropna(); var = compute_var(r, confidence)
    return r[r <= var].mean()

def sharpe(returns, rf=0.0): 
    r = returns.dropna()
    return (r.mean()-rf/252)/(r.std()+1e-9)*np.sqrt(252)

def sortino(returns, rf=0.0):
    r = returns.dropna(); downside = r[r<0].std()
    return (r.mean()-rf/252)/(downside+1e-9)*np.sqrt(252)

def max_drawdown(prices):
    p = prices.dropna(); roll_max = p.cummax()
    dd = (p - roll_max)/roll_max; return dd.min()

def calmar(returns, prices):
    ann = returns.dropna().mean()*252; mdd = abs(max_drawdown(prices))
    return ann/(mdd+1e-9)

# ── portfolio optimization ─────────────────────────────────────────────
def optimize_portfolio(returns_df, method="sharpe"):
    n = returns_df.shape[1]; mu = returns_df.mean()*252
    cov = returns_df.cov()*252; w0 = np.ones(n)/n
    bounds = [(0,1)]*n; cons = [{"type":"eq","fun":lambda w: w.sum()-1}]
    if method == "sharpe":
        def neg_sharpe(w):
            p_ret = w@mu; p_vol = np.sqrt(w@cov@w)
            return -p_ret/(p_vol+1e-9)
        res = minimize(neg_sharpe, w0, bounds=bounds, constraints=cons)
    elif method == "min_vol":
        def port_vol(w): return np.sqrt(w@cov@w)
        res = minimize(port_vol, w0, bounds=bounds, constraints=cons)
    elif method == "equal": return w0
    return res.x if res.success else w0

# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 8px'>
      <div style='font-family:Orbitron,sans-serif;font-size:15px;font-weight:900;
                  letter-spacing:4px;color:#00d4ff;text-shadow:0 0 12px rgba(0,212,255,.4);'>⬡ QUANTUM</div>
      <div style='font-family:Space Mono,monospace;font-size:8px;letter-spacing:2px;color:#2a5a7a;margin-top:2px;'>TERMINAL v3.0 · ALL-IN-ONE</div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    st.markdown("<div class='blabel'>▸ Universe</div>", unsafe_allow_html=True)

    st.markdown("""<div style='font-family:Space Mono,monospace;font-size:9px;color:#4a7a99;line-height:1.8;'>
    Entrez vos tickers séparés par des virgules.<br>
    Actions: AAPL, MSFT, LVMH.PA, AIR.PA<br>
    ETF: SPY, QQQ, IWDA.AS<br>
    Crypto: BTC-USD, ETH-USD<br>
    Forex: EURUSD=X, GBPUSD=X<br>
    Futures: GC=F (Gold), CL=F (Oil)<br>
    Indices: ^GSPC, ^IXIC, ^FCHI
    </div>""", unsafe_allow_html=True)

    raw_input = st.text_area(
        "Tickers (séparés par virgules)",
        value="AAPL, MSFT, NVDA, TSLA",
        height=80,
        help="Exemples: AAPL, LVMH.PA, BTC-USD, ^GSPC, GC=F, EURUSD=X"
    )

    # Parse tickers from text input
    selected = [t.strip().upper() for t in raw_input.replace("\n", ",").split(",") if t.strip()]
    selected = list(dict.fromkeys(selected))[:8]  # deduplicate, max 8

    if not selected:
        selected = ["AAPL"]

    # Validate tickers and show status
    if st.button("✓ Valider les tickers", use_container_width=True):
        st.session_state["validated_tickers"] = selected

    primary  = st.selectbox("Ticker principal (graphique)", selected)

    st.markdown(f"""<div style='font-family:Space Mono,monospace;font-size:9px;color:#2a6a8a;margin-top:4px;'>
    {len(selected)} ticker(s) chargé(s) : {' · '.join(selected)}
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("<div class='blabel'>▸ Fenêtre temporelle</div>", unsafe_allow_html=True)
    PERIODS = {"1 sem":("7d","60m"),"1 mois":("1mo","1d"),"3 mois":("3mo","1d"),
               "6 mois":("6mo","1d"),"1 an":("1y","1d"),"2 ans":("2y","1wk"),"5 ans":("5y","1wk")}
    period_l = st.selectbox("Période", list(PERIODS.keys()), index=3)
    period, interval = PERIODS[period_l]
    chart_type = st.selectbox("Type de graphique", ["Candlestick","OHLC","Line"])

    st.divider()
    st.markdown("<div class='blabel'>▸ Indicateurs</div>", unsafe_allow_html=True)
    show_sma  = st.checkbox("SMA",             value=True)
    show_ema  = st.checkbox("EMA",             value=False)
    show_bb   = st.checkbox("Bollinger Bands", value=True)
    show_vol  = st.checkbox("Volume",          value=True)
    show_rsi  = st.checkbox("RSI",             value=True)
    show_macd = st.checkbox("MACD",            value=True)
    with st.expander("Fenêtres"):
        sma1=st.slider("SMA Fast",5,50,20); sma2=st.slider("SMA Slow",20,200,50)
        ema_w=st.slider("EMA",5,100,21); bb_w=st.slider("BB",10,50,20); rsi_w=st.slider("RSI",5,30,14)

    st.divider()
    st.markdown("<div class='blabel'>▸ Clés API (optionnelles)</div>", unsafe_allow_html=True)
    fred_key  = st.text_input("FRED API Key",       type="password", placeholder="fred.stlouisfed.org")
    av_key    = st.text_input("Alpha Vantage Key",  type="password", placeholder="alphavantage.co")
    news_key  = st.text_input("NewsAPI Key",        type="password", placeholder="newsapi.org")
    forex_key = st.text_input("ExchangeRate Key",   type="password", placeholder="exchangerate-api.com")

    st.divider()
    st.markdown(f"""
    <div style='font-family:Space Mono,monospace;font-size:9px;color:#1e3a55;text-align:center;'>
      <span class='status-dot'></span>LIVE · {datetime.now().strftime('%H:%M:%S')}
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style='display:flex;align-items:baseline;gap:16px;margin-bottom:4px;'>
  <span style='font-family:Orbitron,sans-serif;font-size:24px;font-weight:900;letter-spacing:5px;
               color:#00d4ff;text-shadow:0 0 18px rgba(0,212,255,.35);'>QUANTUM TERMINAL</span>
  <span style='font-family:Space Mono,monospace;font-size:9px;color:#2a5a7a;letter-spacing:2px;'>
    <span class='status-dot'></span>{datetime.now().strftime('%d %b %Y · %H:%M UTC')}
  </span>
</div>
<div style='font-family:Space Mono,monospace;font-size:9px;color:#1e3a55;letter-spacing:1px;margin-bottom:12px;'>
  Market Data · Statistical Analysis · Backtesting · Risk Management · Factor Models · ML/AI
</div>""", unsafe_allow_html=True)

if not selected:
    st.warning("Sélectionnez au moins un ticker dans la barre latérale.")
    st.stop()

# ── live price strip ──────────────────────────────────────────────────
pcols = st.columns(len(selected))
for i, tkr in enumerate(selected):
    info = get_info(tkr)
    price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
    prev  = info.get("previousClose")
    chg   = (price-prev)/prev*100 if price and prev else None
    with pcols[i]:
        st.metric(f"**{tkr}**", f"${price:,.2f}" if price else "N/A",
                  f"{chg:+.2f}%" if chg is not None else None,
                  delta_color="normal" if (chg or 0)>=0 else "inverse")
st.divider()

# ══════════════════════════════════════════════════════════════════════
#  MAIN TABS
# ══════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📈  Chart",
    "🔗  Options",
    "₿   Crypto L2",
    "📊  Statistiques",
    "🎯  Backtesting",
    "⚠️  Risk",
    "🏗️  Portefeuille",
    "🌐  Macro",
    "💱  Forex",
    "📰  News",
    "🏢  Fondamentaux",
    "🔥  Corrélation",
])

# ══════════════════════════════════════════════════════════════════════
#  TAB 0 — CHART & INDICATORS
# ══════════════════════════════════════════════════════════════════════
with tabs[0]:
    blabel("yfinance · OHLCV · Indicateurs Techniques")
    df_raw = get_ohlcv(primary, period, interval)
    if df_raw.empty:
        st.error(f"Aucune donnée pour {primary}.")
    else:
        df = add_indicators(df_raw, sma1, sma2, ema_w, bb_w, rsi_w)
        close = df["Close"].squeeze(); open_ = df["Open"].squeeze()
        high  = df["High"].squeeze();  low   = df["Low"].squeeze()
        vol   = df["Volume"].squeeze()

        rows = 1; rh = [0.55]
        if show_vol:  rows+=1; rh.append(0.12)
        if show_rsi:  rows+=1; rh.append(0.14)
        if show_macd: rows+=1; rh.append(0.19)
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                            vertical_spacing=0.03, row_heights=rh)
        cr = 1
        if chart_type=="Candlestick":
            fig.add_trace(go.Candlestick(x=df.index,open=open_,high=high,low=low,close=close,
                increasing_line_color="#00ff88",decreasing_line_color="#ff4466",
                increasing_fillcolor="#00ff88",decreasing_fillcolor="#ff4466",
                name=primary,line_width=1), row=cr,col=1)
        elif chart_type=="OHLC":
            fig.add_trace(go.Ohlc(x=df.index,open=open_,high=high,low=low,close=close,
                increasing_line_color="#00ff88",decreasing_line_color="#ff4466",name=primary), row=cr,col=1)
        else:
            fig.add_trace(go.Scatter(x=df.index,y=close,mode="lines",
                line=dict(color="#00d4ff",width=1.5),fill="tozeroy",
                fillcolor="rgba(0,212,255,0.04)",name=primary), row=cr,col=1)
        if show_sma:
            fig.add_trace(go.Scatter(x=df.index,y=df[f"SMA{sma1}"],mode="lines",
                line=dict(color="#00ff88",width=0.9),name=f"SMA{sma1}"),row=cr,col=1)
            fig.add_trace(go.Scatter(x=df.index,y=df[f"SMA{sma2}"],mode="lines",
                line=dict(color="#ffaa00",width=0.9,dash="dash"),name=f"SMA{sma2}"),row=cr,col=1)
        if show_ema:
            fig.add_trace(go.Scatter(x=df.index,y=df[f"EMA{ema_w}"],mode="lines",
                line=dict(color="#aa88ff",width=0.9),name=f"EMA{ema_w}"),row=cr,col=1)
        if show_bb:
            fig.add_trace(go.Scatter(x=df.index,y=df["BB_up"],mode="lines",
                line=dict(color="#ffaa00",width=0.7,dash="dot"),name="BB+2σ"),row=cr,col=1)
            fig.add_trace(go.Scatter(x=df.index,y=df["BB_dn"],mode="lines",
                line=dict(color="#ffaa00",width=0.7,dash="dot"),fill="tonexty",
                fillcolor="rgba(255,170,0,0.04)",name="BB-2σ"),row=cr,col=1)
        if show_vol:
            cr+=1
            vc=["#00ff88" if c>=o else "#ff4466" for c,o in zip(close,open_)]
            fig.add_trace(go.Bar(x=df.index,y=vol,marker_color=vc,name="Vol",opacity=0.7),row=cr,col=1)
        if show_rsi:
            cr+=1
            fig.add_trace(go.Scatter(x=df.index,y=df["RSI"],mode="lines",
                line=dict(color="#aa88ff",width=1.2),name="RSI"),row=cr,col=1)
            fig.add_hline(y=70,line_dash="dot",line_color="#ff4466",line_width=0.7,row=cr,col=1)
            fig.add_hline(y=30,line_dash="dot",line_color="#00ff88",line_width=0.7,row=cr,col=1)
            fig.update_yaxes(range=[0,100],row=cr,col=1)
        if show_macd:
            cr+=1
            hc=["#00ff88" if v>=0 else "#ff4466" for v in df["MACD_hist"].fillna(0)]
            fig.add_trace(go.Bar(x=df.index,y=df["MACD_hist"],marker_color=hc,
                name="MACD Hist",opacity=0.8),row=cr,col=1)
            fig.add_trace(go.Scatter(x=df.index,y=df["MACD"],mode="lines",
                line=dict(color="#00d4ff",width=1),name="MACD"),row=cr,col=1)
            fig.add_trace(go.Scatter(x=df.index,y=df["MACD_sig"],mode="lines",
                line=dict(color="#ffaa00",width=1),name="Signal"),row=cr,col=1)
        fig.update_layout(**PLOTLY_BASE, height=max(600,200*rows),
            title=dict(text=f"◈ {primary} — {period_l}",
                       font=dict(family="Orbitron",color="#00d4ff",size=13)),
            xaxis_rangeslider_visible=False,hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        m1,m2,m3,m4,m5,m6 = st.columns(6)
        last=float(close.iloc[-1]); first=float(close.iloc[0]); ret=(last-first)/first*100
        m1.metric("Open (période)", f"${first:,.2f}")
        m2.metric("Dernier",        f"${last:,.2f}")
        m3.metric("Rendement",      f"{ret:+.2f}%")
        m4.metric("High / Low",     f"${float(high.max()):,.2f} / ${float(low.min()):,.2f}")
        m5.metric("Volume moy.",    f"{float(vol.mean())/1e6:.1f}M")
        m6.metric("ATR(14)",        f"${float(df['ATR'].iloc[-1]):,.2f}" if "ATR" in df else "N/A")

        with st.expander("▸ Données brutes"):
            st.dataframe(df.sort_index(ascending=False).head(100), use_container_width=True, height=300)
        st.download_button("⬇ CSV", df.to_csv().encode(), f"{primary}_{period}.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════
#  TAB 1 — OPTIONS CHAIN
# ══════════════════════════════════════════════════════════════════════
with tabs[1]:
    blabel("Options Chain · IV Smile · Open Interest")
    st.markdown("<span class='tag-free'>GRATUIT</span><span class='tag-limit'>Actions US uniquement</span>",
                unsafe_allow_html=True)
    oc1,oc2 = st.columns([2,2])
    opt_t = oc1.text_input("Ticker", value=primary, key="opt_tk")
    opt_type = oc2.radio("Type", ["Calls","Puts","Les deux"], horizontal=True)

    t_obj, _, all_exps = get_options(opt_t.upper().strip())
    if not all_exps:
        st.warning("Aucune option disponible.")
    else:
        sel_exp = st.selectbox("Expiration", all_exps)
        try:
            chain = t_obj.option_chain(sel_exp)
            calls, puts = chain.calls, chain.puts
        except Exception as e:
            st.error(f"Erreur: {e}"); calls=puts=pd.DataFrame()

        spot = (get_info(opt_t.upper().strip()).get("currentPrice") or
                get_info(opt_t.upper().strip()).get("regularMarketPrice"))
        if spot:
            st.markdown(f"<span style='font-family:Space Mono,monospace;font-size:11px;color:#4da8da;'>Spot: <b style='color:#00d4ff;'>${spot:,.2f}</b></span>", unsafe_allow_html=True)

        cols_show = [c for c in ["strike","lastPrice","bid","ask","impliedVolatility",
                                  "openInterest","volume","inTheMoney"] if c in calls.columns]
        def fmt_chain(df_c):
            d = df_c[cols_show].copy()
            if "impliedVolatility" in d:
                d["impliedVolatility"] = (d["impliedVolatility"]*100).round(2).astype(str)+"%"
            return d

        if opt_type in ["Calls","Les deux"] and not calls.empty:
            blabel("Calls"); st.dataframe(fmt_chain(calls), use_container_width=True, height=260)
        if opt_type in ["Puts","Les deux"] and not puts.empty:
            blabel("Puts");  st.dataframe(fmt_chain(puts),  use_container_width=True, height=260)

        if not calls.empty and not puts.empty and "openInterest" in calls.columns:
            c1,c2 = st.columns(2)
            with c1:
                blabel("Open Interest")
                fig_oi = go.Figure()
                fig_oi.add_trace(go.Bar(x=calls["strike"],y=calls["openInterest"],
                    name="Calls OI",marker_color="#00ff88",opacity=0.8))
                fig_oi.add_trace(go.Bar(x=puts["strike"],y=-puts["openInterest"],
                    name="Puts OI",marker_color="#ff4466",opacity=0.8))
                if spot: fig_oi.add_vline(x=spot,line_dash="dot",line_color="#00d4ff")
                fig_oi.update_layout(**PLOTLY_BASE,height=320,barmode="overlay",
                    title=dict(text="◈ OI Calls vs Puts",font=dict(family="Orbitron",color="#00d4ff",size=11)))
                st.plotly_chart(fig_oi, use_container_width=True)
            with c2:
                blabel("IV Smile")
                fig_iv = go.Figure()
                if "impliedVolatility" in calls.columns:
                    iv_c = calls[["strike","impliedVolatility"]].dropna()
                    iv_p = puts[["strike","impliedVolatility"]].dropna()
                    fig_iv.add_trace(go.Scatter(x=iv_c["strike"],y=iv_c["impliedVolatility"]*100,
                        mode="lines+markers",line=dict(color="#00ff88",width=1.5),name="IV Calls"))
                    fig_iv.add_trace(go.Scatter(x=iv_p["strike"],y=iv_p["impliedVolatility"]*100,
                        mode="lines+markers",line=dict(color="#ff4466",width=1.5),name="IV Puts"))
                    if spot: fig_iv.add_vline(x=spot,line_dash="dot",line_color="#00d4ff")
                fig_iv.update_layout(**PLOTLY_BASE,height=320,
                    title=dict(text="◈ IV Smile",font=dict(family="Orbitron",color="#00d4ff",size=11)),
                    yaxis_title="IV (%)")
                st.plotly_chart(fig_iv, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
#  TAB 2 — CRYPTO L2
# ══════════════════════════════════════════════════════════════════════
with tabs[2]:
    blabel("ccxt · Orderbook Level 2 · Multi-Exchange")
    st.markdown("<span class='tag-free'>GRATUIT · Sans clé API</span>", unsafe_allow_html=True)
    if not CCXT_OK:
        st.error("pip install ccxt")
    else:
        cr1,cr2,cr3 = st.columns([2,2,1])
        exch_id = cr1.selectbox("Exchange", ["binance","kraken","okx","bybit","coinbase","kucoin"])
        sym_in  = cr2.text_input("Paire", value="BTC/USDT")
        ob_depth= cr3.slider("Profondeur", 5, 50, 20)
        tf_map  = {"1m":"1m","5m":"5m","15m":"15m","1h":"1h","4h":"4h","1j":"1d"}
        tf_l    = st.selectbox("Timeframe", list(tf_map.keys()), index=3)
        tf      = tf_map[tf_l]

        col_ob, col_chart = st.columns([1,2])
        with col_ob:
            blabel("Order Book L2")
            ob, err = get_ccxt_ob(exch_id, sym_in, ob_depth)
            if err: st.warning(err)
            elif ob:
                bids = pd.DataFrame(ob["bids"], columns=["Prix","Qté"])
                asks = pd.DataFrame(ob["asks"], columns=["Prix","Qté"])
                bids["Cumulé"] = bids["Qté"].cumsum()
                asks["Cumulé"] = asks["Qté"].cumsum()
                mid = (bids["Prix"].iloc[0]+asks["Prix"].iloc[0])/2
                spread = asks["Prix"].iloc[0]-bids["Prix"].iloc[0]
                st.markdown(f"""
                <div style='font-family:Space Mono,monospace;font-size:10px;line-height:2;color:#8aafc8;'>
                  Mid: <b style='color:#00d4ff;'>${mid:,.2f}</b><br>
                  Spread: <b style='color:#ffaa00;'>${spread:.4f} ({spread/mid*100:.4f}%)</b>
                </div>""", unsafe_allow_html=True)
                fig_ob = go.Figure()
                fig_ob.add_trace(go.Bar(x=bids["Cumulé"],y=bids["Prix"],orientation="h",
                    name="Bids",marker_color="rgba(0,255,136,0.6)"))
                fig_ob.add_trace(go.Bar(x=asks["Cumulé"],y=asks["Prix"],orientation="h",
                    name="Asks",marker_color="rgba(255,68,102,0.6)"))
                fig_ob.add_hline(y=mid,line_dash="dot",line_color="#00d4ff",line_width=0.8)
                fig_ob.update_layout(**PLOTLY_BASE,height=400,barmode="overlay",
                    title=dict(text="◈ Depth",font=dict(family="Orbitron",color="#00d4ff",size=11)),
                    xaxis_title="Volume cumulé")
                st.plotly_chart(fig_ob, use_container_width=True)
        with col_chart:
            blabel("OHLCV Crypto")
            df_c, err_c = get_ccxt_ohlcv(exch_id, sym_in, tf, 300)
            if err_c: st.warning(err_c)
            elif not df_c.empty:
                fig_c = make_subplots(rows=2,cols=1,shared_xaxes=True,
                    vertical_spacing=0.03,row_heights=[0.72,0.28])
                fig_c.add_trace(go.Candlestick(x=df_c.index,open=df_c["Open"],high=df_c["High"],
                    low=df_c["Low"],close=df_c["Close"],
                    increasing_line_color="#00ff88",decreasing_line_color="#ff4466",
                    increasing_fillcolor="#00ff88",decreasing_fillcolor="#ff4466",
                    name=sym_in,line_width=1),row=1,col=1)
                vc=["#00ff88" if c>=o else "#ff4466" for c,o in zip(df_c["Close"],df_c["Open"])]
                fig_c.add_trace(go.Bar(x=df_c.index,y=df_c["Volume"],
                    marker_color=vc,name="Vol",opacity=0.7),row=2,col=1)
                fig_c.update_layout(**PLOTLY_BASE,height=440,
                    title=dict(text=f"◈ {sym_in} · {exch_id} · {tf_l}",
                               font=dict(family="Orbitron",color="#00d4ff",size=12)),
                    xaxis_rangeslider_visible=False,hovermode="x unified")
                st.plotly_chart(fig_c, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
#  TAB 3 — STATISTICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════
with tabs[3]:
    blabel("Analyse Statistique & Quantitative")
    stat_tabs = st.tabs(["Régressions","PCA","Distributions","GARCH","Monte Carlo","Cointégration"])

    # fetch all returns
    all_closes = {}
    for t in selected:
        d = get_ohlcv(t, period, "1d")
        if not d.empty: all_closes[t] = d["Close"].squeeze()
    price_df  = pd.DataFrame(all_closes).dropna()
    returns_df = price_df.pct_change().dropna()

    # ── Régressions ──────────────────────────────────────────────────
    with stat_tabs[0]:
        blabel("Régression linéaire · Rolling Beta")
        if len(selected) < 2:
            st.info("Sélectionnez ≥ 2 tickers.")
        else:
            rc1,rc2 = st.columns(2)
            dep_t = rc1.selectbox("Variable dépendante (Y)", selected)
            ind_t = rc2.selectbox("Variable indépendante (X)", [t for t in selected if t!=dep_t])
            y = returns_df[dep_t].dropna(); x = returns_df[ind_t].dropna()
            common = y.index.intersection(x.index); y=y[common]; x=x[common]

            slope, intercept, r, p, se = stats.linregress(x, y)
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Beta (pente)", f"{slope:.4f}")
            m2.metric("Alpha (ordonnée)", f"{intercept*252:.4f} ann.")
            m3.metric("R²", f"{r**2:.4f}")
            m4.metric("p-value", f"{p:.4f}")

            fig_reg = go.Figure()
            fig_reg.add_trace(go.Scatter(x=x,y=y,mode="markers",
                marker=dict(color="#00d4ff",size=4,opacity=0.6),name="Observations"))
            x_line = np.linspace(x.min(),x.max(),100)
            fig_reg.add_trace(go.Scatter(x=x_line,y=slope*x_line+intercept,mode="lines",
                line=dict(color="#ffaa00",width=1.5),name=f"β={slope:.3f}"))
            fig_reg.update_layout(**PLOTLY_BASE,height=380,
                title=dict(text=f"◈ {dep_t} vs {ind_t}",
                           font=dict(family="Orbitron",color="#00d4ff",size=12)),
                xaxis_title=f"{ind_t} returns",yaxis_title=f"{dep_t} returns")
            st.plotly_chart(fig_reg, use_container_width=True)

            # Rolling beta
            blabel("Rolling Beta (60j)")
            roll_beta = pd.Series(dtype=float, index=returns_df.index)
            for i in range(60, len(returns_df)):
                window = returns_df.iloc[i-60:i]
                sl,_,_,_,_ = stats.linregress(window[ind_t], window[dep_t])
                roll_beta.iloc[i] = sl
            roll_beta = roll_beta.dropna()
            fig_rb = go.Figure(go.Scatter(x=roll_beta.index,y=roll_beta,mode="lines",
                line=dict(color="#00d4ff",width=1.5),name="Rolling Beta"))
            fig_rb.add_hline(y=1,line_dash="dot",line_color="#ffaa00",line_width=0.8)
            fig_rb.add_hline(y=0,line_dash="dot",line_color="#4a7a99",line_width=0.8)
            fig_rb.update_layout(**PLOTLY_BASE,height=280,
                title=dict(text="◈ Rolling Beta (60j)",
                           font=dict(family="Orbitron",color="#00d4ff",size=11)))
            st.plotly_chart(fig_rb, use_container_width=True)

    # ── PCA ──────────────────────────────────────────────────────────
    with stat_tabs[1]:
        blabel("Analyse en Composantes Principales (PCA)")
        if len(selected) < 2:
            st.info("Sélectionnez ≥ 2 tickers.")
        else:
            scaler = StandardScaler()
            X_sc = scaler.fit_transform(returns_df.dropna())
            pca = PCA()
            X_pca = pca.fit_transform(X_sc)
            expl = pca.explained_variance_ratio_

            pc1,pc2 = st.columns(2)
            with pc1:
                blabel("Variance expliquée")
                fig_var = go.Figure()
                fig_var.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(len(expl))],
                    y=expl*100,marker_color="#00d4ff",name="Variance"))
                fig_var.add_trace(go.Scatter(x=[f"PC{i+1}" for i in range(len(expl))],
                    y=np.cumsum(expl)*100,mode="lines+markers",
                    line=dict(color="#ffaa00",width=1.5),name="Cumulée"))
                fig_var.update_layout(**PLOTLY_BASE,height=320,
                    title=dict(text="◈ Variance expliquée",
                               font=dict(family="Orbitron",color="#00d4ff",size=11)),
                    yaxis_title="%")
                st.plotly_chart(fig_var, use_container_width=True)
            with pc2:
                blabel("Loadings PC1 vs PC2")
                loadings = pd.DataFrame(pca.components_[:2].T,
                                         columns=["PC1","PC2"], index=returns_df.columns)
                fig_load = go.Figure()
                for j, ticker in enumerate(loadings.index):
                    fig_load.add_trace(go.Scatter(x=[loadings.loc[ticker,"PC1"]],
                        y=[loadings.loc[ticker,"PC2"]],mode="markers+text",
                        text=[ticker],textposition="top center",
                        marker=dict(color=COLORS[j%len(COLORS)],size=10),name=ticker))
                    fig_load.add_shape(type="line",x0=0,y0=0,
                        x1=loadings.loc[ticker,"PC1"],y1=loadings.loc[ticker,"PC2"],
                        line=dict(color=COLORS[j%len(COLORS)],width=1))
                fig_load.add_vline(x=0,line_color="#1a3a55",line_width=0.8)
                fig_load.add_hline(y=0,line_color="#1a3a55",line_width=0.8)
                fig_load.update_layout(**PLOTLY_BASE,height=320,
                    title=dict(text="◈ Loadings",
                               font=dict(family="Orbitron",color="#00d4ff",size=11)))
                st.plotly_chart(fig_load, use_container_width=True)

            blabel("Projection PC1 vs PC2 dans le temps")
            fig_pca = go.Figure(go.Scatter(x=X_pca[:,0],y=X_pca[:,1],mode="markers",
                marker=dict(color=np.arange(len(X_pca)),colorscale="Viridis",
                            size=4,opacity=0.7,showscale=True),name="Observations"))
            fig_pca.update_layout(**PLOTLY_BASE,height=320,
                title=dict(text="◈ Scores PCA",
                           font=dict(family="Orbitron",color="#00d4ff",size=11)),
                xaxis_title=f"PC1 ({expl[0]*100:.1f}%)",
                yaxis_title=f"PC2 ({expl[1]*100:.1f}%)" if len(expl)>1 else "PC2")
            st.plotly_chart(fig_pca, use_container_width=True)

    # ── Distributions ────────────────────────────────────────────────
    with stat_tabs[2]:
        blabel("Distribution des rendements · Tests de normalité")
        dist_t = st.selectbox("Ticker", selected, key="dist_tk")
        r = returns_df[dist_t].dropna() if dist_t in returns_df else pd.Series(dtype=float)
        if not r.empty:
            d1,d2,d3,d4 = st.columns(4)
            d1.metric("Moyenne ann.",  f"{r.mean()*252:.2%}")
            d2.metric("Volatilité ann.", f"{r.std()*np.sqrt(252):.2%}")
            d3.metric("Skewness",      f"{stats.skew(r):.4f}")
            d4.metric("Kurtosis (exc.)", f"{stats.kurtosis(r):.4f}")

            _, p_jb = stats.jarque_bera(r)
            _, p_sw = stats.shapiro(r[:5000])
            st.markdown(f"""
            <div style='font-family:Space Mono,monospace;font-size:10px;color:#8aafc8;line-height:2;'>
              Jarque-Bera p-value: <b style='color:{"#ff4466" if p_jb<0.05 else "#00ff88"};'>{p_jb:.4f}</b>
              {'(non normal)' if p_jb<0.05 else '(normal)'}&nbsp;&nbsp;
              Shapiro-Wilk p-value: <b style='color:{"#ff4466" if p_sw<0.05 else "#00ff88"};'>{p_sw:.4f}</b>
            </div>""", unsafe_allow_html=True)

            fig_dist = make_subplots(rows=1,cols=2,subplot_titles=["Histogramme","QQ-Plot"])
            # Histogram
            fig_dist.add_trace(go.Histogram(x=r,nbinsx=80,
                marker_color="#00d4ff",opacity=0.7,name="Rendements",
                histnorm="probability density"),row=1,col=1)
            x_norm = np.linspace(r.min(),r.max(),200)
            fig_dist.add_trace(go.Scatter(x=x_norm,
                y=stats.norm.pdf(x_norm,r.mean(),r.std()),
                mode="lines",line=dict(color="#ffaa00",width=1.5),name="Normale"),row=1,col=1)
            # QQ
            qq = stats.probplot(r)
            fig_dist.add_trace(go.Scatter(x=qq[0][0],y=qq[0][1],mode="markers",
                marker=dict(color="#00d4ff",size=3),name="QQ"),row=1,col=2)
            fig_dist.add_trace(go.Scatter(x=qq[0][0],
                y=qq[1][0]*qq[0][0]+qq[1][1],mode="lines",
                line=dict(color="#ffaa00",width=1.5),name="Théorique"),row=1,col=2)
            fig_dist.update_layout(**PLOTLY_BASE,height=380,showlegend=False,
                title=dict(text=f"◈ {dist_t} — Distribution rendements",
                           font=dict(family="Orbitron",color="#00d4ff",size=12)))
            st.plotly_chart(fig_dist, use_container_width=True)

    # ── GARCH ────────────────────────────────────────────────────────
    with stat_tabs[3]:
        blabel("GARCH(1,1) · Volatilité conditionnelle")
        if not ARCH_OK:
            st.error("pip install arch")
        else:
            garch_t = st.selectbox("Ticker", selected, key="garch_tk")
            r_g = returns_df[garch_t].dropna()*100 if garch_t in returns_df else pd.Series(dtype=float)
            if not r_g.empty:
                with st.spinner("Fitting GARCH(1,1)…"):
                    try:
                        am = arch_model(r_g, vol="Garch", p=1, q=1, dist="normal")
                        res = am.fit(disp="off")
                        cond_vol = res.conditional_volatility

                        g1,g2,g3,g4 = st.columns(4)
                        g1.metric("ω (omega)", f"{res.params['omega']:.6f}")
                        g2.metric("α (alpha)", f"{res.params['alpha[1]']:.4f}")
                        g3.metric("β (beta)",  f"{res.params['beta[1]']:.4f}")
                        g4.metric("Persist. α+β", f"{res.params['alpha[1]']+res.params['beta[1]']:.4f}")

                        fig_garch = make_subplots(rows=2,cols=1,shared_xaxes=True,
                            vertical_spacing=0.05,row_heights=[0.5,0.5])
                        fig_garch.add_trace(go.Scatter(x=r_g.index,y=r_g,mode="lines",
                            line=dict(color="#00d4ff",width=0.8),name="Returns (%)"),row=1,col=1)
                        fig_garch.add_trace(go.Scatter(x=cond_vol.index,y=cond_vol,mode="lines",
                            line=dict(color="#ff4466",width=1.5),
                            fill="tozeroy",fillcolor="rgba(255,68,102,0.08)",
                            name="Vol. cond. (%)"),row=2,col=1)
                        fig_garch.update_layout(**PLOTLY_BASE,height=460,
                            title=dict(text=f"◈ GARCH(1,1) — {garch_t}",
                                       font=dict(family="Orbitron",color="#00d4ff",size=12)),
                            hovermode="x unified")
                        st.plotly_chart(fig_garch, use_container_width=True)

                        # Forecast
                        blabel("Prévision volatilité (10 jours)")
                        forecast = res.forecast(horizon=10)
                        fvol = np.sqrt(forecast.variance.dropna().iloc[-1].values)
                        fig_fc = go.Figure(go.Bar(x=list(range(1,11)),y=fvol,
                            marker_color="#aa88ff",name="Vol prévue (%)"))
                        fig_fc.update_layout(**PLOTLY_BASE,height=260,
                            title=dict(text="◈ Prévision volatilité GARCH",
                                       font=dict(family="Orbitron",color="#00d4ff",size=11)),
                            xaxis_title="Jours",yaxis_title="Volatilité (%)")
                        st.plotly_chart(fig_fc, use_container_width=True)
                    except Exception as e:
                        st.error(f"GARCH error: {e}")

    # ── Monte Carlo ──────────────────────────────────────────────────
    with stat_tabs[4]:
        blabel("Simulation Monte Carlo · GBM")
        mc1,mc2,mc3 = st.columns(3)
        mc_t    = mc1.selectbox("Ticker", selected, key="mc_tk")
        n_sim   = mc2.slider("Simulations", 100, 2000, 500, step=100)
        n_days  = mc3.slider("Jours à projeter", 30, 365, 252)

        r_mc = returns_df[mc_t].dropna() if mc_t in returns_df else pd.Series(dtype=float)
        if not r_mc.empty:
            mu_d  = r_mc.mean(); sig_d = r_mc.std()
            df_mc = get_ohlcv(mc_t, period, "1d")
            S0    = float(df_mc["Close"].squeeze().iloc[-1]) if not df_mc.empty else 100.0

            with st.spinner(f"Monte Carlo {n_sim} simulations…"):
                np.random.seed(42)
                daily_ret = np.random.normal(mu_d, sig_d, (n_days, n_sim))
                price_paths = S0 * np.cumprod(1+daily_ret, axis=0)

            final = price_paths[-1]
            pct5  = np.percentile(final, 5)
            pct50 = np.percentile(final, 50)
            pct95 = np.percentile(final, 95)

            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Prix actuel",   f"${S0:,.2f}")
            m2.metric("P5 (baissier)", f"${pct5:,.2f}", f"{(pct5/S0-1)*100:+.1f}%")
            m3.metric("Médiane",       f"${pct50:,.2f}", f"{(pct50/S0-1)*100:+.1f}%")
            m4.metric("P95 (haussier)",f"${pct95:,.2f}", f"{(pct95/S0-1)*100:+.1f}%")

            fig_mc = go.Figure()
            for i in range(min(200, n_sim)):
                fig_mc.add_trace(go.Scatter(
                    y=price_paths[:,i], mode="lines",
                    line=dict(color="rgba(0,212,255,0.05)", width=0.5),
                    showlegend=False))
            fig_mc.add_trace(go.Scatter(y=np.percentile(price_paths,95,axis=1),mode="lines",
                line=dict(color="#00ff88",width=2),name="P95"))
            fig_mc.add_trace(go.Scatter(y=np.percentile(price_paths,50,axis=1),mode="lines",
                line=dict(color="#ffaa00",width=2),name="Médiane"))
            fig_mc.add_trace(go.Scatter(y=np.percentile(price_paths,5,axis=1),mode="lines",
                line=dict(color="#ff4466",width=2),name="P5"))
            fig_mc.add_hline(y=S0,line_dash="dot",line_color="#4a7a99",line_width=0.8)
            fig_mc.update_layout(**PLOTLY_BASE,height=420,
                title=dict(text=f"◈ Monte Carlo {n_sim} trajectoires — {mc_t} ({n_days}j)",
                           font=dict(family="Orbitron",color="#00d4ff",size=12)),
                yaxis_title="Prix ($)")
            st.plotly_chart(fig_mc, use_container_width=True)

            # Distribution finale
            fig_fdist = go.Figure()
            fig_fdist.add_trace(go.Histogram(x=final,nbinsx=80,
                marker_color="#00d4ff",opacity=0.7,histnorm="probability density",
                name="Distribution finale"))
            fig_fdist.add_vline(x=S0,line_dash="dot",line_color="#ffaa00",
                annotation_text=f"S₀ ${S0:.0f}")
            fig_fdist.add_vline(x=pct5,line_dash="dot",line_color="#ff4466",
                annotation_text=f"P5 ${pct5:.0f}")
            fig_fdist.add_vline(x=pct95,line_dash="dot",line_color="#00ff88",
                annotation_text=f"P95 ${pct95:.0f}")
            fig_fdist.update_layout(**PLOTLY_BASE,height=300,
                title=dict(text="◈ Distribution des prix finaux",
                           font=dict(family="Orbitron",color="#00d4ff",size=11)))
            st.plotly_chart(fig_fdist, use_container_width=True)

    # ── Cointégration ────────────────────────────────────────────────
    with stat_tabs[5]:
        blabel("Cointégration · Pairs Trading")
        if not SM_OK:
            st.error("pip install statsmodels")
        elif len(selected) < 2:
            st.info("Sélectionnez ≥ 2 tickers.")
        else:
            ci1,ci2 = st.columns(2)
            tA = ci1.selectbox("Actif A", selected, key="ci_a")
            tB = ci2.selectbox("Actif B", [t for t in selected if t!=tA], key="ci_b")
            pA = price_df[tA].dropna(); pB = price_df[tB].dropna()
            idx_c = pA.index.intersection(pB.index); pA=pA[idx_c]; pB=pB[idx_c]

            if len(pA) > 30:
                score, pval, crit = coint(pA, pB)
                adf_spread = adfuller(pA - pB * (pA.mean()/pB.mean()))

                col_ci1, col_ci2, col_ci3 = st.columns(3)
                col_ci1.metric("Score cointégration", f"{score:.4f}")
                col_ci2.metric("p-value",
                    f"{pval:.4f}",
                    "Cointégrés ✓" if pval<0.05 else "Non cointégrés",
                    delta_color="normal" if pval<0.05 else "inverse")
                col_ci3.metric("ADF spread p-value", f"{adf_spread[1]:.4f}")

                # Spread
                hedge = pA.mean()/pB.mean()
                spread = pA - pB*hedge
                z_score = (spread - spread.mean())/spread.std()

                fig_spread = make_subplots(rows=2,cols=1,shared_xaxes=True,
                    vertical_spacing=0.05,row_heights=[0.5,0.5])
                fig_spread.add_trace(go.Scatter(x=spread.index,y=spread,mode="lines",
                    line=dict(color="#00d4ff",width=1.2),name="Spread"),row=1,col=1)
                fig_spread.add_hline(y=spread.mean(),line_dash="dot",
                    line_color="#ffaa00",line_width=0.8,row=1,col=1)
                fig_spread.add_trace(go.Scatter(x=z_score.index,y=z_score,mode="lines",
                    line=dict(color="#aa88ff",width=1.2),name="Z-score"),row=2,col=1)
                for level in [2,-2,1,-1]:
                    col_z = "#ff4466" if abs(level)==2 else "#00ff88"
                    fig_spread.add_hline(y=level,line_dash="dot",
                        line_color=col_z,line_width=0.7,row=2,col=1)
                fig_spread.update_layout(**PLOTLY_BASE,height=400,
                    title=dict(text=f"◈ Spread & Z-score — {tA} vs {tB}",
                               font=dict(family="Orbitron",color="#00d4ff",size=12)),
                    hovermode="x unified")
                st.plotly_chart(fig_spread, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
#  TAB 4 — BACKTESTING
# ══════════════════════════════════════════════════════════════════════
with tabs[4]:
    blabel("Backtesting · Stratégies · Métriques de performance")
    bt_tabs = st.tabs(["SMA Crossover","RSI Mean-Reversion","Momentum","Résultats"])

    bt_ticker = st.selectbox("Ticker pour backtest", selected, key="bt_tk")
    df_bt_raw = get_ohlcv(bt_ticker, "2y", "1d")
    df_bt     = add_indicators(df_bt_raw) if not df_bt_raw.empty else pd.DataFrame()

    def run_backtest(signals, prices, cost=0.001):
        """Generic backtester with transaction costs."""
        pos = signals.shift(1).fillna(0)
        ret = prices.pct_change().fillna(0)
        trade = pos.diff().abs()
        strat_ret = pos*ret - trade*cost
        equity = (1+strat_ret).cumprod()*100
        bh_equity = (1+ret).cumprod()*100
        return strat_ret, equity, bh_equity, pos

    def perf_metrics(ret, equity, prices, rf=0.0):
        r = ret.dropna()
        return {
            "Rendement total":   f"{(equity.iloc[-1]/100-1)*100:+.2f}%",
            "Rendement annualisé": f"{r.mean()*252*100:+.2f}%",
            "Volatilité ann.":   f"{r.std()*np.sqrt(252)*100:.2f}%",
            "Sharpe":            f"{sharpe(r,rf):.3f}",
            "Sortino":           f"{sortino(r,rf):.3f}",
            "Max Drawdown":      f"{max_drawdown(equity)*100:.2f}%",
            "Calmar":            f"{calmar(r,equity):.3f}",
            "Win Rate":          f"{(r>0).mean()*100:.1f}%",
            "Nb trades":         f"{(ret.diff().abs()>0.001).sum()}",
        }

    with bt_tabs[0]:
        blabel("SMA Crossover Strategy")
        if not df_bt.empty:
            bc1,bc2,bc3 = st.columns(3)
            fast = bc1.slider("SMA Fast", 5,50,20,key="bt_f")
            slow = bc2.slider("SMA Slow",20,200,50,key="bt_s")
            cost = bc3.slider("Coût transaction (%)",0.0,0.5,0.1,step=0.05)/100
            close_bt = df_bt["Close"].squeeze()
            sma_f = close_bt.rolling(fast).mean(); sma_s = close_bt.rolling(slow).mean()
            signal = (sma_f > sma_s).astype(int)
            ret_bt, equity_bt, bh_bt, pos = run_backtest(signal, close_bt, cost)
            metrics = perf_metrics(ret_bt, equity_bt, close_bt)
            cols_m = st.columns(len(metrics))
            for i,(k,v) in enumerate(metrics.items()):
                cols_m[i].metric(k,v)
            fig_bt = make_subplots(rows=3,cols=1,shared_xaxes=True,
                vertical_spacing=0.04,row_heights=[0.45,0.3,0.25])
            fig_bt.add_trace(go.Scatter(x=close_bt.index,y=close_bt,mode="lines",
                line=dict(color="#4a7a99",width=1),name="Prix"),row=1,col=1)
            fig_bt.add_trace(go.Scatter(x=sma_f.index,y=sma_f,mode="lines",
                line=dict(color="#00ff88",width=1),name=f"SMA{fast}"),row=1,col=1)
            fig_bt.add_trace(go.Scatter(x=sma_s.index,y=sma_s,mode="lines",
                line=dict(color="#ffaa00",width=1,dash="dash"),name=f"SMA{slow}"),row=1,col=1)
            buys  = close_bt[signal.diff()==1]
            sells = close_bt[signal.diff()==-1]
            fig_bt.add_trace(go.Scatter(x=buys.index,y=buys,mode="markers",
                marker=dict(symbol="triangle-up",color="#00ff88",size=8),name="BUY"),row=1,col=1)
            fig_bt.add_trace(go.Scatter(x=sells.index,y=sells,mode="markers",
                marker=dict(symbol="triangle-down",color="#ff4466",size=8),name="SELL"),row=1,col=1)
            fig_bt.add_trace(go.Scatter(x=equity_bt.index,y=equity_bt,mode="lines",
                line=dict(color="#00d4ff",width=1.5),fill="tozeroy",
                fillcolor="rgba(0,212,255,0.05)",name="Stratégie"),row=2,col=1)
            fig_bt.add_trace(go.Scatter(x=bh_bt.index,y=bh_bt,mode="lines",
                line=dict(color="#4a7a99",width=1,dash="dash"),name="Buy & Hold"),row=2,col=1)
            drawdown_series = (equity_bt/equity_bt.cummax()-1)*100
            fig_bt.add_trace(go.Scatter(x=drawdown_series.index,y=drawdown_series,
                mode="lines",fill="tozeroy",fillcolor="rgba(255,68,102,0.15)",
                line=dict(color="#ff4466",width=1),name="Drawdown"),row=3,col=1)
            fig_bt.update_layout(**PLOTLY_BASE,height=560,
                title=dict(text=f"◈ SMA{fast}/{slow} Crossover — {bt_ticker}",
                           font=dict(family="Orbitron",color="#00d4ff",size=12)),
                hovermode="x unified")
            st.plotly_chart(fig_bt, use_container_width=True)

    with bt_tabs[1]:
        blabel("RSI Mean-Reversion Strategy")
        if not df_bt.empty:
            rc1,rc2,rc3,rc4 = st.columns(4)
            rsi_period = rc1.slider("RSI Period",5,30,14,key="bt_rp")
            ob_level   = rc2.slider("Overbought",60,90,70,key="bt_ob")
            os_level   = rc3.slider("Oversold",10,40,30,key="bt_os")
            cost_r     = rc4.slider("Coût (%)",0.0,0.5,0.1,step=0.05,key="bt_cr")/100
            close_bt   = df_bt["Close"].squeeze()
            delta      = close_bt.diff()
            gain       = delta.clip(lower=0).rolling(rsi_period).mean()
            loss       = (-delta.clip(upper=0)).rolling(rsi_period).mean()
            rsi_bt     = 100-100/(1+gain/loss.replace(0,np.nan))
            signal_rsi = pd.Series(0, index=close_bt.index)
            signal_rsi[rsi_bt < os_level] =  1
            signal_rsi[rsi_bt > ob_level] = -1
            signal_rsi = signal_rsi.replace(0, np.nan).ffill().fillna(0)
            ret_rsi, eq_rsi, bh_rsi, _ = run_backtest(signal_rsi, close_bt, cost_r)
            metrics_rsi = perf_metrics(ret_rsi, eq_rsi, close_bt)
            cols_m = st.columns(len(metrics_rsi))
            for i,(k,v) in enumerate(metrics_rsi.items()): cols_m[i].metric(k,v)
            fig_rsi_bt = make_subplots(rows=3,cols=1,shared_xaxes=True,
                vertical_spacing=0.04,row_heights=[0.35,0.3,0.35])
            fig_rsi_bt.add_trace(go.Scatter(x=rsi_bt.index,y=rsi_bt,mode="lines",
                line=dict(color="#aa88ff",width=1.2),name="RSI"),row=1,col=1)
            fig_rsi_bt.add_hline(y=ob_level,line_dash="dot",line_color="#ff4466",
                line_width=0.8,row=1,col=1)
            fig_rsi_bt.add_hline(y=os_level,line_dash="dot",line_color="#00ff88",
                line_width=0.8,row=1,col=1)
            fig_rsi_bt.add_trace(go.Scatter(x=eq_rsi.index,y=eq_rsi,mode="lines",
                line=dict(color="#00d4ff",width=1.5),name="Stratégie"),row=2,col=1)
            fig_rsi_bt.add_trace(go.Scatter(x=bh_rsi.index,y=bh_rsi,mode="lines",
                line=dict(color="#4a7a99",width=1,dash="dash"),name="Buy & Hold"),row=2,col=1)
            dd_rsi=(eq_rsi/eq_rsi.cummax()-1)*100
            fig_rsi_bt.add_trace(go.Scatter(x=dd_rsi.index,y=dd_rsi,mode="lines",
                fill="tozeroy",fillcolor="rgba(255,68,102,0.15)",
                line=dict(color="#ff4466",width=1),name="Drawdown"),row=3,col=1)
            fig_rsi_bt.update_layout(**PLOTLY_BASE,height=520,
                title=dict(text=f"◈ RSI Mean-Reversion — {bt_ticker}",
                           font=dict(family="Orbitron",color="#00d4ff",size=12)),
                hovermode="x unified")
            st.plotly_chart(fig_rsi_bt, use_container_width=True)

    with bt_tabs[2]:
        blabel("Momentum Strategy")
        if not df_bt.empty:
            mc1,mc2 = st.columns(2)
            mom_period = mc1.slider("Fenêtre momentum (jours)",10,120,60,key="bt_mp")
            cost_m     = mc2.slider("Coût (%)",0.0,0.5,0.1,step=0.05,key="bt_mc")/100
            close_bt   = df_bt["Close"].squeeze()
            momentum   = close_bt.pct_change(mom_period)
            signal_mom = (momentum > 0).astype(int)
            ret_mom, eq_mom, bh_mom, _ = run_backtest(signal_mom, close_bt, cost_m)
            metrics_mom = perf_metrics(ret_mom, eq_mom, close_bt)
            cols_m = st.columns(len(metrics_mom))
            for i,(k,v) in enumerate(metrics_mom.items()): cols_m[i].metric(k,v)
            fig_mom = make_subplots(rows=3,cols=1,shared_xaxes=True,
                vertical_spacing=0.04,row_heights=[0.35,0.35,0.3])
            fig_mom.add_trace(go.Scatter(x=momentum.index,y=momentum*100,mode="lines",
                line=dict(color="#00d4ff",width=1),fill="tozeroy",
                fillcolor="rgba(0,212,255,0.05)",name=f"Momentum {mom_period}j (%)"),row=1,col=1)
            fig_mom.add_hline(y=0,line_color="#4a7a99",line_width=0.8,row=1,col=1)
            fig_mom.add_trace(go.Scatter(x=eq_mom.index,y=eq_mom,mode="lines",
                line=dict(color="#00d4ff",width=1.5),name="Stratégie"),row=2,col=1)
            fig_mom.add_trace(go.Scatter(x=bh_mom.index,y=bh_mom,mode="lines",
                line=dict(color="#4a7a99",width=1,dash="dash"),name="Buy & Hold"),row=2,col=1)
            dd_mom=(eq_mom/eq_mom.cummax()-1)*100
            fig_mom.add_trace(go.Scatter(x=dd_mom.index,y=dd_mom,mode="lines",
                fill="tozeroy",fillcolor="rgba(255,68,102,0.15)",
                line=dict(color="#ff4466",width=1),name="Drawdown"),row=3,col=1)
            fig_mom.update_layout(**PLOTLY_BASE,height=520,
                title=dict(text=f"◈ Momentum ({mom_period}j) — {bt_ticker}",
                           font=dict(family="Orbitron",color="#00d4ff",size=12)),
                hovermode="x unified")
            st.plotly_chart(fig_mom, use_container_width=True)

    with bt_tabs[3]:
        blabel("Comparaison toutes stratégies")
        if not df_bt.empty:
            close_bt = df_bt["Close"].squeeze()
            sma_sig  = (close_bt.rolling(20).mean() > close_bt.rolling(50).mean()).astype(int)
            _,eq_sma,bh,_ = run_backtest(sma_sig, close_bt, 0.001)
            rsi_sig2 = pd.Series(0,index=close_bt.index)
            r_delta  = close_bt.diff(); r_gain=r_delta.clip(lower=0).rolling(14).mean()
            r_loss   = (-r_delta.clip(upper=0)).rolling(14).mean()
            rsi2     = 100-100/(1+r_gain/r_loss.replace(0,np.nan))
            rsi_sig2[rsi2<30]=1; rsi_sig2[rsi2>70]=-1
            rsi_sig2 = rsi_sig2.replace(0,np.nan).ffill().fillna(0)
            _,eq_rsi2,_,_ = run_backtest(rsi_sig2, close_bt, 0.001)
            mom_sig2 = (close_bt.pct_change(60)>0).astype(int)
            _,eq_mom2,_,_ = run_backtest(mom_sig2, close_bt, 0.001)
            fig_cmp = go.Figure()
            for label, eq, color in [("SMA 20/50",eq_sma,"#00d4ff"),
                                      ("RSI MR",   eq_rsi2,"#aa88ff"),
                                      ("Momentum", eq_mom2,"#ffaa00"),
                                      ("Buy & Hold",bh,    "#4a7a99")]:
                fig_cmp.add_trace(go.Scatter(x=eq.index,y=eq,mode="lines",
                    line=dict(color=color,width=1.5),name=label))
            fig_cmp.update_layout(**PLOTLY_BASE,height=400,
                title=dict(text=f"◈ Comparaison stratégies — {bt_ticker}",
                           font=dict(family="Orbitron",color="#00d4ff",size=12)),
                hovermode="x unified",yaxis_title="Equity (base 100)")
            st.plotly_chart(fig_cmp, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
#  TAB 5 — RISK MANAGEMENT
# ══════════════════════════════════════════════════════════════════════
with tabs[5]:
    blabel("Risk Management · VaR · CVaR · Stress Tests · Greeks")
    risk_tabs = st.tabs(["VaR / CVaR","Stress Tests","Greeks Options","Exposition"])

    with risk_tabs[0]:
        blabel("Value at Risk & Conditional VaR")
        risk_t  = st.selectbox("Ticker", selected, key="risk_tk")
        conf_lv = st.slider("Niveau de confiance (%)", 90, 99, 95) / 100
        portfolio_value = st.number_input("Valeur portefeuille ($)", value=100_000, step=10_000)
        r_risk  = returns_df[risk_t].dropna() if risk_t in returns_df else pd.Series(dtype=float)
        if not r_risk.empty:
            var_hist  = compute_var(r_risk, conf_lv, "historical")
            var_param = compute_var(r_risk, conf_lv, "parametric")
            var_cf    = compute_var(r_risk, conf_lv, "cornish_fisher")
            cvar_v    = compute_cvar(r_risk, conf_lv)

            rc1,rc2,rc3,rc4 = st.columns(4)
            rc1.metric(f"VaR Hist. ({conf_lv*100:.0f}%)",
                f"${abs(var_hist*portfolio_value):,.0f}", f"{var_hist*100:.2f}%",delta_color="inverse")
            rc2.metric(f"VaR Param. ({conf_lv*100:.0f}%)",
                f"${abs(var_param*portfolio_value):,.0f}", f"{var_param*100:.2f}%",delta_color="inverse")
            rc3.metric(f"VaR C-F ({conf_lv*100:.0f}%)",
                f"${abs(var_cf*portfolio_value):,.0f}", f"{var_cf*100:.2f}%",delta_color="inverse")
            rc4.metric(f"CVaR ({conf_lv*100:.0f}%)",
                f"${abs(cvar_v*portfolio_value):,.0f}", f"{cvar_v*100:.2f}%",delta_color="inverse")

            fig_var = go.Figure()
            fig_var.add_trace(go.Histogram(x=r_risk*100,nbinsx=80,
                marker_color="#00d4ff",opacity=0.6,name="Returns (%)",
                histnorm="probability density"))
            for v,label,color in [(var_hist,"VaR Hist","#ff4466"),
                                   (var_param,"VaR Param","#ffaa00"),
                                   (cvar_v,"CVaR","#ff88aa")]:
                fig_var.add_vline(x=v*100,line_dash="dot",line_color=color,
                    annotation_text=label,annotation_font_color=color)
            x_fill = np.linspace(r_risk.min()*100, var_hist*100, 200)
            fig_var.add_trace(go.Scatter(x=x_fill,
                y=stats.norm.pdf(x_fill/100,r_risk.mean(),r_risk.std())/100,
                mode="none",fill="tozeroy",fillcolor="rgba(255,68,102,0.2)",name="Tail"))
            fig_var.update_layout(**PLOTLY_BASE,height=380,
                title=dict(text=f"◈ Distribution VaR — {risk_t}",
                           font=dict(family="Orbitron",color="#00d4ff",size=12)),
                xaxis_title="Rendement journalier (%)")
            st.plotly_chart(fig_var, use_container_width=True)

            # Rolling VaR
            blabel("Rolling VaR (60j)")
            roll_var = r_risk.rolling(60).quantile(1-conf_lv)
            fig_rv = go.Figure()
            fig_rv.add_trace(go.Scatter(x=r_risk.index,y=r_risk*100,mode="lines",
                line=dict(color="#00d4ff",width=0.8),opacity=0.5,name="Returns"))
            fig_rv.add_trace(go.Scatter(x=roll_var.index,y=roll_var*100,mode="lines",
                line=dict(color="#ff4466",width=1.5),name=f"Rolling VaR {conf_lv*100:.0f}%"))
            fig_rv.update_layout(**PLOTLY_BASE,height=300,
                title=dict(text="◈ Rolling VaR (60j)",
                           font=dict(family="Orbitron",color="#00d4ff",size=11)),
                yaxis_title="%",hovermode="x unified")
            st.plotly_chart(fig_rv, use_container_width=True)

    with risk_tabs[1]:
        blabel("Stress Tests · Scénarios historiques")
        stress_t = st.selectbox("Ticker", selected, key="stress_tk")
        df_stress = get_ohlcv(stress_t, "5y", "1d")
        if not df_stress.empty:
            r_stress = df_stress["Close"].squeeze().pct_change().dropna()
            scenarios = {
                "COVID Crash (Feb-Mar 2020)": ("2020-02-19","2020-03-23"),
                "Rate Hike 2022":             ("2022-01-03","2022-10-13"),
                "Q4 2018 Selloff":            ("2018-10-01","2018-12-24"),
                "Covid Recovery (2020)":      ("2020-03-23","2020-08-31"),
                "Dot-com (2000-2002)":        ("2000-03-10","2002-10-09"),
            }
            rows_sc = []
            for name, (start, end) in scenarios.items():
                try:
                    mask = (r_stress.index >= start) & (r_stress.index <= end)
                    r_sc = r_stress[mask]
                    if len(r_sc) < 5: continue
                    total = float((1+r_sc).prod()-1)
                    ann   = float(r_sc.mean()*252)
                    vol   = float(r_sc.std()*np.sqrt(252))
                    mdd   = float(max_drawdown((1+r_sc).cumprod()))
                    rows_sc.append({"Scénario":name,"Return":f"{total*100:+.1f}%",
                                    "Ann.":f"{ann*100:+.1f}%","Vol.":f"{vol*100:.1f}%",
                                    "Max DD":f"{mdd*100:.1f}%","Jours":len(r_sc)})
                except: pass

            st.dataframe(pd.DataFrame(rows_sc).set_index("Scénario"),
                         use_container_width=True)

            # Simulation stress -10%, -20%, -30%
            blabel("Simulation pertes sur portefeuille")
            pv = st.number_input("Valeur ($)", value=100_000, step=10_000, key="stress_pv")
            shocks = [-0.05,-0.10,-0.15,-0.20,-0.30,-0.40,-0.50]
            fig_shock = go.Figure(go.Bar(
                x=[f"{s*100:.0f}%" for s in shocks],
                y=[pv*s for s in shocks],
                marker_color=["#ff4466"]*len(shocks),
                text=[f"${pv*s:,.0f}" for s in shocks],
                textposition="outside",
                textfont=dict(family="Space Mono",color="#e8f4fd",size=10),
            ))
            fig_shock.update_layout(**PLOTLY_BASE,height=300,
                title=dict(text="◈ Impact chocs sur portefeuille",
                           font=dict(family="Orbitron",color="#00d4ff",size=11)),
                yaxis_title="Perte ($)")
            st.plotly_chart(fig_shock, use_container_width=True)

    with risk_tabs[2]:
        blabel("Greeks Options (calcul analytique BS)")
        from scipy.stats import norm as norm_dist

        def bs_greeks(S, K, T, r, sigma, opt_type="call"):
            if T <= 0 or sigma <= 0: return {}
            d1 = (np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
            d2 = d1-sigma*np.sqrt(T)
            delta = norm_dist.cdf(d1) if opt_type=="call" else norm_dist.cdf(d1)-1
            gamma = norm_dist.pdf(d1)/(S*sigma*np.sqrt(T))
            theta = (-(S*norm_dist.pdf(d1)*sigma)/(2*np.sqrt(T)) -
                     r*K*np.exp(-r*T)*norm_dist.cdf(d2 if opt_type=="call" else -d2))/365
            vega  = S*norm_dist.pdf(d1)*np.sqrt(T)/100
            rho   = K*T*np.exp(-r*T)*norm_dist.cdf(d2 if opt_type=="call" else -d2)/100
            price = (S*norm_dist.cdf(d1)-K*np.exp(-r*T)*norm_dist.cdf(d2)
                     if opt_type=="call" else
                     K*np.exp(-r*T)*norm_dist.cdf(-d2)-S*norm_dist.cdf(-d1))
            return dict(Price=price,Delta=delta,Gamma=gamma,Theta=theta,Vega=vega,Rho=rho)

        gc1,gc2,gc3,gc4,gc5 = st.columns(5)
        S_in  = gc1.number_input("Spot (S)", value=150.0, step=1.0)
        K_in  = gc2.number_input("Strike (K)", value=150.0, step=1.0)
        T_in  = gc3.number_input("Maturité (jours)", value=30, step=1)
        iv_in = gc4.number_input("IV (σ %)", value=25.0, step=0.5) / 100
        r_in  = gc5.number_input("Taux sans risque (%)", value=5.0, step=0.1) / 100
        opt_type_bs = st.radio("Type", ["call","put"], horizontal=True)

        greeks = bs_greeks(S_in, K_in, T_in/365, r_in, iv_in, opt_type_bs)
        if greeks:
            gm1,gm2,gm3,gm4,gm5,gm6 = st.columns(6)
            gm1.metric("Prix théorique", f"${greeks['Price']:.4f}")
            gm2.metric("Δ Delta", f"{greeks['Delta']:.4f}")
            gm3.metric("Γ Gamma", f"{greeks['Gamma']:.6f}")
            gm4.metric("Θ Theta/j", f"{greeks['Theta']:.4f}")
            gm5.metric("ν Vega/1%", f"{greeks['Vega']:.4f}")
            gm6.metric("ρ Rho/1%", f"{greeks['Rho']:.4f}")

            # Delta vs Spot
            spots = np.linspace(S_in*0.7, S_in*1.3, 100)
            deltas = [bs_greeks(s,K_in,T_in/365,r_in,iv_in,opt_type_bs).get("Delta",0) for s in spots]
            gammas = [bs_greeks(s,K_in,T_in/365,r_in,iv_in,opt_type_bs).get("Gamma",0) for s in spots]
            prices_bs = [bs_greeks(s,K_in,T_in/365,r_in,iv_in,opt_type_bs).get("Price",0) for s in spots]

            fig_greeks = make_subplots(rows=1,cols=3,subplot_titles=["Prix","Delta","Gamma"])
            fig_greeks.add_trace(go.Scatter(x=spots,y=prices_bs,mode="lines",
                line=dict(color="#00d4ff",width=1.5),name="Prix"),row=1,col=1)
            fig_greeks.add_trace(go.Scatter(x=spots,y=deltas,mode="lines",
                line=dict(color="#00ff88",width=1.5),name="Delta"),row=1,col=2)
            fig_greeks.add_trace(go.Scatter(x=spots,y=gammas,mode="lines",
                line=dict(color="#ffaa00",width=1.5),name="Gamma"),row=1,col=3)
            for col in [1,2,3]:
                fig_greeks.add_vline(x=S_in,line_dash="dot",line_color="#4a7a99",
                    line_width=0.8,row=1,col=col)
            fig_greeks.update_layout(**PLOTLY_BASE,height=320,showlegend=False,
                title=dict(text="◈ Greeks vs Spot",
                           font=dict(family="Orbitron",color="#00d4ff",size=11)))
            st.plotly_chart(fig_greeks, use_container_width=True)

    with risk_tabs[3]:
        blabel("Exposition & Métriques de portefeuille")
        if len(selected) >= 2:
            exp_cols = st.columns(len(selected))
            for i, t in enumerate(selected):
                r_t = returns_df[t].dropna() if t in returns_df else pd.Series(dtype=float)
                if not r_t.empty:
                    vol_ann = r_t.std()*np.sqrt(252)*100
                    sharpe_v = sharpe(r_t)
                    mdd_v = max_drawdown(price_df[t]) if t in price_df else 0
                    with exp_cols[i]:
                        st.markdown(f"<div style='font-family:Space Mono,monospace;font-size:10px;color:#00d4ff;'>{t}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='font-family:Space Mono,monospace;font-size:10px;line-height:2;color:#8aafc8;'>Vol ann.: <b>{vol_ann:.1f}%</b><br>Sharpe: <b>{sharpe_v:.2f}</b><br>Max DD: <b>{mdd_v*100:.1f}%</b></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  TAB 6 — PORTFOLIO OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════
with tabs[6]:
    blabel("Optimisation de Portefeuille · Frontière Efficiente · Fama-French")
    port_tabs = st.tabs(["Optimisation","Frontière Efficiente","Factor Model"])

    with port_tabs[0]:
        if len(selected) < 2:
            st.info("Sélectionnez ≥ 2 tickers.")
        else:
            blabel("Allocation optimale")
            opt_method = st.radio("Méthode", ["Max Sharpe","Min Volatilité","Equal Weight"], horizontal=True)
            method_map = {"Max Sharpe":"sharpe","Min Volatilité":"min_vol","Equal Weight":"equal"}
            weights = optimize_portfolio(returns_df, method_map[opt_method])
            mu   = returns_df.mean()*252
            cov  = returns_df.cov()*252
            p_ret  = float(weights@mu)
            p_vol  = float(np.sqrt(weights@cov@weights))
            p_sh   = p_ret/(p_vol+1e-9)

            wm1,wm2,wm3 = st.columns(3)
            wm1.metric("Rendement attendu", f"{p_ret*100:.2f}%")
            wm2.metric("Volatilité",         f"{p_vol*100:.2f}%")
            wm3.metric("Sharpe ratio",        f"{p_sh:.3f}")

            wdf = pd.DataFrame({"Ticker":returns_df.columns,"Poids":weights,"Poids %":weights*100})
            wdf = wdf.sort_values("Poids",ascending=False)

            pw1,pw2 = st.columns([1,1])
            with pw1:
                fig_pie = go.Figure(go.Pie(labels=wdf["Ticker"],values=wdf["Poids"],
                    marker_colors=COLORS[:len(wdf)],
                    textinfo="label+percent",hole=0.4,
                    textfont=dict(family="Space Mono",size=11)))
                fig_pie.update_layout(**PLOTLY_BASE,height=340,
                    title=dict(text="◈ Allocation",
                               font=dict(family="Orbitron",color="#00d4ff",size=11)))
                st.plotly_chart(fig_pie, use_container_width=True)
            with pw2:
                fig_wbar = go.Figure(go.Bar(x=wdf["Ticker"],y=wdf["Poids %"],
                    marker_color=COLORS[:len(wdf)],
                    text=[f"{v:.1f}%" for v in wdf["Poids %"]],textposition="outside",
                    textfont=dict(family="Space Mono",color="#e8f4fd",size=10)))
                fig_wbar.update_layout(**PLOTLY_BASE,height=340,
                    title=dict(text="◈ Poids par actif",
                               font=dict(family="Orbitron",color="#00d4ff",size=11)),
                    yaxis_title="%")
                st.plotly_chart(fig_wbar, use_container_width=True)

            # Performance historique du portefeuille
            blabel("Performance historique du portefeuille optimisé")
            port_returns = (returns_df*weights).sum(axis=1)
            port_equity  = (1+port_returns).cumprod()*100
            fig_port = go.Figure()
            fig_port.add_trace(go.Scatter(x=port_equity.index,y=port_equity,mode="lines",
                line=dict(color="#00d4ff",width=1.5),fill="tozeroy",
                fillcolor="rgba(0,212,255,0.05)",name="Portefeuille"))
            for i,t in enumerate(returns_df.columns):
                eq_t = (1+returns_df[t]).cumprod()*100
                fig_port.add_trace(go.Scatter(x=eq_t.index,y=eq_t,mode="lines",
                    line=dict(color=COLORS[i%len(COLORS)],width=0.7,dash="dot"),name=t))
            fig_port.update_layout(**PLOTLY_BASE,height=360,
                title=dict(text=f"◈ Portefeuille {opt_method} vs actifs individuels",
                           font=dict(family="Orbitron",color="#00d4ff",size=12)),
                hovermode="x unified")
            st.plotly_chart(fig_port, use_container_width=True)

    with port_tabs[1]:
        blabel("Frontière Efficiente (Monte Carlo)")
        if len(selected) < 2:
            st.info("Sélectionnez ≥ 2 tickers.")
        else:
            n_portfolios = st.slider("Nombre de portefeuilles simulés", 500, 5000, 2000, step=500)
            with st.spinner(f"Simulation {n_portfolios} portefeuilles…"):
                np.random.seed(42)
                mu_ef   = returns_df.mean()*252
                cov_ef  = returns_df.cov()*252
                n_assets = len(returns_df.columns)
                rets_sim = []; vols_sim = []; shs_sim = []; ws_sim = []
                for _ in range(n_portfolios):
                    w = np.random.dirichlet(np.ones(n_assets))
                    r_p = float(w@mu_ef)
                    v_p = float(np.sqrt(w@cov_ef@w))
                    rets_sim.append(r_p); vols_sim.append(v_p)
                    shs_sim.append(r_p/v_p); ws_sim.append(w)

            fig_ef = go.Figure()
            fig_ef.add_trace(go.Scatter(x=np.array(vols_sim)*100,
                y=np.array(rets_sim)*100,mode="markers",
                marker=dict(color=shs_sim,colorscale=[
                    [0,"#ff4466"],[0.5,"#ffaa00"],[1,"#00d4ff"]],
                    size=3,opacity=0.7,showscale=True,
                    colorbar=dict(title="Sharpe",
                                  tickfont=dict(family="Space Mono",color="#8aafc8"))),
                name="Portefeuilles"))
            # actifs individuels
            for i,t in enumerate(returns_df.columns):
                r_i=float(mu_ef[t]); v_i=float(np.sqrt(cov_ef.loc[t,t]))
                fig_ef.add_trace(go.Scatter(x=[v_i*100],y=[r_i*100],mode="markers+text",
                    text=[t],textposition="top center",
                    marker=dict(color=COLORS[i%len(COLORS)],size=10,symbol="diamond"),name=t))
            max_sh_idx = np.argmax(shs_sim)
            fig_ef.add_trace(go.Scatter(x=[vols_sim[max_sh_idx]*100],y=[rets_sim[max_sh_idx]*100],
                mode="markers",marker=dict(color="#00ff88",size=14,symbol="star"),
                name="Max Sharpe"))
            fig_ef.update_layout(**PLOTLY_BASE,height=480,
                title=dict(text="◈ Frontière Efficiente de Markowitz",
                           font=dict(family="Orbitron",color="#00d4ff",size=12)),
                xaxis_title="Volatilité (%)",yaxis_title="Rendement attendu (%)")
            st.plotly_chart(fig_ef, use_container_width=True)

    with port_tabs[2]:
        blabel("Factor Model · Alpha/Beta vs benchmark")
        if not SM_OK:
            st.error("pip install statsmodels")
        else:
            bench = st.selectbox("Benchmark", ["SPY","QQQ","^GSPC","^IXIC"], key="bench_tk")
            df_bench = get_ohlcv(bench, period, "1d")
            if df_bench.empty:
                st.warning(f"Impossible de charger {bench}.")
            else:
                r_bench = df_bench["Close"].squeeze().pct_change().dropna()
                rows_fm = []
                for t in selected:
                    r_t = returns_df[t].dropna() if t in returns_df else pd.Series(dtype=float)
                    if r_t.empty: continue
                    idx_c = r_t.index.intersection(r_bench.index)
                    y_t=r_t[idx_c]; x_t=r_bench[idx_c]
                    slope,intercept,r_val,p_val,_ = stats.linregress(x_t,y_t)
                    rows_fm.append({
                        "Ticker": t,
                        "Alpha (ann.)": f"{intercept*252*100:.2f}%",
                        "Beta": f"{slope:.3f}",
                        "R²": f"{r_val**2:.3f}",
                        "p-value Beta": f"{p_val:.4f}",
                        "Treynor": f"{r_t.mean()*252/slope:.3f}" if slope!=0 else "N/A",
                        "Info Ratio": f"{(r_t-r_bench[idx_c]).mean()/((r_t-r_bench[idx_c]).std()+1e-9)*np.sqrt(252):.3f}",
                    })
                st.dataframe(pd.DataFrame(rows_fm).set_index("Ticker"), use_container_width=True)

                # Beta bar chart
                betas = [float(r["Beta"]) for r in rows_fm]
                tickers_fm = [r["Ticker"] for r in rows_fm]
                fig_beta = go.Figure(go.Bar(x=tickers_fm,y=betas,
                    marker_color=["#00ff88" if b<1 else "#ff4466" for b in betas],
                    text=[f"{b:.3f}" for b in betas],textposition="outside",
                    textfont=dict(family="Space Mono",color="#e8f4fd",size=10)))
                fig_beta.add_hline(y=1,line_dash="dot",line_color="#ffaa00",line_width=0.8,
                    annotation_text="Beta = 1",annotation_font_color="#ffaa00")
                fig_beta.update_layout(**PLOTLY_BASE,height=300,
                    title=dict(text=f"◈ Beta vs {bench}",
                               font=dict(family="Orbitron",color="#00d4ff",size=11)),
                    yaxis_title="Beta")
                st.plotly_chart(fig_beta, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
#  TAB 7 — MACRO / FRED
# ══════════════════════════════════════════════════════════════════════
with tabs[7]:
    blabel("FRED · Données Macroéconomiques · 800 000+ séries")
    st.markdown("<span class='tag-key'>Clé gratuite : fred.stlouisfed.org</span>", unsafe_allow_html=True)
    FRED_SERIES = {
        "Fed Funds Rate": "FEDFUNDS", "CPI Inflation": "CPIAUCSL",
        "Core PCE": "PCEPILFE", "PIB USA": "GDP", "Chômage": "UNRATE",
        "10Y Treasury": "DGS10", "2Y Treasury": "DGS2", "Spread 10-2Y": "T10Y2Y",
        "VIX": "VIXCLS", "M2 Money Supply": "M2SL", "Jobless Claims": "ICSA",
        "Housing Starts": "HOUST", "Retail Sales": "RSAFS",
    }
    fred_sel = st.multiselect("Séries FRED", list(FRED_SERIES.keys()),
                               default=["Fed Funds Rate","10Y Treasury","Spread 10-2Y","VIX"])
    custom_fred = st.text_input("Série custom (code FRED)", placeholder="ex: SP500")

    if not fred_key:
        st.info("Entrez votre clé FRED gratuite dans la barre latérale. fred.stlouisfed.org/docs/api/api_key.html")
    elif not FRED_OK:
        st.error("pip install fredapi")
    else:
        ids = [FRED_SERIES[l] for l in fred_sel]
        if custom_fred.strip(): ids.append(custom_fred.strip().upper())
        if ids:
            metric_cols = st.columns(min(len(ids),4))
            fig_fred = go.Figure()
            for i,sid in enumerate(ids):
                s = get_fred(sid, fred_key)
                if s.empty: continue
                fig_fred.add_trace(go.Scatter(x=s.index,y=s.values,mode="lines",
                    name=sid,line=dict(color=COLORS[i%len(COLORS)],width=1.5)))
                if i < len(metric_cols):
                    lv = float(s.iloc[-1]); pv = float(s.iloc[-2]) if len(s)>1 else lv
                    metric_cols[i].metric(sid, f"{lv:.2f}", f"{lv-pv:+.3f}")
            fig_fred.update_layout(**PLOTLY_BASE,height=420,
                title=dict(text="◈ Indicateurs Macro — FRED",
                           font=dict(family="Orbitron",color="#00d4ff",size=12)),
                hovermode="x unified")
            st.plotly_chart(fig_fred, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
#  TAB 8 — FOREX
# ══════════════════════════════════════════════════════════════════════
with tabs[8]:
    blabel("Forex · Taux historiques et live")
    PAIRS = {"EUR/USD":"EURUSD=X","GBP/USD":"GBPUSD=X","USD/JPY":"JPY=X",
             "USD/CHF":"CHF=X","AUD/USD":"AUDUSD=X","USD/CAD":"CAD=X",
             "EUR/GBP":"EURGBP=X","EUR/JPY":"EURJPY=X","BTC/USD":"BTC-USD",
             "XAU/USD":"GC=F","WTI":"CL=F"}
    fx1,fx2,fx3 = st.columns([2,2,1])
    sel_pairs = fx1.multiselect("Paires", list(PAIRS.keys()),
                                default=["EUR/USD","GBP/USD","USD/JPY","XAU/USD"])
    fx_period = fx2.selectbox("Période", ["1mo","3mo","6mo","1y","2y"], index=2, key="fx_p")
    norm_fx   = fx3.checkbox("Normaliser", value=True)

    if forex_key and sel_pairs:
        live_cols = st.columns(min(len(sel_pairs),4))
        for i,pair in enumerate(sel_pairs[:4]):
            parts = pair.split("/")
            if len(parts)==2:
                rate, _ = get_forex_rate(parts[0],parts[1],forex_key)
                if rate and i < len(live_cols):
                    live_cols[i].metric(pair, f"{rate:.4f}")

    closes_fx = {}
    for pair in sel_pairs:
        sym = PAIRS[pair]; d = get_ohlcv(sym, fx_period, "1d")
        if not d.empty: closes_fx[pair] = d["Close"].squeeze()

    if closes_fx:
        pdf = pd.DataFrame(closes_fx).dropna()
        plot_df = (pdf/pdf.iloc[0]*100) if norm_fx else pdf
        fig_fx = go.Figure()
        for i,col in enumerate(plot_df.columns):
            fig_fx.add_trace(go.Scatter(x=plot_df.index,y=plot_df[col],mode="lines",
                name=col,line=dict(color=COLORS[i%len(COLORS)],width=1.5)))
        fig_fx.update_layout(**PLOTLY_BASE,height=400,
            title=dict(text=f"◈ Forex {'normalisé' if norm_fx else 'prix'} — {fx_period}",
                       font=dict(family="Orbitron",color="#00d4ff",size=12)),
            hovermode="x unified",yaxis_title="Base 100" if norm_fx else "Prix")
        st.plotly_chart(fig_fx, use_container_width=True)

        # Recap table
        recap = []
        for pair in sel_pairs:
            sym = PAIRS[pair]; d = get_ohlcv(sym, fx_period, "1d")
            if d.empty: continue
            c = d["Close"].squeeze()
            recap.append({"Paire":pair,"Dernier":f"{float(c.iloc[-1]):.4f}",
                "Variation":f"{(float(c.iloc[-1])/float(c.iloc[0])-1)*100:+.2f}%",
                "High":f"{float(d['High'].squeeze().max()):.4f}",
                "Low":f"{float(d['Low'].squeeze().min()):.4f}",
                "Vol 20j σ":f"{float(c.pct_change().rolling(20).std().iloc[-1]*100):.2f}%"})
        st.dataframe(pd.DataFrame(recap).set_index("Paire"), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
#  TAB 9 — NEWS & SENTIMENT
# ══════════════════════════════════════════════════════════════════════
with tabs[9]:
    blabel("News & Analyse de Sentiment")
    nc1,nc2 = st.columns([3,1])
    news_query = nc1.text_input("Recherche", value=f"{primary} earnings", key="nq")
    news_days  = nc2.slider("Jours", 1, 30, 7, key="nd")

    if not news_key:
        st.info("Clé NewsAPI gratuite : newsapi.org  |  En attendant, actualités via yfinance :")
        try:
            t_news = yf.Ticker(primary)
            for article in (t_news.news or [])[:8]:
                title = article.get("title",""); link = article.get("link","")
                pub   = article.get("publisher","")
                ts    = article.get("providerPublishTime",0)
                dt    = datetime.fromtimestamp(ts).strftime("%d %b %Y") if ts else ""
                sentiment_html = ""
                if VADER_OK and title:
                    s = sia.polarity_scores(title)["compound"]
                    sentiment_html = ("<span style='color:#00ff88;'>▲ POS</span>" if s>=0.05
                                     else "<span style='color:#ff4466;'>▼ NEG</span>" if s<=-0.05
                                     else "<span style='color:#ffaa00;'>◆ NEU</span>")
                st.markdown(f"""<div class='news-card'>
                  <div class='news-title'><a href='{link}' style='color:#e8f4fd;text-decoration:none;' target='_blank'>{title}</a></div>
                  <div class='news-meta'>{pub} · {dt} · {sentiment_html}</div>
                </div>""", unsafe_allow_html=True)
        except: pass
    else:
        articles, err = get_news_api(news_query, news_key, news_days)
        if err: st.error(f"NewsAPI: {err}")
        elif not articles: st.info("Aucun article.")
        else:
            if VADER_OK:
                scores = [sia.polarity_scores(a.get("title","")+(" "+a.get("description","") or ""))["compound"] for a in articles]
                avg = np.mean(scores)
                sm1,sm2,sm3 = st.columns(3)
                sm1.metric("Sentiment moyen", f"{avg:+.3f}")
                sm2.metric("Positifs", sum(1 for s in scores if s>=0.05))
                sm3.metric("Négatifs", sum(1 for s in scores if s<=-0.05))
                fig_sent = go.Figure(go.Bar(x=list(range(len(scores))),y=scores,
                    marker_color=["#00ff88" if s>=0.05 else "#ff4466" if s<=-0.05
                                  else "#ffaa00" for s in scores],name="Score VADER"))
                fig_sent.add_hline(y=0,line_color="#4a7a99",line_width=0.8)
                fig_sent.update_layout(**PLOTLY_BASE,height=220,
                    title=dict(text="◈ Scores VADER",font=dict(family="Orbitron",color="#00d4ff",size=11)))
                st.plotly_chart(fig_sent, use_container_width=True)
            for article in articles[:15]:
                title  = article.get("title",""); url = article.get("url","")
                source = article.get("source",{}).get("name",""); pubdt = article.get("publishedAt","")[:10]
                desc   = article.get("description","") or ""
                sh = ""
                if VADER_OK:
                    sc = sia.polarity_scores(title+" "+desc)["compound"]
                    sh = ("▲ POS" if sc>=0.05 else "▼ NEG" if sc<=-0.05 else "◆ NEU")
                    sh = f"<span style='color:{'#00ff88' if sc>=0.05 else '#ff4466' if sc<=-0.05 else '#ffaa00'};'>{sh}</span>"
                st.markdown(f"""<div class='news-card'>
                  <div class='news-title'><a href='{url}' style='color:#e8f4fd;text-decoration:none;' target='_blank'>{title}</a></div>
                  <div class='news-meta'>{source} · {pubdt} · {sh}</div>
                  <div style='font-family:IBM Plex Sans,sans-serif;font-size:12px;color:#7aafc8;margin-top:6px;'>{desc[:200]}{'…' if len(desc)>200 else ''}</div>
                </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  TAB 10 — FUNDAMENTALS
# ══════════════════════════════════════════════════════════════════════
with tabs[10]:
    blabel("Fondamentaux · Valorisation · Financiers")
    fund_t = st.selectbox("Ticker", selected, key="fund_tk")
    info   = get_info(fund_t)
    if not info:
        st.warning("Données indisponibles.")
    else:
        fc1,fc2,fc3 = st.columns(3)
        with fc1:
            blabel("Entreprise")
            for label, key in [("Nom","longName"),("Secteur","sector"),
                                ("Industrie","industry"),("Pays","country"),("Change","exchange")]:
                st.markdown(f"<span style='color:#4a7a99;font-family:Space Mono,monospace;font-size:10px;'>{label}:</span> <span style='font-family:Space Mono,monospace;font-size:10px;'>{info.get(key,'N/A')}</span>", unsafe_allow_html=True)
            emp = info.get("fullTimeEmployees")
            if isinstance(emp,(int,float)):
                st.markdown(f"<span style='color:#4a7a99;font-family:Space Mono,monospace;font-size:10px;'>Employés:</span> <span style='font-family:Space Mono,monospace;font-size:10px;'>{emp:,.0f}</span>", unsafe_allow_html=True)
        with fc2:
            blabel("Valorisation")
            for k,v in [("Market Cap",fmt_large(info.get("marketCap"))),
                        ("Enterprise Value",fmt_large(info.get("enterpriseValue"))),
                        ("P/E TTM", f"{info.get('trailingPE',0):.2f}" if isinstance(info.get("trailingPE"),float) else "N/A"),
                        ("Fwd P/E", f"{info.get('forwardPE',0):.2f}" if isinstance(info.get("forwardPE"),float) else "N/A"),
                        ("P/B", f"{info.get('priceToBook',0):.2f}" if isinstance(info.get("priceToBook"),float) else "N/A"),
                        ("EV/EBITDA",f"{info.get('enterpriseToEbitda',0):.2f}" if isinstance(info.get("enterpriseToEbitda"),float) else "N/A")]:
                st.markdown(f"<span style='color:#4a7a99;font-family:Space Mono,monospace;font-size:10px;'>{k}:</span> <span style='font-family:Space Mono,monospace;font-size:10px;'>{v}</span>", unsafe_allow_html=True)
        with fc3:
            blabel("Financiers")
            for k,v in [("Revenue",fmt_large(info.get("totalRevenue"))),
                        ("EBITDA",fmt_large(info.get("ebitda"))),
                        ("Net Income",fmt_large(info.get("netIncomeToCommon"))),
                        ("Free Cash Flow",fmt_large(info.get("freeCashflow"))),
                        ("Total Cash",fmt_large(info.get("totalCash"))),
                        ("Total Debt",fmt_large(info.get("totalDebt")))]:
                st.markdown(f"<span style='color:#4a7a99;font-family:Space Mono,monospace;font-size:10px;'>{k}:</span> <span style='font-family:Space Mono,monospace;font-size:10px;'>{v}</span>", unsafe_allow_html=True)

        st.divider()
        blabel("Marges & Croissance")
        m1,m2,m3,m4,m5,m6 = st.columns(6)
        m1.metric("Gross Margin",  pct_fmt(info.get("grossMargins")))
        m2.metric("Op Margin",     pct_fmt(info.get("operatingMargins")))
        m3.metric("Profit Margin", pct_fmt(info.get("profitMargins")))
        m4.metric("Rev. Growth",   pct_fmt(info.get("revenueGrowth")))
        m5.metric("Earn. Growth",  pct_fmt(info.get("earningsGrowth")))
        m6.metric("ROE",           pct_fmt(info.get("returnOnEquity")))

        summary = info.get("longBusinessSummary","")
        if summary:
            with st.expander("▸ Description"):
                st.markdown(f"<p style='font-family:IBM Plex Sans,sans-serif;font-size:13px;color:#8aafc8;line-height:1.7;'>{summary}</p>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  TAB 11 — CORRELATION
# ══════════════════════════════════════════════════════════════════════
with tabs[11]:
    blabel("Matrice de Corrélation · Performance · Distribution")
    if len(selected) < 2:
        st.info("Sélectionnez ≥ 2 tickers.")
    else:
        corr = returns_df.corr()

        cc1,cc2 = st.columns(2)
        with cc1:
            blabel("Heatmap de corrélation")
            fig_hm = go.Figure(go.Heatmap(z=corr.values,x=corr.columns.tolist(),
                y=corr.columns.tolist(),
                colorscale=[[0,"#ff4466"],[0.25,"#aa2244"],[0.5,"#0d1f30"],
                             [0.75,"#004488"],[1,"#00d4ff"]],
                zmid=0,
                text=[[f"{v:.2f}" for v in row] for row in corr.values],
                texttemplate="%{text}",textfont=dict(family="Space Mono",size=11,color="#e8f4fd"),
                hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>ρ=%{z:.4f}<extra></extra>",
                colorbar=dict(tickfont=dict(family="Space Mono",color="#8aafc8"))))
            fig_hm.update_layout(**PLOTLY_BASE,height=420,
                title=dict(text="◈ Corrélation rendements journaliers",
                           font=dict(family="Orbitron",color="#00d4ff",size=12)))
            st.plotly_chart(fig_hm, use_container_width=True)

        with cc2:
            blabel("RSI comparatif")
            rsi_vals=[]; rsi_tks=[]
            for t in selected:
                if t in returns_df:
                    r_t = returns_df[t].dropna()
                    c_t = price_df[t].dropna() if t in price_df else pd.Series(dtype=float)
                    if len(c_t)>=14:
                        delta_t=c_t.diff(); gain_t=delta_t.clip(lower=0).rolling(14).mean()
                        loss_t=(-delta_t.clip(upper=0)).rolling(14).mean()
                        rsi_t=100-100/(1+gain_t/loss_t.replace(0,np.nan))
                        v=float(rsi_t.iloc[-1])
                        if not np.isnan(v): rsi_vals.append(v); rsi_tks.append(t)
            if rsi_vals:
                fig_rsi_bar = go.Figure(go.Bar(x=rsi_tks,y=rsi_vals,
                    marker_color=["#ff4466" if v>70 else "#00ff88" if v<30 else "#00d4ff" for v in rsi_vals],
                    text=[f"{v:.1f}" for v in rsi_vals],textposition="outside",
                    textfont=dict(family="Space Mono",color="#e8f4fd",size=10)))
                fig_rsi_bar.add_hline(y=70,line_dash="dot",line_color="#ff4466",line_width=0.8)
                fig_rsi_bar.add_hline(y=30,line_dash="dot",line_color="#00ff88",line_width=0.8)
                fig_rsi_bar.update_layout(**PLOTLY_BASE,height=420,
                    title=dict(text="◈ RSI(14) comparatif",
                               font=dict(family="Orbitron",color="#00d4ff",size=11)),
                    showlegend=False)
                fig_rsi_bar.update_yaxes(range=[0,100],title_text="RSI")
                st.plotly_chart(fig_rsi_bar, use_container_width=True)

        blabel("Performance normalisée (Base 100)")
        norm_df = (price_df/price_df.iloc[0]*100)
        fig_norm = go.Figure()
        for i,col in enumerate(norm_df.columns):
            fig_norm.add_trace(go.Scatter(x=norm_df.index,y=norm_df[col],mode="lines",
                name=col,line=dict(color=COLORS[i%len(COLORS)],width=1.5)))
        fig_norm.update_layout(**PLOTLY_BASE,height=360,
            title=dict(text="◈ Performance normalisée",
                       font=dict(family="Orbitron",color="#00d4ff",size=12)),
            hovermode="x unified",yaxis_title="Base 100")
        st.plotly_chart(fig_norm, use_container_width=True)

        blabel("Distribution des rendements")
        fig_violin = go.Figure()
        for i,col in enumerate(returns_df.columns):
            fig_violin.add_trace(go.Violin(y=returns_df[col]*100,name=col,
                box_visible=True,meanline_visible=True,
                line_color=COLORS[i%len(COLORS)],opacity=0.6))
        fig_violin.update_layout(**PLOTLY_BASE,height=360,
            title=dict(text="◈ Distribution rendements journaliers (%)",
                       font=dict(family="Orbitron",color="#00d4ff",size=12)),
            yaxis_title="%",violinmode="overlay")
        st.plotly_chart(fig_violin, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════
st.divider()
st.markdown(f"""
<div style='text-align:center;font-family:Space Mono,monospace;font-size:8px;color:#1e3a55;padding:8px 0;'>
  QUANTUM TERMINAL v3.0 · Market Data · Statistics · Backtesting · Risk · Portfolio · Macro · Forex · News · ML ·
  yfinance · ccxt · FRED · Alpha Vantage · NewsAPI · scipy · arch · statsmodels ·
  Usage informatif uniquement · Pas de conseil financier ·
  {datetime.now().strftime('%d %b %Y %H:%M')}
</div>""", unsafe_allow_html=True)
