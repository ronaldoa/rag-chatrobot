"""Streamlit UI for the Jesse Livermore chatbot with chat, backtest, and system settings."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List
from datetime import datetime

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Ensure project root on sys.path for Streamlit execution
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.qa_service import qa_service
from src.config import Settings


# ==========================================
# 1. Base page configuration
# ==========================================
st.set_page_config(
    page_title="aiaio - Jesse Livermore",
    page_icon="🎩",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================
# 2. Helpers
# ==========================================
@st.cache_resource(show_spinner=False)
def load_service():
    """Initialize and return the QA service (cached across reruns)."""
    qa_service.initialize()
    return qa_service


def ensure_state(defaults: Dict):
    """Bootstrap session_state with defaults."""
    st.session_state.setdefault(
        "messages",
        [
            {
                "role": "assistant",
                "content": "Hello, I’m Jesse Livermore. The market is never wrong, only opinions are. How can I help you today?",
            }
        ],
    )
    st.session_state.setdefault("system_prompt", "You are Jesse Livermore...")
    st.session_state.setdefault("model_name", defaults.get("model_path", "livermore-fin-7b"))

    # Sync LLM params into session state for sliders/inputs
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    # UI-centric temperature mirror
    st.session_state.setdefault("temperature", defaults.get("temperature", 0.7))


@st.cache_data(ttl=3600)
def get_stock_data(ticker: str, start, end):
    """Download ticker history with caching."""
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return pd.DataFrame()


def run_livermore_strategy(
    df: pd.DataFrame, fast_ma: int = 50, slow_ma: int = 200, breakout_window: int = 20
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Livermore-inspired signals, returns, and trade log."""
    if df.empty or "Close" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    data = df.copy().sort_index()

    data[f"{fast_ma}MA"] = data["Close"].rolling(window=fast_ma).mean()
    data[f"{slow_ma}MA"] = data["Close"].rolling(window=slow_ma).mean()
    data[f"{breakout_window}High"] = data["Close"].rolling(window=breakout_window).max()
    data[f"{breakout_window}Low"] = data["Close"].rolling(window=breakout_window).min()

    positions: List[int] = []
    current_pos = 0

    for i in range(len(data)):
        price = data["Close"].iloc[i]
        breakout_high = data[f"{breakout_window}High"].shift(1).iloc[i]
        breakout_low = data[f"{breakout_window}Low"].shift(1).iloc[i]
        above_fast = price > data[f"{fast_ma}MA"].iloc[i]
        above_slow = price > data[f"{slow_ma}MA"].iloc[i]

        buy = price > breakout_high and above_fast and above_slow
        sell = price < breakout_low

        if buy:
            current_pos = 1
        elif sell:
            current_pos = -1

        positions.append(current_pos)

    data["Position"] = positions
    data["Buy-and-Hold Return"] = data["Close"].pct_change().fillna(0)
    data["Strategy Return"] = data["Buy-and-Hold Return"] * data["Position"]
    data["Cum_BnH"] = (1 + data["Buy-and-Hold Return"]).cumprod()
    data["Cum_Strat"] = (1 + data["Strategy Return"]).cumprod()

    data["Position_Change"] = data["Position"].diff().fillna(data["Position"])
    trade_log = data[data["Position_Change"] != 0][["Position", "Close"]].copy()
    trade_log["Action"] = trade_log["Position"].apply(
        lambda p: "Buy breakout" if p == 1 else "Exit / Short"
    )
    trade_log.reset_index(inplace=True)
    trade_log.rename(columns={"index": "Date", "Close": "Price"}, inplace=True)

    return data, trade_log


def format_sources(sources: List[Dict]) -> str:
    """Render a markdown-friendly sources block."""
    if not sources:
        return ""
    lines = ["\n\n---\n📚 **References:**"]
    for idx, source in enumerate(sources, start=1):
        line = f"{idx}. `{source.get('source', 'Unknown')}`"
        page = source.get("page")
        if page is not None:
            line += f" (page {page})"
        lines.append(line)
    return "\n".join(lines)


def reset_chat():
    """Reset chat history."""
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello, I’m Jesse Livermore. Let’s restart our market conversation.",
        }
    ]
    st.session_state.chat_status = ""


def handle_chat(prompt: str):
    """Send a question to the QA service and append the response."""
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        with st.spinner("Thinking with local knowledge base..."):
            result = service.ask(prompt, history=st.session_state.messages)
    except Exception as exc:  # noqa: BLE001
        st.session_state.messages.append(
            {"role": "assistant", "content": f"❌ Error: {exc}"}
        )
        return

    answer = result["answer"] + format_sources(result.get("sources"))
    st.session_state.messages.append({"role": "assistant", "content": answer})


def refresh_llm_params():
    """Reload current LLM params from the service into session state."""
    params = service.current_llm_params()
    for key, value in params.items():
        st.session_state[key] = value
    st.toast("LLM parameters reloaded from service.", icon="🔄")


def flatten_config_to_df(data: Dict) -> pd.DataFrame:
    """Flatten health/config payload into a table for display."""
    rows = []
    rows.append({"Category": "System Status", "Parameter": "Status", "Value": data.get("status")})
    rows.append(
        {"Category": "System Status", "Parameter": "Model Loaded", "Value": str(data.get("model_loaded"))}
    )

    model_info = data.get("model_info") or {}
    for key, val in model_info.items():
        rows.append({"Category": "Model Info", "Parameter": key, "Value": str(val)})

    config = data.get("config") or {}
    for key, val in config.items():
        if isinstance(val, list):
            val = ", ".join(val)
        rows.append({"Category": "Configuration", "Parameter": key, "Value": str(val)})

    return pd.DataFrame(rows)


# ==========================================
# 3. Initialization
# ==========================================
service = load_service()
ensure_state(service.current_llm_params())

# Apply pending LLM param refresh before widgets render to avoid widget-state conflicts
if st.session_state.get("_refresh_llm_params"):
    refresh_llm_params()
    st.session_state.pop("_refresh_llm_params", None)

# ==========================================
# 4. Deep CSS customization (core UI logic)
# ==========================================
st.markdown(
    """
    <style>
        /* --- Color variables --- */
        :root {
            --bg-dark: #0f172a;      /* Main background: Slate 900 */
            --sidebar-bg: #1e293b;   /* Sidebar background: Slate 800 */
            --text-primary: #f1f5f9; /* Primary text: Slate 100 */
            --text-secondary: #94a3b8; /* Secondary text: Slate 400 */
            --accent-blue: #3b82f6;  /* Accent: Blue 500 */
            --border-color: #334155; /* Border: Slate 700 */
        }

        /* --- Global style reset --- */
        .stApp {
            background-color: var(--bg-dark);
            color: var(--text-primary);
        }
        /* Force typography to stay bright on dark background */
        body, .stApp, .stApp p, .stApp span, .stApp label, .stMarkdown {
            color: var(--text-primary) !important;
        }

        /* Hide Streamlit default header/menu */
        header, #MainMenu, footer {visibility: hidden;}

        /* Reduce top padding */
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 2rem !important;
        }

        /* --- Sidebar advanced layout (Flexbox hack) --- */
        section[data-testid="stSidebar"] {
            background-color: var(--sidebar-bg);
            border-right: 1px solid var(--border-color);
        }

        /* Make sidebar container a full-height flex column */
        section[data-testid="stSidebar"] > div {
            height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: space-between; /* Distribute top/bottom */
        }

        /* Let upper sidebar area take remaining space and scroll */
        section[data-testid="stSidebar"] > div > div:first-child {
            flex-grow: 1;
            overflow-y: auto;
            padding-bottom: 20px;
        }

        /* --- UI component styling --- */

        /* Inputs & dropdowns */
        .stTextInput > div > div > input,
        .stSelectbox > div > div > div,
        .stNumberInput > div > div > input,
        .stDateInput > div > div > input {
            background-color: var(--sidebar-bg) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 6px;
        }

        /* Chat input container */
        .stChatInputContainer {
            background-color: var(--bg-dark) !important;
            padding-bottom: 20px;
            border-top: 1px solid var(--border-color);
        }
        .stChatInputContainer textarea {
            background-color: var(--sidebar-bg) !important;
            border: 1px solid var(--border-color) !important;
            color: white !important;
        }

        /* Tab headers */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
            background-color: var(--bg-dark);
            padding: 10px 0px;
            border-bottom: 1px solid var(--border-color);
        }
        .stTabs [data-baseweb="tab"] {
            color: var(--text-secondary);
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            color: var(--accent-blue);
            font-weight: 600;
        }

        /* Custom HTML component styles */
        .chat-history-item {
            padding: 10px;
            border-radius: 6px;
            cursor: pointer;
            margin-bottom: 4px;
            transition: background 0.2s;
        }
        .chat-history-item:hover {
            background-color: #334155;
        }
        .history-title { font-size: 13px; color: #e2e8f0; font-weight: 500; }
        .history-date { font-size: 11px; color: #94a3b8; margin-top: 2px; }

        .user-profile {
            padding: 15px 10px;
            border-top: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: space-between;
            background-color: var(--sidebar-bg);
            margin: 0 -1rem; /* Offset padding */
            padding-left: 1.5rem;
            padding-right: 1.5rem;
        }
        .avatar-circle {
            width: 32px; height: 32px;
            background-color: var(--accent-blue);
            border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-weight: bold; font-size: 12px; color: white;
        }
    </style>
""",
    unsafe_allow_html=True,
)


# ==========================================
# 5. Sidebar content
# ==========================================
with st.sidebar:
    st.markdown(
        "<h2 style='color: #818cf8; margin:0; padding:0;'>aiaio</h2>",
        unsafe_allow_html=True,
    )
    st.caption("version: 0.0.5-livermore")
    st.write("")

    if st.button("＋ New Chat", use_container_width=True, type="primary"):
        reset_chat()
        st.rerun()

    st.markdown(
        "<p style='color:#64748b; font-size:12px; font-weight:600; margin-top:20px; text-transform:uppercase;'>Conversations</p>",
        unsafe_allow_html=True,
    )

    # Simple stub list for visual parity; in a real app this could reflect saved threads
    history_items = [
        {"title": "Local QA Session", "date": "Active"},
    ]

    for item in history_items:
        st.markdown(
            f"""
        <div class="chat-history-item">
            <div class="history-title">{item["title"]}</div>
            <div class="history-date">{item["date"]}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    icon_gear = """<svg width="18" height="18" fill="none" stroke="#94a3b8" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"></path><path stroke-linecap="round" stroke-linejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path></svg>"""

    st.markdown(
        f"""
        <div class="user-profile">
            <div style="display:flex; align-items:center; gap:10px;">
                <div class="avatar-circle">JL</div>
                <div style="font-size:14px; font-weight:600; color:#f1f5f9;">Trader Joe</div>
            </div>
            <div style="cursor:pointer;">{icon_gear}</div>
        </div>
    """,
        unsafe_allow_html=True,
    )


# ==========================================
# 7. Main interface (Tabs layout)
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["💬 Chat Interface", "📈 Backtest Dashboard", "📊 System Dashboard", "⚙️ Settings"]
)

# --- TAB 1: Chatbot ---
with tab1:
    col_chat, col_info = st.columns([3, 1])

    with col_chat:
        with st.expander("System Prompt Configuration", expanded=False):
            st.text_area("System Prompt", key="system_prompt", height=68)

        chat_container = st.container(height=500)
        with chat_container:
            for msg in st.session_state.messages:
                avatar = "🧑‍💻" if msg["role"] == "user" else "🎩"
                st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

        if prompt := st.chat_input("Ask Livermore about the market or your docs..."):
            handle_chat(prompt)
            st.rerun()

    with col_info:
        current_temp = st.session_state.get("temperature", 0.7)
        st.markdown(
            """
        <div style="background-color:#1e293b; padding:15px; border-radius:8px; border:1px solid #334155; margin-top:20px;">
            <h4 style="margin-top:0; font-size:14px; color:#94a3b8;">ACTIVE MODEL</h4>
            <div style="font-size:16px; font-weight:bold; color:#f1f5f9; margin-bottom:10px;">"""
            + st.session_state.model_name
            + """</div>
            <div style="font-size:12px; color:#64748b;">
                Temperature: <b>"""
            + str(current_temp)
            + """</b><br>
                Status: <span style="color:#10b981;">Online</span>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

# --- TAB 2: Backtest ---
with tab2:
    with st.container():
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        ticker = c1.text_input("Ticker", value="NVDA").upper()
        start_date = c2.date_input("Start", value=datetime(2020, 1, 1))
        end_date = c3.date_input("End", value=datetime.today())
        c4.write("")
        c4.write("")
        run_btn = c4.button("Run Simulation", type="primary", use_container_width=True)

    with st.container():
        p1, p2, p3 = st.columns(3)
        fast_ma = p1.number_input("Fast MA (trend filter)", min_value=5, max_value=150, value=50, step=5)
        slow_ma = p2.number_input("Slow MA (long trend)", min_value=50, max_value=400, value=200, step=10)
        breakout_window = p3.number_input("Breakout window (days)", min_value=5, max_value=120, value=20, step=5)

    if run_btn:
        if start_date >= end_date:
            st.error("Start date must be before end date.")
        else:
            with st.spinner("Calculating Strategy..."):
                df = get_stock_data(ticker, start_date, end_date)
                if df.empty:
                    st.error("No data found for this ticker.")
                else:
                    results, trades = run_livermore_strategy(
                        df, fast_ma=fast_ma, slow_ma=slow_ma, breakout_window=breakout_window
                    )

                    if results.empty:
                        st.error("Unable to generate strategy output for this date range.")
                    else:
                        strat_ret = results["Cum_Strat"].iloc[-1] - 1
                        bh_ret = results["Cum_BnH"].iloc[-1] - 1
                        delta = strat_ret - bh_ret

                        m1, m2, m3 = st.columns(3)
                        m1.metric("Strategy Return", f"{strat_ret:.2%}")
                        m2.metric("Buy & Hold Return", f"{bh_ret:.2%}")
                        m3.metric("Outperformance", f"{delta:.2%}")

                        price_fig = go.Figure()
                        price_fig.add_trace(
                            go.Scatter(
                                x=results.index,
                                y=results["Close"],
                                name="Close",
                                line=dict(color="#e2e8f0"),
                            )
                        )
                        price_fig.add_trace(
                            go.Scatter(
                                x=results.index,
                                y=results[f"{fast_ma}MA"],
                                name=f"{fast_ma} MA",
                                line=dict(color="#22c55e", width=1.5),
                            )
                        )
                        price_fig.add_trace(
                            go.Scatter(
                                x=results.index,
                                y=results[f"{slow_ma}MA"],
                                name=f"{slow_ma} MA",
                                line=dict(color="#a855f7", width=1.5),
                            )
                        )

                        entries = results[results["Position_Change"] > 0]
                        exits = results[results["Position_Change"] < 0]
                        if not entries.empty:
                            price_fig.add_trace(
                                go.Scatter(
                                    x=entries.index,
                                    y=entries["Close"],
                                    mode="markers",
                                    marker=dict(color="#10b981", size=8, symbol="triangle-up"),
                                    name="Breakout Buy",
                                )
                            )
                        if not exits.empty:
                            price_fig.add_trace(
                                go.Scatter(
                                    x=exits.index,
                                    y=exits["Close"],
                                    mode="markers",
                                    marker=dict(color="#f87171", size=8, symbol="triangle-down"),
                                    name="Exit / Short",
                                )
                            )

                        price_fig.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            height=420,
                            title="Price with MAs & Signals",
                        )

                        equity_fig = go.Figure()
                        equity_fig.add_trace(
                            go.Scatter(
                                x=results.index,
                                y=results["Cum_BnH"],
                                name="Buy & Hold",
                                line=dict(color="#64748b"),
                            )
                        )
                        equity_fig.add_trace(
                            go.Scatter(
                                x=results.index,
                                y=results["Cum_Strat"],
                                name="Livermore Strategy",
                                line=dict(color="#3b82f6", width=2),
                            )
                        )
                        equity_fig.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            height=420,
                            title="Equity Curve (Return Dynamics)",
                        )

                        st.plotly_chart(price_fig, use_container_width=True)
                        st.plotly_chart(equity_fig, use_container_width=True)

                        if not trades.empty:
                            st.markdown("**Trade Log (signals & fills)**")
                            st.dataframe(
                                trades.tail(20),
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "Date": st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD"),
                                    "Price": st.column_config.NumberColumn("Price", format="%.2f"),
                                    "Action": st.column_config.TextColumn("Action"),
                                    "Position": st.column_config.NumberColumn("Position"),
                                },
                            )

# --- TAB 3: System dashboard ---
with tab3:
    st.subheader("System Configuration (Table View)")
    try:
        health = service.health_check()
        st.dataframe(
            flatten_config_to_df(health),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Category": st.column_config.TextColumn("Category", width="medium"),
                "Parameter": st.column_config.TextColumn("Parameter", width="medium"),
                "Value": st.column_config.TextColumn("Current Value", width="large"),
            },
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Health check failed: {exc}")

# --- TAB 4: Settings ---
with tab4:
    st.subheader("System Configuration")
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        with st.container(border=True):
            st.markdown("**Model Parameters**")
            st.text_input("Model Path", key="model_path")
            st.slider("Context Window (n_ctx)", 512, 8192, key="n_ctx", step=256)
            st.slider("Batch Size (n_batch)", 32, 1024, key="n_batch", step=16)
            st.slider("GPU Layers", 0, 64, key="n_gpu_layers", step=1)
            st.slider("CPU Threads", 1, 32, key="n_threads", step=1)

    with col_s2:
        with st.container(border=True):
            st.markdown("**Generation Control**")
            st.slider("Temperature", 0.0, 1.5, key="temperature", step=0.05)
            st.slider("Top P", 0.1, 1.0, key="top_p", step=0.01)
            st.slider("Repeat Penalty", 0.8, 1.5, key="repeat_penalty", step=0.01)
            st.slider("Max Tokens", 64, 4096, key="max_tokens", step=32)

    save_col, load_col = st.columns([2, 1])
    with save_col:
        if st.button("Save Configuration", type="primary"):
            overrides = {
                "model_path": st.session_state.model_path,
                "n_ctx": st.session_state.n_ctx,
                "n_threads": st.session_state.n_threads,
                "n_gpu_layers": st.session_state.n_gpu_layers,
                "n_batch": st.session_state.n_batch,
                "temperature": st.session_state.temperature,
                "top_p": st.session_state.top_p,
                "repeat_penalty": st.session_state.repeat_penalty,
                "max_tokens": st.session_state.max_tokens,
            }
            try:
                service.update_llm_params(overrides)
                st.toast("Settings saved successfully!", icon="✅")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to update settings: {exc}")

    with load_col:
        if st.button("Reload from service", type="primary"):
            st.session_state["_refresh_llm_params"] = True
            st.rerun()
