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
                "content": (
                    "🎩 Hello, I’m Jesse Livermore. The market is never wrong — "
                    "only opinions are. How can I help you today?"
                ),
            }
        ],
    )
    st.session_state.setdefault("system_prompt", "You are Jesse Livermore...")
    st.session_state.setdefault("model_name", defaults.get("model_path", "livermore-fin-7b"))

    # Sync LLM params into session state
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

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


def format_sources(sources: List[Dict]) -> str:
    """Render a markdown-friendly references block."""
    if not sources:
        return ""
    lines = ["\n\n---\n📚 **References:**"]
    for idx, src in enumerate(sources, start=1):
        entry = f"{idx}. `{src.get('source', 'Unknown')}`"
        if src.get("page") is not None:
            entry += f" (page {src['page']})"
        lines.append(entry)
    return "\n".join(lines)


def reset_chat():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "🎩 Hello, I’m Jesse Livermore. "
                "Let’s restart our market conversation."
            ),
        }
    ]


def handle_chat(prompt: str):
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        with st.spinner("Thinking using your knowledge base…"):
            result = service.ask(prompt, history=st.session_state.messages)
    except Exception as exc:
        st.session_state.messages.append(
            {"role": "assistant", "content": f"❌ Error: {exc}"}
        )
        return

    answer = result["answer"] + format_sources(result.get("sources"))
    st.session_state.messages.append({"role": "assistant", "content": answer})


def refresh_llm_params():
    params = service.current_llm_params()
    for key, value in params.items():
        st.session_state[key] = value
    st.toast("LLM parameters reloaded.", icon="🔄")


def flatten_config_to_df(data: Dict) -> pd.DataFrame:
    rows = [
        {"Category": "System Status", "Parameter": "Status", "Value": data.get("status")},
        {"Category": "System Status", "Parameter": "Model Loaded", "Value": str(data.get("model_loaded"))},
    ]

    model_info = data.get("model_info") or {}
    for k, v in model_info.items():
        rows.append({"Category": "Model Info", "Parameter": k, "Value": str(v)})

    cfg = data.get("config") or {}
    for k, v in cfg.items():
        if isinstance(v, list):
            v = ", ".join(v)
        rows.append({"Category": "Configuration", "Parameter": k, "Value": str(v)})

    return pd.DataFrame(rows)


# ==========================================
# 3. Initialization
# ==========================================
service = load_service()
ensure_state(service.current_llm_params())

if st.session_state.get("_refresh_llm_params"):
    refresh_llm_params()
    st.session_state.pop("_refresh_llm_params", None)


# ==========================================
# 4. CSS — merged, corrected, full UI theme
# ==========================================
st.markdown(
    """
<style>

:root {
    --bg-dark: #0f172a;
    --sidebar-bg: #1e293b;
    --text-primary: #ffffff;
    --text-secondary: #94a3b8;
    --accent-blue: #3b82f6;
    --border-color: #334155;
}

/* Global dark theme */
.stApp {
    background-color: var(--bg-dark) !important;
    color: var(--text-primary) !important;
}

header, #MainMenu, footer {
    visibility: hidden;
}

/* ===============================
   Sidebar Layout (True Top + Bottom)
=============================== */
section[data-testid="stSidebar"] {
    background-color: var(--sidebar-bg) !important;
    border-right: 1px solid var(--border-color);
}

section[data-testid="stSidebar"] > div:first-child {
    height: 100vh !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: space-between !important;
    padding-top: 0 !important;
}

/* Sidebar top area */
.sidebar-top {
    padding: 0 12px !important;
}

/* Sidebar bottom area */
.sidebar-bottom {
    padding: 20px 12px !important;
    border-top: 1px solid var(--border-color);
}

/* Avatar bubble */
.avatar-circle {
    width: 32px;
    height: 32px;
    background-color: var(--accent-blue);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: bold;
}

/* Chat history */
.chat-history-item {
    padding: 10px;
    margin-bottom: 4px;
    border-radius: 6px;
    cursor: pointer;
    transition: background 0.2s;
}
.chat-history-item:hover {
    background-color: #334155;
}

/* Chat messages */
.stChatMessageContent {
    color: white !important;
}

/* Chat input */
.stChatInputContainer textarea {
    background-color: var(--sidebar-bg) !important;
    color: white !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 6px !important;
}

/* Labels white */
label, .st-emotion-cache-1wivap2, .st-emotion-cache-10trblm {
    color: white !important;
    font-weight: 600 !important;
}

/* Inputs text color */
input, textarea, select {
    color: white !important;
}

/* ===============================
   Tabs — Version A
=============================== */

/* Tabs container */
.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
    padding: 10px 0;
    border-bottom: 1px solid var(--border-color);
    background-color: var(--bg-dark);
}

/* Unselected tab */
.stTabs [data-baseweb="tab"] {
    color: white !important;
    opacity: 0.6 !important;
    font-weight: 500 !important;
}

/* Selected tab */
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent-blue) !important;
    opacity: 1 !important;
    font-weight: 700 !important;
}

/* Fix text inside tab */
.stTabs span {
    color: white !important;
}

/* ===============================
   Metrics — force white
=============================== */
div[data-testid="stMetricValue"],
div[data-testid="stMetricLabel"] {
    color: white !important;
}

</style>
""",
    unsafe_allow_html=True,
)


# ==========================================
# 5. Sidebar — TRUE top + bottom layout
# ==========================================
with st.sidebar:

    # -------- TOP BLOCK --------
    st.markdown("<div class='sidebar-top'>", unsafe_allow_html=True)

    st.markdown(
        "<h2 style='color:#818cf8; margin:0;'>aiaio</h2>",
        unsafe_allow_html=True,
    )
    st.caption("version: 0.0.5-livermore")

    # New Chat button
    if st.button("＋ New Chat", use_container_width=True, type="primary"):
        reset_chat()
        st.rerun()

    # Conversations label
    st.markdown(
        "<p style='color:#94a3b8; font-size:12px; margin-top:20px; text-transform:uppercase;'>Conversations</p>",
        unsafe_allow_html=True,
    )

    # Example conversation list (placeholder)
    history_items = [
        {"title": "Local QA Session", "date": "Active"},
    ]

    for item in history_items:
        st.markdown(
            f"""
            <div class="chat-history-item">
                <div style="color:white; font-size:14px; font-weight:500;">{item["title"]}</div>
                <div style="color:#94a3b8; font-size:12px;">{item["date"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)   # END TOP BLOCK


    # -------- BOTTOM BLOCK --------
    st.markdown("<div class='sidebar-bottom'>", unsafe_allow_html=True)

    # User Profile area
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:10px;">
            <div class="avatar-circle">TL</div>
            <div style="font-size:14px; font-weight:600; color:white;">Team Livermore</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)  # END BOTTOM BLOCK


# ==========================================
# 6. Main Interface — Tabs
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["💬 Chat Interface", "📈 Backtest Dashboard", "📊 System Dashboard", "⚙️ Settings"]
)


# ----------------------------------------------------
# TAB 1 — CHAT
# ----------------------------------------------------
with tab1:
    col_chat, col_info = st.columns([3, 1])

    with col_chat:
        # System prompt editor
        with st.expander("System Prompt Configuration", expanded=False):
            st.text_area("System Prompt", key="system_prompt", height=68)

        # Chat area
        chat_container = st.container(height=500)
        with chat_container:
            for msg in st.session_state.messages:
                avatar = "🧑‍💻" if msg["role"] == "user" else "🎩"
                st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

        # Chat input
        if prompt := st.chat_input("Ask Livermore about markets or trading..."):
            handle_chat(prompt)
            st.rerun()

    with col_info:
        current_temp = st.session_state.get("temperature", 0.7)

        st.markdown(
            f"""
            <div style="background-color:#1e293b; padding:15px; border-radius:8px; border:1px solid #334155; margin-top:20px;">
                
                <h4 style="margin-top:0; font-size:14px; color:#94a3b8;">ACTIVE MODEL</h4>
                
                <div style="font-size:16px; font-weight:bold; color:white; margin-bottom:10px;">
                    {st.session_state.model_name}
                </div>

                <div style="font-size:12px; color:#94a3b8;">
                    Temperature: <b>{current_temp}</b><br>
                    Status: <span style="color:#10b981;">Online</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ----------------------------------------------------
# TAB 2 — BACKTEST DASHBOARD
# ----------------------------------------------------
with tab2:
    with st.container():
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])

        ticker = c1.text_input("Ticker", value="NVDA").upper()
        start_date = c2.date_input("Start", value=datetime(2020, 1, 1))
        end_date = c3.date_input("End", value=datetime.today())

        c4.write("")
        c4.write("")
        run_btn = c4.button("Run Simulation", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("Calculating Strategy…"):
            df = get_stock_data(ticker, start_date, end_date)

            if not df.empty:
                # Moving averages and breakout conditions
                df["50MA"] = df["Close"].rolling(50).mean()
                df["20High"] = df["Close"].rolling(20).max()

                buy_signal = (df["Close"] > df["20High"].shift(1)) & (df["Close"] > df["50MA"])

                # Build position series
                pos = 0
                positions = []
                for i in range(len(df)):
                    if buy_signal.iloc[i]:
                        pos = 1
                    elif df["Close"].iloc[i] < df["Close"].rolling(20).mean().iloc[i]:
                        pos = 0
                    positions.append(pos)
                df["Position"] = positions

                # Strategy returns
                df["Strat_Ret"] = (df["Close"].pct_change() * df["Position"].shift(1)).fillna(0)
                df["Cum_Strat"] = (1 + df["Strat_Ret"]).cumprod()
                df["Cum_BnH"] = (1 + df["Close"].pct_change()).cumprod()

                # Plot
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df["Cum_BnH"],
                        name="Buy & Hold",
                        line=dict(color="#64748b"),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df["Cum_Strat"],
                        name="Livermore Strategy",
                        line=dict(color="#3b82f6", width=2),
                    )
                )
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=450,
                    title="Equity Curve",
                )
                st.plotly_chart(fig, use_container_width=True)

                # Metrics row
                m1, m2 = st.columns(2)
                m1.metric("Buy & Hold Return", f"{(df['Cum_BnH'].iloc[-1] - 1):.2%}")
                m2.metric(
                    "Strategy Return",
                    f"{(df['Cum_Strat'].iloc[-1] - 1):.2%}",
                    delta=f"{(df['Cum_Strat'].iloc[-1] - df['Cum_BnH'].iloc[-1]):.2%}",
                )
            else:
                st.error("No data found for this ticker.")


# ----------------------------------------------------
# TAB 3 — SYSTEM DASHBOARD
# ----------------------------------------------------
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
                "Value": st.column_config.TextColumn("Value", width="large"),
            },
        )

    except Exception as exc:
        st.error(f"Health check failed: {exc}")


# ----------------------------------------------------
# TAB 4 — SETTINGS
# ----------------------------------------------------
with tab4:
    st.subheader("System Configuration")

    col_s1, col_s2 = st.columns(2)

    # LEFT — Model Parameters
    with col_s1:
        with st.container(border=True):
            st.markdown("**Model Parameters**")

            st.text_input("Model Path", key="model_path")
            st.slider("Context Window (n_ctx)", 512, 8192, key="n_ctx", step=256)
            st.slider("Batch Size (n_batch)", 32, 1024, key="n_batch", step=16)
            st.slider("GPU Layers", 0, 64, key="n_gpu_layers", step=1)
            st.slider("Threads (n_threads)", 1, 32, key="n_threads", step=1)

    # RIGHT — Generation Parameters
    with col_s2:
        with st.container(border=True):
            st.markdown("**Generation Control**")

            st.slider("Temperature", 0.0, 1.5, key="temperature", step=0.05)
            st.slider("Top P", 0.1, 1.0, key="top_p", step=0.01)
            st.slider("Repeat Penalty", 0.8, 1.5, key="repeat_penalty", step=0.01)
            st.slider("Max Tokens", 64, 4096, key="max_tokens", step=32)

    # Save / Load row
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
            except Exception as exc:
                st.error(f"Failed to update settings: {exc}")

    with load_col:
        if st.button("Reload from Service"):
            st.session_state["_refresh_llm_params"] = True
            st.rerun()
