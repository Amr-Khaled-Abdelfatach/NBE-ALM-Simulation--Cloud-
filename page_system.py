# page_system.py
"""Data & Settings page: data explorer, independent calculation & balance sheet audit, data quality, and DB/AI settings."""
import pandas as pd
import streamlit as st

from config import *
from database import table_row_counts
from ui_common import current_session_scenario, show_df


def _tab_explorer(ctx):
    """Browse all loaded SQL views and raw database tables dynamically."""
    data = ctx.data
    sources = {
        f"{FLAT_POSITIONS_VIEW} (SQL View)": ctx.positions,
        f"{FLAT_CASHFLOWS_VIEW} (SQL View)": ctx.cashflows,
        "financial_statements_flat": getattr(ctx, "fs_flat", data.get("fs_flat_raw")),
        "products": data.get("products_raw"),
        "positions": data.get("positions_raw"),
        "cash_flows": data.get("cashflows_raw"),
        "market_data": data.get("market_data_raw"),
        "scenarios": data.get("scenarios_raw"),
        "simulation_runs": data.get("simulation_runs_raw"),
        "simulation_results": data.get("simulation_results_raw"),
    }
    name = st.selectbox("Table / view", list(sources.keys()), key="sys_explorer_table_select")
    df = sources[name] if sources[name] is not None else pd.DataFrame()
    st.caption(f"Rows loaded: {len(df)}")
    show_df(df, height=520)


def _tab_audit(ctx):
    """Run independent mathematical validation on KPIs and verify Balance Sheet double-entry reconciliation."""
    res = current_session_scenario(ctx)
    q = res.detail

    st.markdown("##### 1️⃣ Independent KPI Calculation Validation")
    show_df(res.validation)
    failed = int((res.validation["Status"] == "FAIL").sum()) if not res.validation.empty else 0
    (st.error if failed else st.success)(
        f"{failed} calculation validation check(s) failed."
        if failed
        else "All KPI calculation checks passed within the configured tolerance."
    )

    # Verify Balance Sheet accounting identity (Assets == Liabilities + Equity) across all dynamic periods
    df_bs = getattr(res, "balance_sheet_df", None)
    if df_bs is not None and not df_bs.empty and "code" in df_bs.columns:
        st.markdown("##### 2️⃣ Double-Entry Balance Sheet Reconciliation Audit")
        chk_row = df_bs[df_bs["code"] == "BS_CHECK"]
        if not chk_row.empty:
            meta_cols = {"order", "section", "code", "name_ar", "name_en", "Delta (Impact)"}
            period_cols = [c for c in df_bs.columns if c not in meta_cols]
            bs_audit_rows = []
            for p_col in period_cols:
                discrepancy = float(chk_row[p_col].values[0] or 0.0)
                status = "PASS" if abs(discrepancy) <= 0.05 else "FAIL"
                bs_audit_rows.append({
                    "Reporting / Forecast Period": str(p_col),
                    "Identity Formula": "Total Assets − (Total Liabilities + Total Equity)",
                    "Discrepancy (EGP Mn)": round(discrepancy, 4),
                    "Status": status,
                })
            show_df(pd.DataFrame(bs_audit_rows))

    if q is not None and not q.empty:
        with st.expander("🔍 Inspect Position-Level Repricing & Regulatory Metadata", expanded=False):
            cols = [
                c for c in [
                    "position_id", "product_code", "product_name", "Category",
                    "BS Line Code", "Accounting Class", "Model Currency", "Valuation Basis",
                    "Gold Purity", "HQLA Flag", "Is Liquid Asset", "NSFR ASF Weight", "NSFR RSF Weight",
                    "Before Balance", "After Balance", "Before Rate", "After Rate",
                    "Accrual Days", "First Repricing Date", "Interest Before", "Interest After",
                    "Rate Effect", "Volume Effect", "Valuation Effect", "Investment MTM",
                    "Credit Cost", "Validation Status",
                ]
                if c in q.columns
            ]
            show_df(q[cols], height=420)


def _tab_quality(ctx):
    """Display database table/view row counts and automated data health checks."""
    st.markdown("##### Database Objects & Row Counts")
    if ctx.engine is not None:
        show_df(table_row_counts(ctx.engine))
    st.markdown("##### Engine Data Health Checks")
    show_df(current_session_scenario(ctx).quality)


def _tab_connection(db_connected, connection_error):
    """Configure MySQL connection parameters and trigger immediate reconnection."""
    host, port, database = st.session_state["db_host"], st.session_state["db_port"], st.session_state["db_name"]
    c1, c2 = st.columns(2)
    new_host = c1.text_input("Host", value=st.session_state["db_host"], key="advanced_db_host")
    new_port = c1.number_input("Port", 1, 65535, int(st.session_state["db_port"]), 1, key="advanced_db_port")
    new_db = c2.text_input("Database", value=st.session_state["db_name"], key="advanced_db_name")
    new_user = c2.text_input("User", value=st.session_state["db_user"], key="advanced_db_user")
    new_pw = st.text_input("Password", value=st.session_state["db_password"], type="password", key="advanced_db_password")
    if st.button("Save and reconnect", type="primary", use_container_width=True):
        st.session_state.update({
            "db_host": new_host,
            "db_port": int(new_port),
            "db_name": new_db,
            "db_user": new_user,
            "db_password": new_pw,
        })
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()
    if db_connected:
        st.success(f"Connected to `{database}` on `{host}:{port}`.")
    elif connection_error:
        st.code(connection_error, language="text")


# def _tab_ai():
#     """Configure local Ollama host URL and model name."""
#     ai_host = st.text_input("Ollama URL", value=st.session_state["ollama_host"], key="advanced_ollama_host")
#     ai_model = st.text_input("Ollama model", value=st.session_state["ollama_model"], key="advanced_ollama_model")
#     if st.button("Save AI configuration", use_container_width=True):
#         st.session_state["ollama_host"], st.session_state["ollama_model"] = ai_host, ai_model
#         st.success("AI configuration saved.")



def _tab_ai():
    """Configure AI execution mode (Online Gemini API vs. Offline Local Ollama)."""
    current_mode = str(st.session_state.get("ai_provider", AI_PROVIDER)).lower()
    mode_options = ["Online (Gemini API)", "Offline (Local Ollama)"]
    default_idx = 0 if current_mode == "online" else 1

    selected_mode = st.radio(
        "AI Execution Mode",
        options=mode_options,
        index=default_idx,
        horizontal=True,
        key="sys_ai_mode_radio",
    )
    is_online = "Online" in selected_mode

    if is_online:
        st.caption("🌐 **Online Mode:** Uses Google Gemini Cloud API (zero local GPU/VRAM usage).")
        gem_key = st.text_input(
            "Gemini API Key",
            value=st.session_state.get("gemini_api_key", GEMINI_API_KEY),
            type="password",
            placeholder="AIzaSy...",
            key="advanced_gemini_key",
        )
        gem_model = st.text_input(
            "Gemini Model",
            value=st.session_state.get("gemini_model", GEMINI_MODEL),
            key="advanced_gemini_model",
        )
        if st.button("Save AI configuration", type="primary", use_container_width=True):
            st.session_state["ai_provider"] = "online"
            st.session_state["gemini_api_key"] = gem_key.strip()
            st.session_state["gemini_model"] = gem_model.strip()
            for k in list(st.session_state.keys()):
                if str(k).startswith("ai_insight_"):
                    del st.session_state[k]
            st.success(f"AI configuration saved: Online Mode ({gem_model.strip()}).")
    else:
        st.caption("💻 **Offline Mode:** Uses local Ollama server on your machine.")
        ai_host = st.text_input(
            "Ollama URL",
            value=st.session_state.get("ollama_host", OLLAMA_HOST),
            key="advanced_ollama_host",
        )
        ai_model = st.text_input(
            "Ollama Model",
            value=st.session_state.get("ollama_model", OLLAMA_MODEL),
            key="advanced_ollama_model",
        )
        if st.button("Save AI configuration", type="primary", use_container_width=True):
            st.session_state["ai_provider"] = "offline"
            st.session_state["ollama_host"] = ai_host.strip()
            st.session_state["ollama_model"] = ai_model.strip()
            st.success(f"AI configuration saved: Offline Mode ({ai_model.strip()}).")

def page_system(ctx, db_connected, connection_error):
    """Main entry point for the Data & Settings page."""
    st.markdown('<div class="section-header">⚙️ Data & Settings</div>', unsafe_allow_html=True)
    t = st.tabs([
        "Data explorer",
        "Calculation audit",
        "Data quality",
        "Database connection",
        "AI Configurations",
    ])
    with t[0]:
        _tab_explorer(ctx) if db_connected else st.info("Connect to the database first.")
    with t[1]:
        _tab_audit(ctx)
    with t[2]:
        _tab_quality(ctx)
    with t[3]:
        _tab_connection(db_connected, connection_error)
    with t[4]:
        _tab_ai()