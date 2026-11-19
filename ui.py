# ui.py
"""Main Streamlit application router, state initializer, and unified context builder for NBE ALM."""
from datetime import date
from types import SimpleNamespace

import pandas as pd
import streamlit as st

from config import *
from calculations import prepare_inputs
from database import connect_existing_database, load_alm_data
from scenarios import scenario_summary, template_values
from ui_common import CUSTOM_CSS, render_page_header
from page_dashboard import page_dashboard
from page_scenarios import page_scenarios
from page_documentation import page_documentation
from page_system import page_system
from market_feed import render_live_market_feed_panel


def _init_state():
    """Initialize required Streamlit session_state keys using dynamic defaults from config.py."""
    defaults = {
        "db_host": DB_HOST,
        "db_port": int(DB_PORT),
        "db_name": DB_NAME,
        "db_user": DB_USER,
        "db_password": DB_PASSWORD,
        "active_template": "Combined Stress Scenario",
        "builder_values": template_values("Combined Stress Scenario"),
        "current_params": dict(SCENARIO_DEFAULTS),
        "current_simulation_date": date.today(),
        "current_horizon_days": DEFAULT_HORIZON_DAYS,
        "current_scenario_name": "Combined Stress Scenario",
        "current_scenario_description": "",
        "ollama_host": OLLAMA_HOST,
        "ollama_model": OLLAMA_MODEL,
        "selected_page": "Dashboard",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


@st.cache_data(ttl=DATA_CACHE_TTL_SECONDS, show_spinner=False)
def _cached_load(_engine, host, port, db_name):
    """Cache database table and SQL view reads across Streamlit reruns."""
    return load_alm_data(_engine)


def _build_context():
    """Connect to MySQL, load all tables/views, and assemble the unified execution context."""
    host = st.session_state["db_host"]
    port = int(st.session_state["db_port"])
    db_name = st.session_state["db_name"]
    user = st.session_state["db_user"]
    password = st.session_state["db_password"]

    engine = None
    db_connected = False
    connection_error = ""
    data = {}

    try:
        engine = connect_existing_database(host, port, db_name, user, password)
        data = _cached_load(engine, host, port, db_name)
        db_connected = True
    except Exception as exc:
        connection_error = str(exc)

    positions = data.get("positions_flat", pd.DataFrame())
    cashflows = data.get("cashflows_flat", pd.DataFrame())
    market_data = data.get("market_data_raw", pd.DataFrame())
    fs_flat = data.get("fs_flat_raw", pd.DataFrame())
    scenarios = data.get("scenarios_raw", pd.DataFrame())
    runs = data.get("simulation_runs_raw", pd.DataFrame())
    results = data.get("simulation_results_raw", pd.DataFrame())
    source_notes = data.get("source_notes", {})

    prep = prepare_inputs(positions, cashflows, market_data, fs_flat=fs_flat)
    stamp = f"{len(positions)}_{len(cashflows)}_{len(market_data)}_{len(fs_flat)}_{len(scenarios)}"

    ctx = SimpleNamespace(
        engine=engine,
        data=data,
        raw=data,
        positions=positions,
        cashflows=cashflows,
        market_data=market_data,
        fs_flat=fs_flat,
        scenarios=scenarios,
        runs=runs,
        results=results,
        source_notes=source_notes,
        prep=prep,
        stamp=stamp,
    )
    return ctx, db_connected, connection_error


def _render_sidebar(db_connected: bool):
    """Render the navigation sidebar and external documentation link using configured brand assets."""
    with st.sidebar:
        if LOGO_SIDEBAR_PATH and os.path.exists(LOGO_SIDEBAR_PATH):
            _, col_logo, _ = st.columns([0.4, 2.2, 0.4])
            with col_logo:
                st.image(LOGO_SIDEBAR_PATH, use_container_width=True)
        else:
            st.markdown("### 🏦 NBE ALM Center")
            

        st.markdown("---")

        current_page = st.session_state.get("selected_page", "Dashboard")
        for page_label in PAGES:
            icon = PAGE_ICONS.get(page_label, "")
            btn_label = f"{icon} {page_label}".strip()
            is_active = current_page == page_label
            if st.button(
                btn_label,
                key=f"nav_btn_{page_label}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):
                st.session_state["selected_page"] = page_label
                st.rerun()

        st.markdown("---")
        status_txt = "🟢 Database Connected" if db_connected else "🔴 Database Disconnected"
        st.caption(status_txt)

        if DOCS_ENABLED and DOCS_BASE_URL:
            st.markdown(
                f'<a href="{DOCS_BASE_URL}" target="_blank" style="display:block; text-align:center; '
                f'background:#EAF5EF; color:#005B2D; padding:8px 12px; border-radius:10px; '
                f'text-decoration:none; font-weight:700; font-size:0.82rem; border:1px solid #BFE0CC;">'
                f'📘 Project Documentation ↗</a>',
                unsafe_allow_html=True,
            )


def main():
    """Main entry point for the Streamlit application."""
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout=PAGE_LAYOUT,
        initial_sidebar_state=INITIAL_SIDEBAR_STATE,
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    _init_state()

    ctx, db_connected, connection_error = _build_context()
    _render_sidebar(db_connected)

    status_label = f"Connected ({len(ctx.positions)} positions)" if db_connected else "Disconnected"
    render_page_header(
        APP_NAME,
        "Asset & Liability Management — Deterministic Stress Testing, Multi-Year Statements & AI Analytics",
        status=status_label,
    )

    history = scenario_summary(ctx.scenarios) if not ctx.scenarios.empty else pd.DataFrame()
    selected = st.session_state.get("selected_page", "Dashboard")

    if selected == "Dashboard":
        page_dashboard(ctx, history)
    elif selected == "Scenarios":
        page_scenarios(ctx, history)
    # elif selected == "Documentation":
    #     page_documentation(ctx)
    elif selected == "Live Market Feed":
        render_live_market_feed_panel(ctx)
    elif selected == "Data & Settings":
        page_system(ctx, db_connected, connection_error)
    else:
        page_dashboard(ctx, history)


def run_app():
    """Main application entry point for app.py runners."""
    main()


if __name__ == "__main__":
    run_app()