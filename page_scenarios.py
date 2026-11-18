# page_scenarios.py
"""Scenarios page: create & run scenarios across all 10 database drivers, view saved scenarios, and compare results."""
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from config import *
from calculations import latest_market_date
from database import persist_simulation
from scenarios import (
    build_saved_template_values,
    build_scenario_parameters,
    individual_driver_to_params,
    parse_free_type_scenario_with_ollama,
    resolve_saved_template,
    scenario_params_from_row,
    template_values,
)
from ui_common import (
    evaluate_saved_scenario,
    format_large_number,
    kpi_card,
    run_scenario,
    show_df,
    style_nbe_figure,
    tone_for,
)

_BUILDER_PREFIXES = (
    "builder_param_",
    "builder_individual",
    "builder_free_type_text",
    "builder_scenario_",
)


# ----------------------------------------------------------------------------
# Session state & template synchronization helpers
# ----------------------------------------------------------------------------
def apply_template(template_name: str):
    """Reset the scenario builder state to match the chosen template defaults."""
    st.session_state["active_template"] = template_name
    st.session_state["builder_values"] = template_values(template_name)
    st.session_state["scenario_action"] = "new"
    st.session_state.pop("ai_insight_dashboard_current", None)
    st.session_state.pop("ai_insight_dashboard_dashboard_current", None)
    for key in list(st.session_state.keys()):
        if str(key).startswith(_BUILDER_PREFIXES):
            del st.session_state[key]
    for key in ["free_type_text", "free_type_parsed", "free_type_parsed_source"]:
        st.session_state.pop(key, None)


def _on_template_change():
    """Callback triggered when the user selects a different scenario template."""
    apply_template(st.session_state["workspace_template_selector"])


def _load_saved_into_builder():
    """Populate the scenario builder widgets from a selected saved scenario row."""
    scen = st.session_state.get("_scenarios_raw")
    sname = st.session_state.get("builder_old_scenario")
    if scen is None or sname is None or scen.empty:
        return
    match = scen[scen["scenario_name"].astype(str) == str(sname)]
    if match.empty:
        return
    row = match.iloc[0]
    params = scenario_params_from_row(row)
    template = resolve_saved_template(str(row.get("scenario_type", "")))
    apply_template(template)
    st.session_state["scenario_action"] = "old"
    st.session_state["workspace_template_selector"] = template
    values = build_saved_template_values(template, params)
    st.session_state["builder_values"] = values
    for k in PARAM_SPECS:
        st.session_state[f"builder_param_{k}"] = float(values.get(k, SCENARIO_DEFAULTS[k]))
    if template == "Individual Parameter Scenario":
        label = values["individual_parameter"]
        st.session_state["builder_individual_parameter"] = label
        st.session_state[f"builder_individual_value_{INDIVIDUAL_DRIVERS[label]}"] = float(values["individual_value"])
    st.session_state["builder_scenario_name"] = str(row.get("scenario_name", ""))
    st.session_state["builder_scenario_description"] = str(row.get("description") or "")


def _on_run_mode_change():
    """Switch between creating a new scenario and rerunning an existing saved scenario."""
    if st.session_state.get("builder_run_mode") == "Rerun Saved Scenario":
        scen = st.session_state.get("_scenarios_raw")
        if scen is not None and not scen.empty and st.session_state.get("builder_old_scenario") is None:
            st.session_state["builder_old_scenario"] = str(scen["scenario_name"].dropna().iloc[0])
        _load_saved_into_builder()


def _set_horizon(days: int):
    """Update the simulation horizon in days when a preset button is clicked."""
    st.session_state["builder_horizon"] = int(days)


def _ensure_state(key: str, initial):
    """Initialize a session_state key only if it does not already exist."""
    if key not in st.session_state:
        st.session_state[key] = initial


def _number_param(container, key: str, values: dict) -> float:
    """Render a numeric input widget for a single scenario parameter from PARAM_SPECS."""
    label, unit, lo, hi, default, step, _group = PARAM_SPECS[key]
    wkey = f"builder_param_{key}"
    _ensure_state(wkey, float(min(max(float(values.get(key, default)), lo), hi)))
    return container.number_input(
        f"{label} ({unit})",
        min_value=float(lo),
        max_value=float(hi),
        step=float(step),
        key=wkey,
    )


def _param_grid(keys: list, values: dict, per_row: int = 3) -> dict:
    """Render a multi-column grid of numeric parameter inputs."""
    out = {}
    for i in range(0, len(keys), per_row):
        chunk = keys[i:i + per_row]
        for col, k in zip(st.columns(per_row), chunk):
            out[k] = _number_param(col, k, values)
    return out


# ----------------------------------------------------------------------------
# Tab 1: Create & Run Scenario
# ----------------------------------------------------------------------------
def _tab_create(ctx, history):
    """Render the interactive scenario builder, live KPI preview, and database save controls."""
    st.session_state["_scenarios_raw"] = ctx.scenarios
    names = list(PREDEFINED_TEMPLATE_NAMES)
    _ensure_state("workspace_template_selector", st.session_state.get("active_template", "Combined Stress Scenario"))
    st.selectbox("1 - Scenario template", names, key="workspace_template_selector", on_change=_on_template_change)
    active = st.session_state["workspace_template_selector"]
    if st.session_state.get("active_template") != active:
        apply_template(active)
    template = SCENARIO_TEMPLATES[active]
    values = st.session_state["builder_values"]
    st.info(f"**{active}** — {template['description']}")

    st.markdown("##### 2 - Simulation window")
    m1, m2, m3 = st.columns(3)
    simulation_date = m1.date_input("Simulation date", value=date.today(), key="builder_simulation_date")
    _ensure_state("builder_horizon", DEFAULT_HORIZON_DAYS)
    horizon_days = m2.number_input("Horizon (days)", 1, 3650, step=1, key="builder_horizon")
    _ensure_state("builder_run_mode", "New Scenario")
    run_mode = m3.selectbox(
        "Run mode",
        ["New Scenario", "Rerun Saved Scenario"],
        key="builder_run_mode",
        on_change=_on_run_mode_change,
    )

    preset_cols = st.columns(len(HORIZON_PRESETS))
    for col, (label, days) in zip(preset_cols, HORIZON_PRESETS.items()):
        col.button(label, key=f"horizon_preset_{days}", use_container_width=True, on_click=_set_horizon, args=(days,))

    existing_id = None
    if run_mode == "Rerun Saved Scenario":
        if not history.empty:
            saved_names = history["scenario_name"].astype(str).tolist()
            _ensure_state("builder_old_scenario", saved_names[0] if saved_names else "")
            selected_name = st.selectbox(
                "Saved scenario",
                saved_names,
                key="builder_old_scenario",
                on_change=_load_saved_into_builder,
            )
            match_row = history[history["scenario_name"].astype(str) == str(selected_name)]
            if not match_row.empty:
                existing_id = int(match_row["scenario_id"].values[0])
        else:
            st.warning("No saved scenarios are available.")

    st.markdown(f"##### 3 - Scenario Assumptions ({len(PARAM_SPECS)} Database Drivers)")
    params = dict(SCENARIO_DEFAULTS)
    if active == "Base Case":
        st.caption("The base case has no stress assumptions.")
    elif active == "Individual Parameter Scenario":
        drivers = list(INDIVIDUAL_DRIVERS.keys())
        _ensure_state("builder_individual_parameter", values.get("individual_parameter", drivers[0]))
        driver = st.selectbox("Driver to stress", drivers, key="builder_individual_parameter")
        key = INDIVIDUAL_DRIVERS[driver]
        label, unit, lo, hi, default, step, _g = PARAM_SPECS[key]
        vkey = f"builder_individual_value_{key}"
        _ensure_state(
            vkey,
            float(min(max(values.get("individual_value", 0.0) if values.get("individual_parameter") == driver else 0.0, lo), hi)),
        )
        value = st.number_input(f"{label} ({unit})", min_value=float(lo), max_value=float(hi), step=float(step), key=vkey)
        params = individual_driver_to_params(driver, value)
    elif active == "Free Type Scenario":
        text_in = st.text_area(
            "Describe the scenario",
            value=st.session_state.get("free_type_text", ""),
            height=100,
            key="builder_free_type_text",
            placeholder="Example: Interest rates +300 bps; USD/EGP +15%; gold price -10%; wholesale spread +100 bps; deposit outflows 10%.",
        )
        st.session_state["free_type_text"] = text_in
        if st.button("🧠 Extract assumptions", type="primary"):
            if text_in.strip():
                host = st.session_state.get("ollama_host", OLLAMA_HOST)
                model = st.session_state.get("ollama_model", OLLAMA_MODEL)
                parsed = parse_free_type_scenario_with_ollama(host, model, text_in)
                st.session_state["free_type_parsed"] = parsed
                st.session_state["free_type_parsed_source"] = text_in
        parsed = st.session_state.get("free_type_parsed")
        if parsed and st.session_state.get("free_type_parsed_source") == text_in:
            rows = []
            for k, v in parsed.get("parameters", {}).items():
                if v is not None and k in PARAM_SPECS:
                    params[k] = float(v)
                    rows.append([PARAM_SPECS[k][0], v, PARAM_SPECS[k][1]])
            if rows:
                show_df(pd.DataFrame(rows, columns=["Parameter", "Value", "Unit"]))
    else:
        group_titles = {
            "Market": "Market shocks (Policy rate, FX, Gold, Wholesale funding spread)",
            "Balance sheet & liquidity": "Balance sheet & deposit run-off",
            "Macro": "Macroeconomic shocks (Inflation, Unemployment, GDP)",
        }
        for group, title in group_titles.items():
            keys = [k for k in template["visible"] if k in PARAM_SPECS and PARAM_SPECS[k][6] == group]
            if keys:
                st.markdown(f"**{title}**")
                params.update(_param_grid(keys, values, min(len(keys), 4)))

    params = build_scenario_parameters(params)

    st.markdown("##### 4 - Name, Run & Save")
    _ensure_state("builder_scenario_name", f"{active} - {simulation_date}")
    scenario_name = st.text_input("Scenario name", key="builder_scenario_name")
    scenario_description = st.text_area(
        "Description",
        placeholder="Optional business description...",
        height=70,
        key="builder_scenario_description",
    )

    st.session_state.update({
        "current_params": params,
        "current_simulation_date": simulation_date,
        "current_horizon_days": int(horizon_days),
        "current_scenario_name": scenario_name,
        "current_scenario_description": scenario_description,
    })
    res = run_scenario(ctx, params, simulation_date, int(horizon_days))

    c1, c2 = st.columns(2)
    run_clicked = c1.button(
        "▶ Run & save simulation",
        type="primary",
        use_container_width=True,
        disabled=ctx.engine is None,
    )
    if c2.button("↺ Reset parameters", use_container_width=True):
        apply_template(active)
        st.rerun()
    if run_clicked:
        run_id, scenario_id, msgs = persist_simulation(
            ctx.engine,
            scenario_name,
            template["scenario_type"],
            simulation_date,
            int(horizon_days),
            params,
            res.comparison,
            res.risk,
            scenario_description=scenario_description,
            base_as_of_datetime=latest_market_date(ctx.market_data),
            existing_scenario_id=existing_id,
            ai_explanation=None,
        )
        st.session_state.update({
            "last_run_id": run_id,
            "last_scenario_id": scenario_id,
            "last_save_messages": msgs,
        })
        if run_id:
            st.cache_data.clear()
    for ok, msg in st.session_state.get("last_save_messages", []):
        (st.success if ok else st.warning)(msg)

    st.markdown("##### Live Result Preview")
    b, a = res.before, res.after
    cols = st.columns(4)
    with cols[0]:
        kpi_card(
            "NII",
            format_large_number(a["NII"]),
            f"{a['NII'] - b['NII']:+,.0f} EGP" if abs(a["NII"] - b["NII"]) > 1e-6 else None,
            tone_for(a["NII"] - b["NII"], True),
        )
    with cols[1]:
        kpi_card("NIM", f"{a['NIM %']:.2f}%", None, "neutral", f"base {b['NIM %']:.2f}%")
    with cols[2]:
        kpi_card(
            "LCR Proxy (30d)",
            "N/A" if pd.isna(a["CF LCR %"]) else f"{a['CF LCR %']:,.1f}%",
            None,
            "neutral",
            f"base {b['CF LCR %']:,.1f}%" if pd.notna(b["CF LCR %"]) else "",
        )
    with cols[3]:
        kpi_card(
            "Total P&L & value impact",
            format_large_number(a["PnL Total Impact"]),
            None,
            tone_for(a["PnL Total Impact"], True),
            f"overall risk: {res.risk_level}",
        )


# ----------------------------------------------------------------------------
# Tab 2: Saved Scenarios Inspector
# ----------------------------------------------------------------------------
def _tab_saved(ctx, history):
    """Display the table of saved scenarios and inspect any selected scenario's assumptions."""
    if history.empty:
        st.info("No saved scenarios found.")
        return
    show_df(history, height=280)
    names = history["scenario_name"].astype(str).tolist()
    selected_name = st.selectbox("Scenario name", names, key="history_scenario_name")
    row = ctx.scenarios[ctx.scenarios["scenario_name"].astype(str) == str(selected_name)]
    if row.empty:
        return
    params = scenario_params_from_row(row.iloc[0])
    st.markdown("##### Saved Assumptions")
    show_df(
        pd.DataFrame(
            [[PARAM_SPECS[k][0], v, PARAM_SPECS[k][1]] for k, v in params.items() if k in PARAM_SPECS],
            columns=["Parameter", "Value", "Unit"],
        )
    )


# ----------------------------------------------------------------------------
# Tab 3: Multi-Scenario Comparison (Dynamic Evaluation & Charts)
# ----------------------------------------------------------------------------
def _tab_compare(ctx, history):
    """Compare multiple saved scenarios side-by-side using unified engine evaluations."""
    if history.empty or ctx.scenarios.empty:
        st.info("No saved scenarios found.")
        return

    names = history["scenario_name"].astype(str).tolist()
    selected_names = st.multiselect("Scenarios to compare", names, default=[], key="comparison_scenarios")
    if not selected_names:
        return

    # Evaluate each selected saved scenario dynamically through the unified engine for 100% consistency
    comp_records = []
    for sname in selected_names:
        match = ctx.scenarios[ctx.scenarios["scenario_name"].astype(str) == str(sname)]
        if match.empty:
            continue
        s_row = match.iloc[0]
        sid = int(s_row["scenario_id"])
        scn_lbl = f"{sname} (ID {sid})"
        ev = evaluate_saved_scenario(ctx, s_row)
        for _, r in ev.comparison.iterrows():
            comp_records.append({
                "scenario_label": scn_lbl,
                "clean_metric": str(r["KPI"]),
                "scenario_value_num": float(r["After"]) if pd.notna(r["After"]) else np.nan,
                "unit": str(r["Unit"]),
            })
        # Also include key financial statement KPIs (ROE, ROA, NSFR, Cost-to-Income)
        k_ev = getattr(ev, "kpis", {}) or {}
        for extra_lbl, extra_val in [
            ("Return on Equity (ROE %)", k_ev.get("roe_scn")),
            ("Return on Assets (ROA %)", k_ev.get("roa_scn")),
            ("NSFR Proxy %", k_ev.get("nsfr_scn")),
            ("Cost-to-Income %", k_ev.get("cti_scn")),
        ]:
            if extra_val is not None and pd.notna(extra_val):
                comp_records.append({
                    "scenario_label": scn_lbl,
                    "clean_metric": extra_lbl,
                    "scenario_value_num": float(extra_val),
                    "unit": "%",
                })

    res = pd.DataFrame(comp_records).dropna(subset=["scenario_value_num"])
    if res.empty:
        st.info("No KPI results found for the selected scenarios.")
        return

    st.markdown("##### Executive KPI Comparison Table")
    pivot = (
        res.pivot_table(index="clean_metric", columns="scenario_label", values="scenario_value_num", aggfunc="first")
        .reset_index()
        .rename(columns={"clean_metric": "Metric Name"})
    )
    show_df(pivot)

    is_pct = res["unit"].astype(str).str.contains("%") | res["clean_metric"].str.contains("%|NIM|LCR|NSFR|ROE|ROA", case=False, regex=True)

    # A. Percentage & Ratio Comparisons
    pct_df = res[is_pct].copy()
    if not pct_df.empty:
        st.markdown("---")
        st.markdown("##### Percentage & Ratio Scenario Comparisons")
        pct_kpis = pct_df["clean_metric"].unique()

        for i in range(0, len(pct_kpis), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(pct_kpis):
                    kpi_name = pct_kpis[i + j]
                    sub_pct = pct_df[pct_df["clean_metric"] == kpi_name].drop_duplicates(subset=["scenario_label"]).copy()
                    sub_pct["Formatted_Pct"] = sub_pct["scenario_value_num"].apply(lambda x: f"{x:.2f}%")

                    with cols[j]:
                        st.markdown(f"**{kpi_name}**")
                        fig_pct = px.bar(
                            sub_pct,
                            x="scenario_label",
                            y="scenario_value_num",
                            color="scenario_label",
                            text="Formatted_Pct",
                            barmode="group",
                            color_discrete_sequence=NBE_PALETTE,
                        )
                        fig_pct = style_nbe_figure(fig_pct, 320)
                        fig_pct.update_traces(
                            textposition="outside",
                            cliponaxis=False,
                            textfont=dict(size=12, color="#111827"),
                        )
                        fig_pct.update_layout(
                            title="",
                            xaxis=dict(title="", showticklabels=True),
                            yaxis=dict(
                                title="Percentage (%)",
                                zeroline=True,
                                zerolinewidth=1.5,
                                zerolinecolor="#374151",
                            ),
                            showlegend=False,
                            margin=dict(l=50, r=20, t=35, b=40),
                            height=320,
                        )
                        st.plotly_chart(fig_pct, use_container_width=True, key=f"compare_pct_chart_{i}_{j}")

    # B. Financial Amount KPI Comparisons (dynamic Billion / Million scaling)
    amt = res[~is_pct].copy()
    amt["Formatted"] = amt["scenario_value_num"].apply(format_large_number)

    if not amt.empty:
        st.markdown("---")
        st.markdown("##### Financial Amount KPI Scenario Comparisons")
        kpis = amt["clean_metric"].unique()

        for i in range(0, len(kpis), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(kpis):
                    kpi_name = kpis[i + j]
                    sub_df = amt[amt["clean_metric"] == kpi_name].drop_duplicates(subset=["scenario_label"]).copy()
                    if sub_df.empty:
                        continue

                    max_raw = float(sub_df["scenario_value_num"].abs().max())
                    if max_raw >= 1_000_000_000:
                        scale_div, y_title, y_suffix, tick_fmt = 1_000_000_000.0, "Billion EGP", " B", ",.2f"
                    else:
                        scale_div, y_title, y_suffix, tick_fmt = 1_000_000.0, "Million EGP", " M", ",.1f"

                    sub_df["Scaled_Val"] = sub_df["scenario_value_num"] / scale_div
                    max_abs = max(float(sub_df["Scaled_Val"].abs().max()), 1e-6)

                    def boost_val(v):
                        if v == 0 or pd.isna(v):
                            return 0.0
                        sign = 1.0 if v > 0 else -1.0
                        ratio = abs(v) / max_abs
                        boosted_ratio = max(ratio, 0.18) if ratio < 0.18 else ratio
                        return sign * boosted_ratio * max_abs

                    sub_df["Plot_Y"] = sub_df["Scaled_Val"].apply(boost_val) if len(sub_df) > 1 else sub_df["Scaled_Val"]

                    min_y = float(sub_df["Plot_Y"].min())
                    max_y = float(sub_df["Plot_Y"].max())
                    span = max(abs(min_y), abs(max_y), 1e-6)
                    upper_bound = max(max_y, 0.0) + (span * 0.35)
                    lower_bound = (min_y - (span * 0.35)) if min_y < 0 else 0.0

                    with cols[j]:
                        st.markdown(f"**{kpi_name}**")
                        fig = px.bar(
                            sub_df,
                            x="scenario_label",
                            y="Plot_Y",
                            color="scenario_label",
                            text="Formatted",
                            barmode="group",
                            color_discrete_sequence=NBE_PALETTE,
                        )
                        fig = style_nbe_figure(fig, 380)
                        fig.update_traces(
                            textposition="outside",
                            cliponaxis=False,
                            textfont=dict(size=12, color="#111827"),
                            hovertemplate="<b>%{x}</b><br>Value: %{text}<extra></extra>",
                        )
                        fig.update_layout(
                            title="",
                            xaxis=dict(title="", showticklabels=True),
                            yaxis=dict(
                                title=y_title,
                                range=[lower_bound, upper_bound],
                                tickformat=tick_fmt,
                                ticksuffix=y_suffix,
                                zeroline=True,
                                zerolinewidth=2,
                                zerolinecolor="#374151",
                            ),
                            showlegend=False,
                            margin=dict(l=60, r=20, t=40, b=40),
                            height=380,
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"compare_chart_amt_clean_{i}_{j}")


def page_scenarios(ctx, history):
    """Main entry point for the Scenarios page."""
    st.markdown('<div class="section-header">Scenarios</div>', unsafe_allow_html=True)
    t1, t2, t3 = st.tabs(["Create & run", "Saved scenarios", "Compare scenarios"])
    with t1:
        _tab_create(ctx, history)
    with t2:
        _tab_saved(ctx, history)
    with t3:
        _tab_compare(ctx, history)