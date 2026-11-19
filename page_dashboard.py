# page_dashboard.py
"""
Executive Dashboard: 4-per-row KPI grid (20 KPIs), Dynamic Financial Statements Impact Bar,
Outlined Multi-Year & ML Forecast Statements, Analytics Tabs, and PDF/PPTX/Excel Exporters.
100% Data-Driven: Zero hardcoded rates, IDs, or reporting dates.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import text

from config import *
from calculations import valuation_rates
from analytics import (
    build_full_row_by_row_scenario_table,
    export_row_by_row_with_formulas_xlsx,
    exposure_tables,
    liquidity_ladder,
    management_insights,
    pnl_components,
    position_sensitivity,
)
from ui_common import (
    active_driver_chips,
    attribution,
    current_session_scenario,
    evaluate_saved_scenario,
    format_large_number,
    kpi_card,
    render_ai_scenario_panel,
    show_df,
    style_nbe_figure,
    tone_for,
)
from reporting import (
    REPORTING_AVAILABLE,
    REPORTING_MISSING_ERROR,
    generate_pdf_report,
    generate_pptx_report,
    generate_zip_bundle,
)


def _fmt_delta(value, unit=""):
    """Format numeric variance with explicit +/- sign and unit."""
    if value is None or pd.isna(value) or abs(value) < 1e-12:
        return None
    sign = "+" if value > 0 else "-"
    return f"{sign}{format_large_number(abs(value))}{unit}" if unit == "" else f"{value:+,.2f}{unit}"


@st.dialog("KPI Audit Breakdown", width="large")
def _open_kpi_audit_dialog(kpi_payload):
    """Render an interactive modal dialog showing the master formula and component breakdown for any clicked KPI."""
    label = kpi_payload.get("label") or kpi_payload.get("title") or "KPI Audit"
    val = kpi_payload.get("value") or ""
    subtitle = kpi_payload.get("subtitle") or ""

    st.markdown(
        f'<div style="font-size:1.35rem; font-weight:800; margin-top:2px;">'
        f'{label} — <span style="color:#F7941D;">{val}</span></div>',
        unsafe_allow_html=True,
    )
    if subtitle:
        st.caption(subtitle)

    st.markdown("#### 1️⃣ Mathematical Formula & Step-by-Step Substitution")
    st.info(f"**∑ Master Formula:** `{kpi_payload.get('formula', '')}`")
    if kpi_payload.get("substitution"):
        st.code(kpi_payload["substitution"], language="text")

    st.markdown("#### 2️⃣ Sub-Component Breakdown (Before vs. After vs. Variance)")
    comp_df = kpi_payload.get("components_df")
    if (comp_df is None or comp_df.empty) and kpi_payload.get("rows"):
        comp_df = pd.DataFrame([
            {
                "Sub-Component": r[0],
                "Before (Base)": f"{r[1]:,.2f}" if isinstance(r[1], (int, float)) else str(r[1]),
                "After (Scenario)": f"{r[2]:,.2f}" if isinstance(r[2], (int, float)) else str(r[2]),
                "Variance (Δ)": f"{r[3]:+,.2f}" if isinstance(r[3], (int, float)) else str(r[3]),
            }
            for r in kpi_payload["rows"]
        ])
    if comp_df is not None and not comp_df.empty:
        show_df(comp_df, height=min(360, 45 + len(comp_df) * 36))


def _kpi_grid(res):
    """Render the 20 executive KPI cards (5 rows x 4 columns) plus the live Financial Statements Impact summary bar."""
    b, a = res.before, res.after
    horizon = max(int(res.horizon), 1)
    ann_factor = 365.0 / horizon

    k = dict(a) if isinstance(a, dict) else {}
    k_obj = getattr(res, "kpis", None)
    if isinstance(k_obj, dict):
        k.update(k_obj)
    elif hasattr(k_obj, "__dict__"):
        k.update(k_obj.__dict__)

    nii_pct = ((a["NII"] - b["NII"]) / abs(b["NII"]) * 100) if b.get("NII") else np.nan
    delta_nii = float(a.get("NII", 0.0) - b.get("NII", 0.0))
    earning_assets_b = float(b.get("Earning Assets", b.get("Loans", 0.0) + b.get("Investments", 0.0)))
    earning_assets_a = float(a.get("Earning Assets", a.get("Loans", 0.0) + a.get("Investments", 0.0)))

    cof_b = float(k.get("cof_fs_base", b.get("Cost of Funds %", 0.0)))
    cof_a = float(k.get("cof_fs_scn", a.get("Cost of Funds %", 0.0)))

    cards_config = [
        # Row 1: Core Earnings, Yield, Cost of Funds & Spread
        {
            "key": "nii",
            "card": ("NII (HORIZON)", format_large_number(a["NII"]), _fmt_delta(delta_nii), tone_for(delta_nii, True), f"{nii_pct:+.1f}% vs base" if pd.notna(nii_pct) else "", "Interest Income − Interest Expense"),
            "dialog": {
                "title": "Net Interest Income (NII)",
                "formula": "NII = Interest Income (Loans + Investments) − Interest Expense (Deposits + Funding)",
                "substitution": f"Base NII = {format_large_number(b['NII'])}\nScenario NII = {format_large_number(a['NII'])}\nΔNII = {format_large_number(delta_nii)}",
                "rows": [
                    ("Interest Income (EGP)", b.get("Interest Income", 0.0), a.get("Interest Income", 0.0), a.get("Interest Income", 0.0) - b.get("Interest Income", 0.0)),
                    ("Interest Expense (EGP)", b.get("Interest Expense", 0.0), a.get("Interest Expense", 0.0), a.get("Interest Expense", 0.0) - b.get("Interest Expense", 0.0)),
                    ("Net Interest Income (EGP)", b["NII"], a["NII"], delta_nii),
                ],
            },
        },
        {
            "key": "nim",
            "card": ("NIM %", f"{a['NIM %']:.2f}%", _fmt_delta((a["NIM %"] - b["NIM %"]) * 100, " bps"), tone_for(a["NIM %"] - b["NIM %"], True), f"Base: {b['NIM %']:.2f}%", "[NII × (365/Horizon)] ÷ Earning Assets"),
            "dialog": {
                "title": "Net Interest Margin (NIM %)",
                "formula": "NIM % = [Annualized NII ÷ Earning Assets (Excl. Non-Earning Cash)] × 100",
                "substitution": f"Base NIM = {b['NIM %']:.2f}% | Scenario NIM = {a['NIM %']:.2f}%",
                "rows": [
                    ("Annualized NII (EGP)", b["NII"] * ann_factor, a["NII"] * ann_factor, delta_nii * ann_factor),
                    ("Earning Assets (EGP)", earning_assets_b, earning_assets_a, earning_assets_a - earning_assets_b),
                    ("NIM (%)", b["NIM %"], a["NIM %"], a["NIM %"] - b["NIM %"]),
                ],
            },
        },
        {
            "key": "yield_ea",
            "card": ("YIELD ON EARNING ASSETS", f"{k.get('yield_ea_scn', 0.0):.2f}%", f"{(k.get('yield_ea_scn', 0.0) - k.get('yield_ea_base', 0.0)) * 100:+.1f} bps", tone_for(k.get("yield_ea_scn", 0.0) - k.get("yield_ea_base", 0.0), True), f"Base: {k.get('yield_ea_base', 0.0):.2f}%", "Interest Income ÷ Avg Earning Assets"),
            "dialog": {
                "title": "Yield on Earning Assets (%)",
                "formula": "Yield % = (Annualized Interest Income ÷ Average Earning Assets) × 100",
                "substitution": f"Base = {k.get('yield_ea_base', 0.0):.2f}% | Scenario = {k.get('yield_ea_scn', 0.0):.2f}%",
                "rows": [
                    ("Earning Assets (EGP Mn)", k.get("earning_assets_base_mn", 0.0), k.get("earning_assets_scn_mn", 0.0), k.get("earning_assets_scn_mn", 0.0) - k.get("earning_assets_base_mn", 0.0)),
                    ("Yield (%)", k.get("yield_ea_base", 0.0), k.get("yield_ea_scn", 0.0), k.get("yield_ea_scn", 0.0) - k.get("yield_ea_base", 0.0)),
                ],
            },
        },
        {
            "key": "cof",
            "card": ("COST OF FUNDS & SPREAD", f"{cof_a:.2f}%", f"{(cof_a - cof_b) * 100:+.1f} bps", tone_for(cof_a - cof_b, False), f"Spread: {k.get('spread_scn', 0.0):.2f}% (Base CoF: {cof_b:.2f}%)", "Interest Expense ÷ Costed Liabilities"),
            "dialog": {
                "title": "Cost of Funds (COF %) & Net Interest Spread",
                "formula": "COF % = (Annualized Interest Expense ÷ Costed Liabilities [Excl. Non-Int CA]) × 100\nSpread % = Yield on Earning Assets % − Cost of Funds %",
                "substitution": f"Base COF = {cof_b:.2f}% | Scenario COF = {cof_a:.2f}%\nNet Interest Spread = {k.get('spread_scn', 0.0):.2f}%",
                "rows": [
                    ("Cost of Funds (%)", cof_b, cof_a, cof_a - cof_b),
                    ("Yield on Earning Assets (%)", k.get("yield_ea_base", 0.0), k.get("yield_ea_scn", 0.0), k.get("yield_ea_scn", 0.0) - k.get("yield_ea_base", 0.0)),
                    ("Net Interest Spread (%)", k.get("spread_base", 0.0), k.get("spread_scn", 0.0), k.get("spread_scn", 0.0) - k.get("spread_base", 0.0)),
                ],
            },
        },

        # Row 2: Total P&L Impact, Repricing Gap, ΔEVE / Equity, and Cost-to-Income & Net Profit
        {
            "key": "pnl",
            "card": ("TOTAL P&L IMPACT", format_large_number(a["PnL Total Impact"]), _fmt_delta(a["PnL Total Impact"]), tone_for(a["PnL Total Impact"], True), "ΔNII + FX/Gold + MTM − ECL", "Total Scenario P&L & Valuation Impact"),
            "dialog": {
                "title": "Total P&L & Value Impact",
                "formula": "Total Impact = ΔNII + FX/Gold Revaluation + Fair-Value MTM − Macro Credit Cost",
                "substitution": f"ΔNII = {format_large_number(delta_nii)} | Reval = {format_large_number(a.get('PnL FX Gold Revaluation', 0.0))} | MTM = {format_large_number(a.get('PnL Investment MTM', 0.0))} | Credit = {format_large_number(a.get('PnL Credit Cost', 0.0))}",
                "rows": [
                    ("NII Change (EGP)", 0.0, delta_nii, delta_nii),
                    ("FX & Gold Revaluation (EGP)", 0.0, a.get("PnL FX Gold Revaluation", 0.0), a.get("PnL FX Gold Revaluation", 0.0)),
                    ("Fair-Value Investment MTM (EGP)", 0.0, a.get("PnL Investment MTM", 0.0), a.get("PnL Investment MTM", 0.0)),
                    ("Macro Credit Cost (EGP)", 0.0, a.get("PnL Credit Cost", 0.0), a.get("PnL Credit Cost", 0.0)),
                    ("Total P&L Impact (EGP)", 0.0, a["PnL Total Impact"], a["PnL Total Impact"]),
                ],
            },
        },
        {
            "key": "repricing_gap_card",
            "card": ("REPRICING GAP (12M)", format_large_number(a["Repricing Gap"]), _fmt_delta(a["Repricing Gap"] - b["Repricing Gap"]), "neutral", f"±100bp Sens: {k.get('nii_sens_100bp', 0.0):+.2f}% NII", "RSA − RSL Repricing in Horizon"),
            "dialog": {
                "title": "Repricing Gap & NII Sensitivity per 100bp",
                "formula": "Repricing Gap = Repricing Assets (RSA) − Repricing Liabilities (RSL)\nNII Sensitivity % = (Repricing Gap × 1%) ÷ Base NII",
                "substitution": f"Base Gap = {k.get('rep_gap_base_mn', 0.0):,.1f} M EGP | Scenario Gap = {k.get('rep_gap_scn_mn', 0.0):,.1f} M EGP",
                "rows": [
                    ("Repricing Gap (EGP Mn)", k.get("rep_gap_base_mn", 0.0), k.get("rep_gap_scn_mn", 0.0), k.get("rep_gap_scn_mn", 0.0) - k.get("rep_gap_base_mn", 0.0)),
                    ("±100bp NII Sensitivity (%)", k.get("nii_sens_100bp", 0.0), k.get("nii_sens_100bp", 0.0), 0.0),
                ],
            },
        },
        {
            "key": "fvoci_eve",
            "card": ("ΔEVE / EQUITY (FVOCI)", f"{k.get('eve_eq_scn', 0.0):+.2f}%", f"{k.get('fvoci_mtm_mn', 0.0):+,.1f} M EGP", tone_for(k.get("eve_eq_scn", 0.0), True), "FVOCI Reserve MTM Impact", "−(ΔRate × Duration × FVOCI) ÷ Equity"),
            "dialog": {
                "title": "Economic Value of Equity Impact (ΔEVE / Equity %)",
                "formula": "ΔEVE / Equity % = (FVOCI Securities Duration MTM ÷ Total Equity) × 100",
                "substitution": f"FVOCI MTM = {k.get('fvoci_mtm_mn', 0.0):+,.2f} M EGP | ΔEVE/Equity = {k.get('eve_eq_scn', 0.0):+.2f}%",
                "rows": [
                    ("FVOCI MTM Impact (EGP Mn)", 0.0, k.get("fvoci_mtm_mn", 0.0), k.get("fvoci_mtm_mn", 0.0)),
                    ("Total Equity (EGP Mn)", k.get("total_eq_base_mn", 0.0), k.get("total_eq_scn_mn", 0.0), k.get("total_eq_scn_mn", 0.0) - k.get("total_eq_base_mn", 0.0)),
                ],
            },
        },
        {
            "key": "cost_to_income",
            "card": ("COST-TO-INCOME & PROFIT", f"{k.get('cti_scn', 0.0):.2f}%", f"ΔNet Profit: {k.get('delta_net_profit_mn', 0.0):+,.1f} M", tone_for(k.get("cti_base", 0.0) - k.get("cti_scn", 0.0), True), f"Net Profit: {k.get('scn_net_profit_mn', 0.0)/1e3:,.2f}B EGP", "|G&A| ÷ Operating Income"),
            "dialog": {
                "title": "Cost-to-Income Ratio & Net Profit After Tax",
                "formula": "CTI % = (|G&A Expenses| ÷ Operating Income) × 100\nΔNet Profit = ΔEBT × (1 − Effective Tax Rate)",
                "substitution": f"Base CTI = {k.get('cti_base', 0.0):.2f}% | Scenario CTI = {k.get('cti_scn', 0.0):.2f}%\nNet Profit = {k.get('scn_net_profit_mn', 0.0):,.1f} M EGP",
                "rows": [
                    ("Operating Income (EGP Mn)", k.get("op_inc_base_mn", 0.0), k.get("op_inc_scn_mn", 0.0), k.get("op_inc_scn_mn", 0.0) - k.get("op_inc_base_mn", 0.0)),
                    ("Cost-to-Income (%)", k.get("cti_base", 0.0), k.get("cti_scn", 0.0), k.get("cti_scn", 0.0) - k.get("cti_base", 0.0)),
                    ("Net Profit After Tax (EGP Mn)", k.get("base_net_profit_mn", 0.0), k.get("scn_net_profit_mn", 0.0), k.get("delta_net_profit_mn", 0.0)),
                ],
            },
        },

        # Row 3: Liquidity Ratios (30d LCR Proxy, Liquid Assets/Deposits, CASA, LDR)
        {
            "key": "lcr",
            "card": ("LCR PROXY (30D STRESS)", "N/A" if pd.isna(a["CF LCR %"]) else f"{a['CF LCR %']:,.1f}%", None if pd.isna(a["CF LCR %"]) or pd.isna(b["CF LCR %"]) else _fmt_delta(a["CF LCR %"] - b["CF LCR %"], " pp"), tone_for(a.get("CF LCR %", 0.0) - b.get("CF LCR %", 0.0), True), f"Base: {b.get('CF LCR %', 0.0):,.1f}%", "HQLA ÷ (30d Outflows − 30d Inflows)"),
            "dialog": {
                "title": "Liquidity Coverage Ratio Proxy (LCR % — 30-Day Window)",
                "formula": "LCR Proxy % = HQLA Stock ÷ Max(30d Outflows + Runoff − 30d Inflows, 1) × 100",
                "substitution": f"HQLA = {format_large_number(a.get('CF HQLA', 0.0))} | 30d Net Outflow = {format_large_number(a.get('LCR Net Outflow', 0.0))} | LCR = {a.get('CF LCR %', 0.0):,.2f}%",
                "rows": [
                    ("HQLA Stock (EGP)", b.get("CF HQLA", 0.0), a.get("CF HQLA", 0.0), a.get("CF HQLA", 0.0) - b.get("CF HQLA", 0.0)),
                    ("30d Gross Outflow + Runoff (EGP)", b.get("LCR Outflow", 0.0), a.get("LCR Outflow", 0.0), a.get("LCR Outflow", 0.0) - b.get("LCR Outflow", 0.0)),
                    ("30d Scheduled Inflows (EGP)", b.get("LCR Inflow", 0.0), a.get("LCR Inflow", 0.0), a.get("LCR Inflow", 0.0) - b.get("LCR Inflow", 0.0)),
                    ("30d Stressed Net Outflow (EGP)", b.get("LCR Net Outflow", 0.0), a.get("LCR Net Outflow", 0.0), a.get("LCR Net Outflow", 0.0) - b.get("LCR Net Outflow", 0.0)),
                    ("LCR Proxy (%)", b.get("CF LCR %", 0.0), a.get("CF LCR %", 0.0), a.get("CF LCR %", 0.0) - b.get("CF LCR %", 0.0)),
                ],
            },
        },
        {
            "key": "liq_assets_dep",
            "card": ("LIQUID ASSETS / DEPOSITS", f"{k.get('liq_dep_scn', 0.0):.2f}%", f"{(k.get('liq_dep_scn', 0.0) - k.get('liq_dep_base', 0.0)):+.2f} pp", tone_for(k.get("liq_dep_scn", 0.0) - k.get("liq_dep_base", 0.0), True), f"Base: {k.get('liq_dep_base', 0.0):.2f}%", "(Cash + Banks + T-Bills) ÷ Deposits"),
            "dialog": {
                "title": "Liquid Assets to Customer Deposits Ratio (%)",
                "formula": "Liquid / Deposits % = [Liquid Assets (Is_Liquid_Asset=1) ÷ Customer Deposits] × 100",
                "substitution": f"Base = {k.get('liq_dep_base', 0.0):.2f}% | Scenario = {k.get('liq_dep_scn', 0.0):.2f}%",
                "rows": [
                    ("Liquid Assets (EGP Mn)", k.get("liq_assets_base_mn", 0.0), k.get("liq_assets_scn_mn", 0.0), k.get("liq_assets_scn_mn", 0.0) - k.get("liq_assets_base_mn", 0.0)),
                    ("Customer Deposits (EGP Mn)", k.get("dep_base_mn", 0.0), k.get("dep_scn_mn", 0.0), k.get("dep_scn_mn", 0.0) - k.get("dep_base_mn", 0.0)),
                ],
            },
        },
        {
            "key": "casa_ratio",
            "card": ("CASA RATIO", f"{k.get('casa_ratio_scn', 0.0):.2f}%", f"{(k.get('casa_ratio_scn', 0.0) - k.get('casa_ratio_base', 0.0)):+.2f} pp", "neutral", f"Base: {k.get('casa_ratio_base', 0.0):.2f}%", "(Current + Savings) ÷ Total Deposits"),
            "dialog": {
                "title": "CASA Ratio (Low-Cost Deposits Share %)",
                "formula": "CASA % = [(Current + Savings Accounts) ÷ Total Customer Deposits] × 100",
                "substitution": f"CASA Deposits = {k.get('casa_dep_scn_mn', 0.0):,.1f} M EGP ({k.get('casa_ratio_scn', 0.0):.2f}%)",
                "rows": [
                    ("CASA Deposits (EGP Mn)", k.get("casa_dep_base_mn", 0.0), k.get("casa_dep_scn_mn", 0.0), k.get("casa_dep_scn_mn", 0.0) - k.get("casa_dep_base_mn", 0.0)),
                    ("CASA Ratio (%)", k.get("casa_ratio_base", 0.0), k.get("casa_ratio_scn", 0.0), k.get("casa_ratio_scn", 0.0) - k.get("casa_ratio_base", 0.0)),
                ],
            },
        },
        {
            "key": "ldr",
            "card": ("LOAN / DEPOSIT (LDR)", f"{a['Loan Deposit %']:.1f}%", _fmt_delta(a["Loan Deposit %"] - b["Loan Deposit %"], " pp"), "neutral", f"Base: {b['Loan Deposit %']:.1f}%", "(Total Loans ÷ Total Deposits) × 100"),
            "dialog": {
                "title": "Loan-to-Deposit Ratio (LDR %)",
                "formula": "LDR % = (Stressed Net Loans ÷ Stressed Customer Deposits) × 100",
                "substitution": f"Base LDR = {b['Loan Deposit %']:.2f}% | Scenario LDR = {a['Loan Deposit %']:.2f}%",
                "rows": [
                    ("Total Loans (EGP)", b["Loans"], a["Loans"], a["Loans"] - b["Loans"]),
                    ("Total Deposits (EGP)", b["Deposits"], a["Deposits"], a["Deposits"] - b["Deposits"]),
                ],
            },
        },

        # Row 4: Return on Equity, Return on Assets, Leverage, and Net Open FX Position
        {
            "key": "roe",
            "card": ("RETURN ON EQUITY (ROE)", f"{k.get('roe_scn', 0.0):.2f}%", f"{(k.get('roe_scn', 0.0) - k.get('roe_base', 0.0)):+.2f} pp", tone_for(k.get("roe_scn", 0.0) - k.get("roe_base", 0.0), True), f"Base: {k.get('roe_base', 0.0):.2f}%", "(Annualized Net Profit ÷ Avg Equity) × 100"),
            "dialog": {
                "title": "Return on Equity (ROE %)",
                "formula": "ROE % = (Annualized Net Profit After Tax ÷ Average Total Equity) × 100",
                "substitution": f"Base ROE = {k.get('roe_base', 0.0):.2f}% | Scenario ROE = {k.get('roe_scn', 0.0):.2f}%",
                "rows": [
                    ("Net Profit After Tax (EGP Mn)", k.get("base_net_profit_mn", 0.0), k.get("scn_net_profit_mn", 0.0), k.get("delta_net_profit_mn", 0.0)),
                    ("Total Equity (EGP Mn)", k.get("total_eq_base_mn", 0.0), k.get("total_eq_scn_mn", 0.0), k.get("total_eq_scn_mn", 0.0) - k.get("total_eq_base_mn", 0.0)),
                ],
            },
        },
        {
            "key": "roa",
            "card": ("RETURN ON ASSETS (ROA)", f"{k.get('roa_scn', 0.0):.2f}%", f"{(k.get('roa_scn', 0.0) - k.get('roa_base', 0.0)):+.2f} pp", tone_for(k.get("roa_scn", 0.0) - k.get("roa_base", 0.0), True), f"Base: {k.get('roa_base', 0.0):.2f}%", "(Annualized Net Profit ÷ Avg Assets) × 100"),
            "dialog": {
                "title": "Return on Assets (ROA %)",
                "formula": "ROA % = (Annualized Net Profit After Tax ÷ Average Total Assets) × 100",
                "substitution": f"Base ROA = {k.get('roa_base', 0.0):.2f}% | Scenario ROA = {k.get('roa_scn', 0.0):.2f}%",
                "rows": [
                    ("Net Profit After Tax (EGP Mn)", k.get("base_net_profit_mn", 0.0), k.get("scn_net_profit_mn", 0.0), k.get("delta_net_profit_mn", 0.0)),
                    ("Total Assets (EGP Mn)", k.get("total_assets_base_mn", 0.0), k.get("total_assets_scn_mn", 0.0), k.get("total_assets_scn_mn", 0.0) - k.get("total_assets_base_mn", 0.0)),
                ],
            },
        },
        {
            "key": "eq_assets",
            "card": ("EQUITY / ASSETS", f"{k.get('eq_ast_scn', 0.0):.2f}%", f"{(k.get('eq_ast_scn', 0.0) - k.get('eq_ast_base', 0.0)):+.2f} pp", tone_for(k.get("eq_ast_scn", 0.0) - k.get("eq_ast_base", 0.0), True), f"Base: {k.get('eq_ast_base', 0.0):.2f}%", "(Total Equity ÷ Total Assets) × 100"),
            "dialog": {
                "title": "Capital Leverage Ratio (Equity / Assets %)",
                "formula": "Equity / Assets % = (Total Equity ÷ Total Assets) × 100",
                "substitution": f"Base = {k.get('eq_ast_base', 0.0):.2f}% | Scenario = {k.get('eq_ast_scn', 0.0):.2f}%",
                "rows": [
                    ("Total Equity (EGP Mn)", k.get("total_eq_base_mn", 0.0), k.get("total_eq_scn_mn", 0.0), k.get("total_eq_scn_mn", 0.0) - k.get("total_eq_base_mn", 0.0)),
                    ("Total Assets (EGP Mn)", k.get("total_assets_base_mn", 0.0), k.get("total_assets_scn_mn", 0.0), k.get("total_assets_scn_mn", 0.0) - k.get("total_assets_base_mn", 0.0)),
                ],
            },
        },
        {
            "key": "nop_equity",
            "card": ("NET OPEN FX POS / EQUITY", f"{k.get('nop_eq_scn', 0.0):+.2f}%", f"{(k.get('nop_eq_scn', 0.0) - k.get('nop_eq_base', 0.0)):+.2f} pp", "neutral", f"NOP: {k.get('nop_scn_mn', 0.0):+,.0f} M EGP", "(FCY Assets − FCY Liab) ÷ Equity"),
            "dialog": {
                "title": "Net Open Foreign Currency Position to Equity (%)",
                "formula": "NOP / Equity % = [(Foreign Currency Assets − Foreign Currency Liabilities) ÷ Total Equity] × 100",
                "substitution": f"Base NOP = {k.get('nop_base_mn', 0.0):+,.1f} M EGP | Scenario NOP = {k.get('nop_scn_mn', 0.0):+,.1f} M EGP",
                "rows": [
                    ("Net Open FX Position (EGP Mn)", k.get("nop_base_mn", 0.0), k.get("nop_scn_mn", 0.0), k.get("nop_scn_mn", 0.0) - k.get("nop_base_mn", 0.0)),
                    ("NOP / Equity (%)", k.get("nop_eq_base", 0.0), k.get("nop_eq_scn", 0.0), k.get("nop_eq_scn", 0.0) - k.get("nop_eq_base", 0.0)),
                ],
            },
        },

        # Row 5: Structural & Regulatory Ratios (NSFR Proxy, Sovereign Concentration, FX Wholesale, Contingent/Equity)
        {
            "key": "nsfr_proxy",
            "card": ("NSFR PROXY (BASEL III)", f"{k.get('nsfr_scn', 0.0):.1f}%", f"{k.get('nsfr_scn', 0.0) - k.get('nsfr_base', 0.0):+.2f} pp", tone_for(k.get("nsfr_scn", 0.0) - 100.0, True), f"Base: {k.get('nsfr_base', 0.0):.1f}% (Min 100%)", "Available Stable Funding ÷ Required Stable Funding"),
            "dialog": {
                "title": "Net Stable Funding Ratio Proxy (NSFR %)",
                "formula": "NSFR % = [ASF (Equity + ∑ Balance × NSFR_ASF_Weight) ÷ RSF (∑ Balance × NSFR_RSF_Weight)] × 100",
                "substitution": f"ASF = {k.get('asf_scn_mn', 0.0):,.0f} M EGP | RSF = {k.get('rsf_scn_mn', 0.0):,.0f} M EGP | NSFR = {k.get('nsfr_scn', 0.0):.2f}%",
                "rows": [
                    ("Available Stable Funding - ASF (EGP Mn)", k.get("asf_base_mn", 0.0), k.get("asf_scn_mn", 0.0), k.get("asf_scn_mn", 0.0) - k.get("asf_base_mn", 0.0)),
                    ("Required Stable Funding - RSF (EGP Mn)", k.get("rsf_base_mn", 0.0), k.get("rsf_scn_mn", 0.0), k.get("rsf_scn_mn", 0.0) - k.get("rsf_base_mn", 0.0)),
                    ("NSFR Proxy (%)", k.get("nsfr_base", 0.0), k.get("nsfr_scn", 0.0), k.get("nsfr_scn", 0.0) - k.get("nsfr_base", 0.0)),
                ],
            },
        },
        {
            "key": "sov_assets",
            "card": ("SOVEREIGN DEBT / ASSETS", f"{k.get('sov_ast_scn', 0.0):.2f}%", f"{k.get('sov_ast_scn', 0.0) - k.get('sov_ast_base', 0.0):+.2f} pp", "neutral", f"Base: {k.get('sov_ast_base', 0.0):.2f}%", "(FVOCI + Amortised Cost Securities) ÷ Total Assets"),
            "dialog": {
                "title": "Sovereign Debt Concentration to Total Assets (%)",
                "formula": "Sovereign / Assets % = [(FVOCI + Amortised Cost Securities) ÷ Total Assets] × 100",
                "substitution": f"Base = {k.get('sov_ast_base', 0.0):.2f}% | Scenario = {k.get('sov_ast_scn', 0.0):.2f}%",
                "rows": [
                    ("Sovereign / Assets (%)", k.get("sov_ast_base", 0.0), k.get("sov_ast_scn", 0.0), k.get("sov_ast_scn", 0.0) - k.get("sov_ast_base", 0.0)),
                ],
            },
        },
        {
            "key": "fx_wsf_liab",
            "card": ("FX WHOLESALE / LIAB", f"{k.get('wsf_liab_scn', 0.0):.2f}%", f"{k.get('wsf_liab_scn', 0.0) - k.get('wsf_liab_base', 0.0):+.2f} pp", tone_for(k.get("wsf_liab_base", 0.0) - k.get("wsf_liab_scn", 0.0), True), f"Base: {k.get('wsf_liab_base', 0.0):.2f}%", "Other Borrowings ÷ Total Liabilities"),
            "dialog": {
                "title": "FX Wholesale Funding to Total Liabilities (%)",
                "formula": "FX Wholesale / Liabilities % = (Other Borrowings ÷ Total Liabilities) × 100",
                "substitution": f"Base = {k.get('wsf_liab_base', 0.0):.2f}% | Scenario = {k.get('wsf_liab_scn', 0.0):.2f}%",
                "rows": [
                    ("Other Borrowings (EGP Mn)", k.get("borrow_base_mn", 0.0), k.get("borrow_scn_mn", 0.0), k.get("borrow_scn_mn", 0.0) - k.get("borrow_base_mn", 0.0)),
                    ("Total Liabilities (EGP Mn)", k.get("total_liab_base_mn", 0.0), k.get("total_liab_scn_mn", 0.0), k.get("total_liab_scn_mn", 0.0) - k.get("total_liab_base_mn", 0.0)),
                ],
            },
        },
        {
            "key": "off_bs_eq",
            "card": ("CONTINGENT (OFF-BS) / EQUITY", f"{k.get('off_eq_scn', 0.0):.1f}%", f"{k.get('off_eq_scn', 0.0) - k.get('off_eq_base', 0.0):+.1f} pp", "neutral", f"Credit Cost: {a.get('Credit Cost Rate (%)', 0.0)*100:.0f} bps", "Contingent LGs & LCs ÷ Total Equity"),
            "dialog": {
                "title": "Off-Balance Sheet Exposure to Equity (%) & Macro Credit Cost",
                "formula": "Off-BS / Equity % = (Contingent Liabilities ÷ Total Equity) × 100",
                "substitution": f"Base = {k.get('off_eq_base', 0.0):.1f}% | Scenario = {k.get('off_eq_scn', 0.0):.1f}%",
                "rows": [
                    ("Off-BS / Equity (%)", k.get("off_eq_base", 0.0), k.get("off_eq_scn", 0.0), k.get("off_eq_scn", 0.0) - k.get("off_eq_base", 0.0)),
                ],
            },
        },
    ]

    for offset in range(0, len(cards_config), 4):
        cols = st.columns(4)
        for col, item in zip(cols, cards_config[offset:offset + 4]):
            with col:
                kpi_card(*item["card"])
                if st.button(" ", key=f"btn_audit_{item['key']}", use_container_width=True):
                    d = dict(item.get("dialog", {}))
                    d.setdefault("label", item["card"][0])
                    d.setdefault("value", item["card"][1])
                    d.setdefault("delta", item["card"][2] if len(item["card"]) > 2 else "")
                    _open_kpi_audit_dialog(d)

    # Dynamic Financial Statements Impact Summary Bar (zero hardcoded fallbacks)
    ast_b, ast_s = float(k.get("total_assets_base_mn", 0.0)), float(k.get("total_assets_scn_mn", 0.0))
    liab_b, liab_s = float(k.get("total_liab_base_mn", 0.0)), float(k.get("total_liab_scn_mn", 0.0))
    eq_b, eq_s = float(k.get("total_eq_base_mn", 0.0)), float(k.get("total_eq_scn_mn", 0.0))
    np_b, np_s = float(k.get("base_net_profit_mn", 0.0)), float(k.get("scn_net_profit_mn", 0.0))

    with st.container(border=True):
        st.markdown("**🏛️ ملخص تأثير السيناريو المباشر على القوائم المالية (Financial Statements Impact Summary — EGP Mn)**")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("إجمالي الأصول (Total Assets)", f"{ast_s:,.0f} M", f"{ast_s - ast_b:+,.0f} M EGP")
        s2.metric("إجمالي الالتزامات (Total Liabilities)", f"{liab_s:,.0f} M", f"{liab_s - liab_b:+,.0f} M EGP", delta_color="inverse")
        s3.metric("إجمالي حقوق الملكية (Total Equity)", f"{eq_s:,.0f} M", f"{eq_s - eq_b:+,.0f} M EGP")
        s4.metric("صافي الربح السنوي بعد الضريبة", f"{np_s:,.0f} M", f"{np_s - np_b:+,.0f} M EGP")


def _extract_year_tag(col_str: str, suffix: str) -> str:
    """Dynamically format any period string into institutional financial notation (e.g., 2023A, 2025E, 2026F)."""
    import re
    m = re.search(r"(20\d{2})", str(col_str))
    yr = m.group(1) if m else str(col_str)
    return f"{yr}{suffix}"


def _render_outlined_statement(
    df: pd.DataFrame,
    hist_cols: list,
    base_col: str,
    scn_col: str,
    delta_col: str,
    base_label: str,
    fcst_cols: list,
    show_ml: bool = True,
):
    """Render hierarchical Outlined financial statements with a symmetric ±2Y or ±3Y column window."""
    def _fmt(v):
        return "—" if (v is None or pd.isna(v)) else f"{float(v):,.2f}"

    def _fmtd(v):
        return "—" if (v is None or pd.isna(v)) else f"{float(v):+,.2f}"

    hist_hdr_html = "".join(
        f'<span style="flex:0.92; text-align:right; font-size:0.82rem; font-family:monospace;">{_extract_year_tag(hc, "A")}</span>'
        for hc in hist_cols
    )
    ml_hdr_html = (
        "".join(
            f'<span style="flex:0.98; text-align:right; font-size:0.82rem; color:#86EFAC; font-family:monospace;">{_extract_year_tag(fc, "F")}</span>'
            for fc in fcst_cols
        )
        if show_ml else ""
    )

    for sec in df["section"].unique():
        sec_df = df[df["section"] == sec]
        with st.container(border=True):
            hdr_html = (
                '<div style="background:#005B2D; color:#FFFFFF; padding:8px 14px; border-radius:8px; font-weight:700; display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">'
                f'<span style="flex:2.3;">📂 {sec}</span>'
                f'{hist_hdr_html}'
                f'<span style="flex:1.00; text-align:right; font-size:0.82rem; font-family:monospace; color:#FDE68A;">{base_label}</span>'
                '<span style="flex:1.00; text-align:right; font-size:0.80rem; color:#F7941D;">Scenario</span>'
                '<span style="flex:0.92; text-align:right; font-size:0.80rem;">Delta (Δ)</span>'
                f'{ml_hdr_html}'
                '</div>'
            )

            rows_html = []
            for _, r in sec_df.iterrows():
                code = str(r.get("code", ""))
                src = str(r.get("source", ""))
                is_tot = any(t in code for t in ("TOT", "CHECK", "NII", "EBT", "NET_PROFIT")) or src in ("TOTAL", "SUBTOTAL", "CHECK")
                d_val = float(r.get(delta_col, 0.0) or 0.0)
                d_col = "#007A3D" if d_val > 1e-4 else ("#C0392B" if d_val < -1e-4 else "#64748B")
                bg = "background:#EAF5EF; border-left:4px solid #007A3D; font-weight:800; color:#005B2D;" if is_tot else "border-bottom:1px solid #EDF2EF; padding-left:14px;"
                pad = "padding:8px 12px; border-radius:6px; margin-top:4px;" if is_tot else "padding:6px 12px;"

                hist_cells = "".join(
                    f'<div style="flex:0.92; text-align:right; font-family:monospace; font-size:0.85rem;">{_fmt(r.get(hc))}</div>'
                    for hc in hist_cols
                )
                ml_cells = (
                    "".join(
                        f'<div style="flex:0.98; text-align:right; font-family:monospace; font-size:0.85rem; color:#0F5132; background:rgba(0,122,61,0.04); padding:2px 4px; border-radius:4px;">{_fmt(r.get(fc))}</div>'
                        for fc in fcst_cols
                    )
                    if show_ml else ""
                )
                row_str = (
                    f'<div style="display:flex; justify-content:space-between; align-items:center; {bg} {pad}">'
                    f'<div style="flex:2.3;"><div>{r["name_ar"]}</div><div style="font-size:0.72rem; opacity:0.7; font-weight:400;">{r["name_en"]}</div></div>'
                    f'{hist_cells}'
                    f'<div style="flex:1.00; text-align:right; font-family:monospace; font-size:0.85rem; font-weight:600;">{_fmt(r.get(base_col))}</div>'
                    f'<div style="flex:1.00; text-align:right; font-family:monospace; font-size:0.87rem; font-weight:700; color:#005B2D;">{_fmt(r.get(scn_col))}</div>'
                    f'<div style="flex:0.92; text-align:right; font-family:monospace; font-size:0.85rem; font-weight:700; color:{d_col};">{_fmtd(d_val)}</div>'
                    f'{ml_cells}'
                    '</div>'
                )
                rows_html.append(row_str)

            st.markdown(
                f'<div class="fs-scroll-wrapper"><div class="fs-table-inner">{hdr_html}{"".join(rows_html)}</div></div>',
                unsafe_allow_html=True,
            )


def _render_fs_tab(res):
    """Render the Financial Statements tab with a smart symmetric window selector (±2Y or ±3Y around Base)."""
    df_bs = getattr(res, "balance_sheet_df", None)
    df_is = getattr(res, "income_statement_df", None)
    if df_bs is None or df_is is None:
        st.warning("لم يتم تحميل القوائم المالية بعد.")
        return

    meta = getattr(res, "fs_meta", None) or (getattr(res, "kpis", {}) or {}).get("fs_meta", {})

    all_hist_cols = [c for c in meta.get("hist_cols", []) if c in df_bs.columns]
    all_fcst_cols = [c for c in meta.get("fcst_cols", []) if c in df_bs.columns]
    bs_base_col = meta.get("bs_base_col", "2025E")

    # Fallback if meta lists are not populated
    if not all_hist_cols or not all_fcst_cols:
        meta_fixed = {"order", "section", "code", "name_ar", "name_en", "Scenario (After)", "Delta (Impact)", bs_base_col}
        dyn_c = [c for c in df_bs.columns if c not in meta_fixed]
        all_hist_cols = [c for c in dyn_c if str(c).endswith("A")]
        all_fcst_cols = [c for c in dyn_c if str(c).endswith("F")]

    base_pt = str(meta.get("is_base_pt", "6M")).upper()
    base_suffix = "A" if base_pt == "FY" and not meta.get("is_interim_base", True) else "E"
    base_lbl = _extract_year_tag(bs_base_col, base_suffix)

    # Smart Control Row: Timeline Bar + Window Size Selector (±2Y / ±3Y) + Forecast Toggle
    c_bar, c_win, c_tog = st.columns([3.5, 1.1, 0.9], vertical_alignment="center")
    with c_win:
        win_choice = st.radio(
            "Horizon Window",
            options=[2, 3],
            index=1 if len(all_hist_cols) >= 3 else 0,
            format_func=lambda n: f"±{n} Years",
            horizontal=True,
            label_visibility="collapsed",
            key="fs_sym_window_n",
        )
    n_win = int(win_choice)
    active_hist_cols = all_hist_cols[-n_win:] if len(all_hist_cols) >= n_win else all_hist_cols
    active_fcst_cols = all_fcst_cols[:n_win] if len(all_fcst_cols) >= n_win else all_fcst_cols

    with c_tog:
        f_start = _extract_year_tag(active_fcst_cols[0], "F") if active_fcst_cols else "F1"
        f_end = _extract_year_tag(active_fcst_cols[-1], "F") if active_fcst_cols else "F2"
        show_ml = st.toggle(f"{f_start}–{f_end}", value=True, key="toggle_ml_fs")

    with c_bar:
        h_pills = '<span style="color:#94A3B8;">—</span>'.join(
            f'<span style="background:#E2E8F0; color:#1E293B; padding:3px 9px; border-radius:6px; font-weight:700; font-family:monospace; font-size:0.81rem;">{_extract_year_tag(hc, "A")}</span>'
            for hc in active_hist_cols
        )
        f_pills = (
            '<span style="color:#94A3B8;">—</span>'
            + '<span style="color:#94A3B8;">—</span>'.join(
                f'<span style="background:#005B2D; color:#FFFFFF; padding:3px 9px; border-radius:6px; font-weight:700; font-family:monospace; font-size:0.81rem;">{_extract_year_tag(fc, "F")}</span>'
                for fc in active_fcst_cols
            )
            if show_ml and active_fcst_cols else ""
        )
        bar_html = (
            '<div style="display:flex; align-items:center; justify-content:space-between; background:#F8FAFC; border:1px solid #E2E8F0; padding:7px 14px; border-radius:10px;">'
            '<div style="display:flex; align-items:center; gap:5px; direction:ltr; flex-wrap:wrap;">'
            f'{h_pills}'
            '<span style="color:#94A3B8;">—</span>'
            f'<span style="background:#FEF3C7; color:#92400E; border:1px solid #FDE68A; padding:3px 9px; border-radius:6px; font-weight:700; font-family:monospace; font-size:0.81rem;">{base_lbl}</span>'
            f'{f_pills}'
            '</div>'
            '<div style="font-size:0.74rem; color:#64748B; font-weight:600; direction:ltr; white-space:nowrap;">'
            '<b>A</b> Actual &nbsp;|&nbsp; <b>E</b> Estimated &nbsp;|&nbsp; <b>F</b> Forecasted'
            '</div>'
            '</div>'
        )
        st.markdown(bar_html, unsafe_allow_html=True)

    sub_bs, sub_is = st.tabs([
        "🏛️ الميزانية العمومية (Balance Sheet)",
        "📈 قائمة الدخل (Income Statement)",
    ])
    with sub_bs:
        _render_outlined_statement(
            df_bs, active_hist_cols, bs_base_col, "Scenario (After)", "Delta (Impact)",
            base_lbl, active_fcst_cols, show_ml=show_ml
        )
    with sub_is:
        _render_outlined_statement(
            df_is, active_hist_cols, "base_mn", "scn_mn", "delta_mn",
            base_lbl, active_fcst_cols, show_ml=show_ml
        )


def _before_after_bar(df, x, before_col, after_col, title, height=360):
    """Create a grouped NBE-styled bar chart comparing Before vs. After values."""
    long = df.melt(id_vars=x, value_vars=[before_col, after_col], var_name="Stage", value_name="Value")
    long["Stage"] = long["Stage"].str.replace(r" ?(Balance|Interest)", "", regex=True).replace({"": "Value"})
    long["Label"] = long["Value"].apply(format_large_number)
    fig = px.bar(long, x=x, y="Value", color="Stage", barmode="group", text="Label", title=title,
                 color_discrete_sequence=[NBE_GREEN, NBE_ORANGE])
    fig.update_traces(textposition="outside", cliponaxis=False)
    return style_nbe_figure(fig, height)


def _waterfall(labels, values, title, total_label):
    """Create an NBE-styled Plotly Waterfall chart for NII bridge and P&L attribution."""
    measure = ["relative"] * len(values) + ["total"]
    fig = go.Figure(go.Waterfall(
        x=labels + [total_label], measure=measure, y=list(values) + [sum(values)],
        text=[format_large_number(v) for v in list(values) + [sum(values)]],
        increasing=dict(marker=dict(color=NBE_GREEN)), decreasing=dict(marker=dict(color="#C0392B")),
        totals=dict(marker=dict(color=NBE_DARK_GREEN)),
    ))
    fig.update_layout(title=title)
    return style_nbe_figure(fig)


@st.fragment
def _render_report_export_section(ctx, res, b, a, fig_nii_bridge, fig_pnl_comp, fig_portfolio, fig_currency, insights):
    """Isolated Streamlit fragment for building and downloading PDF, PPTX, and ZIP executive reports."""
    st.markdown("---")
    with st.expander("📄 Export Custom Executive Report (PDF / PPTX / Both)", expanded=False):
        if not REPORTING_AVAILABLE:
            st.warning(f"To enable PDF and PPTX export, install the required packages:\n`pip install reportlab python-pptx kaleido`\n({REPORTING_MISSING_ERROR})")
            return

        st.caption("1️⃣ اختر المؤشرات والرسوم البيانية، 2️⃣ حدد الصيغة المطلوبة، ثم اضغط Generate لتوليد التقرير التنفيذي فوراً.")

        k = dict(a) if isinstance(a, dict) else {}
        k_obj = getattr(res, "kpis", None)
        if isinstance(k_obj, dict):
            k.update(k_obj)

        cof_b = float(k.get("cof_fs_base", b.get("Cost of Funds %", 0.0)))
        cof_a = float(k.get("cof_fs_scn", a.get("Cost of Funds %", 0.0)))

        kpi_items = [
            {"label": "NII (Horizon)", "before": format_large_number(b["NII"]), "after": format_large_number(a["NII"]), "delta": _fmt_delta(a["NII"] - b["NII"])},
            {"label": "Net Interest Margin (NIM %)", "before": f"{b['NIM %']:.2f}%", "after": f"{a['NIM %']:.2f}%", "delta": _fmt_delta((a["NIM %"] - b["NIM %"]) * 100, " bps")},
            {"label": "Yield on Earning Assets %", "before": f"{k.get('yield_ea_base', 0.0):.2f}%", "after": f"{k.get('yield_ea_scn', 0.0):.2f}%", "delta": f"{(k.get('yield_ea_scn', 0.0) - k.get('yield_ea_base', 0.0)) * 100:+.1f} bps"},
            {"label": "Cost of Funds %", "before": f"{cof_b:.2f}%", "after": f"{cof_a:.2f}%", "delta": f"{(cof_a - cof_b) * 100:+.1f} bps"},
            {"label": "Net Interest Spread %", "before": f"{k.get('spread_base', 0.0):.2f}%", "after": f"{k.get('spread_scn', 0.0):.2f}%", "delta": f"{(k.get('spread_scn', 0.0) - k.get('spread_base', 0.0)) * 100:+.1f} bps"},
            {"label": "Total P&L & Value Impact", "before": "0.00", "after": format_large_number(a["PnL Total Impact"]), "delta": format_large_number(a["PnL Total Impact"])},
            {"label": "LCR Proxy % (30d Stress)", "before": f"{b.get('CF LCR %', 0.0):,.1f}%", "after": f"{a.get('CF LCR %', 0.0):,.1f}%", "delta": _fmt_delta(a.get("CF LCR %", 0.0) - b.get("CF LCR %", 0.0), " pp")},
            {"label": "NSFR Proxy % (Basel III)", "before": f"{k.get('nsfr_base', 0.0):.1f}%", "after": f"{k.get('nsfr_scn', 0.0):.1f}%", "delta": f"{k.get('nsfr_scn', 0.0) - k.get('nsfr_base', 0.0):+.2f} pp"},
            {"label": "Liquid Assets / Deposits %", "before": f"{k.get('liq_dep_base', 0.0):.2f}%", "after": f"{k.get('liq_dep_scn', 0.0):.2f}%", "delta": f"{k.get('liq_dep_scn', 0.0) - k.get('liq_dep_base', 0.0):+.2f} pp"},
            {"label": "CASA Ratio %", "before": f"{k.get('casa_ratio_base', 0.0):.2f}%", "after": f"{k.get('casa_ratio_scn', 0.0):.2f}%", "delta": f"{k.get('casa_ratio_scn', 0.0) - k.get('casa_ratio_base', 0.0):+.2f} pp"},
            {"label": "Loan-to-Deposit Ratio (LDR %)", "before": f"{b['Loan Deposit %']:.1f}%", "after": f"{a['Loan Deposit %']:.1f}%", "delta": _fmt_delta(a["Loan Deposit %"] - b["Loan Deposit %"], " pp")},
            {"label": "Return on Equity (ROE %)", "before": f"{k.get('roe_base', 0.0):.2f}%", "after": f"{k.get('roe_scn', 0.0):.2f}%", "delta": f"{k.get('roe_scn', 0.0) - k.get('roe_base', 0.0):+.2f} pp"},
            {"label": "Return on Assets (ROA %)", "before": f"{k.get('roa_base', 0.0):.2f}%", "after": f"{k.get('roa_scn', 0.0):.2f}%", "delta": f"{k.get('roa_scn', 0.0) - k.get('roa_base', 0.0):+.2f} pp"},
            {"label": "Equity / Assets %", "before": f"{k.get('eq_ast_base', 0.0):.2f}%", "after": f"{k.get('eq_ast_scn', 0.0):.2f}%", "delta": f"{k.get('eq_ast_scn', 0.0) - k.get('eq_ast_base', 0.0):+.2f} pp"},
            {"label": "Sovereign Debt / Assets %", "before": f"{k.get('sov_ast_base', 0.0):.2f}%", "after": f"{k.get('sov_ast_scn', 0.0):.2f}%", "delta": f"{k.get('sov_ast_scn', 0.0) - k.get('sov_ast_base', 0.0):+.2f} pp"},
            {"label": "Cost-to-Income Ratio %", "before": f"{k.get('cti_base', 0.0):.2f}%", "after": f"{k.get('cti_scn', 0.0):.2f}%", "delta": f"{k.get('cti_scn', 0.0) - k.get('cti_base', 0.0):+.2f} pp"},
            {"label": "Total Assets (EGP Mn)", "before": f"{k.get('total_assets_base_mn', 0.0):,.0f} M", "after": f"{k.get('total_assets_scn_mn', 0.0):,.0f} M", "delta": f"{k.get('total_assets_scn_mn', 0.0) - k.get('total_assets_base_mn', 0.0):+,.0f} M"},
            {"label": "Annualized Net Profit (EGP Mn)", "before": f"{k.get('base_net_profit_mn', 0.0):,.0f} M", "after": f"{k.get('scn_net_profit_mn', 0.0):,.0f} M", "delta": f"{k.get('delta_net_profit_mn', 0.0):+,.0f} M"},
        ]
        all_kpi_names = [item["label"] for item in kpi_items]

        available_charts = {
            "NII Bridge (Base vs Scenario)": fig_nii_bridge,
            "P&L and Value Impact Components": fig_pnl_comp,
            "Portfolio Before vs After": fig_portfolio,
        }
        if fig_currency is not None:
            available_charts["Balance by Currency (EGP)"] = fig_currency

        with st.form(key="report_builder_form"):
            col1, col2 = st.columns(2)
            with col1:
                chosen_kpis = st.multiselect("1. Choose KPIs to include", options=all_kpi_names, default=[], key="rep_kpis")
            with col2:
                chosen_charts = st.multiselect("2. Choose Charts to include", options=list(available_charts.keys()), default=[], key="rep_charts")

            o1, o2, o3 = st.columns(3)
            inc_comp = o1.checkbox("Include Statements & Comparison Tables", value=True, key="rep_inc_comp")
            inc_ai = o2.checkbox("Include Management / AI Insight", value=True, key="rep_inc_ai")
            rep_format = o3.radio("3. Output Format", options=["PDF", "PPTX", "Both (PDF & PPTX)"], horizontal=True, key="rep_fmt")

            submitted = st.form_submit_button("⚙️ Generate Report File(s)", use_container_width=True)

        if submitted:
            with st.spinner("Generating executive report file(s)..."):
                selected_kpis_payload = [item for item in kpi_items if item["label"] in chosen_kpis]
                selected_charts_payload = {name: available_charts[name] for name in chosen_charts}

                ai_text = st.session_state.get("ai_insight_dashboard_current")
                run_id = st.session_state.get("last_run_id")

                if not ai_text and ctx.engine is not None:
                    try:
                        if run_id:
                            q_sql = text("SELECT ai_explanation FROM simulation_runs WHERE run_id = :rid")
                            df_run = pd.read_sql(q_sql, ctx.engine, params={"rid": int(run_id)})
                        else:
                            q_sql = text("""
                                SELECT sr.run_id, sr.ai_explanation
                                FROM simulation_runs sr
                                JOIN scenarios s ON sr.scenario_id = s.scenario_id
                                WHERE s.scenario_name = :sname
                                ORDER BY sr.run_id DESC LIMIT 1
                            """)
                            df_run = pd.read_sql(q_sql, ctx.engine, params={"sname": str(res.name)})
                        if not df_run.empty:
                            if not run_id and "run_id" in df_run.columns:
                                run_id = int(df_run.iloc[0]["run_id"])
                            if pd.notna(df_run.iloc[0]["ai_explanation"]):
                                ai_text = str(df_run.iloc[0]["ai_explanation"]).strip()
                    except Exception:
                        pass

                if not ai_text:
                    ai_text = "\n".join(insights)

                ai_payload = ai_text if inc_ai else None
                comp_payload = res.comparison.drop(columns=["Code"], errors="ignore") if inc_comp else None
                bs_payload = getattr(res, "balance_sheet_df", None) if inc_comp else None
                is_payload = getattr(res, "income_statement_df", None) if inc_comp else None

                safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(res.name))
                base_filename = f"NBE_ALM_Report_{safe_name}_{res.sim_date:%Y%m%d}"
                win_choice = str(st.session_state.get("fs_sym_window_n", "±2 Years"))
                win_years = 3 if "3" in str(win_choice) else 2
                st.session_state["fs_active_window_years"] = win_years
                active_win_years = int(st.session_state.get("fs_active_window_years", 2))

                pdf_b, pptx_b = None, None
                if rep_format in ("PDF", "Both (PDF & PPTX)"):
                    pdf_b = generate_pdf_report(
                        res.name, res.type, res.sim_date.strftime("%Y-%m-%d"), res.horizon,
                        selected_kpis_payload, selected_charts_payload, res.comparison, res.risk, ai_text,
                        params=res.params, engine=ctx.engine, run_id=run_id,
                        balance_sheet_df=bs_payload, income_statement_df=is_payload,
                        window_years=active_win_years,
                    )
                if rep_format in ("PPTX", "Both (PDF & PPTX)"):
                    pptx_b = generate_pptx_report(
                        res.name, res.type, res.sim_date.strftime("%Y-%m-%d"), res.horizon,
                        selected_kpis_payload, selected_charts_payload, res.comparison, ai_text,
                        params=res.params, engine=ctx.engine, run_id=run_id,
                        balance_sheet_df=bs_payload, income_statement_df=is_payload,
                        window_years=active_win_years,
                    )

                st.session_state["rep_pdf_bytes"] = pdf_b
                st.session_state["rep_pptx_bytes"] = pptx_b
                st.session_state["rep_ready_mode"] = rep_format
                st.session_state["rep_ready_name"] = base_filename
            st.success("Report ready for download!")

        r_mode = st.session_state.get("rep_ready_mode")
        r_name = st.session_state.get("rep_ready_name", "NBE_ALM_Report")
        r_pdf = st.session_state.get("rep_pdf_bytes")
        r_pptx = st.session_state.get("rep_pptx_bytes")

        if r_mode == "PDF" and r_pdf:
            st.download_button("⬇️ Download PDF Report (.pdf)", data=r_pdf, file_name=f"{r_name}.pdf", mime="application/pdf", use_container_width=True, key="dl_pdf_single")
        elif r_mode == "PPTX" and r_pptx:
            st.download_button("⬇️ Download PowerPoint (.pptx)", data=r_pptx, file_name=f"{r_name}.pptx", mime="application/vnd.openxmlformats-officedocument.presentationml.presentation", use_container_width=True, key="dl_pptx_single")
        elif r_mode == "Both (PDF & PPTX)" and r_pdf and r_pptx:
            zip_b = generate_zip_bundle(r_pdf, r_pptx, r_name)
            d1, d2, d3 = st.columns(3)
            d1.download_button("📦 Download Both (ZIP)", data=zip_b, file_name=f"{r_name}_Bundle.zip", mime="application/zip", use_container_width=True, key="dl_zip_bundle")
            d2.download_button("⬇️ Download PDF Only", data=r_pdf, file_name=f"{r_name}.pdf", mime="application/pdf", use_container_width=True, key="dl_pdf_part")
            d3.download_button("⬇️ Download PPTX Only", data=r_pptx, file_name=f"{r_name}.pptx", mime="application/vnd.openxmlformats-officedocument.presentationml.presentation", use_container_width=True, key="dl_pptx_part")


def page_dashboard(ctx, history):
    """Main entry point for rendering the Executive Dashboard page."""
    res = current_session_scenario(ctx)
    if not history.empty and not ctx.scenarios.empty:
        options = {"Current session scenario": 0}
        for _, r in history.sort_values("scenario_id").iterrows():
            options[f"{r.get('scenario_name', 'Scenario')} (ID {int(r['scenario_id'])})"] = int(r["scenario_id"])
        choice = st.selectbox("Scenario shown on the dashboard", list(options.keys()), key="dashboard_scenario_selector")
        if options[choice] != 0:
            row = ctx.scenarios[pd.to_numeric(ctx.scenarios["scenario_id"]) == options[choice]].iloc[0]
            res = evaluate_saved_scenario(ctx, row)

    if ctx.positions.empty:
        st.warning("No positions loaded. Connect to the database in Data & Settings.")
        return

    # Pull FX and Gold (24K & 21K) valuation rates dynamically from market_data by instrument_code
    vr = valuation_rates(ctx.market_data, as_of=getattr(res, "base_as_of", None))
    inst_map = dict(zip(vr["instrument_code"].astype(str).str.upper(), vr["market_rate"])) if not vr.empty and "instrument_code" in vr.columns else {}
    ccy_map = dict(zip(vr["base_currency"].astype(str).str.upper(), vr["market_rate"])) if not vr.empty else {}

    fx_shock = float(res.params.get("fx_shock_pct", 0.0))
    gold_shock = float(res.params.get("gold_shock_pct", 0.0))

    base_fx_rate = float(inst_map.get("FX_USD_EGP", ccy_map.get("USD", 0.0)))
    base_gold24_rate = float(inst_map.get("GOLD_24K_EGP", ccy_map.get("GOLD", 0.0)))
    base_gold21_rate = float(inst_map.get("GOLD_21K_EGP", ccy_map.get("GOLD_21K", 0.0)))

    affected_fx_rate = base_fx_rate * (1.0 + fx_shock / 100.0)
    affected_gold24_rate = base_gold24_rate * (1.0 + gold_shock / 100.0)
    affected_gold21_rate = base_gold21_rate * (1.0 + gold_shock / 100.0)

    fx_color = "#28a745" if fx_shock > 0 else ("#dc3545" if fx_shock < 0 else "#666")
    gold_color = "#28a745" if gold_shock > 0 else ("#dc3545" if gold_shock < 0 else "#666")

    gold21_badge_html = (
        f'<div style="background: rgba(255,255,255,0.8); padding: 6px 12px; border-radius: 8px; border: 1px solid #e0e0e0; font-size: 13px;">'
        f'<span style="color: #555; font-weight: 600;">🪙 Gold (21K):</span> '
        f'<span style="color: #666;">{base_gold21_rate:,.2f} ➔</span> '
        f'<span style="color: #111; font-weight: bold;">{affected_gold21_rate:,.2f}</span> '
        f'<span style="color: {gold_color}; font-size: 11px; font-weight: bold;">({gold_shock:+.1f}%)</span>'
        f'</div>'
        if base_gold21_rate > 0 else ""
    )

    st.markdown(
        f'''<div class="scn-banner">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap: 15px;">
                <div>
                    <div class="scn-title">{res.name}</div>
                    <div class="scn-meta">Simulation date {res.sim_date:%Y-%m-%d} &nbsp;|&nbsp; Horizon {res.horizon} days
                    (to {(res.sim_date + pd.Timedelta(days=res.horizon)):%Y-%m-%d}) &nbsp;|&nbsp; EGP-equivalent</div>
                </div>
                <div style="display:flex; align-items:center; gap: 12px; flex-wrap:wrap;">
                    <div style="background: rgba(255,255,255,0.8); padding: 6px 12px; border-radius: 8px; border: 1px solid #e0e0e0; font-size: 13px;">
                        <span style="color: #555; font-weight: 600;">💱 USD:</span>
                        <span style="color: #666;">{base_fx_rate:,.2f} ➔</span>
                        <span style="color: #111; font-weight: bold;">{affected_fx_rate:,.2f}</span>
                        <span style="color: {fx_color}; font-size: 11px; font-weight: bold;">({fx_shock:+.1f}%)</span>
                    </div>
                    <div style="background: rgba(255,255,255,0.8); padding: 6px 12px; border-radius: 8px; border: 1px solid #e0e0e0; font-size: 13px;">
                        <span style="color: #555; font-weight: 600;">🪙 Gold (24K):</span>
                        <span style="color: #666;">{base_gold24_rate:,.2f} ➔</span>
                        <span style="color: #111; font-weight: bold;">{affected_gold24_rate:,.2f}</span>
                        <span style="color: {gold_color}; font-size: 11px; font-weight: bold;">({gold_shock:+.1f}%)</span>
                    </div>
                    {gold21_badge_html}
                </div>
            </div>
            {active_driver_chips(res.params)}
        </div>''',
        unsafe_allow_html=True,
    )

    _kpi_grid(res)

    attr = attribution(ctx, res)
    insights = management_insights(res.before, res.after, attr)

    b, a = res.before, res.after
    fig_nii_bridge = _waterfall(
        ["Base NII", "Rate effect", "Volume / FX effect"],
        [b["NII"], a["NII Rate Effect"], a["NII Volume Effect"]],
        "NII bridge: base -> scenario (EGP)", "Scenario NII",
    )
    comp = pnl_components(b, a)
    fig_pnl_comp = _waterfall(comp["Component"].tolist(), comp["Impact"].tolist(), "P&L and value impact (EGP)", "Total impact")

    port = pd.DataFrame({
        "Portfolio": ["Loans", "Deposits", "Funding", "Investments"],
        "Before": [b["Loans"], b["Deposits"], b["Funding"], b["Investments"]],
        "After": [a["Loans"], a["Deposits"], a["Funding"], a["Investments"]],
    })
    fig_portfolio = _before_after_bar(port, "Portfolio", "Before", "After", "Portfolio before vs after (EGP)")
    expo = exposure_tables(res.detail)
    fig_currency = _before_after_bar(expo["currency"], "Model Currency", "Before Balance", "After Balance", "Balance by currency (EGP)") if "currency" in expo else None

    tabs = st.tabs([
        "1. 🏛️ Financial Statements (Outlined)",
        "2. 📊 Overview & AI Explanation",
        "3. 💼 Balances & Exposures",
        "4. 💧 Liquidity & Row-by-Row Audit",
    ])

    with tabs[0]:
        _render_fs_tab(res)

    with tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(fig_nii_bridge, use_container_width=True, key="dash_fig_nii_bridge")
        with c2:
            st.plotly_chart(fig_pnl_comp, use_container_width=True, key="dash_fig_pnl_comp")
        if not attr.empty:
            st.markdown("#### Driver Attribution (Standalone vs Combined)")
            show_df(attr)
        st.markdown("#### KPI Comparison Table")
        show_df(res.comparison.drop(columns=["Code", "Source"], errors="ignore"))
        render_ai_scenario_panel(ctx, res, res.name, res.type, panel_key="dashboard_current", insights=insights)

    with tabs[2]:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(fig_portfolio, use_container_width=True, key="dash_fig_portfolio")
        with c2:
            if fig_currency is not None:
                st.plotly_chart(fig_currency, use_container_width=True, key="dash_fig_currency")
        top_pos_df = position_sensitivity(res.detail, top=10)
        if not top_pos_df.empty:
            st.markdown("#### Top 10 Positions by Total Financial Impact")
            show_df(top_pos_df)

    with tabs[3]:
        lad = liquidity_ladder(ctx.cashflows, res.params, res.sim_date, res.horizon, stressed=True)
        if not lad.empty:
            fig_lad = go.Figure()
            fig_lad.add_bar(x=lad["Bucket"], y=lad["Inflow"], name="Inflow", marker_color=NBE_GREEN)
            fig_lad.add_bar(x=lad["Bucket"], y=-lad["Outflow"], name="Outflow", marker_color=NBE_ORANGE)
            fig_lad.add_trace(go.Scatter(x=lad["Bucket"], y=lad["Cumulative Net"], name="Cumulative net", mode="lines+markers", line=dict(color=NBE_DARK_GREEN, width=3)))
            fig_lad.update_layout(title="Maturity Ladder by Bucket (EGP)", barmode="relative")
            st.plotly_chart(style_nbe_figure(fig_lad, 360), use_container_width=True, key="dash_fig_liquidity_ladder")

        st.markdown("#### Detailed Row-by-Row Scenario Impact Analysis")
        full_row_table = build_full_row_by_row_scenario_table(res.detail, res.params, res.horizon, sim_date=res.sim_date)
        st.dataframe(full_row_table, use_container_width=True)
        win_val = st.session_state.get("fs_sym_window_n", 2)
        active_win_years = 3 if "3" in str(win_val) else 2
        excel_bytes = export_row_by_row_with_formulas_xlsx(
            full_row_table,
            horizon_days=res.horizon,
            df_bs=getattr(res, "balance_sheet_df", None),
            df_is=getattr(res, "income_statement_df", None),
            res=res,
            window_years=active_win_years,
            cashflows=ctx.cashflows,
        )
        st.download_button(
            label="⬇️ Download Full ALM & Financial Statements Workbook (4 Sheets with Live Excel Formulas)",
            data=excel_bytes,
            file_name="NBE_ALM_Full_Model_and_Statements_Formulas.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_row_tab_xlsx_formulas",
        )

    _render_report_export_section(
        ctx, res, b, a, fig_nii_bridge, fig_pnl_comp, fig_portfolio, fig_currency, insights
    )