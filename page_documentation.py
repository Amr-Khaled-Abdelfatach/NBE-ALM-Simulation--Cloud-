# page_documentation.py
"""
Interactive ALM Methodology, Live 20-KPI Formula Audit, 10 Scenario Drivers,
Financial Statements Architecture, and SQL View Documentation Page.
100% Data-Driven: Dynamically reflects all parameters from `config.py` and live
values from the current simulation session without any hardcoded constants or legacy keys.
"""
import numpy as np
import pandas as pd
import streamlit as st

from config import (
    CREDIT_COST_BPS_PER_GDP_DROP_PP,
    CREDIT_COST_BPS_PER_UNEMP_PP,
    DAYS_PER_YEAR,
    FLAT_CASHFLOWS_VIEW,
    FLAT_POSITIONS_VIEW,
    LCR_WINDOW_DAYS,
    MACRO_CODES,
    PAGE_DOCS,
    PARAM_SPECS,
    REQUIRED_TABLES,
    UNIFIED_BS_VIEW,
    VALUATION_INSTRUMENTS,
)
from ui_common import (
    current_session_scenario,
    doc_button,
    format_large_number,
    kpi_card,
    show_df,
    tone_for,
)


def _fmt_pct(val, decimals: int = 2) -> str:
    """Safely format a numeric percentage value."""
    if val is None or pd.isna(val):
        return "N/A"
    return f"{float(val):,.{decimals}f}%"


def _fmt_mn(val) -> str:
    """Safely format EGP Million values."""
    if val is None or pd.isna(val):
        return "N/A"
    return f"{float(val):,.1f} M EGP"


def page_documentation(ctx):
    """Render the comprehensive, live-linked ALM documentation and methodology guide."""
    st.markdown(
        '<div class="section-header">📘 ALM Methodology, 20-KPI Formula Reference & SQL Architecture</div>',
        unsafe_allow_html=True,
    )
    doc_button(PAGE_DOCS.get("Documentation", ""), "Open External Online Documentation")

    res = current_session_scenario(ctx)
    b, a, p = res.before, res.after, res.params
    k = dict(a) if isinstance(a, dict) else {}
    k_obj = getattr(res, "kpis", None)
    if isinstance(k_obj, dict):
        k.update(k_obj)

    meta = getattr(res, "fs_meta", {}) or k.get("fs_meta", {})

    # ------------------------------------------------------------------------
    # Top Live Engine Status Cards (4 Executive Anchors)
    # ------------------------------------------------------------------------
    st.markdown("##### ⚡ Live Engine State (Synchronized with Current Session Scenario)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        delta_nii = float(a.get("NII", 0.0) - b.get("NII", 0.0))
        kpi_card(
            "SCENARIO NII (HORIZON)",
            format_large_number(a.get("NII", 0.0)),
            f"{delta_nii:+,.0f} EGP" if abs(delta_nii) > 1e-4 else None,
            tone_for(delta_nii, True),
            f"Base: {format_large_number(b.get('NII', 0.0))} ({res.horizon}d)",
        )
    with c2:
        kpi_card(
            "SCENARIO NIM %",
            _fmt_pct(a.get("NIM %", 0.0)),
            f"{(a.get('NIM %', 0.0) - b.get('NIM %', 0.0)) * 100:+.1f} bps",
            tone_for(a.get("NIM %", 0.0) - b.get("NIM %", 0.0), True),
            f"Base NIM: {_fmt_pct(b.get('NIM %', 0.0))}",
        )
    with c3:
        lcr_a = a.get("CF LCR %", np.nan)
        lcr_b = b.get("CF LCR %", np.nan)
        kpi_card(
            f"LCR PROXY ({LCR_WINDOW_DAYS}D STRESS)",
            _fmt_pct(lcr_a, 1),
            f"{lcr_a - lcr_b:+.1f} pp" if pd.notna(lcr_a) and pd.notna(lcr_b) else None,
            tone_for((lcr_a - lcr_b) if pd.notna(lcr_a) and pd.notna(lcr_b) else 0.0, True),
            f"NSFR Proxy: {_fmt_pct(k.get('nsfr_scn', 0.0), 1)}",
        )
    with c4:
        ast_s = float(k.get("total_assets_scn_mn", 0.0))
        ast_b = float(k.get("total_assets_base_mn", 0.0))
        kpi_card(
            "TOTAL ASSETS (SCENARIO)",
            f"{ast_s:,.0f} M",
            f"{ast_s - ast_b:+,.0f} M EGP" if abs(ast_s - ast_b) > 1e-4 else None,
            "neutral",
            f"Base Period: {meta.get('base_date_str', 'Latest SQL')}",
        )

    tabs = st.tabs([
        "1. 📐 All 20 Dashboard KPIs & Live Formulas",
        "2. 🎛️ 10 Database Scenario Drivers",
        "3. 🏛️ Financial Statements, IFRS 9 & ML Forecast",
        "4. 🗄️ SQL Views (24K/21K Gold & Schema)",
    ])

    # ========================================================================
    # TAB 1: All 20 Dashboard KPIs with Mathematical Formulas & Live Values
    # ========================================================================
    with tabs[0]:
        st.markdown("#### Complete 20-KPI Reference Table (5 Dashboard Rows × 4 Cards)")
        st.caption(
            "Every KPI below is computed dynamically from SQL views (`vw_alm_flat_table`, `vw_alm_cashflows_flat`) "
            "and `financial_statements_flat` with zero hardcoded position or product IDs."
        )

        cof_b = float(k.get("cof_fs_base", b.get("Cost of Funds %", 0.0)))
        cof_a = float(k.get("cof_fs_scn", a.get("Cost of Funds %", 0.0)))

        kpi_doc_rows = [
            # Row 1
            (
                "Row 1 — Card 1",
                "NII (Horizon)",
                "∑ Interest_Assets − ∑ Interest_Liabilities",
                format_large_number(b.get("NII", 0.0)),
                format_large_number(a.get("NII", 0.0)),
                "Accrued over simulation horizon with rollover at maturity.",
            ),
            (
                "Row 1 — Card 2",
                "NIM %",
                "[NII × (365 / Horizon) ÷ Earning Assets] × 100",
                _fmt_pct(b.get("NIM %", 0.0)),
                _fmt_pct(a.get("NIM %", 0.0)),
                "Earning Assets dynamically exclude `BS_AST_CASH` (Cash & CBE Reserve).",
            ),
            (
                "Row 1 — Card 3",
                "Yield on Earning Assets %",
                "(Annualized Interest Income ÷ Average Earning Assets) × 100",
                _fmt_pct(k.get("yield_ea_base", 0.0)),
                _fmt_pct(k.get("yield_ea_scn", 0.0)),
                "Uses average earning assets between prior fiscal year and current period.",
            ),
            (
                "Row 1 — Card 4",
                "Cost of Funds % & Spread",
                "(Annualized Interest Expense ÷ Costed Liabilities) × 100",
                f"CoF: {cof_b:.2f}% | Sprd: {k.get('spread_base', 0.0):.2f}%",
                f"CoF: {cof_a:.2f}% | Sprd: {k.get('spread_scn', 0.0):.2f}%",
                "Excludes non-interest-bearing deposits (`Rate_Type = Non-Interest`).",
            ),
            # Row 2
            (
                "Row 2 — Card 5",
                "Total P&L Impact",
                "ΔNII + FX/Gold Revaluation + Fair-Value MTM − Macro Credit Cost",
                "0.00",
                format_large_number(a.get("PnL Total Impact", 0.0)),
                "Combines horizon earnings impact, valuation gains/losses, and macro ECL.",
            ),
            (
                "Row 2 — Card 6",
                "Repricing Gap (12M)",
                "Repricing Assets (RSA) − Repricing Liabilities (RSL)",
                format_large_number(b.get("Repricing Gap", 0.0)),
                format_large_number(a.get("Repricing Gap", 0.0)),
                f"±100bp NII Sensitivity = {k.get('nii_sens_100bp', 0.0):+.2f}% of Base NII.",
            ),
            (
                "Row 2 — Card 7",
                "ΔEVE / Equity (FVOCI %)",
                "[−(ΔRate × Duration × FVOCI_Balance) ÷ Total Equity] × 100",
                "0.00%",
                f"{k.get('eve_eq_scn', 0.0):+.2f}% ({_fmt_mn(k.get('fvoci_mtm_mn', 0.0))})",
                "Applies strictly to `Accounting_Class IN ('FVOCI','FVTPL')` excluding Gold.",
            ),
            (
                "Row 2 — Card 8",
                "Cost-to-Income & Net Profit",
                "CTI % = (|G&A Expenses| ÷ Operating Income) × 100",
                f"CTI: {k.get('cti_base', 0.0):.2f}% | NP: {_fmt_mn(k.get('base_net_profit_mn', 0.0))}",
                f"CTI: {k.get('cti_scn', 0.0):.2f}% | NP: {_fmt_mn(k.get('scn_net_profit_mn', 0.0))}",
                "Net Profit adjusts dynamically for effective income tax rate.",
            ),
            # Row 3
            (
                "Row 3 — Card 9",
                f"LCR Proxy ({LCR_WINDOW_DAYS}d Stress)",
                "HQLA Stock ÷ Max(30d Outflows + Runoff − 30d Inflows, 1) × 100",
                _fmt_pct(b.get("CF LCR %", np.nan), 1),
                _fmt_pct(a.get("CF LCR %", np.nan), 1),
                "HQLA identified via `HQLA_Flag = 1`, `BS_AST_CASH`, or Sovereign/Gold family.",
            ),
            (
                "Row 3 — Card 10",
                "Liquid Assets / Deposits %",
                "[∑ Balance(Is_Liquid_Asset = 1) ÷ Customer Deposits] × 100",
                _fmt_pct(k.get("liq_dep_base", 0.0)),
                _fmt_pct(k.get("liq_dep_scn", 0.0)),
                "Driven by the `is_liquid_asset` column on `products` (Cash, Banks, T-Bills).",
            ),
            (
                "Row 3 — Card 11",
                "CASA Ratio %",
                "[(Current Accounts + Savings Accounts) ÷ Total Deposits] × 100",
                _fmt_pct(k.get("casa_ratio_base", 0.0)),
                _fmt_pct(k.get("casa_ratio_scn", 0.0)),
                "Identified dynamically via `Product_Type` / `Product_Family` (`CASA`).",
            ),
            (
                "Row 3 — Card 12",
                "Loan / Deposit Ratio (LDR %)",
                "(Total Net Customer Loans ÷ Total Customer Deposits) × 100",
                _fmt_pct(b.get("Loan Deposit %", 0.0), 1),
                _fmt_pct(a.get("Loan Deposit %", 0.0), 1),
                "Reflects loan growth, deposit growth, deposit run-off, and FX revaluation.",
            ),
            # Row 4
            (
                "Row 4 — Card 13",
                "Return on Equity (ROE %)",
                "(Annualized Net Profit After Tax ÷ Average Total Equity) × 100",
                _fmt_pct(k.get("roe_base", 0.0)),
                _fmt_pct(k.get("roe_scn", 0.0)),
                "Denominator averages prior year SQL equity and current period equity.",
            ),
            (
                "Row 4 — Card 14",
                "Return on Assets (ROA %)",
                "(Annualized Net Profit After Tax ÷ Average Total Assets) × 100",
                _fmt_pct(k.get("roa_base", 0.0)),
                _fmt_pct(k.get("roa_scn", 0.0)),
                "Denominator averages prior year SQL assets and current period assets.",
            ),
            (
                "Row 4 — Card 15",
                "Equity / Assets %",
                "(Total Equity ÷ Total Assets) × 100",
                _fmt_pct(k.get("eq_ast_base", 0.0)),
                _fmt_pct(k.get("eq_ast_scn", 0.0)),
                "Captures balance sheet capital leverage including FVOCI reserve & profit shifts.",
            ),
            (
                "Row 4 — Card 16",
                "Net Open FX Pos / Equity %",
                "[(Total FCY Assets − Total FCY Liabilities) ÷ Total Equity] × 100",
                f"{k.get('nop_eq_base', 0.0):+.2f}% ({_fmt_mn(k.get('nop_base_mn', 0.0))})",
                f"{k.get('nop_eq_scn', 0.0):+.2f}% ({_fmt_mn(k.get('nop_scn_mn', 0.0))})",
                "Includes all non-EGP foreign currencies (`Currency != 'EGP'` & `Basis = 'FX'`).",
            ),
            # Row 5
            (
                "Row 5 — Card 17",
                "NSFR Proxy (Basel III %)",
                "[(Total Equity + ∑ Bal × NSFR_ASF_Weight) ÷ (∑ Bal × NSFR_RSF_Weight)] × 100",
                _fmt_pct(k.get("nsfr_base", 0.0), 1),
                _fmt_pct(k.get("nsfr_scn", 0.0), 1),
                "Reads `nsfr_asf_weight` and `nsfr_rsf_weight` per product directly from SQL.",
            ),
            (
                "Row 5 — Card 18",
                "Sovereign Debt / Assets %",
                "[(FVOCI Securities + Amortised Cost Securities) ÷ Total Assets] × 100",
                _fmt_pct(k.get("sov_ast_base", 0.0)),
                _fmt_pct(k.get("sov_ast_scn", 0.0)),
                "Tracks sovereign debt concentration (`BS_AST_FVOCI` + `BS_AST_AMORT`).",
            ),
            (
                "Row 5 — Card 19",
                "FX Wholesale / Liabilities %",
                "(Other Borrowings `BS_LIAB_BORROW` ÷ Total Liabilities) × 100",
                _fmt_pct(k.get("wsf_liab_base", 0.0)),
                _fmt_pct(k.get("wsf_liab_scn", 0.0)),
                "Measures reliance on foreign wholesale borrowings.",
            ),
            (
                "Row 5 — Card 20",
                "Contingent (Off-BS) / Equity %",
                "(Contingent Liabilities `BS_OFF_CONTINGENT` ÷ Total Equity) × 100",
                _fmt_pct(k.get("off_eq_base", 0.0), 1),
                _fmt_pct(k.get("off_eq_scn", 0.0), 1),
                "Tracks LGs & LCs exposure relative to capital base.",
            ),
        ]

        show_df(
            pd.DataFrame(
                kpi_doc_rows,
                columns=[
                    "Dashboard Position",
                    "KPI Name",
                    "Master Formula",
                    "Live Base (Before)",
                    "Live Scenario (After)",
                    "Data-Driven Rule (Zero Hardcoding)",
                ],
            ),
            height=680,
        )

    # ========================================================================
    # TAB 2: All 10 Database Scenario Drivers
    # ========================================================================
    with tabs[1]:
        st.markdown(f"#### All {len(PARAM_SPECS)} Scenario Drivers (`scenarios` Table Columns)")
        st.caption(
            "Every scenario parameter below is read dynamically from `PARAM_SPECS` in `config.py` "
            "and maps 1-to-1 to a dedicated column in the MySQL `scenarios` table."
        )

        db_col_map = {
            "rate_change_pp": "policy_rate_shock_bps (stored in bps; 1 pp = 100 bps)",
            "fx_shock_pct": "fx_shock_pct",
            "gold_shock_pct": "gold_shock_pct",
            "wsf_spread_shock_bps": "wsf_spread_shock_bps",
            "deposit_runoff_pct": "deposit_runoff_pct",
            "loan_growth_pct": "loan_growth_pct",
            "deposit_growth_pct": "deposit_growth_pct",
            "inflation_shock_pct": "inflation_shock_pct",
            "unemployment_shock_pct": "unemployment_shock_pct",
            "gdp_shock_pct": "gdp_shock_pct",
        }

        driver_rows = []
        for idx, (key, (label, unit, lo, hi, default, step, group)) in enumerate(PARAM_SPECS.items(), start=1):
            cur_val = float(p.get(key, default))
            driver_rows.append({
                "#": idx,
                "Driver Group": group,
                "Parameter Label": label,
                "Engine Key": key,
                "MySQL `scenarios` Column": db_col_map.get(key, key),
                "Unit": unit,
                "Allowed Range": f"[{lo:g}, {hi:g}]",
                "Step": step,
                "Current Session Value": f"{cur_val:+g} {unit}",
            })
        show_df(pd.DataFrame(driver_rows), height=420)

        st.markdown("##### Direct Transmission Channels of the 10 Drivers")
        st.markdown(
            f"""
            - **Policy / Market Rate Shock (`rate_change_pp`):** Reprices floating-rate positions on their `Next_Repricing_Date` and fixed-rate positions maturing within the horizon (`ROLLOVER_AT_MATURITY = True`), while revaluing `FVOCI` debt securities via modified duration.
            - **FX Shock (`fx_shock_pct`):** Revalues all positions and scheduled cash flows where `Valuation_Basis = 'FX'`, generating both an NII volume effect and a direct FX revaluation P&L effect (`IS_TRADING`).
            - **Gold Price Shock (`gold_shock_pct`):** Revalues both **24K** (`GOLD_24K_EGP`) and **21K** (`GOLD_21K_EGP`) gold positions where `Valuation_Basis = 'Gold'`, flowing into `BS_AST_FVTPL` and trading/revaluation income.
            - **FX Wholesale Funding Spread Shock (`wsf_spread_shock_bps`):** Adds a funding spread shock (in bps) specifically to foreign wholesale borrowings (`BS_Line_Code = 'BS_LIAB_BORROW'`).
            - **Deposit Run-off (`deposit_runoff_pct`):** Reduces customer deposit balances (`BS_LIAB_DEP`) and adds an immediate 30-day stressed cash outflow (`CF Runoff Amount`) to the `LCR Proxy` denominator.
            - **Loan & Deposit Growth (`loan_growth_pct`, `deposit_growth_pct`):** Scales loan (`BS_AST_LOANS`) and deposit (`BS_LIAB_DEP`) volumes and their corresponding interest accruals.
            - **Macroeconomic Shocks (`inflation_shock_pct`, `unemployment_shock_pct`, `gdp_shock_pct`):** Updates real policy rates and computes Expected Credit Loss (`ECL`) charges at **`+{CREDIT_COST_BPS_PER_UNEMP_PP:.0f} bps`** of loans per `+1 pp` unemployment rise and **`+{CREDIT_COST_BPS_PER_GDP_DROP_PP:.0f} bps`** per `−1 pp` GDP drop.
            """
        )

    # ========================================================================
    # TAB 3: Financial Statements, IFRS 9 & ML Forecast Methodology
    # ========================================================================
    with tabs[2]:
        st.markdown("#### Dynamic Financial Statements Engine (`fs_engine.py`)")
        st.markdown(
            f"""
            1. **Zero-Hardcoded Period Detection (`_resolve_dynamic_periods`):**
               - Inspects `financial_statements_flat.report_date` to automatically resolve the two historical comparison years (**`{meta.get('hist_col_1', 'Hist Y-2')}`**, **`{meta.get('hist_col_2', 'Hist Y-1')}`**), the baseline reporting date (**`{meta.get('bs_base_col', 'Base')}`**), and the two future forecast years (**`{meta.get('fcst_col_1', 'Forecast +1Y')}`**, **`{meta.get('fcst_col_2', 'Forecast +2Y')}`**).
               - Automatically reads `period_type` (`{meta.get('is_base_pt', '6M')}`) for the base year Income Statement and applies the exact annualization multiplier (**`{meta.get('is_ann_mult', 2.0):.2f}x`**).
            2. **IFRS 9 Accounting Class Separation (`Accounting_Class` Column):**
               - **`FVOCI` (Fair Value Through OCI):** Duration-based interest rate MTM impacts **`BS_AST_FVOCI`** on the asset side and **`BS_EQ_FVOCI` (Fair Value Reserve)** in Equity — leaving the Income Statement untouched.
               - **`FVTPL` (Fair Value Through P&L):** Gold and trading revaluations flow directly into the Income Statement (`IS_TRADING` / `IS_INV_GAINS`) and subsequently into Retained/Period Net Profit (`BS_EQ_NET_PROFIT`).
               - **`AMORTISED_COST`:** Sovereign and corporate bonds held to collect contractual cash flows (`BS_AST_AMORT`) are immune to interest rate MTM.
            3. **Dynamic Effective Tax Rate & Double-Entry Clearing:**
               - Reads `EFFECTIVE_TAX_RATE` dynamically from `market_data` / `vw_alm_flat_table` (or base SQL tax-to-EBT ratio) to compute `ΔTax = −ΔEBT × Effective_Tax_Rate` and update `BS_LIAB_TAX`.
               - Reconciles net cash/settlement flows through `BS_AST_OTHER` so that **`BS_CHECK = Total Assets − (Total Liabilities + Total Equity) = 0.00`** across all historical, base, scenario, and ML forecast columns.
            4. **L2 Ridge Regression + Macro Elasticity Forecasting (`_ml_ridge_forecast_series`):**
               - Fits a regularized linear trend (`λ = 0.8`) across the 3 historical/base observations per line item, blends the damped slope (`55%`) with macroeconomic drift (`45%`), projects 2 future years from the stressed scenario state, and enforces all accounting subtotal identities.
            """
        )

    # ========================================================================
    # TAB 4: Upgraded SQL View (`vw_alm_flat_table`) & Database Architecture
    # ========================================================================
    with tabs[3]:
        st.markdown("#### Upgraded SQL View (`vw_alm_flat_table`) & 24K/21K Gold Valuation Logic")
        st.markdown(
            """
            The SQL View **`vw_alm_flat_table`** serves as the single analytical position source for Python:
            - **Partitioned Market Rate Lookup (`rankedmarketdata`):** Uses `ROW_NUMBER() OVER (PARTITION BY instrument_code ORDER BY as_of_datetime DESC, market_data_id DESC)` so every FX pair (`FX_USD_EGP`, `FX_EUR_EGP`, `FX_GBP_EGP`, `FX_SAR_EGP`) and each Gold purity (`GOLD_24K_EGP`, `GOLD_21K_EGP`) is fetched independently without `base_currency` collisions.
            - **24K vs. 21K Gold Purity Support (`pos.gold_purity`):**
              - Positions with `Valuation_Basis = 'Gold'` and `Gold_Purity = '24K'` join to `GOLD_24K_EGP`.
              - Positions with `Valuation_Basis = 'Gold'` and `Gold_Purity = '21K'` join to `GOLD_21K_EGP`.
            - **Embedded Regulatory & Accounting Metadata:** Exposes `BS_Line_Code`, `Accounting_Class`, `NSFR_ASF_Weight`, `NSFR_RSF_Weight`, `Is_Liquid_Asset`, and `Effective_Tax_Rate` directly on every position row.
            """
        )

        val_codes_str = ", ".join(VALUATION_INSTRUMENTS)
        macro_codes_str = ", ".join(f"{k}: {v}" for k, v in MACRO_CODES.items())

        schema_rows = [
            ("vw_alm_flat_table", "SQL View", f"Master position view with 24K/21K Gold, FX rates ({val_codes_str}), macro indicators, BS_Line_Code, Accounting_Class, and Basel III NSFR weights."),
            (FLAT_CASHFLOWS_VIEW, "SQL View", "Scheduled cash-flow view converting native principal & interest flows into EGP across ALM maturity buckets."),
            (UNIFIED_BS_VIEW, "SQL View", "Unified Balance Sheet view reconciling position aggregates against `financial_statements_flat`."),
            ("products", "Base Table", "Product catalog storing repricing frequencies, `nsfr_asf_weight`, `nsfr_rsf_weight`, and `is_liquid_asset`."),
            ("positions", "Base Table", "Active balance sheet & off-balance sheet positions storing `bs_line_code`, `accounting_class`, `gold_purity`, `rate_type`, and dates."),
            ("cash_flows", "Base Table", "Contractual inflow/outflow schedules and `hqla_flag` per position."),
            ("market_data", "Base Table", f"Time-series store for valuation instruments ({val_codes_str}) and macro codes ({macro_codes_str})."),
            ("financial_statements_flat", "Base Table", "Multi-year historical and baseline Balance Sheet & Income Statement lines in EGP Millions."),
            ("scenarios / simulation_runs / simulation_results", "Audit Tables", "Stores all 10 scenario drivers, execution timestamps, KPI/Risk outputs, and cached Ollama (Phi-3) explanations."),
        ]
        show_df(pd.DataFrame(schema_rows, columns=["Database Object", "Object Type", "Architectural Role"]), height=380)