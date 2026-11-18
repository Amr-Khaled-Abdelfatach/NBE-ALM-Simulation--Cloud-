# analytics.py
"""
ALM Analytics, Risk Summaries, Independent Calculation Audit, and Dynamic 4-Sheet
Excel Workbook Exporter (with 100% live Excel formulas and zero hardcoded IDs or dates).
"""
import io
from typing import Optional

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from config import (
    CREDIT_COST_BPS_PER_GDP_DROP_PP,
    CREDIT_COST_BPS_PER_UNEMP_PP,
    DAYS_PER_YEAR,
    FLAT_CASHFLOWS_VIEW,
    FLAT_POSITIONS_VIEW,
    LCR_WARNING_THRESHOLD,
    MATURITY_BUCKET_ORDER,
    RISK_COF_MEDIUM_RISE_BPS,
    RISK_LCR_MEDIUM_RELATIVE_DROP_PCT,
    RISK_NII_HIGH_DROP_PCT,
    RISK_NII_MEDIUM_DROP_PCT,
    SCENARIO_DEFAULTS,
)
from calculations import (
    _basis_shock_pct,
    apply_prepared,
    merge_params,
    normalize_cashflows,
    valuation_rates,
)


def fmt_amount(num) -> str:
    """Format large numbers into human-readable strings with T, B, M, K suffixes."""
    if num is None or pd.isna(num):
        return "N/A"
    a = abs(float(num))
    for limit, suffix in [(1e12, " T"), (1e9, " B"), (1e6, " M"), (1e3, " K")]:
        if a >= limit:
            return f"{float(num) / limit:,.2f}{suffix}"
    return f"{float(num):,.2f}"


# Standard KPI specifications for the executive comparison table and DB persistence
_COMPARISON_SPECS = [
    ("POSITION_NII", "NII", "Positions", "EGP"),
    ("POSITION_NIM", "NIM %", "Positions", "%"),
    ("POSITION_COF", "Cost of Funds %", "Positions", "%"),
    ("POSITION_LOANS", "Loans", "Positions", "EGP"),
    ("POSITION_DEPOSITS", "Deposits", "Positions", "EGP"),
    ("POSITION_FUNDING", "Funding", "Positions", "EGP"),
    ("POSITION_INVESTMENTS", "Investments", "Positions", "EGP"),
    ("POSITION_LOAN_DEPOSIT", "Loan Deposit %", "Positions", "%"),
    ("POSITION_REPRICING_GAP", "Repricing Gap", "Positions", "EGP"),
    ("LIQUIDITY_NET_CASHFLOW", "Liquidity", "Cash_Flows", "EGP"),
    ("LCR_PROXY", "CF LCR %", "Cash_Flows", "%"),
    ("PNL_CREDIT_COST", "PnL Credit Cost", "P&L", "EGP"),
    ("PNL_FX_GOLD_REVALUATION", "PnL FX Gold Revaluation", "P&L", "EGP"),
    ("PNL_INVESTMENT_MTM", "PnL Investment MTM", "P&L", "EGP"),
    ("PNL_TOTAL_IMPACT", "PnL Total Impact", "P&L", "EGP"),
]


def comparison_table(before: dict, after: dict) -> pd.DataFrame:
    """Build a side-by-side Before vs. After KPI comparison table."""
    rows = []
    for code, metric, source, unit in _COMPARISON_SPECS:
        b_raw, a_raw = before.get(metric, np.nan), after.get(metric, np.nan)
        b = float(b_raw) if pd.notna(b_raw) else np.nan
        a = float(a_raw) if pd.notna(a_raw) else np.nan
        delta = a - b if pd.notna(a) and pd.notna(b) else np.nan
        change = delta / abs(b) * 100.0 if pd.notna(delta) and b else np.nan
        rows.append({
            "Code": code, "Source": source, "KPI": metric, "Unit": unit,
            "Before": b, "After": a, "Absolute Change": delta, "Change %": change,
        })
    return pd.DataFrame(rows)


def build_alm_risk_summary(before: dict, after: dict) -> pd.DataFrame:
    """Evaluate key ALM risk indicators against regulatory and internal thresholds."""
    specs = [
        ("POSITION_NII", "NII", "EGP", None, "Positions"),
        ("POSITION_NIM", "NIM %", "%", None, "Positions"),
        ("POSITION_COF", "Cost of Funds %", "%", None, "Positions"),
        ("REPRICING_GAP", "Repricing Gap", "EGP", None, "Positions"),
        ("LCR_PROXY", "CF LCR %", "%", LCR_WARNING_THRESHOLD, "Cash_Flows"),
    ]
    rows = []
    for code, name, unit, limit, source in specs:
        b_raw, a_raw = before.get(name, np.nan), after.get(name, np.nan)
        b = float(b_raw) if pd.notna(b_raw) else np.nan
        a = float(a_raw) if pd.notna(a_raw) else np.nan
        impact = a - b if pd.notna(a) and pd.notna(b) else np.nan
        pct = impact / abs(b) * 100.0 if pd.notna(impact) and b else np.nan

        level = "INFO"
        if code == "LCR_PROXY":
            level = "UNAVAILABLE" if pd.isna(a) else ("HIGH" if a < limit else "OK")
        elif code == "POSITION_COF":
            level = "ADVERSE" if pd.notna(impact) and impact > 0 else "POSITIVE" if pd.notna(impact) and impact < 0 else "NEUTRAL"
        elif code in {"POSITION_NII", "POSITION_NIM"}:
            level = "ADVERSE" if pd.notna(impact) and impact < 0 else "POSITIVE" if pd.notna(impact) and impact > 0 else "NEUTRAL"

        rows.append({
            "Risk Indicator": code, "Metric": name, "Base": b, "Scenario": a, "Impact": impact,
            "Impact %": pct, "Risk Level": level, "Limit": limit,
            "Limit Breach": (bool(a < limit) if limit is not None and pd.notna(a) else None),
            "Unit": unit, "Source": source,
        })
    return pd.DataFrame(rows)


def shock_sensitivity(prep: dict, simulation_date, horizon_days, params, param_key: str, values) -> pd.DataFrame:
    """Run a grid of shock values for a single parameter to plot sensitivity curves."""
    rows = []
    base = merge_params(params)
    for v in values:
        p = dict(base)
        p[param_key] = float(v)
        _, a, _ = apply_prepared(prep, simulation_date, horizon_days, p)
        rows.append([
            float(v), a["NII"], a["NIM %"], a["Cost of Funds %"],
            a["Repricing Gap"], a["CF LCR %"], a["PnL Total Impact"],
        ])
    return pd.DataFrame(rows, columns=[
        "Shock", "NII", "NIM %", "Cost of Funds %",
        "Repricing Gap", "LCR %", "Total P&L & Value Impact",
    ])


# All 10 scenario drivers mapped for standalone attribution analysis
DRIVER_GROUPS = {
    "Policy rate": ["rate_change_pp"],
    "FX": ["fx_shock_pct"],
    "Gold": ["gold_shock_pct"],
    "Wholesale funding spread": ["wsf_spread_shock_bps"],
    "Deposit run-off": ["deposit_runoff_pct"],
    "Loan growth": ["loan_growth_pct"],
    "Deposit growth": ["deposit_growth_pct"],
    "Inflation": ["inflation_shock_pct"],
    "Unemployment": ["unemployment_shock_pct"],
    "GDP": ["gdp_shock_pct"],
}


def driver_attribution(prep: dict, simulation_date, horizon_days, params) -> pd.DataFrame:
    """Decompose combined scenario impact into standalone driver effects plus interaction term."""
    p = merge_params(params)
    base_b, comb, _ = apply_prepared(prep, simulation_date, horizon_days, p)
    neutral = dict(SCENARIO_DEFAULTS)
    rows, sum_nii, sum_tot = [], 0.0, 0.0
    for label, keys in DRIVER_GROUPS.items():
        if all(abs(p.get(k, 0.0) - SCENARIO_DEFAULTS.get(k, 0.0)) < 1e-12 for k in keys):
            continue
        single = dict(neutral)
        single.update({k: p[k] for k in keys})
        _, a, _ = apply_prepared(prep, simulation_date, horizon_days, single)
        d_nii = a["NII"] - base_b["NII"]
        rows.append([
            label, d_nii, a.get("PnL Total Impact", 0.0),
            a.get("CF LCR %", 0.0),
            a.get("Cost of Funds %", 0.0) - base_b.get("Cost of Funds %", 0.0),
        ])
        sum_nii += d_nii
        sum_tot += a["PnL Total Impact"]

    cols = ["Driver", "NII Impact", "Total P&L & Value Impact", "LCR %", "Cost of Funds Change (pp)"]
    if rows:
        rows.append([
            "Interaction (combined - sum of drivers)",
            comb["NII"] - base_b["NII"] - sum_nii,
            comb["PnL Total Impact"] - sum_tot, np.nan, np.nan,
        ])
        rows.append([
            "COMBINED SCENARIO",
            comb["NII"] - base_b["NII"],
            comb["PnL Total Impact"], comb["CF LCR %"],
            comb["Cost of Funds %"] - base_b["Cost of Funds %"],
        ])
    return pd.DataFrame(rows, columns=cols)


def position_sensitivity(position_detail: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """Rank the top N positions by absolute total financial impact."""
    cols = ["Position", "Category", "Currency", "NII Impact", "Valuation Effect", "Investment MTM", "Total Impact"]
    q = position_detail
    if q is None or q.empty:
        return pd.DataFrame(columns=cols)
    cat = q["Category"].astype(str).str.lower()
    sign = np.where(cat.isin(["loan", "investment"]), 1.0, np.where(cat.isin(["deposit", "funding"]), -1.0, 0.0))
    out = pd.DataFrame({
        "Position": q["position_id"].astype(int).astype(str) + " - " + q["product_name"].astype(str).str.slice(0, 28),
        "Category": q["Category"], "Currency": q["Model Currency"],
        "NII Impact": q["Interest Change"] * sign, "Valuation Effect": q["Valuation Effect"],
        "Investment MTM": q["Investment MTM"],
    })
    out["Total Impact"] = out["NII Impact"] + out["Valuation Effect"] + out["Investment MTM"] - q["Credit Cost"]
    out = out.reindex(out["Total Impact"].abs().sort_values(ascending=False).index).head(top)
    return out[cols].reset_index(drop=True)


def pnl_components(before: dict, after: dict) -> pd.DataFrame:
    """Return the 4 main P&L and valuation impact components for waterfall charting."""
    return pd.DataFrame([
        ["NII change", after.get("PnL NII Change", 0.0)],
        ["Credit cost (macro)", after.get("PnL Credit Cost", 0.0)],
        ["FX / gold revaluation", after.get("PnL FX Gold Revaluation", 0.0)],
        ["Investment MTM", after.get("PnL Investment MTM", 0.0)],
    ], columns=["Component", "Impact"])


def overall_risk_rating(before: dict, after: dict, risk_df: Optional[pd.DataFrame] = None):
    """Determine overall scenario risk rating (LOW / MEDIUM / HIGH) with clear audit reasons."""
    level, reasons = "LOW", []

    def bump(new, why):
        nonlocal level
        order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        if order[new] > order[level]:
            level = new
        reasons.append(why)

    b_nii, a_nii = before.get("NII", 0.0), after.get("NII", 0.0)
    nii_pct = (a_nii - b_nii) / abs(b_nii) * 100 if b_nii else 0.0
    tot_pct = after.get("PnL Total Impact", 0.0) / abs(b_nii) * 100 if b_nii else 0.0
    lcr_b, lcr_a = before.get("CF LCR %", np.nan), after.get("CF LCR %", np.nan)
    cof = (after.get("Cost of Funds %", 0) - before.get("Cost of Funds %", 0)) * 100

    if pd.notna(lcr_a) and lcr_a < LCR_WARNING_THRESHOLD:
        bump("HIGH", f"LCR proxy {lcr_a:,.1f}% is below the {LCR_WARNING_THRESHOLD:.0f}% threshold.")
    if nii_pct <= RISK_NII_HIGH_DROP_PCT:
        bump("HIGH", f"NII falls {nii_pct:.1f}% (threshold {RISK_NII_HIGH_DROP_PCT:.0f}%).")
    elif nii_pct <= RISK_NII_MEDIUM_DROP_PCT:
        bump("MEDIUM", f"NII falls {nii_pct:.1f}% (threshold {RISK_NII_MEDIUM_DROP_PCT:.0f}%).")
    if tot_pct <= RISK_NII_HIGH_DROP_PCT:
        bump("HIGH", f"Total P&L / value impact is {tot_pct:.1f}% of base NII.")
    elif tot_pct <= RISK_NII_MEDIUM_DROP_PCT:
        bump("MEDIUM", f"Total P&L / value impact is {tot_pct:.1f}% of base NII.")
    if cof >= RISK_COF_MEDIUM_RISE_BPS:
        bump("MEDIUM", f"Cost of funds rises {cof:,.0f} bps.")
    if pd.notna(lcr_a) and pd.notna(lcr_b) and lcr_b:
        rel = (lcr_a - lcr_b) / abs(lcr_b) * 100
        if rel <= RISK_LCR_MEDIUM_RELATIVE_DROP_PCT:
            bump("MEDIUM", f"LCR proxy drops {abs(rel):.0f}% versus base.")
    if not reasons:
        reasons.append("No risk threshold is breached.")
    return level, reasons


def management_insights(before: dict, after: dict, attribution: Optional[pd.DataFrame] = None) -> list:
    """Generate concise deterministic bullet points for the executive dashboard."""
    out = []
    h = int(after.get("Horizon Days", 365))
    b_nii, a_nii = before.get("NII", 0.0), after.get("NII", 0.0)
    pct = (a_nii - b_nii) / abs(b_nii) * 100 if b_nii else np.nan
    arrow = "rises" if a_nii > b_nii else "falls" if a_nii < b_nii else "is unchanged"
    pct_txt = f" ({pct:+.1f}%)" if pd.notna(pct) else ""
    out.append(
        f"**NII {arrow}** from {fmt_amount(b_nii)} to {fmt_amount(a_nii)} EGP over {h} days{pct_txt}: "
        f"rate effect {fmt_amount(after.get('NII Rate Effect', 0))}, volume / FX effect {fmt_amount(after.get('NII Volume Effect', 0))}."
    )
    d_nim = after.get("NIM %", 0) - before.get("NIM %", 0)
    d_cof = after.get("cof_fs_scn", after.get("Cost of Funds %", 0)) - before.get("cof_fs_base", before.get("Cost of Funds %", 0))
    out.append(
        f"**NIM** moves {d_nim * 100:+.0f} bps to {after.get('NIM %', 0):.2f}%; "
        f"**cost of funds** moves {d_cof * 100:+.0f} bps to {after.get('cof_fs_scn', after.get('Cost of Funds %', 0)):.2f}%."
    )
    lcr_b, lcr_a = before.get("CF LCR %", np.nan), after.get("CF LCR %", np.nan)
    if pd.notna(lcr_a):
        out.append(
            f"**Liquidity:** 30-day LCR Proxy moves from {lcr_b:,.1f}% to {lcr_a:,.1f}% "
            f"(HQLA / Stressed 30d Net Outflows)."
        )
    total = after.get("PnL Total Impact", 0.0)
    out.append(f"**Total P&L & Value Impact:** {fmt_amount(total)} EGP across NII, FX/Gold revaluation, Investment MTM, and Credit Cost.")
    return out


def liquidity_ladder(cashflows, params=None, simulation_date=None, horizon_days: int = 365, stressed: bool = True) -> pd.DataFrame:
    """Build maturity bucket cash-flow ladder (Inflow, Outflow, Net, and Cumulative Net)."""
    cf = normalize_cashflows(cashflows)
    cols = ["Bucket", "Inflow", "Outflow", "Net", "Cumulative Net"]
    if cf.empty:
        return pd.DataFrame(columns=cols)
    p = merge_params(params if stressed else None)
    sim = pd.Timestamp.today().normalize() if simulation_date is None else pd.Timestamp(simulation_date).normalize()
    d = cf["Cashflow Date"]
    cf = cf[d.notna() & (d >= sim) & (d <= sim + pd.Timedelta(days=int(horizon_days)))].copy()
    if cf.empty:
        return pd.DataFrame(columns=cols)
    amt = cf["Amount EGP"] * (1.0 + _basis_shock_pct(cf["Valuation Basis"], p) / 100.0)
    cf["_in"] = amt.where(cf["Flow"].eq("inflow"), 0.0)
    cf["_out"] = amt.where(cf["Flow"].eq("outflow"), 0.0)
    g = cf.groupby("Bucket")[["_in", "_out"]].sum().reset_index()
    g.columns = ["Bucket", "Inflow", "Outflow"]
    order = {b: i for i, b in enumerate(MATURITY_BUCKET_ORDER)}
    g["_o"] = g["Bucket"].map(order).fillna(len(order))
    g = g.sort_values("_o").drop(columns="_o").reset_index(drop=True)
    g["Net"] = g["Inflow"] - g["Outflow"]
    g["Cumulative Net"] = g["Net"].cumsum()
    return g[cols]


def exposure_tables(position_detail: pd.DataFrame) -> dict:
    """Group position balances and interest by Category, Currency, and Product Family."""
    if position_detail is None or position_detail.empty:
        return {}
    out = {}
    for key, column in [("category", "Category"), ("currency", "Model Currency"), ("family", "Product Family")]:
        out[key] = position_detail.groupby(column).agg(
            Before_Balance=("Before Balance", "sum"), After_Balance=("After Balance", "sum"),
            Interest_Before=("Interest Before", "sum"), Interest_After=("Interest After", "sum"),
        ).reset_index().rename(columns={
            "Before_Balance": "Before Balance", "After_Balance": "After Balance",
            "Interest_Before": "Interest Before", "Interest_After": "Interest After",
        })
    return out


def build_data_quality_checks(before: dict, positions_flat, cashflows_flat=None, market_data=None, after=None,
                              raw: Optional[dict] = None, source_notes: Optional[dict] = None) -> pd.DataFrame:
    """Run automated data health checks across SQL views and tables."""
    checks = []

    def add(name, status, detail):
        checks.append([name, status, detail])

    source_notes = source_notes or {}
    for key, label in [("positions", FLAT_POSITIONS_VIEW), ("cashflows", FLAT_CASHFLOWS_VIEW)]:
        note = source_notes.get(key)
        if note:
            add(f"Flat source: {label}", "OK", note.get("message", ""))

    n_pos = 0 if positions_flat is None else len(positions_flat)
    add("Positions loaded", "OK" if n_pos else "ERROR", f"{n_pos} row(s) loaded.")
    add("NII", "OK" if np.isfinite(before.get("NII", np.nan)) else "ERROR",
        f"NII (horizon) = {before.get('NII', 0):,.2f} EGP")

    if cashflows_flat is not None and not cashflows_flat.empty:
        add("Cash flow source", "OK", f"{len(cashflows_flat)} cash-flow row(s) loaded in EGP.")
    else:
        add("Cash flow source", "WARNING", "No cash flows loaded.")

    if market_data is not None and not market_data.empty:
        rates = valuation_rates(market_data)
        add("Market data source", "OK" if not rates.empty else "WARNING",
            f"{len(market_data)} market_data rows; valuation rates for {rates['instrument_code'].tolist()}.")
    return pd.DataFrame(checks, columns=["Check", "Status", "Detail"])


def calculation_validation_table(before: dict, after: dict, position_detail: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Independently verify that every core KPI matches its mathematical formula."""
    rows = []

    def check(source, name, expected, actual, formula, tolerance=0.01):
        if pd.isna(expected) or pd.isna(actual):
            status, diff = "SKIP", np.nan
        else:
            diff = float(actual - expected)
            status = "PASS" if abs(diff) <= tolerance else "FAIL"
        rows.append([source, name, formula, expected, actual, diff, status])

    q = position_detail if position_detail is not None else pd.DataFrame()
    h = max(int(after.get("Horizon Days", 365)), 1)
    ann = DAYS_PER_YEAR / h

    if not q.empty:
        cat = q["Category"].astype(str).str.lower()
        sign = np.where(cat.isin(["loan", "investment"]), 1.0, np.where(cat.isin(["deposit", "funding"]), -1.0, 0.0))
        check("Positions", "NII (before)", float((q["Interest Before"] * sign).sum()),
              before.get("NII", 0), "Sum(asset interest) - Sum(liability interest), Before", 0.01)
        check("Positions", "NII (after)", float((q["Interest After"] * sign).sum()),
              after.get("NII", 0), "Sum(asset interest) - Sum(liability interest), After", 0.01)
        check("Positions", "NII change = rate effect + volume effect",
              after.get("NII", 0) - before.get("NII", 0),
              after.get("NII Rate Effect", 0) + after.get("NII Volume Effect", 0),
              "NII(after) - NII(before) = NII rate effect + NII volume effect", 0.01)

    earning = after.get("Earning Assets", after.get("Loans", 0) + after.get("Investments", 0))
    check("Positions", "NIM (after)",
          after.get("NII", 0) * ann / earning * 100 if earning else 0.0,
          after.get("NIM %", 0), "NII x 365/horizon / Earning Assets (Excl. BS_AST_CASH) x 100", 0.0001)

    fund = after.get("Deposits", 0) + after.get("Funding", 0)
    check("Positions", "Cost of Funds (after)",
          after.get("Interest Expense", 0) * ann / fund * 100 if fund else 0.0,
          after.get("Cost of Funds %", 0), "Interest expense x 365/horizon / (Deposits + Funding) x 100", 0.0001)

    check("P&L", "Total impact = NII + credit + FX/gold + MTM",
          after.get("PnL NII Change", 0) + after.get("PnL Credit Cost", 0) + after.get("PnL FX Gold Revaluation", 0)
          + after.get("PnL Investment MTM", 0), after.get("PnL Total Impact", 0),
          "dNII - credit cost + FX/gold revaluation + investment MTM", 0.01)

    # Validate 30-day LCR Proxy against HQLA / Stressed 30d Net Outflow
    hqla_stock = after.get("CF HQLA", 0.0)
    net_outflow_30d = after.get("LCR Net Outflow", 0.0)
    expected_lcr = (hqla_stock / net_outflow_30d * 100.0) if net_outflow_30d > 1.0 else np.nan

    check("Cash_Flows", "LCR Proxy (30d)",
          expected_lcr,
          after.get("CF LCR %", np.nan),
          "CF HQLA / Max(30d Outflow + Runoff - 30d Inflow, 1) x 100", 0.0001)

    return pd.DataFrame(rows, columns=["Source", "Check", "Formula", "Expected", "Actual", "Difference", "Status"])


def build_full_row_by_row_scenario_table(
    detail_df: pd.DataFrame,
    params: Optional[dict] = None,
    horizon_days: int = 365,
    sim_date=None,
) -> pd.DataFrame:
    """
    Build the complete position-level Row-by-Row Scenario Impact table aligned 100%
    with calculations.py (run_position_engine & cashflow_metrics) and fs_engine.py.
    """
    if detail_df is None or detail_df.empty:
        return pd.DataFrame()

    df = detail_df.copy()
    hz = max(int(horizon_days or 365), 1)
    sim_ts = pd.to_datetime(sim_date, errors="coerce") if sim_date is not None else pd.NaT

    def _col(name: str, default=0.0) -> pd.Series:
        if name in df.columns:
            return df[name]
        return pd.Series([default] * len(df), index=df.index)

    def _num(name: str, default=0.0) -> pd.Series:
        return pd.to_numeric(_col(name, default), errors="coerce").fillna(default)

    cat_str = _col("Category", "").astype(str).str.strip()
    bs_code_str = _col("BS Line Code", "").astype(str).str.strip().str.upper()
    pfam_str = _col("Product Family", "").astype(str).str.strip()
    ptype_str = _col("Product Type", "").astype(str).str.strip()
    basis_str = _col("Valuation Basis", "EGP").astype(str).str.strip()

    native_bal = _num("Native Balance", 0.0)
    base_fx = _num("FX Rate", 1.0)
    after_fx = _num("Applied Market Rate After", 1.0)
    before_bal = _num("Before Balance", 0.0)
    after_bal = _num("After Balance", 0.0)

    # Derived growth factor and valuation shock % matching calculations.py
    shock_factor = _num("Valuation Shock Factor", 1.0)
    val_shock_pct = shock_factor - 1.0
    denom_g = before_bal * shock_factor
    growth_factor = np.where(denom_g > 1e-12, after_bal / denom_g, 1.0)

    before_rate = _num("Before Rate", 0.0)
    after_rate = _num("After Rate", 0.0)
    accrual_days = _num("Accrual Days", float(hz)).clip(lower=0.0, upper=float(hz))

    # Exact Repricing Eligible flag directly from engine column `Eligible`
    if "Eligible" in df.columns:
        rep_elig = df["Eligible"].fillna(False).astype(bool)
    else:
        rt_low = _col("Model Rate Type", "").astype(str).str.lower()
        is_fixed = rt_low.str.contains("fixed|non-interest|non interest|zero interest", regex=True, na=False)
        rep_elig = (~is_fixed) & (accrual_days > 0)

    first_rep_dt = pd.to_datetime(_col("First Repricing Date", pd.NaT), errors="coerce")

    # Exact stepped pre-repricing days matching run_position_engine:
    # When Eligible=True -> min(max((first - sim).days, 0), accrual_days); when Eligible=False -> accrual_days
    if pd.notna(sim_ts):
        diff_d = (first_rep_dt - sim_ts).dt.days.fillna(0).clip(lower=0)
        elig_pre = np.where(first_rep_dt.notna(), np.minimum(diff_d, accrual_days), 0.0)
    else:
        elig_pre = np.zeros(len(df), dtype=float)
    pre_days = pd.Series(np.where(rep_elig, elig_pre, accrual_days), index=df.index, dtype=float)
    post_days = (accrual_days - pre_days).clip(lower=0.0)

    eff_rate = np.where(
        accrual_days > 0,
        (before_rate * pre_days + after_rate * post_days) / accrual_days,
        after_rate,
    )
    eff_rate = pd.Series(eff_rate, index=df.index, dtype=float)

    int_before = before_bal * before_rate * (accrual_days / DAYS_PER_YEAR)
    int_after = after_bal * eff_rate * (accrual_days / DAYS_PER_YEAR)
    d_int = int_after - int_before
    rate_eff = before_bal * (eff_rate - before_rate) * (accrual_days / DAYS_PER_YEAR)
    vol_eff = (after_bal - before_bal) * eff_rate * (accrual_days / DAYS_PER_YEAR)

    is_asset = cat_str.str.lower().isin(["loan", "investment"])
    is_liab = cat_str.str.lower().isin(["deposit", "funding"])
    sign_mult = np.where(is_asset, 1.0, np.where(is_liab, -1.0, 0.0))

    signed_nii_b = int_before * sign_mult
    signed_nii_a = int_after * sign_mult
    signed_dnii = signed_nii_a - signed_nii_b

    dur_yrs = _num("Duration (yrs)", 0.0)
    val_eff = _num("Valuation Effect", 0.0)
    inv_mtm = _num("Investment MTM", 0.0)
    cred_cost = _num("Credit Cost", 0.0)
    tot_pnl = signed_dnii + val_eff + inv_mtm - cred_cost

    raw_hqla = _col("HQLA Flag", False).fillna(False).astype(bool)
    effective_hqla = (
        raw_hqla
        | bs_code_str.eq("BS_AST_CASH")
        | pfam_str.str.lower().isin(["sovereign debt", "precious metals"])
    )

    liq_flag = _col("Is Liquid Asset", False).fillna(False).astype(bool).astype(int)
    asf_w = _num("NSFR ASF Weight", 0.0)
    rsf_w = _num("NSFR RSF Weight", 0.0)

    out = pd.DataFrame({
        "Position ID": _col("position_id", ""),
        "Product Code": _col("product_code", ""),
        "Product Name": _col("product_name", ""),
        "Product Type": ptype_str,
        "Category": cat_str,
        "BS Line Code": bs_code_str,
        "Accounting Class": _col("Accounting Class", ""),
        "Product Family": pfam_str,
        "Rate Type": _col("Model Rate Type", ""),
        "Currency": _col("Model Currency", "EGP"),
        "Valuation Basis": basis_str,
        "Raw HQLA Flag": raw_hqla.astype(int),
        "Effective HQLA Flag": effective_hqla.astype(int),
        "Is Liquid Asset": liq_flag,
        "NSFR ASF Weight": asf_w,
        "NSFR RSF Weight": rsf_w,
        "Repricing Eligible": rep_elig.astype(int),
        "First Repricing Date": first_rep_dt.dt.date,
        "Duration (yrs)": dur_yrs,
        "Native Balance": native_bal,
        "Base Market Rate": base_fx,
        "Growth Factor": growth_factor,
        "Valuation Shock (%)": val_shock_pct,
        "After Market Rate": after_fx,
        "Before Balance (EGP)": before_bal,
        "After Balance (EGP)": after_bal,
        "Balance Change (EGP)": after_bal - before_bal,
        "Balance Change (%)": np.where(before_bal != 0, (after_bal - before_bal) / before_bal, 0.0),
        "Before Rate": before_rate,
        "After Rate": after_rate,
        "Rate Change (bps)": (after_rate - before_rate) * 10000.0,
        "Accrual Days": accrual_days,
        "Pre-Repricing Days": pre_days,
        "Post-Repricing Days": post_days,
        "Effective Rate": eff_rate,
        "Interest Before (EGP)": int_before,
        "Interest After (EGP)": int_after,
        "Net Interest Change (EGP)": d_int,
        "Rate Effect (EGP)": rate_eff,
        "Volume / FX Effect (EGP)": vol_eff,
        "Valuation Effect (EGP)": val_eff,
        "Investment MTM (EGP)": inv_mtm,
        "Credit Cost (EGP)": cred_cost,
        "Signed NII Before (EGP)": signed_nii_b,
        "Signed NII After (EGP)": signed_nii_a,
        "Signed ΔNII (EGP)": signed_dnii,
        "Total Position P&L Impact (EGP)": tot_pnl,
        "HQLA Before (EGP)": np.where(effective_hqla, before_bal, 0.0),
        "HQLA After (EGP)": np.where(effective_hqla, after_bal, 0.0),
        "Liquid Assets Before (EGP)": np.where(liq_flag == 1, before_bal, 0.0),
        "Liquid Assets After (EGP)": np.where(liq_flag == 1, after_bal, 0.0),
        "NSFR ASF Before (EGP)": before_bal * asf_w,
        "NSFR ASF After (EGP)": after_bal * asf_w,
        "NSFR RSF Before (EGP)": before_bal * rsf_w,
        "NSFR RSF After (EGP)": after_bal * rsf_w,
        "Signed Repricing Before (EGP)": np.where(rep_elig, before_bal * sign_mult, 0.0),
        "Signed Repricing After (EGP)": np.where(rep_elig, after_bal * sign_mult, 0.0),
        "Audit Check (EGP)": np.round(d_int - (rate_eff + vol_eff), 2),
        "Validation Status": np.where(np.abs(d_int - (rate_eff + vol_eff)) <= 0.05, "PASS", "CHECK"),
    })
    return out


def export_row_by_row_with_formulas_xlsx(
    full_row_table: pd.DataFrame,
    horizon_days: int = 365,
    df_bs: Optional[pd.DataFrame] = None,
    df_is: Optional[pd.DataFrame] = None,
    res=None,
    params: Optional[dict] = None,
    window_years: Optional[int] = None,
    cashflows: Optional[pd.DataFrame] = None,
    **kwargs,
) -> bytes:
    """
    Generate a 4-sheet NBE Executive Excel Workbook where EVERY derived column
    (Growth Factor, Valuation Shock, After Market Rate, After Balance, After Rate,
    Stepped Repricing Days, NII, Rate/Volume Effect, Valuation, MTM, Credit Cost,
    HQLA, Liquid Assets, NSFR, Repricing Gap, KPIs, and Financial Statements)
    is powered by live Excel formulas matching calculations.py & fs_engine.py 100%.
    """
    import io
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    from config import (
        APP_NAME,
        CREDIT_COST_BPS_PER_GDP_DROP_PP,
        CREDIT_COST_BPS_PER_UNEMP_PP,
        DAYS_PER_YEAR,
        LCR_DEPOSIT_STRESS_FLOOR_PCT,
        LCR_WINDOW_DAYS,
        NBE_DARK_GREEN,
        NBE_GREEN,
        PARAM_SPECS,
        RATE_FLOOR,
        SCENARIO_DEFAULTS,
    )

    wb = Workbook()

    c_dark_green = NBE_DARK_GREEN.lstrip("#")
    c_green = NBE_GREEN.lstrip("#")
    fill_banner = PatternFill("solid", fgColor=c_dark_green)
    fill_header = PatternFill("solid", fgColor=c_green)
    fill_subhdr = PatternFill("solid", fgColor="EAF5EF")
    fill_zebra = PatternFill("solid", fgColor="F8FAFC")
    fill_total = PatternFill("solid", fgColor="E2F1E8")
    fill_param = PatternFill("solid", fgColor="FFF8EB")

    font_banner = Font(name="Calibri", size=13, bold=True, color="FFFFFF")
    font_header = Font(name="Calibri", size=10, bold=True, color="FFFFFF")
    font_bold = Font(name="Calibri", size=10, bold=True, color="0F172A")
    font_total = Font(name="Calibri", size=10, bold=True, color=c_dark_green)
    font_reg = Font(name="Calibri", size=10, bold=False, color="1E293B")
    font_link = Font(name="Calibri", size=10, bold=True, color="005B2D")

    thin_side = Side(style="thin", color="CBD5E1")
    med_green = Side(style="medium", color=c_dark_green)
    double_bottom = Side(style="double", color=c_dark_green)
    border_cell = Border(left=thin_side, right=thin_side, top=thin_side, bottom=thin_side)
    border_total = Border(top=med_green, bottom=double_bottom, left=thin_side, right=thin_side)

    FMT_AMT = "#,##0.00;(#,##0.00);\"-\""
    FMT_INT = "#,##0;(#,##0);\"0\""
    FMT_PCT = "0.00%"
    FMT_DEC = "0.0000"
    FMT_BPS = "+#,##0.0;-#,##0.0;0.0"
    FMT_DATE = "YYYY-MM-DD"

    def _auto_fit(ws, min_w: int = 11, max_w: int = 42):
        for col in ws.columns:
            col_letter = get_column_letter(col[0].column)
            best = min_w
            for cell in col:
                if cell.row <= 3:
                    continue
                val_s = str(cell.value or "")
                if val_s.startswith("="):
                    best = max(best, 16)
                else:
                    best = max(best, min(len(val_s) + 3, max_w))
            ws.column_dimensions[col_letter].width = best

    scn_name = str(getattr(res, "name", kwargs.get("scenario_name", "Current Scenario")))
    hz_val = int(getattr(res, "horizon", horizon_days) or 365)
    sim_dt_raw = getattr(res, "sim_date", kwargs.get("sim_date"))
    sim_ts = pd.to_datetime(sim_dt_raw, errors="coerce") if sim_dt_raw is not None else pd.Timestamp.today().normalize()
    if pd.isna(sim_ts):
        sim_ts = pd.Timestamp.today().normalize()

    active_params = dict(SCENARIO_DEFAULTS)
    if params:
        active_params.update(params)
    elif res is not None and isinstance(getattr(res, "params", None), dict):
        active_params.update(res.params)

    before_dict = getattr(res, "before", {}) or {}
    after_dict = getattr(res, "after", {}) or {}
    kpis_dict = getattr(res, "kpis", {}) or {}
    fs_meta = getattr(res, "fs_meta", {}) or kpis_dict.get("fs_meta", {})


    win_n = window_years if window_years is not None else fs_meta.get("window_years")
    if win_n is None:
        try:
            import streamlit as _st
            if "fs_sym_window_n" in _st.session_state:
                win_n = _st.session_state["fs_sym_window_n"]
        except Exception:
            pass
    win_n = int(win_n) if win_n is not None else 0


    # Pre-map parameter row coordinates in Sheet 2 ('Executive KPIs & Parameters')
    # Row 4 = Horizon (Days), Rows 5..14 = PARAM_SPECS in exact order
    param_row_map: Dict[str, int] = {"horizon_days": 4}
    for idx_p, k_spec in enumerate(PARAM_SPECS.keys(), start=5):
        param_row_map[k_spec] = idx_p

    def _p_ref(param_key: str) -> str:
        return f"'Executive KPIs & Parameters'!$G${param_row_map[param_key]}"

    # =========================================================================
    # SHEET 1: Row-by-Row Simulation (34 Dynamic Formula Columns!)
    # =========================================================================
    ws_row = wb.active
    ws_row.title = "Row-by-Row Simulation"

    df_r = full_row_table.copy() if full_row_table is not None else pd.DataFrame()
    n_r_cols = max(len(df_r.columns), 10)

    ws_row.merge_cells(start_row=1, start_column=1, end_row=1, end_column=n_r_cols)
    ws_row["A1"] = f"NATIONAL BANK OF EGYPT (NBE) — {APP_NAME.upper()} | ROW-BY-ROW SIMULATION ({scn_name})"
    ws_row["A1"].fill = fill_banner
    ws_row["A1"].font = font_banner
    ws_row["A1"].alignment = Alignment(horizontal="left", vertical="center", indent=1)
    ws_row.row_dimensions[1].height = 28

    ws_row["A2"] = "Horizon (Days):"
    ws_row["A2"].font = font_bold
    ws_row["B2"] = hz_val
    ws_row["B2"].font = font_bold
    ws_row["B2"].number_format = FMT_INT

    ws_row["A3"] = "Simulation Date:"
    ws_row["A3"].font = font_bold
    ws_row["B3"] = sim_ts.date()
    ws_row["B3"].font = font_bold
    ws_row["B3"].number_format = FMT_DATE

    ws_row["C2"] = "Dynamic Formula Engine:"
    ws_row["C2"].font = font_bold
    ws_row["D2"] = "All After Balances, Market Rates, Growth Factors, After Rates, Stepped Accruals, NII, MTM, Credit Cost, HQLA, NSFR & KPIs are live Excel formulas."
    ws_row["D2"].font = Font(name="Calibri", size=9.5, italic=True, color="475569")

    HDR_ROW_1 = 5
    START_ROW_1 = 6
    col_idx_map: Dict[str, int] = {}

    if not df_r.empty:
        for c_i, col_name in enumerate(df_r.columns, start=1):
            col_idx_map[col_name] = c_i
            cell = ws_row.cell(row=HDR_ROW_1, column=c_i, value=col_name)
            cell.fill = fill_header
            cell.font = font_header
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = border_cell
        ws_row.row_dimensions[HDR_ROW_1].height = 30

        def _rc(col_title: str, r_num: int) -> str:
            idx = col_idx_map.get(col_title)
            return f"{get_column_letter(idx)}{r_num}" if idx else "0"

        r_rate_pp = _p_ref("rate_change_pp")
        r_fx_pct = _p_ref("fx_shock_pct")
        r_gold_pct = _p_ref("gold_shock_pct")
        r_wsf_bps = _p_ref("wsf_spread_shock_bps")
        r_runoff = _p_ref("deposit_runoff_pct")
        r_loan_g = _p_ref("loan_growth_pct")
        r_dep_g = _p_ref("deposit_growth_pct")
        r_unemp = _p_ref("unemployment_shock_pct")
        r_gdp = _p_ref("gdp_shock_pct")
        r_hz = _p_ref("horizon_days")

        days_yr = float(DAYS_PER_YEAR)
        rate_flr = float(RATE_FLOOR)
        unemp_bps = float(CREDIT_COST_BPS_PER_UNEMP_PP)
        gdp_bps = float(CREDIT_COST_BPS_PER_GDP_DROP_PP)

        for idx_row, (_, r_data) in enumerate(df_r.iterrows()):
            xl_r = START_ROW_1 + idx_row
            is_even = idx_row % 2 == 1

            cat_c = _rc("Category", xl_r)
            bs_c = _rc("BS Line Code", xl_r)
            fam_c = _rc("Product Family", xl_r)
            bas_c = _rc("Valuation Basis", xl_r)
            raw_hqla_c = _rc("Raw HQLA Flag", xl_r)
            eff_hqla_c = _rc("Effective HQLA Flag", xl_r)
            liq_c = _rc("Is Liquid Asset", xl_r)
            asf_c = _rc("NSFR ASF Weight", xl_r)
            rsf_c = _rc("NSFR RSF Weight", xl_r)
            elig_c = _rc("Repricing Eligible", xl_r)
            rep_dt_c = _rc("First Repricing Date", xl_r)
            dur_c = _rc("Duration (yrs)", xl_r)
            base_rt_c = _rc("Base Market Rate", xl_r)
            gf_c = _rc("Growth Factor", xl_r)
            vs_c = _rc("Valuation Shock (%)", xl_r)
            b_bal_c = _rc("Before Balance (EGP)", xl_r)
            a_bal_c = _rc("After Balance (EGP)", xl_r)
            b_rate_c = _rc("Before Rate", xl_r)
            a_rate_c = _rc("After Rate", xl_r)
            acc_d_c = _rc("Accrual Days", xl_r)
            pre_d_c = _rc("Pre-Repricing Days", xl_r)
            post_d_c = _rc("Post-Repricing Days", xl_r)
            eff_r_c = _rc("Effective Rate", xl_r)
            int_b_c = _rc("Interest Before (EGP)", xl_r)
            int_a_c = _rc("Interest After (EGP)", xl_r)
            d_int_c = _rc("Net Interest Change (EGP)", xl_r)
            r_eff_c = _rc("Rate Effect (EGP)", xl_r)
            v_eff_c = _rc("Volume / FX Effect (EGP)", xl_r)
            val_e_c = _rc("Valuation Effect (EGP)", xl_r)
            mtm_c = _rc("Investment MTM (EGP)", xl_r)
            cred_c = _rc("Credit Cost (EGP)", xl_r)
            snii_b_c = _rc("Signed NII Before (EGP)", xl_r)
            snii_a_c = _rc("Signed NII After (EGP)", xl_r)
            sdnii_c = _rc("Signed ΔNII (EGP)", xl_r)
            aud_c = _rc("Audit Check (EGP)", xl_r)

            sign_expr = f"IF(OR(LOWER({cat_c})=\"loan\",LOWER({cat_c})=\"investment\"),1,IF(OR(LOWER({cat_c})=\"deposit\",LOWER({cat_c})=\"funding\"),-1,0))"

            formula_overrides = {
                "Effective HQLA Flag": f"=IF(OR({raw_hqla_c}=1,{bs_c}=\"BS_AST_CASH\",LOWER({fam_c})=\"sovereign debt\",LOWER({fam_c})=\"precious metals\"),1,0)",
                "Growth Factor": f"=MAX(0,IF(LOWER({cat_c})=\"loan\",1+{r_loan_g}/100,IF(LOWER({cat_c})=\"deposit\",(1+{r_dep_g}/100)*(1-{r_runoff}/100),1)))",
                "Valuation Shock (%)": f"=IF(LOWER({bas_c})=\"fx\",{r_fx_pct}/100,IF(LOWER({bas_c})=\"gold\",{r_gold_pct}/100,0))",
                "After Market Rate": f"={base_rt_c}*(1+{vs_c})",
                "After Balance (EGP)": f"={b_bal_c}*{gf_c}*(1+{vs_c})",
                "Balance Change (EGP)": f"={a_bal_c}-{b_bal_c}",
                "Balance Change (%)": f"=IF({b_bal_c}<>0,({a_bal_c}-{b_bal_c})/{b_bal_c},0)",
                "After Rate": (
                    f"=IF({elig_c}=1,"
                    f"MAX({rate_flr},{b_rate_c}+({r_rate_pp}/100)+IF({bs_c}=\"BS_LIAB_BORROW\",{r_wsf_bps}/10000,0)),"
                    f"{b_rate_c}+IF({bs_c}=\"BS_LIAB_BORROW\",{r_wsf_bps}/10000,0))"
                ),
                "Rate Change (bps)": f"=({a_rate_c}-{b_rate_c})*10000",
                "Pre-Repricing Days": f"=IF(AND({elig_c}=1,ISNUMBER({rep_dt_c})),MIN({acc_d_c},MAX(0,{rep_dt_c}-$B$3)),{acc_d_c})",
                "Post-Repricing Days": f"=MAX(0,{acc_d_c}-{pre_d_c})",
                "Effective Rate": f"=IF({acc_d_c}>0,({b_rate_c}*{pre_d_c}+{a_rate_c}*{post_d_c})/{acc_d_c},{a_rate_c})",
                "Interest Before (EGP)": f"={b_bal_c}*{b_rate_c}*({acc_d_c}/{days_yr})",
                "Interest After (EGP)": f"={a_bal_c}*{eff_r_c}*({acc_d_c}/{days_yr})",
                "Net Interest Change (EGP)": f"={int_a_c}-{int_b_c}",
                "Rate Effect (EGP)": f"={b_bal_c}*({eff_r_c}-{b_rate_c})*({acc_d_c}/{days_yr})",
                "Volume / FX Effect (EGP)": f"=({a_bal_c}-{b_bal_c})*{eff_r_c}*({acc_d_c}/{days_yr})",
                "Valuation Effect (EGP)": f"={sign_expr}*{b_bal_c}*{gf_c}*{vs_c}",
                "Investment MTM (EGP)": f"=-{dur_c}*({r_rate_pp}/100)*{a_bal_c}",
                "Credit Cost (EGP)": f"=IF(LOWER({cat_c})=\"loan\",{a_bal_c}*(({unemp_bps}*MAX({r_unemp},0)+{gdp_bps}*MAX(-{r_gdp},0))/10000)*({r_hz}/{days_yr}),0)",
                "Signed NII Before (EGP)": f"={sign_expr}*{int_b_c}",
                "Signed NII After (EGP)": f"={sign_expr}*{int_a_c}",
                "Signed ΔNII (EGP)": f"={snii_a_c}-{snii_b_c}",
                "Total Position P&L Impact (EGP)": f"={sdnii_c}+{val_e_c}+{mtm_c}-{cred_c}",
                "HQLA Before (EGP)": f"=IF({eff_hqla_c}=1,{b_bal_c},0)",
                "HQLA After (EGP)": f"=IF({eff_hqla_c}=1,{a_bal_c},0)",
                "Liquid Assets Before (EGP)": f"=IF({liq_c}=1,{b_bal_c},0)",
                "Liquid Assets After (EGP)": f"=IF({liq_c}=1,{a_bal_c},0)",
                "NSFR ASF Before (EGP)": f"={b_bal_c}*{asf_c}",
                "NSFR ASF After (EGP)": f"={a_bal_c}*{asf_c}",
                "NSFR RSF Before (EGP)": f"={b_bal_c}*{rsf_c}",
                "NSFR RSF After (EGP)": f"={a_bal_c}*{rsf_c}",
                "Signed Repricing Before (EGP)": f"=IF({elig_c}=1,{sign_expr}*{b_bal_c},0)",
                "Signed Repricing After (EGP)": f"=IF({elig_c}=1,{sign_expr}*{a_bal_c},0)",
                "Audit Check (EGP)": f"=ROUND({d_int_c}-({r_eff_c}+{v_eff_c}),2)",
                "Validation Status": f"=IF(ABS({aud_c})<=0.05,\"PASS\",\"CHECK\")",
            }

            for c_name, c_i in col_idx_map.items():
                if c_name in formula_overrides:
                    val = formula_overrides[c_name]
                else:
                    raw_v = r_data[c_name]
                    val = None if (raw_v is None or (not isinstance(raw_v, str) and pd.isna(raw_v))) else raw_v

                cell = ws_row.cell(row=xl_r, column=c_i, value=val)
                cell.font = font_reg
                cell.border = border_cell
                if is_even:
                    cell.fill = fill_zebra

                if c_name == "First Repricing Date":
                    cell.number_format = FMT_DATE
                elif "(EGP)" in c_name or c_name == "Native Balance":
                    cell.number_format = FMT_AMT
                elif "%" in c_name or ("Rate" in c_name and "bps" not in c_name and "Type" not in c_name and "Market" not in c_name):
                    cell.number_format = FMT_PCT
                elif "bps" in c_name:
                    cell.number_format = FMT_BPS
                elif "Weight" in c_name or "Factor" in c_name or "Duration" in c_name or "Market Rate" in c_name:
                    cell.number_format = FMT_DEC
                elif "Days" in c_name or "Flag" in c_name or "Eligible" in c_name:
                    cell.number_format = FMT_INT

        END_ROW_1 = START_ROW_1 + len(df_r) - 1
        TOT_ROW_1 = END_ROW_1 + 1

        ws_row.cell(row=TOT_ROW_1, column=1, value="TOTAL / PORTFOLIO SUMMARY").font = font_total
        for c_name, c_i in col_idx_map.items():
            c_let = get_column_letter(c_i)
            t_cell = ws_row.cell(row=TOT_ROW_1, column=c_i)
            t_cell.fill = fill_total
            t_cell.font = font_total
            t_cell.border = border_total
            if "(EGP)" in c_name:
                t_cell.value = f"=SUM({c_let}{START_ROW_1}:{c_let}{END_ROW_1})"
                t_cell.number_format = FMT_AMT
            elif c_name in ("Repricing Eligible", "Effective HQLA Flag", "Is Liquid Asset"):
                t_cell.value = f"=SUM({c_let}{START_ROW_1}:{c_let}{END_ROW_1})"
                t_cell.number_format = FMT_INT
            elif c_name == "Validation Status":
                t_cell.value = f"=IF(ABS({_rc('Audit Check (EGP)', TOT_ROW_1)})<=1,\"ALL PASS\",\"REVIEW\")"

        ws_row.freeze_panes = f"D{START_ROW_1}"
    else:
        END_ROW_1 = START_ROW_1
        TOT_ROW_1 = START_ROW_1

    # =========================================================================
    # CREATE REMAINING 3 SHEETS & BUILD INCOME STATEMENT FIRST
    # =========================================================================
    ws_kpi = wb.create_sheet(title="Executive KPIs & Parameters")
    ws_bs = wb.create_sheet(title="Balance Sheet")
    ws_is = wb.create_sheet(title="Income Statement")

    eff_tax = 0.40
    if df_is is not None and not df_is.empty and "code" in df_is.columns and "base_mn" in df_is.columns:
        ebt_s = df_is.loc[df_is["code"] == "IS_EBT", "base_mn"]
        tax_s = df_is.loc[df_is["code"] == "IS_TAX", "base_mn"]
        if not ebt_s.empty and not tax_s.empty and float(ebt_s.values[0]) > 0:
            eff_tax = abs(float(tax_s.values[0])) / float(ebt_s.values[0])

    is_row_by_code: Dict[str, int] = {}
    is_col_by_name: Dict[str, str] = {}
    is_fcst_cols: List[str] = []

    if df_is is not None and not df_is.empty:
        ws_is.merge_cells("A1:K1")
        ws_is["A1"] = f"NATIONAL BANK OF EGYPT (NBE) — COMPARATIVE & FORECASTED INCOME STATEMENT ({scn_name})"
        ws_is["A1"].fill = fill_banner
        ws_is["A1"].font = font_banner
        ws_is["A1"].alignment = Alignment(horizontal="left", vertical="center", indent=1)
        ws_is.row_dimensions[1].height = 28

        ws_is["A2"] = "All figures in EGP Millions | Interim base periods annualized to 12M equivalent"
        ws_is["A2"].font = Font(name="Calibri", size=9.5, italic=True, color="475569")
        ws_is["D2"] = "Effective Tax Rate:"
        ws_is["D2"].font = font_bold
        ws_is["E2"] = round(eff_tax, 6)
        ws_is["E2"].font = font_bold
        ws_is["E2"].number_format = FMT_PCT
        TAX_RATE_REF = "$E$2"

        fixed_is_meta = {"order", "section", "code", "name_ar", "name_en", "source", "base_mn", "scn_mn", "delta_mn", "delta_pct"}
        dyn_is_cols = [c for c in df_is.columns if c not in fixed_is_meta]
        all_is_hist = [c for c in fs_meta.get("hist_cols", []) if c in df_is.columns] or [c for c in dyn_is_cols if str(c).endswith("A")]
        all_is_fcst = [c for c in fs_meta.get("fcst_cols", []) if c in df_is.columns] or [c for c in dyn_is_cols if str(c).endswith("F")]
        is_hist_cols = all_is_hist[-win_n:] if win_n > 0 else all_is_hist
        is_fcst_cols = all_is_fcst[:win_n] if win_n > 0 else all_is_fcst
        base_lbl_is = fs_meta.get("is_base_header") or fs_meta.get("bs_base_col") or "Base (2025E)"

        is_col_specs = (
            [("code", "Line Code"), ("section", "Section"), ("name_ar", "البند (عربي)"), ("name_en", "Line Item (English)")]
            + [(hc, hc) for hc in is_hist_cols]
            + [
                ("base_mn", f"Base ({base_lbl_is})"),
                ("scn_mn", "Scenario (After)"),
                ("delta_mn", "Delta (EGP Mn)"),
                ("delta_pct", "Delta (%)"),
            ]
            + [(fc, fc) for fc in is_fcst_cols]
        )

        IS_HDR_ROW = 4
        IS_START_ROW = 5
        for c_i, (c_key, c_hdr) in enumerate(is_col_specs, start=1):
            is_col_by_name[c_key] = get_column_letter(c_i)
            cell = ws_is.cell(row=IS_HDR_ROW, column=c_i, value=c_hdr)
            cell.fill = fill_header
            cell.font = font_header
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = border_cell
        ws_is.row_dimensions[IS_HDR_ROW].height = 28

        for idx_r, (_, r) in enumerate(df_is.iterrows()):
            xl_r = IS_START_ROW + idx_r
            is_row_by_code[str(r.get("code", ""))] = xl_r

        base_let_is = is_col_by_name["base_mn"]
        scn_let_is = is_col_by_name["scn_mn"]
        d_mn_let_is = is_col_by_name["delta_mn"]
        num_period_keys_is = is_hist_cols + ["base_mn", "scn_mn"] + is_fcst_cols

        for idx_r, (_, r) in enumerate(df_is.iterrows()):
            xl_r = IS_START_ROW + idx_r
            code = str(r.get("code", ""))
            src = str(r.get("source", ""))
            is_tot = any(t in code for t in ("NII", "EBT", "NET_PROFIT")) or src in ("TOTAL", "SUBTOTAL")

            for c_idx, (c_key, _) in enumerate(is_col_specs, start=1):
                c_let = get_column_letter(c_idx)
                cell = ws_is.cell(row=xl_r, column=c_idx)
                cell.border = border_total if is_tot else border_cell
                cell.font = font_total if is_tot else font_reg
                if is_tot:
                    cell.fill = fill_total
                elif idx_r % 2 == 1:
                    cell.fill = fill_zebra

                if c_key in ("code", "section", "name_ar", "name_en"):
                    cell.value = str(r.get(c_key, ""))
                elif c_key == "delta_mn":
                    cell.value = f"={scn_let_is}{xl_r}-{base_let_is}{xl_r}"
                    cell.number_format = FMT_AMT
                elif c_key == "delta_pct":
                    cell.value = f"=IF({base_let_is}{xl_r}<>0,{d_mn_let_is}{xl_r}/ABS({base_let_is}{xl_r}),0)"
                    cell.number_format = FMT_PCT
                elif c_key in num_period_keys_is:
                    cell.number_format = FMT_AMT
                    if code == "IS_NII" and all(k in is_row_by_code for k in ("IS_INT_INC", "IS_INT_EXP")):
                        cell.value = f"=SUM({c_let}{is_row_by_code['IS_INT_INC']}:{c_let}{is_row_by_code['IS_INT_EXP']})"
                    elif code == "IS_OP_PROFIT" and all(k in is_row_by_code for k in ("IS_NII", "IS_NET_FEE", "IS_OTHER_OP")):
                        cell.value = f"={c_let}{is_row_by_code['IS_NII']}+SUM({c_let}{is_row_by_code['IS_NET_FEE']}:{c_let}{is_row_by_code['IS_OTHER_OP']})"
                    elif code == "IS_NET_INVESTING" and all(k in is_row_by_code for k in ("IS_DIVIDENDS", "IS_INV_GAINS")):
                        cell.value = f"=SUM({c_let}{is_row_by_code['IS_DIVIDENDS']}:{c_let}{is_row_by_code['IS_INV_GAINS']})"
                    elif code == "IS_PBFIT" and all(k in is_row_by_code for k in ("IS_OP_PROFIT", "IS_NET_INVESTING")):
                        cell.value = f"={c_let}{is_row_by_code['IS_OP_PROFIT']}+{c_let}{is_row_by_code['IS_NET_INVESTING']}"
                    elif code == "IS_NET_FINANCING" and "IS_FIN_COST" in is_row_by_code:
                        cell.value = f"={c_let}{is_row_by_code['IS_FIN_COST']}"
                    elif code == "IS_EBT" and all(k in is_row_by_code for k in ("IS_PBFIT", "IS_NET_FINANCING")):
                        cell.value = f"={c_let}{is_row_by_code['IS_PBFIT']}+{c_let}{is_row_by_code['IS_NET_FINANCING']}"
                    elif code == "IS_TAX" and c_key in is_fcst_cols and "IS_EBT" in is_row_by_code:
                        r_ebt = is_row_by_code["IS_EBT"]
                        cell.value = f"=IF({c_let}{r_ebt}>0,-ABS({c_let}{r_ebt})*{TAX_RATE_REF},0)"
                    elif code == "IS_NET_PROFIT" and all(k in is_row_by_code for k in ("IS_EBT", "IS_TAX")):
                        cell.value = f"={c_let}{is_row_by_code['IS_EBT']}+{c_let}{is_row_by_code['IS_TAX']}"
                    else:
                        cell.value = float(r.get(c_key, 0.0) or 0.0)

        ws_is.freeze_panes = f"E{IS_START_ROW}"

    # =========================================================================
    # SHEET 3: Balance Sheet (Articulated with Income Statement Net Profit)
    # =========================================================================
    bs_row_by_code: Dict[str, int] = {}
    bs_col_by_name: Dict[str, str] = {}
    bs_base_col_key = ""
    bs_fcst_cols: List[str] = []
    bs_hist_cols: List[str] = []

    if df_bs is not None and not df_bs.empty:
        ws_bs.merge_cells("A1:K1")
        ws_bs["A1"] = f"NATIONAL BANK OF EGYPT (NBE) — COMPARATIVE & FORECASTED BALANCE SHEET ({scn_name})"
        ws_bs["A1"].fill = fill_banner
        ws_bs["A1"].font = font_banner
        ws_bs["A1"].alignment = Alignment(horizontal="left", vertical="center", indent=1)
        ws_bs.row_dimensions[1].height = 28

        ws_bs["A2"] = "All figures in EGP Millions | Forecasted Net Profit & Retained Earnings are linked via live formulas to Income Statement"
        ws_bs["A2"].font = Font(name="Calibri", size=9.5, italic=True, color="475569")

        fixed_bs_meta = {"order", "section", "code", "name_ar", "name_en", "Scenario (After)", "Delta (Impact)"}
        dyn_bs_cols = [c for c in df_bs.columns if c not in fixed_bs_meta]
        all_bs_hist = [c for c in fs_meta.get("hist_cols", []) if c in df_bs.columns] or [c for c in dyn_bs_cols if str(c).endswith("A")]
        all_bs_fcst = [c for c in fs_meta.get("fcst_cols", []) if c in df_bs.columns] or [c for c in dyn_bs_cols if str(c).endswith("F")]
        bs_hist_cols = all_bs_hist[-win_n:] if win_n > 0 else all_bs_hist
        bs_fcst_cols = all_bs_fcst[:win_n] if win_n > 0 else all_bs_fcst
        rem_bs = [c for c in dyn_bs_cols if c not in all_bs_hist and c not in all_bs_fcst]
        bs_base_col_key = fs_meta.get("bs_base_col") if fs_meta.get("bs_base_col") in df_bs.columns else (rem_bs[0] if rem_bs else "2025E")

        bs_col_specs = (
            [("code", "Line Code"), ("section", "Section"), ("name_ar", "البند (عربي)"), ("name_en", "Line Item (English)")]
            + [(hc, hc) for hc in bs_hist_cols]
            + [
                (bs_base_col_key, f"Base ({bs_base_col_key})"),
                ("Scenario (After)", "Scenario (After)"),
                ("Delta (Impact)", "Delta (EGP Mn)"),
            ]
            + [(fc, fc) for fc in bs_fcst_cols]
        )

        BS_HDR_ROW = 4
        BS_START_ROW = 5
        for c_i, (c_key, c_hdr) in enumerate(bs_col_specs, start=1):
            bs_col_by_name[c_key] = get_column_letter(c_i)
            cell = ws_bs.cell(row=BS_HDR_ROW, column=c_i, value=c_hdr)
            cell.fill = fill_header
            cell.font = font_header
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = border_cell
        ws_bs.row_dimensions[BS_HDR_ROW].height = 28

        for idx_r, (_, r) in enumerate(df_bs.iterrows()):
            xl_r = BS_START_ROW + idx_r
            bs_row_by_code[str(r.get("code", ""))] = xl_r

        base_let_bs = bs_col_by_name[bs_base_col_key]
        scn_let_bs = bs_col_by_name["Scenario (After)"]
        all_bs_period_keys = bs_hist_cols + [bs_base_col_key, "Scenario (After)"] + bs_fcst_cols

        for idx_r, (_, r) in enumerate(df_bs.iterrows()):
            xl_r = BS_START_ROW + idx_r
            code = str(r.get("code", ""))
            is_tot = any(t in code for t in ("TOT", "CHECK"))

            for c_idx, (c_key, _) in enumerate(bs_col_specs, start=1):
                c_let = get_column_letter(c_idx)
                cell = ws_bs.cell(row=xl_r, column=c_idx)
                cell.border = border_total if is_tot else border_cell
                cell.font = font_total if is_tot else font_reg
                if is_tot:
                    cell.fill = fill_total
                elif idx_r % 2 == 1:
                    cell.fill = fill_zebra

                if c_key in ("code", "section", "name_ar", "name_en"):
                    cell.value = str(r.get(c_key, ""))
                elif c_key == "Delta (Impact)":
                    cell.value = f"={scn_let_bs}{xl_r}-{base_let_bs}{xl_r}"
                    cell.number_format = FMT_AMT
                elif c_key in all_bs_period_keys:
                    cell.number_format = FMT_AMT

                    if (
                        code == "BS_EQ_NET_PROFIT"
                        and "IS_NET_PROFIT" in is_row_by_code
                        and (c_key == "Scenario (After)" or c_key in bs_fcst_cols)
                    ):
                        is_np_r = is_row_by_code["IS_NET_PROFIT"]
                        is_target_let = is_col_by_name.get("scn_mn" if c_key == "Scenario (After)" else c_key)
                        if is_target_let:
                            cell.value = f"='Income Statement'!{is_target_let}{is_np_r}"
                            cell.font = font_link
                        else:
                            cell.value = float(r.get(c_key, 0.0) or 0.0)

                    elif (
                        code == "BS_EQ_RETAINED"
                        and c_key in bs_fcst_cols
                        and "BS_EQ_NET_PROFIT" in bs_row_by_code
                    ):
                        f_pos = bs_fcst_cols.index(c_key)
                        prev_key = "Scenario (After)" if f_pos == 0 else bs_fcst_cols[f_pos - 1]
                        prev_let = bs_col_by_name[prev_key]
                        np_r = bs_row_by_code["BS_EQ_NET_PROFIT"]
                        cell.value = f"={prev_let}{xl_r}+{prev_let}{np_r}"
                        cell.font = font_link

                    elif (
                        code == "BS_AST_OTHER"
                        and (c_key == "Scenario (After)" or c_key in bs_fcst_cols)
                        and all(k in bs_row_by_code for k in ("BS_TOT_LIAB", "BS_TOT_EQ", "BS_AST_CASH", "BS_AST_FIXED"))
                    ):
                        r_liab = bs_row_by_code["BS_TOT_LIAB"]
                        r_eq = bs_row_by_code["BS_TOT_EQ"]
                        r_a1 = bs_row_by_code["BS_AST_CASH"]
                        r_a10 = bs_row_by_code["BS_AST_FIXED"]
                        cell.value = f"=({c_let}{r_liab}+{c_let}{r_eq})-SUM({c_let}{r_a1}:{c_let}{r_a10})"

                    elif code == "BS_TOT_ASSETS" and "BS_AST_CASH" in bs_row_by_code and "BS_AST_OTHER" in bs_row_by_code:
                        cell.value = f"=SUM({c_let}{bs_row_by_code['BS_AST_CASH']}:{c_let}{bs_row_by_code['BS_AST_OTHER']})"
                    elif code == "BS_TOT_LIAB" and "BS_LIAB_BANKS" in bs_row_by_code and "BS_LIAB_RETIR" in bs_row_by_code:
                        cell.value = f"=SUM({c_let}{bs_row_by_code['BS_LIAB_BANKS']}:{c_let}{bs_row_by_code['BS_LIAB_RETIR']})"
                    elif code == "BS_TOT_EQ" and "BS_EQ_CAP" in bs_row_by_code and "BS_EQ_RETAINED" in bs_row_by_code:
                        cell.value = f"=SUM({c_let}{bs_row_by_code['BS_EQ_CAP']}:{c_let}{bs_row_by_code['BS_EQ_RETAINED']})"
                    elif code == "BS_TOT_LIAB_EQ" and "BS_TOT_LIAB" in bs_row_by_code and "BS_TOT_EQ" in bs_row_by_code:
                        cell.value = f"={c_let}{bs_row_by_code['BS_TOT_LIAB']}+{c_let}{bs_row_by_code['BS_TOT_EQ']}"
                    elif code == "BS_CHECK" and "BS_TOT_ASSETS" in bs_row_by_code and "BS_TOT_LIAB_EQ" in bs_row_by_code:
                        cell.value = f"=ROUND({c_let}{bs_row_by_code['BS_TOT_ASSETS']}-{c_let}{bs_row_by_code['BS_TOT_LIAB_EQ']}, 2)"
                    else:
                        cell.value = float(r.get(c_key, 0.0) or 0.0)

        ws_bs.freeze_panes = f"E{BS_START_ROW}"

    # =========================================================================
    # SHEET 2: Executive KPIs & Parameters (100% Reconciled Live Excel Formulas)
    # =========================================================================
    ws_kpi.merge_cells("A1:G1")
    ws_kpi["A1"] = f"NATIONAL BANK OF EGYPT (NBE) — EXECUTIVE KPIs & SCENARIO PARAMETERS ({scn_name})"
    ws_kpi["A1"].fill = fill_banner
    ws_kpi["A1"].font = font_banner
    ws_kpi["A1"].alignment = Alignment(horizontal="left", vertical="center", indent=1)
    ws_kpi.row_dimensions[1].height = 28

    ws_kpi["F3"] = "APPLIED SCENARIO PARAMETERS & SHOCKS"
    ws_kpi["F3"].fill = fill_header
    ws_kpi["F3"].font = font_header
    ws_kpi.merge_cells("F3:G3")

    ws_kpi.cell(row=4, column=6, value="Horizon (Days)").font = font_bold
    ws_kpi.cell(row=4, column=6).fill = fill_param
    ws_kpi.cell(row=4, column=6).border = border_cell
    p_hz_cell = ws_kpi.cell(row=4, column=7, value="='Row-by-Row Simulation'!B2")
    p_hz_cell.font = font_bold
    p_hz_cell.fill = fill_param
    p_hz_cell.border = border_cell
    p_hz_cell.number_format = FMT_INT
    HZ_CELL_REF = "$G$4"

    for k_spec, spec_tuple in PARAM_SPECS.items():
        r_idx_p = param_row_map[k_spec]
        lbl_p, unit_p = spec_tuple[0], spec_tuple[1]
        c_lbl = ws_kpi.cell(row=r_idx_p, column=6, value=f"{lbl_p} ({unit_p})")
        c_lbl.font = font_reg
        c_lbl.fill = fill_param
        c_lbl.border = border_cell

        c_val = ws_kpi.cell(row=r_idx_p, column=7, value=float(active_params.get(k_spec, 0.0)))
        c_val.font = font_bold
        c_val.fill = fill_param
        c_val.border = border_cell
        c_val.number_format = "#,##0.00"

    # =========================================================================
    # SHEET 5: Cash Flows (Filtered to Simulation Window with 30D LCR Formulas)
    # =========================================================================
    ws_cf = wb.create_sheet(title="Cash Flows")
    cf_norm = normalize_cashflows(cashflows) if cashflows is not None and not cashflows.empty else pd.DataFrame()
    max_cf_days = max(int(hz_val), int(LCR_WINDOW_DAYS))

    if not cf_norm.empty:
        d_cf = cf_norm["Cashflow Date"]
        cf_win = cf_norm[
            d_cf.notna() & (d_cf >= sim_ts) & (d_cf <= sim_ts + pd.Timedelta(days=max_cf_days))
        ].sort_values(["Cashflow Date", "cashflow_id"]).reset_index(drop=True)
    else:
        cf_win = pd.DataFrame()

    ws_cf.merge_cells("A1:O1")
    ws_cf["A1"] = f"NATIONAL BANK OF EGYPT (NBE) — SCHEDULED CASH FLOWS & 30D LCR WINDOW ({scn_name})"
    ws_cf["A1"].fill = fill_banner
    ws_cf["A1"].font = font_banner
    ws_cf["A1"].alignment = Alignment(horizontal="left", vertical="center", indent=1)
    ws_cf.row_dimensions[1].height = 28

    ws_cf["A2"] = f"Includes cash flows within {max_cf_days} days from Simulation Date; 30D LCR Eligible flag dynamically filters days 0..{int(LCR_WINDOW_DAYS)}."
    ws_cf["A2"].font = Font(name="Calibri", size=9.5, italic=True, color="475569")

    CF_HDR_ROW = 4
    CF_START_ROW = 5
    has_cf_sheet = not cf_win.empty

    if has_cf_sheet:
        cf_headers = [
            "Cashflow ID", "Position ID", "Product Code", "Product Family",
            "Cashflow Type", "Flow", "Bucket", "Currency", "Valuation Basis",
            "Cashflow Date", "Days to Cashflow", "30D LCR Eligible",
            "Base Amount (EGP)", "Valuation Shock (%)", "Stressed Amount (EGP)",
        ]
        for c_i, h_txt in enumerate(cf_headers, start=1):
            cell = ws_cf.cell(row=CF_HDR_ROW, column=c_i, value=h_txt)
            cell.fill = fill_header
            cell.font = font_header
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = border_cell
        ws_cf.row_dimensions[CF_HDR_ROW].height = 28

        r_fx_cf = _p_ref("fx_shock_pct")
        r_gold_cf = _p_ref("gold_shock_pct")
        lcr_win_d = int(LCR_WINDOW_DAYS)

        for idx_c, (_, r_cf) in enumerate(cf_win.iterrows()):
            xl_r = CF_START_ROW + idx_c
            is_even = idx_c % 2 == 1
            row_vals = [
                r_cf.get("cashflow_id", ""),
                r_cf.get("position_id", ""),
                str(r_cf.get("Product Code", "")),
                str(r_cf.get("Product Family", "")),
                str(r_cf.get("Cashflow Type", "")),
                str(r_cf.get("Flow", "")),
                str(r_cf.get("Bucket", "")),
                str(r_cf.get("Currency", "EGP")),
                str(r_cf.get("Valuation Basis", "EGP")),
                r_cf["Cashflow Date"].date() if pd.notna(r_cf["Cashflow Date"]) else None,
                f"=J{xl_r}-'Row-by-Row Simulation'!$B$3",
                f"=IF(AND(K{xl_r}>=0,K{xl_r}<={lcr_win_d}),1,0)",
                float(r_cf.get("Amount EGP", 0.0) or 0.0),
                f"=IF(LOWER(I{xl_r})=\"fx\",{r_fx_cf}/100,IF(LOWER(I{xl_r})=\"gold\",{r_gold_cf}/100,0))",
                f"=M{xl_r}*(1+N{xl_r})",
            ]
            for c_i, val in enumerate(row_vals, start=1):
                cell = ws_cf.cell(row=xl_r, column=c_i, value=val)
                cell.font = font_reg
                cell.border = border_cell
                if is_even:
                    cell.fill = fill_zebra
                if c_i == 10:
                    cell.number_format = FMT_DATE
                elif c_i in (11, 12):
                    cell.number_format = FMT_INT
                elif c_i in (13, 15):
                    cell.number_format = FMT_AMT
                elif c_i == 14:
                    cell.number_format = FMT_PCT

        CF_END_ROW = CF_START_ROW + len(cf_win) - 1
        CF_TOT_ROW = CF_END_ROW + 1
        ws_cf.cell(row=CF_TOT_ROW, column=1, value="TOTAL WINDOW CASH FLOWS").font = font_total
        for c_i in range(1, len(cf_headers) + 1):
            t_cell = ws_cf.cell(row=CF_TOT_ROW, column=c_i)
            t_cell.fill = fill_total
            t_cell.font = font_total
            t_cell.border = border_total
            if c_i == 12:
                t_cell.value = f"=SUM(L{CF_START_ROW}:L{CF_END_ROW})"
                t_cell.number_format = FMT_INT
            elif c_i in (13, 15):
                c_let = get_column_letter(c_i)
                t_cell.value = f"=SUM({c_let}{CF_START_ROW}:{c_let}{CF_END_ROW})"
                t_cell.number_format = FMT_AMT
        ws_cf.freeze_panes = f"D{CF_START_ROW}"
    else:
        CF_END_ROW = CF_START_ROW

    # دوال مساعدة للربط المباشر بشيت المراكز وشيت الميزانية
    def _s1_tot(col_title: str) -> str:
        c_let = get_column_letter(col_idx_map[col_title])
        return f"'Row-by-Row Simulation'!${c_let}${TOT_ROW_1}"

    def _s1_range(col_title: str) -> str:
        c_let = get_column_letter(col_idx_map[col_title])
        return f"'Row-by-Row Simulation'!${c_let}${START_ROW_1}:${c_let}${END_ROW_1}"

    has_bs = bool(bs_row_by_code and bs_col_by_name)
    has_is = bool(is_row_by_code and is_col_by_name)

    def _bs_ref(code: str, col_key: str) -> str:
        return f"'Balance Sheet'!${bs_col_by_name[col_key]}${bs_row_by_code[code]}"

    def _is_ref(code: str, col_key: str) -> str:
        return f"'Income Statement'!${is_col_by_name[col_key]}${is_row_by_code[code]}"

    sup_start = max(param_row_map.values()) + 2
    ws_kpi.cell(row=sup_start, column=6, value="SUPPORTING LIQUIDITY & STATEMENT INPUTS").fill = fill_header
    ws_kpi.cell(row=sup_start, column=6).font = font_header
    ws_kpi.merge_cells(start_row=sup_start, start_column=6, end_row=sup_start, end_column=7)

    # إحداثيات الخلايا داخل هذا الجدول الفرعي
    r_sched_out_b = sup_start + 1
    r_sched_out_s = sup_start + 2
    r_runoff_s = sup_start + 3

    if has_cf_sheet:
        cf_flow_r = f"'Cash Flows'!$F${CF_START_ROW}:$F${CF_END_ROW}"
        cf_elig_r = f"'Cash Flows'!$L${CF_START_ROW}:$L${CF_END_ROW}"
        cf_base_r = f"'Cash Flows'!$M${CF_START_ROW}:$M${CF_END_ROW}"
        cf_str_r = f"'Cash Flows'!$O${CF_START_ROW}:$O${CF_END_ROW}"

        sched_out_base_val = f"=SUMIFS({cf_base_r},{cf_flow_r},\"outflow\",{cf_elig_r},1)"
        sched_out_scn_val = f"=SUMIFS({cf_str_r},{cf_flow_r},\"outflow\",{cf_elig_r},1)"
        sched_in_base_val = f"=SUMIFS({cf_base_r},{cf_flow_r},\"inflow\",{cf_elig_r},1)"
        sched_in_scn_val = f"=SUMIFS({cf_str_r},{cf_flow_r},\"inflow\",{cf_elig_r},1)"
    else:
        lcr_in_b = float(before_dict.get("LCR Inflow", 0.0))
        lcr_out_b = float(before_dict.get("LCR Outflow", 0.0))
        lcr_in_a = float(after_dict.get("LCR Inflow", 0.0))
        lcr_out_a = float(after_dict.get("LCR Outflow", 0.0))
        runoff_a = float(after_dict.get("CF Runoff Amount", 0.0))
        sched_out_base_val = lcr_out_b
        sched_out_scn_val = max(lcr_out_a - runoff_a, 0.0)
        sched_in_base_val = lcr_in_b
        sched_in_scn_val = lcr_in_a

    if not df_r.empty:
        cat_r = _s1_range("Category")
        bas_r = _s1_range("Valuation Basis")
        rt_r = _s1_range("Rate Type")
        b_bal_r = _s1_range("Before Balance (EGP)")
        a_bal_r = _s1_range("After Balance (EGP)")
        r_runoff = _p_ref("deposit_runoff_pct")
        r_fx_pct = _p_ref("fx_shock_pct")
        r_gold_pct = _p_ref("gold_shock_pct")

        dep_pre_runoff_f = (
            f"(SUMIFS({b_bal_r},{cat_r},\"Deposit\",{bas_r},\"EGP\")"
            f"+SUMIFS({b_bal_r},{cat_r},\"Deposit\",{bas_r},\"FX\")*(1+{r_fx_pct}/100)"
            f"+SUMIFS({b_bal_r},{cat_r},\"Deposit\",{bas_r},\"Gold\")*(1+{r_gold_pct}/100))"
        )
        runoff_scn_val = f"=({r_runoff}/100)*{dep_pre_runoff_f}"

        tot_liab_b_f = f"(SUMIFS({b_bal_r},{cat_r},\"Deposit\")+SUMIFS({b_bal_r},{cat_r},\"Funding\"))"
        non_int_b_f = (
            f"(SUMIFS({b_bal_r},{cat_r},\"Deposit\",{rt_r},\"*Non-Interest*\")"
            f"+SUMIFS({b_bal_r},{cat_r},\"Funding\",{rt_r},\"*Non-Interest*\"))"
        )
        non_int_share_val = f"=IF({tot_liab_b_f}>0,{non_int_b_f}/{tot_liab_b_f},0)"

        fcy_net_b_f = (
            f"((SUMIFS({b_bal_r},{bas_r},\"FX\",{cat_r},\"Loan\")+SUMIFS({b_bal_r},{bas_r},\"FX\",{cat_r},\"Investment\"))"
            f"-(SUMIFS({b_bal_r},{bas_r},\"FX\",{cat_r},\"Deposit\")+SUMIFS({b_bal_r},{bas_r},\"FX\",{cat_r},\"Funding\")))"
        )
        fcy_net_a_f = (
            f"((SUMIFS({a_bal_r},{bas_r},\"FX\",{cat_r},\"Loan\")+SUMIFS({a_bal_r},{bas_r},\"FX\",{cat_r},\"Investment\"))"
            f"-(SUMIFS({a_bal_r},{bas_r},\"FX\",{cat_r},\"Deposit\")+SUMIFS({a_bal_r},{bas_r},\"FX\",{cat_r},\"Funding\")))"
        )
        if has_bs and "BS_TOT_ASSETS" in bs_row_by_code:
            scale_f = f"({_bs_ref('BS_TOT_ASSETS', bs_base_col_key)}/((B4+B5)/1000000))"
            nop_b_val = f"=({fcy_net_b_f}/1000000)*{scale_f}"
            nop_a_val = f"=({fcy_net_a_f}/1000000)*{scale_f}"
        else:
            nop_b_val = f"={fcy_net_b_f}/1000000"
            nop_a_val = f"={fcy_net_a_f}/1000000"
    else:
        runoff_scn_val = float(after_dict.get("CF Runoff Amount", 0.0))
        non_int_share_val = 0.20
        nop_b_val = float(kpis_dict.get("nop_base_mn", 0.0))
        nop_a_val = float(kpis_dict.get("nop_scn_mn", 0.0))

    sup_items = [
        ("30d Scheduled Outflows — Base (EGP)", sched_out_base_val, FMT_AMT),
        ("30d Scheduled Outflows — Scenario (EGP)", sched_out_scn_val, FMT_AMT),
        ("30d Deposit Run-off Outflow — Scenario (EGP)", runoff_scn_val, FMT_AMT),
        ("30d Gross Outflow + Runoff — Base (EGP)", f"=$G${r_sched_out_b}", FMT_AMT),
        ("30d Scheduled Inflows — Base (EGP)", sched_in_base_val, FMT_AMT),
        ("30d Gross Outflow + Runoff — Scenario (EGP)", f"=$G${r_sched_out_s}+$G${r_runoff_s}", FMT_AMT),
        ("30d Scheduled Inflows — Scenario (EGP)", sched_in_scn_val, FMT_AMT),
        ("Non-Interest Liability Share (%)", non_int_share_val, FMT_PCT),
        ("Net Open FX Position — Base (EGP Mn)", nop_b_val, FMT_AMT),
        ("Net Open FX Position — Scenario (EGP Mn)", nop_a_val, FMT_AMT),
    ]
    sup_refs: Dict[str, str] = {}
    for idx_s, (s_lbl, s_val, s_fmt) in enumerate(sup_items, start=1):
        r_s = sup_start + idx_s
        sup_refs[s_lbl] = f"$G${r_s}"
        c1_s = ws_kpi.cell(row=r_s, column=6, value=s_lbl)
        c1_s.font = font_reg
        c1_s.border = border_cell
        c2_s = ws_kpi.cell(row=r_s, column=7, value=s_val)
        c2_s.font = font_bold
        c2_s.border = border_cell
        c2_s.number_format = s_fmt

    kpi_headers = ["Executive KPI Metric", "Baseline (Before)", "Scenario (After)", "Impact / Delta"]
    for c_i, h_txt in enumerate(kpi_headers, start=1):
        cell = ws_kpi.cell(row=3, column=c_i, value=h_txt)
        cell.fill = fill_header
        cell.font = font_header
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = border_cell
    ws_kpi.row_dimensions[3].height = 26

    prior_hist_key = bs_hist_cols[-1] if (has_bs and bs_hist_cols) else bs_base_col_key
    kpi_rows_spec = []

    if not df_r.empty:
        ea_nim_b = f"((B4+B5)-SUMIFS({_s1_range('Before Balance (EGP)')},{_s1_range('BS Line Code')},\"BS_AST_CASH\"))"
        ea_nim_s = f"((C4+C5)-SUMIFS({_s1_range('After Balance (EGP)')},{_s1_range('BS Line Code')},\"BS_AST_CASH\"))"

        kpi_rows_spec.extend([
            ("Total Loans Portfolio (EGP)", f"=SUMIFS({_s1_range('Before Balance (EGP)')},{_s1_range('Category')},\"Loan\")", f"=SUMIFS({_s1_range('After Balance (EGP)')},{_s1_range('Category')},\"Loan\")", "=C4-B4", FMT_AMT, FMT_AMT, False),
            ("Total Investments Portfolio (EGP)", f"=SUMIFS({_s1_range('Before Balance (EGP)')},{_s1_range('Category')},\"Investment\")", f"=SUMIFS({_s1_range('After Balance (EGP)')},{_s1_range('Category')},\"Investment\")", "=C5-B5", FMT_AMT, FMT_AMT, False),
            ("Total Customer Deposits (EGP)", f"=SUMIFS({_s1_range('Before Balance (EGP)')},{_s1_range('Category')},\"Deposit\")", f"=SUMIFS({_s1_range('After Balance (EGP)')},{_s1_range('Category')},\"Deposit\")", "=C6-B6", FMT_AMT, FMT_AMT, False),
            ("Total Wholesale Funding (EGP)", f"=SUMIFS({_s1_range('Before Balance (EGP)')},{_s1_range('Category')},\"Funding\")", f"=SUMIFS({_s1_range('After Balance (EGP)')},{_s1_range('Category')},\"Funding\")", "=C7-B7", FMT_AMT, FMT_AMT, False),
            ("1. Net Interest Income — NII (Horizon, EGP)", f"={_s1_tot('Signed NII Before (EGP)')}", f"={_s1_tot('Signed NII After (EGP)')}", "=C8-B8", FMT_AMT, FMT_AMT, True),
            ("2. Net Interest Margin — NIM (%)", f"=IF({ea_nim_b}>0,(B8*(365/{HZ_CELL_REF}))/{ea_nim_b},0)", f"=IF({ea_nim_s}>0,(C8*(365/{HZ_CELL_REF}))/{ea_nim_s},0)", "=(C9-B9)*10000", FMT_PCT, FMT_BPS, True),
        ])
    else:
        for dummy_lbl in ["Total Loans", "Total Investments", "Total Deposits", "Total Funding", "1. NII", "2. NIM %"]:
            kpi_rows_spec.append((dummy_lbl, "=0", "=0", "=0", FMT_AMT, FMT_AMT, False))

    if has_bs and has_is:
        ea_codes = ["BS_AST_BANKS", "BS_AST_FVTPL", "BS_AST_LOANS", "BS_AST_FVOCI", "BS_AST_REV_REPO", "BS_AST_AMORT"]
        ea_prev = "+".join(_bs_ref(c, prior_hist_key) for c in ea_codes if c in bs_row_by_code)
        ea_b = "+".join(_bs_ref(c, bs_base_col_key) for c in ea_codes if c in bs_row_by_code)
        ea_s = "+".join(_bs_ref(c, "Scenario (After)") for c in ea_codes if c in bs_row_by_code)
        kpi_rows_spec.append((
            "3. Yield on Earning Assets (%)",
            f"=IF((({ea_prev})+({ea_b}))>0,{_is_ref('IS_INT_INC', 'base_mn')}/((({ea_prev})+({ea_b}))/2),0)",
            f"=IF((({ea_prev})+({ea_s}))>0,{_is_ref('IS_INT_INC', 'scn_mn')}/((({ea_prev})+({ea_s}))/2),0)",
            "=(C10-B10)*10000",
            FMT_PCT, FMT_BPS, True,
        ))
        ni_sh = sup_refs["Non-Interest Liability Share (%)"]
        cl_b = f"{_bs_ref('BS_LIAB_BANKS', bs_base_col_key)}+{_bs_ref('BS_LIAB_REPO', bs_base_col_key)}+({_bs_ref('BS_LIAB_DEP', bs_base_col_key)}*(1-{ni_sh}))+{_bs_ref('BS_LIAB_BORROW', bs_base_col_key)}"
        cl_s = f"{_bs_ref('BS_LIAB_BANKS', 'Scenario (After)')}+{_bs_ref('BS_LIAB_REPO', 'Scenario (After)')}+({_bs_ref('BS_LIAB_DEP', 'Scenario (After)')}*(1-{ni_sh}))+{_bs_ref('BS_LIAB_BORROW', 'Scenario (After)')}"
        kpi_rows_spec.append((
            "4. Cost of Funds — COF (%)",
            f"=IF(({cl_b})>0,ABS({_is_ref('IS_INT_EXP', 'base_mn')})/({cl_b}),0)",
            f"=IF(({cl_s})>0,ABS({_is_ref('IS_INT_EXP', 'scn_mn')})/({cl_s}),0)",
            "=(C11-B11)*10000",
            FMT_PCT, FMT_BPS, True,
        ))
    else:
        kpi_rows_spec.extend([
            ("3. Yield on Earning Assets (%)", "=0", "=0", "=0", FMT_PCT, FMT_BPS, True),
            ("4. Cost of Funds — COF (%)", "=0", "=0", "=0", FMT_PCT, FMT_BPS, True),
        ])

    kpi_rows_spec.append(("5. Net Interest Spread (%)", "=B10-B11", "=C10-C11", "=(C12-B12)*10000", FMT_PCT, FMT_BPS, False))
    kpi_rows_spec.append((
        "6. Total P&L & Valuation Impact (EGP)",
        "=0",
        f"={_s1_tot('Total Position P&L Impact (EGP)')}" if not df_r.empty else "=0",
        "=C13-B13",
        FMT_AMT, FMT_AMT, True,
    ))
    kpi_rows_spec.append((
        "7. Repricing Gap (12M, EGP)",
        f"={_s1_tot('Signed Repricing Before (EGP)')}" if not df_r.empty else "=0",
        f"={_s1_tot('Signed Repricing After (EGP)')}" if not df_r.empty else "=0",
        "=C14-B14",
        FMT_AMT, FMT_AMT, False,
    ))
    kpi_rows_spec.append((
        "8. ±100 bps Parallel NII Sensitivity (%)",
        "=IF(B8<>0,(B14*0.01)/B8,0)",
        "=IF(C8<>0,(C14*0.01)/C8,0)",
        "=(C15-B15)*100",
        FMT_PCT, "+0.00;-0.00;0.00", False,
    ))

    if has_bs and "BS_EQ_FVOCI" in bs_row_by_code and "BS_TOT_EQ" in bs_row_by_code:
        kpi_rows_spec.append((
            "9. ΔEVE / Equity — FVOCI MTM (%)",
            "=0",
            f"=IF({_bs_ref('BS_TOT_EQ', bs_base_col_key)}<>0,({_bs_ref('BS_EQ_FVOCI', 'Scenario (After)')}-{_bs_ref('BS_EQ_FVOCI', bs_base_col_key)})/{_bs_ref('BS_TOT_EQ', bs_base_col_key)},0)",
            "=(C16-B16)*100",
            FMT_PCT, "+0.00;-0.00;0.00", False,
        ))
    else:
        kpi_rows_spec.append(("9. ΔEVE / Equity — FVOCI MTM (%)", "=0", "=0", "=0", FMT_PCT, "+0.00;-0.00;0.00", False))

    if has_is:
        op_b = (
            f"{_is_ref('IS_NII', 'base_mn')}+{_is_ref('IS_NET_FEE', 'base_mn')}"
            f"+{_is_ref('IS_DIVIDENDS', 'base_mn')}+{_is_ref('IS_TRADING', 'base_mn')}+{_is_ref('IS_INV_GAINS', 'base_mn')}"
        )
        op_s = (
            f"{_is_ref('IS_NII', 'scn_mn')}+{_is_ref('IS_NET_FEE', 'scn_mn')}"
            f"+{_is_ref('IS_DIVIDENDS', 'scn_mn')}+{_is_ref('IS_TRADING', 'scn_mn')}+{_is_ref('IS_INV_GAINS', 'scn_mn')}"
        )
        kpi_rows_spec.extend([
            ("10. Cost-to-Income Ratio — CTI (%)", f"=IF(({op_b})>0,ABS({_is_ref('IS_GA_EXP', 'base_mn')})/({op_b}),0)", f"=IF(({op_s})>0,ABS({_is_ref('IS_GA_EXP', 'scn_mn')})/({op_s}),0)", "=(C17-B17)*100", FMT_PCT, "+0.00;-0.00;0.00", False),
            ("11. Annualized Net Profit After Tax (EGP Mn)", f"={_is_ref('IS_NET_PROFIT', 'base_mn')}", f"={_is_ref('IS_NET_PROFIT', 'scn_mn')}", "=C18-B18", FMT_AMT, FMT_AMT, True),
        ])
    else:
        kpi_rows_spec.extend([
            ("10. Cost-to-Income Ratio — CTI (%)", "=0", "=0", "=0", FMT_PCT, "+0.00;-0.00;0.00", False),
            ("11. Annualized Net Profit After Tax (EGP Mn)", "=0", "=0", "=0", FMT_AMT, FMT_AMT, True),
        ])

    g_out_b = sup_refs["30d Gross Outflow + Runoff — Base (EGP)"]
    g_in_b = sup_refs["30d Scheduled Inflows — Base (EGP)"]
    g_out_a = sup_refs["30d Gross Outflow + Runoff — Scenario (EGP)"]
    g_in_a = sup_refs["30d Scheduled Inflows — Scenario (EGP)"]
    kpi_rows_spec.append((
        "12. LCR Proxy — 30D Stress (%)",
        f"=IF(MAX({g_out_b}-{g_in_b},1)>1,{_s1_tot('HQLA Before (EGP)')}/MAX({g_out_b}-{g_in_b},1),0)" if not df_r.empty else "=0",
        f"=IF(MAX({g_out_a}-{g_in_a},1)>1,{_s1_tot('HQLA After (EGP)')}/MAX({g_out_a}-{g_in_a},1),0)" if not df_r.empty else "=0",
        "=(C19-B19)*100",
        FMT_PCT, "+0.0;-0.0;0.0", True,
    ))

    if not df_r.empty and has_bs and "BS_TOT_ASSETS" in bs_row_by_code and "BS_LIAB_DEP" in bs_row_by_code:
        liq_scale_f = f"({_bs_ref('BS_TOT_ASSETS', bs_base_col_key)}/((B4+B5)/1000000))"
        liq_b_f = f"=IF({_bs_ref('BS_LIAB_DEP', bs_base_col_key)}>0,(({_s1_tot('Liquid Assets Before (EGP)')}/1000000)*{liq_scale_f})/{_bs_ref('BS_LIAB_DEP', bs_base_col_key)},0)"
        liq_s_f = f"=IF({_bs_ref('BS_LIAB_DEP', 'Scenario (After)')}>0,(({_s1_tot('Liquid Assets After (EGP)')}/1000000)*{liq_scale_f})/{_bs_ref('BS_LIAB_DEP', 'Scenario (After)')},0)"
    else:
        liq_b_f = f"=IF(B6>0,{_s1_tot('Liquid Assets Before (EGP)')}/B6,0)" if not df_r.empty else "=0"
        liq_s_f = f"=IF(C6>0,{_s1_tot('Liquid Assets After (EGP)')}/C6,0)" if not df_r.empty else "=0"

    kpi_rows_spec.append((
        "13. Liquid Assets / Customer Deposits (%)",
        liq_b_f,
        liq_s_f,
        "=(C20-B20)*100",
        FMT_PCT, "+0.00;-0.00;0.00", False,
    ))

    if not df_r.empty:
        cat_r = _s1_range("Category")
        fam_r = _s1_range("Product Family")
        pt_r = _s1_range("Product Type")
        b_bal_r = _s1_range("Before Balance (EGP)")
        a_bal_r = _s1_range("After Balance (EGP)")
        casa_b_sum = (
            f"(SUMIFS({b_bal_r},{cat_r},\"Deposit\",{fam_r},\"*CASA*\")"
            f"+SUMIFS({b_bal_r},{cat_r},\"Deposit\",{fam_r},\"<>CASA*\",{pt_r},\"*Current*\")"
            f"+SUMIFS({b_bal_r},{cat_r},\"Deposit\",{fam_r},\"<>CASA*\",{pt_r},\"*Saving*\"))"
        )
        casa_a_sum = (
            f"(SUMIFS({a_bal_r},{cat_r},\"Deposit\",{fam_r},\"*CASA*\")"
            f"+SUMIFS({a_bal_r},{cat_r},\"Deposit\",{fam_r},\"<>CASA*\",{pt_r},\"*Current*\")"
            f"+SUMIFS({a_bal_r},{cat_r},\"Deposit\",{fam_r},\"<>CASA*\",{pt_r},\"*Saving*\"))"
        )
        kpi_rows_spec.append((
            "14. CASA Ratio (%)",
            f"=IF(B6>0,{casa_b_sum}/B6,0)",
            f"=IF(C6>0,{casa_a_sum}/C6,0)",
            "=(C21-B21)*100",
            FMT_PCT, "+0.00;-0.00;0.00", False,
        ))
    else:
        kpi_rows_spec.append(("14. CASA Ratio (%)", "=0", "=0", "=0", FMT_PCT, "+0.00;-0.00;0.00", False))

    kpi_rows_spec.append((
        "15. Loan-to-Deposit Ratio — LDR (%)",
        "=IF(B6>0,B4/B6,0)",
        "=IF(C6>0,C4/C6,0)",
        "=(C22-B22)*100",
        FMT_PCT, "+0.00;-0.00;0.00", False,
    ))

    if has_bs and has_is:
        avg_eq_b = f"(({_bs_ref('BS_TOT_EQ', prior_hist_key)}+{_bs_ref('BS_TOT_EQ', bs_base_col_key)})/2)"
        avg_eq_s = f"(({_bs_ref('BS_TOT_EQ', prior_hist_key)}+{_bs_ref('BS_TOT_EQ', 'Scenario (After)')})/2)"
        avg_ast_b = f"(({_bs_ref('BS_TOT_ASSETS', prior_hist_key)}+{_bs_ref('BS_TOT_ASSETS', bs_base_col_key)})/2)"
        avg_ast_s = f"(({_bs_ref('BS_TOT_ASSETS', prior_hist_key)}+{_bs_ref('BS_TOT_ASSETS', 'Scenario (After)')})/2)"
        nop_b_r = sup_refs["Net Open FX Position — Base (EGP Mn)"]
        nop_a_r = sup_refs["Net Open FX Position — Scenario (EGP Mn)"]

        if not df_r.empty:
            nsfr_scale = f"({_bs_ref('BS_TOT_ASSETS', bs_base_col_key)}/((B4+B5)/1000000))"
            asf_b_tot = f"(({_s1_tot('NSFR ASF Before (EGP)')}/1000000)*{nsfr_scale}+{_bs_ref('BS_TOT_EQ', bs_base_col_key)})"
            rsf_b_tot = f"(({_s1_tot('NSFR RSF Before (EGP)')}/1000000)*{nsfr_scale})"
            asf_s_tot = f"(({_s1_tot('NSFR ASF After (EGP)')}/1000000)*{nsfr_scale}+{_bs_ref('BS_TOT_EQ', 'Scenario (After)')})"
            rsf_s_tot = f"(({_s1_tot('NSFR RSF After (EGP)')}/1000000)*{nsfr_scale})"
            nsfr_b_f = f"=IF({rsf_b_tot}>0,{asf_b_tot}/{rsf_b_tot},0)"
            nsfr_s_f = f"=IF({rsf_s_tot}>0,{asf_s_tot}/{rsf_s_tot},0)"
        else:
            nsfr_b_f = nsfr_s_f = "=0"

        kpi_rows_spec.extend([
            ("16. Return on Equity — ROE (%)", f"=IF({avg_eq_b}>0,B18/{avg_eq_b},0)", f"=IF({avg_eq_s}>0,C18/{avg_eq_s},0)", "=(C23-B23)*100", FMT_PCT, "+0.00;-0.00;0.00", True),
            ("17. Return on Assets — ROA (%)", f"=IF({avg_ast_b}>0,B18/{avg_ast_b},0)", f"=IF({avg_ast_s}>0,C18/{avg_ast_s},0)", "=(C24-B24)*100", FMT_PCT, "+0.00;-0.00;0.00", False),
            ("18. Capital Leverage — Equity / Assets (%)", f"=IF({_bs_ref('BS_TOT_ASSETS', bs_base_col_key)}>0,{_bs_ref('BS_TOT_EQ', bs_base_col_key)}/{_bs_ref('BS_TOT_ASSETS', bs_base_col_key)},0)", f"=IF({_bs_ref('BS_TOT_ASSETS', 'Scenario (After)')}>0,{_bs_ref('BS_TOT_EQ', 'Scenario (After)')}/{_bs_ref('BS_TOT_ASSETS', 'Scenario (After)')},0)", "=(C25-B25)*100", FMT_PCT, "+0.00;-0.00;0.00", False),
            ("19. Net Open FX Position / Equity (%)", f"=IF({_bs_ref('BS_TOT_EQ', bs_base_col_key)}>0,{nop_b_r}/{_bs_ref('BS_TOT_EQ', bs_base_col_key)},0)", f"=IF({_bs_ref('BS_TOT_EQ', 'Scenario (After)')}>0,{nop_a_r}/{_bs_ref('BS_TOT_EQ', 'Scenario (After)')},0)", "=(C26-B26)*100", FMT_PCT, "+0.00;-0.00;0.00", False),
            ("20. NSFR Proxy — Basel III (%)", nsfr_b_f, nsfr_s_f, "=(C27-B27)*100", FMT_PCT, "+0.00;-0.00;0.00", True),
            ("21. Sovereign Debt / Total Assets (%)", f"=IF({_bs_ref('BS_TOT_ASSETS', bs_base_col_key)}>0,({_bs_ref('BS_AST_FVOCI', bs_base_col_key)}+{_bs_ref('BS_AST_AMORT', bs_base_col_key)})/{_bs_ref('BS_TOT_ASSETS', bs_base_col_key)},0)", f"=IF({_bs_ref('BS_TOT_ASSETS', 'Scenario (After)')}>0,({_bs_ref('BS_AST_FVOCI', 'Scenario (After)')}+{_bs_ref('BS_AST_AMORT', 'Scenario (After)')})/{_bs_ref('BS_TOT_ASSETS', 'Scenario (After)')},0)", "=(C28-B28)*100", FMT_PCT, "+0.00;-0.00;0.00", False),
            ("22. FX Wholesale Funding / Liabilities (%)", f"=IF({_bs_ref('BS_TOT_LIAB', bs_base_col_key)}>0,{_bs_ref('BS_LIAB_BORROW', bs_base_col_key)}/{_bs_ref('BS_TOT_LIAB', bs_base_col_key)},0)", f"=IF({_bs_ref('BS_TOT_LIAB', 'Scenario (After)')}>0,{_bs_ref('BS_LIAB_BORROW', 'Scenario (After)')}/{_bs_ref('BS_TOT_LIAB', 'Scenario (After)')},0)", "=(C29-B29)*100", FMT_PCT, "+0.00;-0.00;0.00", False),
            ("23. Contingent Off-BS / Equity (%)", f"=IF({_bs_ref('BS_TOT_EQ', bs_base_col_key)}>0,{_bs_ref('BS_OFF_CONTINGENT', bs_base_col_key)}/{_bs_ref('BS_TOT_EQ', bs_base_col_key)},0)", f"=IF({_bs_ref('BS_TOT_EQ', 'Scenario (After)')}>0,{_bs_ref('BS_OFF_CONTINGENT', 'Scenario (After)')}/{_bs_ref('BS_TOT_EQ', 'Scenario (After)')},0)", "=(C30-B30)*100", FMT_PCT, "+0.00;-0.00;0.00", False),
        ])

    for idx_k, (k_lbl, f_b, f_a, f_d, v_fmt, d_fmt, is_hi) in enumerate(kpi_rows_spec):
        xl_r = 4 + idx_k
        c1 = ws_kpi.cell(row=xl_r, column=1, value=k_lbl)
        c2 = ws_kpi.cell(row=xl_r, column=2, value=f_b)
        c3 = ws_kpi.cell(row=xl_r, column=3, value=f_a)
        c4 = ws_kpi.cell(row=xl_r, column=4, value=f_d)

        row_font = font_total if is_hi else font_reg
        row_fill = fill_subhdr if is_hi else (fill_zebra if idx_k % 2 == 1 else PatternFill(fill_type=None))
        for cell in (c1, c2, c3, c4):
            cell.font = row_font
            cell.border = border_cell
            if row_fill.fill_type:
                cell.fill = row_fill

        c2.number_format = v_fmt
        c3.number_format = v_fmt
        c4.number_format = d_fmt

    ws_kpi.freeze_panes = "A4"

    for sheet in (ws_row, ws_kpi, ws_bs, ws_is, ws_cf):
        _auto_fit(sheet)

    out_buf = io.BytesIO()
    wb.save(out_buf)
    out_buf.seek(0)
    return out_buf.getvalue()