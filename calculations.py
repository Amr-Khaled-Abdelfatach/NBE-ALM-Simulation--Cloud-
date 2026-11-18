# calculations.py
"""Deterministic ALM calculation engine for position repricing, liquidity, and P&L impact."""
import re
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import (
    CREDIT_COST_BPS_PER_GDP_DROP_PP,
    CREDIT_COST_BPS_PER_UNEMP_PP,
    DAYS_PER_YEAR,
    EXCLUDE_MATURED_POSITIONS,
    FREQUENCY_WORD_DAYS,
    FUNDING_SEGMENTS,
    LCR_WINDOW_DAYS,
    MACRO_CODES,
    MODEL_AMOUNT_UNIT,
    RATE_FLOOR,
    ROLLOVER_AT_MATURITY,
    SCENARIO_DEFAULTS,
    TENOR_DAY_FACTORS,
    VALUATION_INSTRUMENTS,
)
from fs_engine import build_financial_statements_and_kpis


# ----------------------------------------------------------------------------
# Basic DataFrame and date utility functions
# ----------------------------------------------------------------------------
def col(df: pd.DataFrame, name: str) -> Optional[str]:
    """Find a column name in a DataFrame case-insensitively."""
    if df is None or df.empty:
        return None
    lookup = {str(c).lower(): c for c in df.columns}
    return lookup.get(name.lower())


def to_num(series, default=0.0):
    """Safely convert a pandas Series to numeric values, filling NaNs with default."""
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _series(df: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    """Return a column Series if it exists, otherwise a constant Series of the same length."""
    if name in df.columns:
        return df[name]
    return pd.Series(default, index=df.index)


def _str(df: pd.DataFrame, name: str, default: str = "") -> pd.Series:
    """Return a clean string Series with missing values replaced by default."""
    return _series(df, name, default).astype("object").where(lambda s: s.notna(), default).astype(str)


def latest_date(df: pd.DataFrame, column: str) -> Optional[pd.Timestamp]:
    """Get the most recent valid timestamp from a DataFrame column."""
    if df is None or df.empty or column not in df.columns:
        return None
    values = pd.to_datetime(df[column], errors="coerce").dropna()
    return values.max() if not values.empty else None


def _norm_date(value=None) -> pd.Timestamp:
    """Normalize any date input to midnight Timestamp (defaults to today)."""
    ts = pd.Timestamp.today() if value is None else pd.Timestamp(value)
    return ts.normalize()


def tenor_to_days(value, default=None):
    """Convert frequency or tenor strings (e.g., 'Monthly', '3M', '1Y') into days."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    text = str(value).strip().upper()
    if not text:
        return default
    if text in FREQUENCY_WORD_DAYS:
        days = FREQUENCY_WORD_DAYS[text]
        return default if days is None else int(days)
    m = re.search(r"(\d+(?:\.\d+)?)\s*([DWMY])", text)
    if not m:
        return default
    return int(round(float(m.group(1)) * TENOR_DAY_FACTORS[m.group(2)]))


def normalize_market_yield(value):
    """Ensure interest rates/yields are stored as decimals (e.g. 27.25 -> 0.2725)."""
    try:
        x = float(value)
    except Exception:
        return np.nan
    if not np.isfinite(x):
        return np.nan
    return x / 100.0 if abs(x) > 2.0 else x


def merge_params(params: Optional[dict]) -> Dict[str, float]:
    """Merge incoming scenario parameters with baseline defaults."""
    out = dict(SCENARIO_DEFAULTS)
    if params:
        for k, v in params.items():
            if k in out and v is not None:
                try:
                    out[k] = float(v)
                except (TypeError, ValueError):
                    pass
    return out


# ----------------------------------------------------------------------------
# Market data & macroeconomic snapshot helpers
# ----------------------------------------------------------------------------
def latest_market_date(market_data: pd.DataFrame) -> Optional[pd.Timestamp]:
    """Return the latest timestamp available in the market_data table."""
    return latest_date(market_data, "as_of_datetime")


def valuation_rates(market_data: pd.DataFrame, as_of=None) -> pd.DataFrame:
    """
    Fetch the latest FX and Gold (21K & 24K) valuation rates in EGP partitioned by instrument_code
    so GOLD_21K_EGP and GOLD_24K_EGP never overwrite each other.
    """
    cols = ["instrument_code", "base_currency", "market_rate"]
    need = {"instrument_code", "base_currency", "value"}
    if market_data is None or market_data.empty or not need <= set(market_data.columns):
        return pd.DataFrame(columns=cols)
    md = market_data[market_data["instrument_code"].isin(VALUATION_INSTRUMENTS)].copy()
    if md.empty:
        return pd.DataFrame(columns=cols)
    md["_d"] = pd.to_datetime(_series(md, "as_of_datetime"), errors="coerce")

    # Filter market rates up to the scenario reference date if provided
    if as_of is not None and pd.notna(as_of):
        cutoff = pd.to_datetime(as_of, errors="coerce")
        if pd.notna(cutoff):
            filtered = md[md["_d"] <= cutoff]
            if not filtered.empty:
                md = filtered

    md["_id"] = pd.to_numeric(_series(md, "market_data_id", 0), errors="coerce").fillna(0)
    md["value"] = pd.to_numeric(md["value"], errors="coerce")
    md = md.dropna(subset=["value"]).sort_values(["_d", "_id"])
    latest = md.groupby("instrument_code", as_index=False).tail(1).copy()

    # Standardize base_currency alias so callers looking up 'GOLD' get 24K by default while 'GOLD_21K' is also available
    latest["base_currency"] = np.where(
        latest["instrument_code"].astype(str).str.upper() == "GOLD_24K_EGP",
        "GOLD",
        np.where(
            latest["instrument_code"].astype(str).str.upper() == "GOLD_21K_EGP",
            "GOLD_21K",
            latest["base_currency"].astype(str).str.upper(),
        ),
    )
    return latest[["instrument_code", "base_currency", "value"]].rename(columns={"value": "market_rate"}).reset_index(drop=True)


def macro_snapshot(market_data: pd.DataFrame) -> Dict[str, float]:
    """Extract the latest policy rate, inflation, unemployment, GDP growth, and tax rate from market_data."""
    out = {k: np.nan for k in MACRO_CODES}
    out["as_of"] = None
    if market_data is None or market_data.empty or "instrument_code" not in market_data.columns:
        return out
    md = market_data.copy()
    md["_d"] = pd.to_datetime(_series(md, "as_of_datetime"), errors="coerce")
    md["_id"] = pd.to_numeric(_series(md, "market_data_id", 0), errors="coerce").fillna(0)
    for key, code in MACRO_CODES.items():
        sub = md[md["instrument_code"] == code].sort_values(["_d", "_id"])
        if sub.empty:
            continue
        out[key] = normalize_market_yield(sub.iloc[-1]["value"])
        d = sub.iloc[-1]["_d"]
        if pd.notna(d) and (out["as_of"] is None or d > out["as_of"]):
            out["as_of"] = d
    return out


def market_history(market_data: pd.DataFrame, instrument_code: str) -> pd.DataFrame:
    """Return time-series history for a single market_data instrument code."""
    if market_data is None or market_data.empty or "instrument_code" not in market_data.columns:
        return pd.DataFrame(columns=["as_of_datetime", "value"])
    h = market_data[market_data["instrument_code"] == instrument_code].copy()
    h["as_of_datetime"] = pd.to_datetime(h["as_of_datetime"], errors="coerce")
    h["value"] = pd.to_numeric(h["value"], errors="coerce")
    return h.sort_values("as_of_datetime")[["as_of_datetime", "value"]].reset_index(drop=True)


# ----------------------------------------------------------------------------
# SQL View normalization (vw_alm_flat_table & vw_alm_cashflows_flat)
# ----------------------------------------------------------------------------
def position_category(position_type: pd.Series, segment: pd.Series) -> pd.Series:
    """Map raw position_type and segment into standardized ALM categories."""
    ptype = position_type.astype("object").where(position_type.notna(), "").astype(str).str.strip().str.title()
    seg = segment.astype("object").where(segment.notna(), "").astype(str).str.strip().str.lower()
    funding = ptype.eq("Deposit") & seg.isin([s.lower() for s in FUNDING_SEGMENTS])
    cat = ptype.mask(funding, "Funding")
    return cat.mask(cat.eq(""), "Other")


def normalize_positions(flat: pd.DataFrame) -> pd.DataFrame:
    """Standardize vw_alm_flat_table into the position DataFrame used by the simulation engine."""
    if flat is None or flat.empty:
        return pd.DataFrame()
    x = flat.copy()
    status = _str(x, "Status", "Active").str.strip().str.lower()
    x = x[status.isin(["active", ""])].copy().reset_index(drop=True)
    if x.empty:
        return pd.DataFrame()

    bal_native = to_num(_series(x, "Original_Balance_Or_Grams", 0.0))
    fx_rate = to_num(_series(x, "Applied_Market_Rate", 1.0), 1.0)
    egp = pd.to_numeric(_series(x, "Evaluated_Balance_EGP"), errors="coerce")

    model = pd.DataFrame({
        "position_id": pd.to_numeric(x["Position_ID"], errors="coerce").astype("Int64"),
        "product_id": pd.to_numeric(_series(x, "Product_ID"), errors="coerce").astype("Int64"),
        "product_code": _str(x, "Product_Code"),
        "product_name": _str(x, "Product_Name"),
        "Product Type": _str(x, "Product_Type"),
        "Product Family": _str(x, "Product_Family"),
        "Category": position_category(_series(x, "ALM_Position_Category"), _series(x, "Customer_Segment")),
        "Customer Segment": _str(x, "Customer_Segment"),
        "Model Currency": _str(x, "Currency", "EGP"),
        "Valuation Basis": _str(x, "Valuation_Basis", "EGP"),
        "Gold Purity": _str(x, "Gold_Purity", "24K"),
        "Native Balance": bal_native,
        "FX Rate": fx_rate,
        "Model Balance": egp.fillna(bal_native * fx_rate),
        "Model Rate": to_num(_series(x, "Contractual_Interest_Rate", 0.0)),
        "Model Rate Type": _str(x, "Rate_Type", "Unknown"),
        "Model Start Date": pd.to_datetime(_series(x, "Start_Date"), errors="coerce"),
        "Model Maturity Date": pd.to_datetime(_series(x, "Maturity_Date"), errors="coerce"),
        "Model Repricing Date": pd.to_datetime(_series(x, "Next_Repricing_Date"), errors="coerce"),
        "Repricing Frequency": _str(x, "Repricing_Frequency"),
        "HQLA Flag": to_num(_series(x, "HQLA_Liquidity_Flag", 0)).astype(bool),
        "Status": _str(x, "Status", "Active"),
        # Newly exposed dynamic metadata columns from upgraded SQL View
        "BS Line Code": _str(x, "BS_Line_Code", ""),
        "Accounting Class": _str(x, "Accounting_Class", "AMORTISED_COST"),
        "NSFR ASF Weight": to_num(_series(x, "NSFR_ASF_Weight", 0.0)),
        "NSFR RSF Weight": to_num(_series(x, "NSFR_RSF_Weight", 0.0)),
        "Is Liquid Asset": to_num(_series(x, "Is_Liquid_Asset", 0)).astype(bool),
        "Effective Tax Rate": to_num(_series(x, "Effective_Tax_Rate", np.nan), np.nan),
    })
    model["Repricing Frequency Days"] = model["Repricing Frequency"].map(tenor_to_days).astype(float)
    return model


def normalize_cashflows(flat: pd.DataFrame) -> pd.DataFrame:
    """Standardize vw_alm_cashflows_flat into the cash-flow DataFrame used for liquidity metrics."""
    cols = [
        "cashflow_id", "position_id", "Cashflow Date", "Flow", "Amount EGP",
        "Principal EGP", "Interest EGP", "Valuation Basis", "Currency",
        "HQLA Flag", "Bucket", "Cashflow Type", "Product Code", "Product Family",
    ]
    if flat is None or flat.empty:
        return pd.DataFrame(columns=cols)
    x = flat
    return pd.DataFrame({
        "cashflow_id": _series(x, "Cashflow_ID"),
        "position_id": _series(x, "Position_ID"),
        "Cashflow Date": pd.to_datetime(_series(x, "Cashflow_Date"), errors="coerce"),
        "Flow": _str(x, "Flow_Type").str.strip().str.lower(),
        "Amount EGP": to_num(_series(x, "Total_Amount_EGP", 0.0)),
        "Principal EGP": to_num(_series(x, "Principal_EGP", 0.0)),
        "Interest EGP": to_num(_series(x, "Interest_EGP", 0.0)),
        "Valuation Basis": _str(x, "Valuation_Basis", "EGP"),
        "Currency": _str(x, "Currency", "EGP"),
        "HQLA Flag": to_num(_series(x, "HQLA_Flag", 0)).astype(bool),
        "Bucket": _str(x, "ALM_Maturity_Bucket", "Unbucketed"),
        "Cashflow Type": _str(x, "Cashflow_Type"),
        "Product Code": _str(x, "Product_Code"),
        "Product Family": _str(x, "Product_Family"),
    })[cols]


# ----------------------------------------------------------------------------
# Core Position Repricing & Valuation Engine (100% Data-Driven, Zero IDs)
# ----------------------------------------------------------------------------
def _basis_shock_pct(basis: pd.Series, p: Dict[str, float]) -> np.ndarray:
    """Return the applicable valuation shock percentage (FX % or Gold %) per position."""
    b = basis.astype(str).str.strip().str.lower()
    return np.select([b.eq("fx"), b.eq("gold")], [p.get("fx_shock_pct", 0.0), p.get("gold_shock_pct", 0.0)], default=0.0)


def _credit_cost_rate_decimal(p: Dict[str, float]) -> float:
    """Compute macro credit-cost rate from adverse unemployment (+) and GDP (-) shocks."""
    unemp_up = max(p["unemployment_shock_pct"], 0.0)
    gdp_down = max(-p["gdp_shock_pct"], 0.0)
    return (CREDIT_COST_BPS_PER_UNEMP_PP * unemp_up + CREDIT_COST_BPS_PER_GDP_DROP_PP * gdp_down) / 10000.0


def run_position_engine(q: pd.DataFrame, simulation_date, horizon_days: int, params: Optional[dict] = None) -> pd.DataFrame:
    """Simulate per-position Before/After balances, interest, MTM, FX/Gold revaluation, and credit cost using pure DB attributes."""
    if q is None or q.empty:
        return pd.DataFrame()
    p = merge_params(params)
    sim = _norm_date(simulation_date)
    horizon = max(int(horizon_days), 0)
    h_end = sim + pd.Timedelta(days=horizon)
    out = q.copy()
    cat = out["Category"].astype(str).str.lower()
    bs_code = _str(out, "BS Line Code", "").str.strip().str.upper()
    acc_cls = _str(out, "Accounting Class", "AMORTISED_COST").str.strip().str.upper()

    # 1. Determine accrual window (with rollover at maturity if enabled)
    mat = out["Model Maturity Date"]
    matured = mat.notna() & (mat <= sim)
    mat_in_window = mat.notna() & (mat > sim) & (mat < h_end)
    accrual_end = pd.Series(h_end, index=out.index) if ROLLOVER_AT_MATURITY else mat.where(mat_in_window, h_end)
    days = (accrual_end - sim).dt.days.clip(lower=0).astype(float)
    excluded = matured if EXCLUDE_MATURED_POSITIONS else pd.Series(False, index=out.index)
    days = days.where(~matured, 0.0)

    # 2. Identify repricing eligibility and first repricing date within the horizon
    rt = out["Model Rate Type"].astype(str).str.lower()
    fixed = rt.str.contains("fixed|non-interest|non interest|zero interest", regex=True, na=False)
    floating = rt.str.contains("floating|variable|repric|adjustable", regex=True, na=False)
    eligible_type = ~fixed & (floating | cat.isin(["loan", "deposit", "funding", "investment"]))

    fd = out["Repricing Frequency Days"].astype(float).where(lambda s: s > 0)
    rep = out["Model Repricing Date"]
    need_roll = rep.notna() & (rep < sim)
    gap = (sim - rep).dt.days.astype(float)
    rolled = rep + pd.to_timedelta(np.ceil(gap / fd), unit="D")
    first = rep.where(~need_roll, rolled).where(~(need_roll & fd.isna()), sim).fillna(sim)

    reset_event = eligible_type & (first < accrual_end)
    maturity_event = ROLLOVER_AT_MATURITY & ~eligible_type & mat_in_window
    has_event = (days > 0) & (reset_event | maturity_event)
    first = first.where(eligible_type, mat)

    t1 = np.minimum(((first - sim).dt.days.astype(float)).clip(lower=0), days)
    t1 = pd.Series(np.where(has_event, t1, days), index=out.index)
    after_first = ((accrual_end - first).dt.days.astype(float)).clip(lower=0)
    per_freq = np.where(fd.notna() & eligible_type, np.ceil(after_first / fd.fillna(1.0)), 1.0)
    n_events = pd.Series(np.where(has_event, per_freq, 0.0), index=out.index).astype(int)

    # 3. Apply interest rate shock + wholesale funding spread shock dynamically to BS_LIAB_BORROW positions
    delta_rate = p["rate_change_pp"] / 100.0
    wsf_extra_rate = np.where(bs_code.eq("BS_LIAB_BORROW"), p.get("wsf_spread_shock_bps", 0.0) / 10000.0, 0.0)
    r0 = out["Model Rate"].astype(float)
    r1 = pd.Series(np.where(has_event, np.maximum(RATE_FLOOR, r0 + delta_rate + wsf_extra_rate), r0 + wsf_extra_rate), index=out.index)
    rate_profile = (r0 * days + (r1 - r0) * (days - t1)) / DAYS_PER_YEAR

    # 4. Calculate stressed balances (loan growth, deposit growth/run-off, and FX/Gold revaluation)
    b0 = out["Model Balance"].astype(float).where(~excluded, 0.0)
    growth = np.select(
        [cat.eq("loan"), cat.eq("deposit")],
        [
            1.0 + p["loan_growth_pct"] / 100.0,
            (1.0 + p["deposit_growth_pct"] / 100.0) * (1.0 - p["deposit_runoff_pct"] / 100.0),
        ],
        default=1.0,
    )
    growth = np.maximum(growth, 0.0)
    shock_factor = 1.0 + _basis_shock_pct(out["Valuation Basis"], p) / 100.0
    out["Applied Market Rate After"] = out["FX Rate"] * shock_factor
    b1 = b0 * growth * shock_factor

    # 5. Decompose interest change into Rate Effect and Volume/FX Effect
    interest_before = b0 * ((r0 * days) / DAYS_PER_YEAR)
    interest_rate_only = b0 * rate_profile
    interest_after = b1 * rate_profile

    # 6. Compute FX/Gold revaluation, Fair-Value Securities MTM (strictly via Accounting Class FVOCI/FVTPL), and Credit Cost
    sign = np.where(cat.isin(["loan", "investment"]), 1.0, np.where(cat.isin(["deposit", "funding"]), -1.0, 0.0))
    valuation_effect = sign * b0 * growth * (shock_factor - 1.0)

    # Pure accounting rule: Duration MTM applies only to fair-value debt securities (FVOCI / FVTPL excluding Gold)
    inv_fair_value = (
        acc_cls.isin(["FVOCI", "FVTPL"])
        & ~out["Valuation Basis"].astype(str).str.lower().eq("gold")
        & ~excluded
    )
    dur_years = ((first - sim).dt.days.astype(float).clip(lower=0) / DAYS_PER_YEAR / (1.0 + r0)).where(first.notna(), 0.0)
    mtm = pd.Series(np.where(inv_fair_value, -dur_years * delta_rate * b1, 0.0), index=out.index)

    cc_rate = _credit_cost_rate_decimal(p)
    credit_cost = pd.Series(np.where(cat.eq("loan"), b1 * cc_rate * horizon / DAYS_PER_YEAR, 0.0), index=out.index)

    out["Valuation Effect"] = valuation_effect
    out["Duration (yrs)"] = dur_years.where(inv_fair_value, 0.0)
    out["Investment MTM"] = mtm
    out["Credit Cost"] = credit_cost
    out["Excluded (matured)"] = excluded & matured
    out["Accrual Days"] = days
    out["Eligible"] = has_event
    out["First Repricing Date"] = first.where(has_event)
    out["Repricing Event Count"] = n_events
    out["Repricing Fraction"] = np.where(has_event & (days > 0), (days - t1) / days.where(days > 0, 1.0), 0.0)
    out["Before Balance"] = b0
    out["After Balance"] = b1
    out["Valuation Shock Factor"] = shock_factor
    out["Before Rate"] = r0
    out["After Rate"] = r1
    out["Applied Shock"] = r1 - r0
    out["Annual Interest Before"] = b0 * r0
    out["Annual Interest After"] = b1 * r1
    out["Interest Before"] = interest_before
    out["Interest After"] = interest_after
    out["Interest Change"] = interest_after - interest_before
    out["Rate Effect"] = interest_rate_only - interest_before
    out["Volume Effect"] = interest_after - interest_rate_only
    out["Annual Interest Change"] = out["Interest Change"] * DAYS_PER_YEAR / max(horizon, 1)
    finite = np.isfinite(out["Interest After"]) & np.isfinite(out["Interest Before"])
    out["Validation Status"] = np.where(finite, "PASS", "FAIL")
    return out


def summarize_positions(q: pd.DataFrame, phase: str, horizon_days: int) -> Dict[str, float]:
    """Aggregate position-level balances, interest income/expense, NIM %, CoF %, and Repricing Gap using DB attributes."""
    keys = [
        "NII", "Interest Income", "Interest Expense",
        "Loan Interest", "Investment Interest",
        "Deposit Interest", "Funding Interest",
        "Loans", "Deposits", "Funding", "Investments", "Earning Assets",
        "NIM %", "Cost of Funds %", "Loan Deposit %",
        "Repricing Gap", "HQLA Stock",
    ]
    if q is None or q.empty:
        return {**{k: 0.0 for k in keys}, "Horizon Days": int(horizon_days)}
    bal = f"{phase} Balance"
    intr_col = f"Interest {phase}"
    cat = q["Category"].astype(str).str.lower()
    bs_code = _str(q, "BS Line Code", "").str.strip().str.upper()
    isl, isd, isf, isi = cat.eq("loan"), cat.eq("deposit"), cat.eq("funding"), cat.eq("investment")

    li, ii = float(q.loc[isl, intr_col].sum()), float(q.loc[isi, intr_col].sum())
    di, fi = float(q.loc[isd, intr_col].sum()), float(q.loc[isf, intr_col].sum())
    loans, deposits = float(q.loc[isl, bal].sum()), float(q.loc[isd, bal].sum())
    funding, invest = float(q.loc[isf, bal].sum()), float(q.loc[isi, bal].sum())

    # Exclude non-earning Cash & CBE Reserve (BS_AST_CASH) dynamically from Earning Assets
    non_earning_cash = float(q.loc[bs_code.eq("BS_AST_CASH"), bal].sum())
    earning = max((loans + invest) - non_earning_cash, 0.0)

    income, expense = li + ii, di + fi
    nii = income - expense
    fund_base = deposits + funding
    ann = DAYS_PER_YEAR / max(int(horizon_days), 1)
    elig = q["Eligible"].astype(bool)
    gap = float(q.loc[(isl | isi) & elig, bal].sum()) - float(q.loc[(isd | isf) & elig, bal].sum())
    hqla = float(q.loc[isi & q["HQLA Flag"].astype(bool), bal].sum())
    return {
        "NII": nii, "Interest Income": income, "Interest Expense": expense,
        "Loan Interest": li, "Investment Interest": ii,
        "Deposit Interest": di, "Funding Interest": fi,
        "Loans": loans, "Deposits": deposits, "Funding": funding,
        "Investments": invest, "Earning Assets": earning,
        "NIM %": nii * ann / earning * 100.0 if earning else 0.0,
        "Cost of Funds %": expense * ann / fund_base * 100.0 if fund_base else 0.0,
        "Loan Deposit %": loans / deposits * 100.0 if deposits else 0.0,
        "Repricing Gap": gap, "HQLA Stock": hqla,
        "Horizon Days": int(horizon_days),
    }


# ----------------------------------------------------------------------------
# Cash-flow Liquidity & LCR Proxy Engine (Aligned with INV_Final_1)
# ----------------------------------------------------------------------------
def _window_mask(cf: pd.DataFrame, sim: pd.Timestamp, days: int) -> pd.Series:
    """Filter cash flows falling within [sim, sim + days]."""
    d = cf["Cashflow Date"]
    return d.notna() & (d >= sim) & (d <= sim + pd.Timedelta(days=int(days)))


def cashflow_metrics(cf: pd.DataFrame, q: pd.DataFrame, simulation_date, horizon_days: int,
                     params: Optional[dict] = None, stressed: bool = False) -> Dict[str, float]:
    """
    Calculate 30-day LCR Proxy and horizon liquidity metrics using dynamic DB flags & categories:
    LCR Proxy % = HQLA Stock / (30d Outflows - 30d Inflows) * 100
    """
    p = merge_params(params)
    sim = _norm_date(simulation_date)
    use = p if stressed else merge_params(None)

    cat = q["Category"].astype(str).str.lower() if q is not None and not q.empty else pd.Series(dtype=str)
    if q is not None and not q.empty:
        bal_col = "After Balance" if stressed else "Before Balance"
        bs_code = _str(q, "BS Line Code", "").str.strip().str.upper()
        pfam = _str(q, "Product Family", "").str.strip().str.lower()
        # HQLA is determined purely by HQLA Flag, Cash/CBE line (BS_AST_CASH), or Sovereign Debt / Precious Metals family
        is_hqla = (
            q["HQLA Flag"].astype(bool)
            | bs_code.eq("BS_AST_CASH")
            | pfam.isin(["sovereign debt", "precious metals"])
        )
        hqla = float(q.loc[is_hqla, bal_col].sum())
        total_deposits = float(q.loc[cat.eq("deposit"), bal_col].sum())
        deposits_pre_runoff = float((q.loc[cat.eq("deposit"), "Before Balance"] *
                                     q.loc[cat.eq("deposit"), "Valuation Shock Factor"]).sum())
    else:
        hqla = total_deposits = deposits_pre_runoff = 0.0

    runoff_extra = (use["deposit_runoff_pct"] / 100.0) * deposits_pre_runoff

    if cf is None or cf.empty:
        in_h = out_h = in_l = out_l = 0.0
        cf_asof = None
    else:
        amt = cf["Amount EGP"] * (1.0 + _basis_shock_pct(cf["Valuation Basis"], use) / 100.0)
        is_in, is_out = cf["Flow"].eq("inflow"), cf["Flow"].eq("outflow")
        wh = _window_mask(cf, sim, horizon_days)
        wl = _window_mask(cf, sim, LCR_WINDOW_DAYS)
        in_h, out_h = float(amt[wh & is_in].sum()), float(amt[wh & is_out].sum())
        in_l, out_l = float(amt[wl & is_in].sum()), float(amt[wl & is_out].sum())
        cf_asof = latest_date(cf, "Cashflow Date")

    out_h += runoff_extra
    gross_30d_outflow = out_l + runoff_extra

    # Exact INV_Final_1 formula: HQLA / (Outflows - Inflows) over 30 days
    net_30d_outflow = max(gross_30d_outflow - in_l, 1.0)
    lcr_proxy = (hqla / net_30d_outflow * 100.0) if net_30d_outflow > 1.0 else np.nan
    hqla_to_deposits = (hqla / total_deposits * 100.0) if total_deposits > 0 else np.nan

    return {
        "CF Inflow": in_h,
        "CF Outflow": out_h,
        "CF Net Cashflow": in_h - out_h,
        "CF Runoff Amount": runoff_extra,
        "CF HQLA": hqla,
        "CF LCR %": lcr_proxy,
        "HQLA to Deposits %": hqla_to_deposits,
        "LCR Inflow": in_l,
        "LCR Outflow": gross_30d_outflow,
        "LCR Net Outflow": net_30d_outflow,
        "Liquidity": in_h - out_h,
        "CF As Of": cf_asof,
    }


# ----------------------------------------------------------------------------
# Full Scenario Execution Pipeline
# ----------------------------------------------------------------------------
def _macro_block(macro: Dict[str, float], p: Dict[str, float]) -> Dict[str, dict]:
    """Build Before and After macroeconomic indicator dictionaries."""
    def pct(v):
        return float(v) * 100.0 if pd.notna(v) else np.nan
    pol, inf = pct(macro["policy_rate"]), pct(macro["inflation"])
    une, gdp = pct(macro["unemployment"]), pct(macro["gdp_growth"])
    pol_s = pol + p["rate_change_pp"] if pd.notna(pol) else np.nan
    inf_s = inf + p["inflation_shock_pct"] if pd.notna(inf) else np.nan
    return {
        "before": {
            "Macro Policy Rate %": pol, "Macro Inflation %": inf,
            "Macro Unemployment %": une, "Macro GDP Growth %": gdp,
            "Macro Real Policy Rate %": (pol - inf) if pd.notna(pol) and pd.notna(inf) else np.nan,
        },
        "after": {
            "Macro Policy Rate %": pol_s, "Macro Inflation %": inf_s,
            "Macro Unemployment %": (une + p["unemployment_shock_pct"]) if pd.notna(une) else np.nan,
            "Macro GDP Growth %": (gdp + p["gdp_shock_pct"]) if pd.notna(gdp) else np.nan,
            "Macro Real Policy Rate %": (pol_s - inf_s) if pd.notna(pol_s) and pd.notna(inf_s) else np.nan,
        },
    }


def prepare_inputs(positions: pd.DataFrame, cashflows: Optional[pd.DataFrame] = None,
                   market_data: Optional[pd.DataFrame] = None,
                   fs_flat: Optional[pd.DataFrame] = None) -> dict:
    """Pre-normalize positions, cash flows, market rates, and financial statements from SQL."""
    market_data = market_data if market_data is not None else pd.DataFrame()
    fs_flat = fs_flat if fs_flat is not None else pd.DataFrame()
    return {
        "q0": normalize_positions(positions if positions is not None else pd.DataFrame()),
        "cf": normalize_cashflows(cashflows if cashflows is not None else pd.DataFrame()),
        "macro": macro_snapshot(market_data),
        "valuation": latest_market_date(market_data),
        "market_data": market_data,
        "fs_flat": fs_flat,
    }


def apply_scenario(positions: pd.DataFrame, cashflows: Optional[pd.DataFrame] = None,
                   market_data: Optional[pd.DataFrame] = None, simulation_date=None,
                   horizon_days: int = 365, params: Optional[dict] = None,
                   fs_flat: Optional[pd.DataFrame] = None):
    """Run a scenario directly from raw DataFrames."""
    return apply_prepared(prepare_inputs(positions, cashflows, market_data, fs_flat), simulation_date, horizon_days, params)


def apply_prepared(prep: dict, simulation_date=None, horizon_days: int = 365, params: Optional[dict] = None, base_as_of=None):
    """Execute a full deterministic simulation run from prepared inputs."""
    p = merge_params(params)
    sim = _norm_date(simulation_date)
    horizon = max(int(horizon_days), 1)
    q0, cf = prep["q0"].copy(), prep["cf"]

    # Re-evaluate baseline FX and Gold (21K / 24K) rates by instrument_code if a historical reference date is given
    md = prep.get("market_data")
    ref_date = base_as_of if (base_as_of is not None and pd.notna(base_as_of)) else None
    if md is not None and not md.empty and ref_date is not None and not q0.empty:
        vr = valuation_rates(md, as_of=ref_date)
        if not vr.empty:
            inst_map = dict(zip(vr["instrument_code"].astype(str).str.upper(), vr["market_rate"]))
            ccy_upper = q0["Model Currency"].astype(str).str.upper()
            basis_lower = q0["Valuation Basis"].astype(str).str.lower()
            purity_upper = _str(q0, "Gold Purity", "24K").str.strip().str.upper()

            inst_key = np.where(
                basis_lower.eq("gold") & purity_upper.eq("21K"),
                "GOLD_21K_EGP",
                np.where(
                    basis_lower.eq("gold"),
                    "GOLD_24K_EGP",
                    np.where(ccy_upper.eq("EGP"), "EGP", "FX_" + ccy_upper + "_EGP"),
                ),
            )
            mapped_rates = pd.Series(inst_key, index=q0.index).map(lambda k: 1.0 if k == "EGP" else inst_map.get(k, np.nan))
            q0["FX Rate"] = mapped_rates.fillna(q0["FX Rate"])
            q0["Model Balance"] = q0["Native Balance"] * q0["FX Rate"]

    q = run_position_engine(q0, sim, horizon, p)

    before = summarize_positions(q, "Before", horizon)
    after = summarize_positions(q, "After", horizon)
    before.update(cashflow_metrics(cf, q, sim, horizon, p, stressed=False))
    after.update(cashflow_metrics(cf, q, sim, horizon, p, stressed=True))
    after["CF LCR Proxy Base %"] = before.get("CF LCR %", np.nan)
    after["Liquidity Base"] = before.get("CF Net Cashflow", 0.0)

    macro = _macro_block(prep["macro"], p)
    before.update(macro["before"])
    after.update(macro["after"])

    valuation = prep["valuation"]
    for d in (before, after):
        d["Position As Of"] = valuation if valuation is not None else sim
        d["Simulation Date"] = sim
        d["Baseline Unit"] = MODEL_AMOUNT_UNIT
        d["Model Amount Unit"] = MODEL_AMOUNT_UNIT

    def delta(cats, col_):
        if q is None or q.empty:
            return 0.0
        return float(q.loc[q["Category"].astype(str).str.lower().isin(cats), col_].sum())

    def nii_effect(col_):
        if q is None or q.empty:
            return 0.0
        c = q["Category"].astype(str).str.lower()
        sign = np.where(c.isin(["loan", "investment"]), 1.0, np.where(c.isin(["deposit", "funding"]), -1.0, 0.0))
        return float((q[col_] * sign).sum())

    credit = float(q["Credit Cost"].sum()) if not q.empty else 0.0
    reval = float(q["Valuation Effect"].sum()) if not q.empty else 0.0
    mtm = float(q["Investment MTM"].sum()) if not q.empty else 0.0
    d_nii = after["NII"] - before["NII"]
    cc_rate_pct = _credit_cost_rate_decimal(p) * 100.0

    before.update({
        "PnL NII Change": 0.0, "PnL Credit Cost": 0.0,
        "PnL FX Gold Revaluation": 0.0, "PnL Investment MTM": 0.0, "PnL Total Impact": 0.0,
        "Credit Cost Rate (%)": 0.0,
    })
    after.update({
        "PnL NII Change": d_nii,
        "PnL Credit Cost": -credit if credit else 0.0,
        "PnL FX Gold Revaluation": reval,
        "PnL Investment MTM": mtm,
        "PnL Total Impact": d_nii - credit + reval + mtm,
        "Credit Cost Rate (%)": cc_rate_pct,
        "Loan Interest Change": delta(["loan"], "Interest Change"),
        "Deposit Interest Change": delta(["deposit"], "Interest Change"),
        "Funding Interest Change": delta(["funding"], "Interest Change"),
        "Investment Interest Change": delta(["investment"], "Interest Change"),
        "NII Rate Effect": nii_effect("Rate Effect"),
        "NII Volume Effect": nii_effect("Volume Effect"),
        "Repricing Gap Base": before["Repricing Gap"],
        "Repricing Gap After": after["Repricing Gap"],
        "Repricing Eligible Positions": float(q["Eligible"].sum()) if not q.empty else 0.0,
        "Repricing Events Total": float(q["Repricing Event Count"].sum()) if not q.empty else 0.0,
        "Excluded Matured Positions": float(q["Excluded (matured)"].sum()) if not q.empty else 0.0,
        "Rate Shock (pp)": p["rate_change_pp"],
        "FX Shock (%)": p["fx_shock_pct"],
        "Gold Shock (%)": p["gold_shock_pct"],
        "FX Wholesale Spread Shock (bps)": p.get("wsf_spread_shock_bps", 0.0),
        "Deposit Run-off (%)": p["deposit_runoff_pct"],
        "Loan Growth (%)": p["loan_growth_pct"],
        "Deposit Growth (%)": p["deposit_growth_pct"],
        "Inflation Shock (pp)": p["inflation_shock_pct"],
        "Unemployment Shock (pp)": p["unemployment_shock_pct"],
        "GDP Shock (pp)": p["gdp_shock_pct"],
    })

    # Build multi-year dynamic financial statements and all 22 KPIs directly from SQL & positions
    fs_bundle = build_financial_statements_and_kpis(
        pos_detail_df=q,
        params=p,
        before_dict=before,
        after_dict=after,
        horizon_days=horizon,
        fs_flat_df=prep.get("fs_flat"),
        macro_dict=prep.get("macro"),
        sim_date=sim,
    )
    fs_kpis = fs_bundle["fs_kpis"]

    before.update(fs_kpis)
    after.update(fs_kpis)
    after["_fs_bundle"] = fs_bundle

    return before, after, q


def build_model_baseline(positions: pd.DataFrame, cashflows: pd.DataFrame,
                         market_data: Optional[pd.DataFrame] = None,
                         simulation_date=None, horizon_days: int = 365) -> Dict[str, float]:
    """Helper to compute baseline KPIs with neutral assumptions."""
    before, _, _ = apply_scenario(positions, cashflows, market_data, simulation_date, horizon_days, None)
    return before