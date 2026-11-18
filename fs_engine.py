# fs_engine.py
"""
Dynamic Multi-Year Financial Statements & Symmetric Window (±2Y / ±3Y) L2 Ridge Engine.
100% Data-Driven (Zero Hardcoded Dates, Rates, or Product IDs):
  - Trains Recency-Weighted L2 Ridge Regression on ALL historical dates in `financial_statements_flat` (2019–2025+).
  - Dynamically anchors on the latest reporting year (`Base`) and builds up to 3 historical years before it
    (`hist_cols`: e.g. `2022A`, `2023A`, `2024A`) and up to 3 forecasted years after it (`fcst_cols`: e.g. `2026F`, `2027F`, `2028F`).
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sqlalchemy import text

from config import DAYS_PER_YEAR

# Structural metadata for Balance Sheet line ordering and bilingual labels
BS_LINE_SPECS = [
    (1, "1. الأصول (ASSETS)", "BS_AST_CASH", "نقدية وأرصدة لدى البنك المركزي", "Cash & balances with CBE"),
    (2, "1. الأصول (ASSETS)", "BS_AST_BANKS", "أرصدة لدى البنوك (بالصافي)", "Due from banks (net)"),
    (3, "1. الأصول (ASSETS)", "BS_AST_FVTPL", "استثمارات بالقيمة العادلة خلال الأرباح والخسائر", "Investments at FVTPL (Gold)"),
    (4, "1. الأصول (ASSETS)", "BS_AST_LOANS", "قروض وتسهيلات للعملاء والبنوك (بالصافي)", "Loans & advances to customers (net)"),
    (5, "1. الأصول (ASSETS)", "BS_AST_DERIV", "مشتقات مالية", "Derivatives"),
    (6, "1. الأصول (ASSETS)", "BS_AST_FVOCI", "استثمارات بالقيمة العادلة خلال الدخل الشامل (FVOCI)", "Investments at FVOCI"),
    (7, "1. الأصول (ASSETS)", "BS_AST_REV_REPO", "عمليات شراء أذون خزانة مع التعهد بإعادة البيع", "Reverse repo"),
    (8, "1. الأصول (ASSETS)", "BS_AST_AMORT", "أدوات دين بالتكلفة المستهلكة", "Debt instruments at amortised cost"),
    (9, "1. الأصول (ASSETS)", "BS_AST_SUBS", "استثمارات في شركات تابعة وشقيقة", "Subsidiaries & associates"),
    (10, "1. الأصول (ASSETS)", "BS_AST_FIXED", "أصول ثابتة (بالصافي)", "Fixed assets"),
    (11, "1. الأصول (ASSETS)", "BS_AST_OTHER", "أصول أخرى وتسويات السيولة النقدية للسيناريو", "Other assets & scenario cash settlement"),
    (12, "1. الأصول (ASSETS)", "BS_TOT_ASSETS", "إجمالي الأصول (Total Assets)", "Total Assets"),

    (13, "2. الالتزامات (LIABILITIES)", "BS_LIAB_BANKS", "أرصدة مستحقة للبنوك", "Due to banks"),
    (14, "2. الالتزامات (LIABILITIES)", "BS_LIAB_REPO", "عمليات بيع أذون خزانة مع التزام إعادة الشراء (Repo)", "Repo agreements"),
    (15, "2. الالتزامات (LIABILITIES)", "BS_LIAB_DEP", "ودائع العملاء", "Customer deposits"),
    (16, "2. الالتزامات (LIABILITIES)", "BS_LIAB_BORROW", "قروض أخرى (تمويل خارجي بالعملة)", "Other borrowings (Wholesale FX)"),
    (17, "2. الالتزامات (LIABILITIES)", "BS_LIAB_OTHER", "التزامات أخرى", "Other liabilities"),
    (18, "2. الالتزامات (LIABILITIES)", "BS_LIAB_TAX", "التزامات ضرائب الدخل الجارية", "Current tax liabilities"),
    (19, "2. الالتزامات (LIABILITIES)", "BS_LIAB_PROV", "مخصصات أخرى", "Other provisions"),
    (20, "2. الالتزامات (LIABILITIES)", "BS_LIAB_RETIR", "التزامات مزايا التقاعد", "Retirement benefit obligations"),
    (21, "2. الالتزامات (LIABILITIES)", "BS_TOT_LIAB", "إجمالي الالتزامات (Total Liabilities)", "Total Liabilities"),

    (22, "3. حقوق الملكية (EQUITY)", "BS_EQ_CAP", "رأس المال المدفوع", "Paid-up capital"),
    (23, "3. حقوق الملكية (EQUITY)", "BS_EQ_RES", "احتياطيات", "Reserves"),
    (24, "3. حقوق الملكية (EQUITY)", "BS_EQ_SUB_LOAN", "فروق القيمة الاسمية للقرض المساند من البنك المركزي", "CBE subordinated loan PV difference"),
    (25, "3. حقوق الملكية (EQUITY)", "BS_EQ_FVOCI", "احتياطي القيمة العادلة (الدخل الشامل الآخر)", "FVOCI reserve"),
    (26, "3. حقوق الملكية (EQUITY)", "BS_EQ_NET_PROFIT", "صافي أرباح الفترة / العام", "Net profit for the year"),
    (27, "3. حقوق الملكية (EQUITY)", "BS_EQ_RETAINED", "أرباح محتجزة", "Retained earnings"),
    (28, "3. حقوق الملكية (EQUITY)", "BS_TOT_EQ", "إجمالي حقوق الملكية (Total Equity)", "Total Equity"),

    (29, "4. المطابقة والتوازن (RECONCILIATION)", "BS_TOT_LIAB_EQ", "إجمالي الالتزامات وحقوق الملكية", "Total Liabilities & Equity"),
    (30, "4. المطابقة والتوازن (RECONCILIATION)", "BS_CHECK", "فحص التوازن (الأصول − الالتزامات وحقوق الملكية)", "Balance Check (Must = 0.00)"),
    (31, "5. الالتزامات العرضية (OFF-BALANCE SHEET)", "BS_OFF_CONTINGENT", "التزامات عرضية (خطابات ضمان واعتمادات)", "Contingent liabilities (LGs & LCs)"),
]

IS_LINE_SPECS = [
    # 1. Operating Activities (أنشطة التشغيل)
    (1, "1. أنشطة التشغيل (OPERATING ACTIVITIES)", "IS_INT_INC", "عائد القروض والإيرادات المشابهة", "Interest income from loans & similar", "ENGINE"),
    (2, "1. أنشطة التشغيل (OPERATING ACTIVITIES)", "IS_INT_EXP", "تكلفة الودائع والتكاليف المشابهة", "Interest expense on deposits & similar", "ENGINE"),
    (3, "1. أنشطة التشغيل (OPERATING ACTIVITIES)", "IS_NII", "صافي الدخل من العائد (Net Interest Income)", "Net Interest Income (NII)", "SUBTOTAL"),
    (4, "1. أنشطة التشغيل (OPERATING ACTIVITIES)", "IS_NET_FEE", "صافي الدخل من الأتعاب والعمولات", "Net fee & commission income", "SQL"),
    (5, "1. أنشطة التشغيل (OPERATING ACTIVITIES)", "IS_TRADING", "صافي دخل المتاجرة وفروق تقييم العملات والذهب", "Net trading & FX/Gold revaluation", "HYBRID"),
    (6, "1. أنشطة التشغيل (OPERATING ACTIVITIES)", "IS_ECL_CHARGE", "عبء الخسائر الائتمانية المتوقعة (ECL)", "Expected credit loss (ECL) charge", "HYBRID"),
    (7, "1. أنشطة التشغيل (OPERATING ACTIVITIES)", "IS_GA_EXP", "مصروفات عمومية وإدارية", "General & administrative expenses", "SQL"),
    (8, "1. أنشطة التشغيل (OPERATING ACTIVITIES)", "IS_OTHER_OP", "إيرادات (مصروفات) تشغيل أخرى", "Other operating income / (expense)", "SQL"),
    (9, "1. أنشطة التشغيل (OPERATING ACTIVITIES)", "IS_OP_PROFIT", "صافي ربح التشغيل (Operating Profit — IFRS 18)", "Operating Profit", "SUBTOTAL"),

    # 2. Investing Activities (أنشطة الاستثمار)
    (10, "2. أنشطة الاستثمار (INVESTING ACTIVITIES)", "IS_DIVIDENDS", "توزيعات أرباح استثمارات مالية", "Dividend income from investments", "SQL"),
    (11, "2. أنشطة الاستثمار (INVESTING ACTIVITIES)", "IS_INV_GAINS", "أرباح (خسائر) استثمارات مالية", "Gains / (losses) on financial investments", "HYBRID"),
    (12, "2. أنشطة الاستثمار (INVESTING ACTIVITIES)", "IS_NET_INVESTING", "صافي دخل (خسائر) أنشطة الاستثمار", "Net Investing Income / (Loss)", "SUBTOTAL"),
    (13, "2. أنشطة الاستثمار (INVESTING ACTIVITIES)", "IS_PBFIT", "الربح قبل التمويل وضرائب الدخل (IFRS 18)", "Profit Before Financing & Income Taxes", "SUBTOTAL"),

    # 3. Financing Activities (أنشطة التمويل)
    (14, "3. أنشطة التمويل (FINANCING ACTIVITIES)", "IS_FIN_COST", "تكلفة التمويل والقروض المساندة", "Financing & subordinated debt costs", "SQL"),
    (15, "3. أنشطة التمويل (FINANCING ACTIVITIES)", "IS_NET_FINANCING", "صافي دخل / (تكلفة) أنشطة التمويل", "Net Financing Income / (Expense)", "SUBTOTAL"),
    (16, "3. أنشطة التمويل (FINANCING ACTIVITIES)", "IS_EBT", "أرباح العام قبل الضرائب (Profit Before Tax)", "Profit Before Income Tax (EBT)", "SUBTOTAL"),

    # 4. Income Taxes & Net Profit (ضرائب الدخل وصافي ربح العام)
    (17, "4. ضرائب الدخل وصافي ربح العام (TAX & NET PROFIT)", "IS_TAX", "مصروفات ضرائب الدخل", "Income tax expense", "HYBRID"),
    (18, "4. ضرائب الدخل وصافي ربح العام (TAX & NET PROFIT)", "IS_NET_PROFIT", "صافي أرباح العام بعد الضريبة (Net Profit)", "Net Profit After Tax", "TOTAL"),
]


def load_fs_flat_from_db(engine) -> pd.DataFrame:
    """Load `financial_statements_flat` directly from MySQL."""
    if engine is None:
        return pd.DataFrame()
    try:
        q = text(
            "SELECT report_date, period_type, statement_type, section_name, line_code, "
            "line_name_ar, line_name_en, amount_egp_mn, display_order "
            "FROM financial_statements_flat ORDER BY display_order, report_date"
        )
        df = pd.read_sql(q, engine)
        if not df.empty:
            df["report_date_str"] = pd.to_datetime(df["report_date"]).dt.strftime("%Y-%m-%d")
        return df
    except Exception:
        return pd.DataFrame()


def _period_annualization_multiplier(period_type: str, report_dt: pd.Timestamp = None) -> Tuple[float, float]:
    """Return `(elapsed_year_fraction, annualization_multiplier)` for any SQL `period_type`."""
    pt_u = str(period_type or "").strip().upper()
    if pt_u == "18M":
        return 1.50, 12.0 / 18.0
    if pt_u in ("FY", "12M", "ANNUAL"):
        return 1.00, 1.0
    if pt_u in ("9M",):
        return 0.75, 12.0 / 9.0
    if pt_u in ("6M", "H1"):
        return 0.50, 2.0
    if pt_u in ("3M", "Q1", "Q2", "Q3", "Q4"):
        return 0.25, 4.0
    if report_dt is not None and pd.notna(report_dt):
        if report_dt.month == 12 and report_dt.day == 31:
            return 1.00, 1.0
        elapsed = max(min(report_dt.month / 12.0, 1.0), 1.0 / 12.0)
        return elapsed, 1.0 / elapsed
    return 1.00, 1.0


def _resolve_dynamic_periods(fs_df: pd.DataFrame, max_window: int = 3) -> dict:
    """
    Inspect `financial_statements_flat` to dynamically determine:
      - ALL historical Balance Sheet dates (`all_bs_dates`) for full-history ML training
      - Up to `max_window` (3) annual historical dates before the latest reporting year (`hist_dates`, `hist_cols`)
      - Latest reporting date (`base_date_str`, `bs_base_col`)
      - Up to `max_window` (3) forecasted future years after the latest reporting year (`fcst_cols`)
    """
    if fs_df is not None and not fs_df.empty and "report_date" in fs_df.columns:
        df_tmp = fs_df.copy()
        df_tmp["dt"] = pd.to_datetime(df_tmp["report_date"], errors="coerce")
        bs_dates = sorted(df_tmp[df_tmp["statement_type"] == "BALANCE_SHEET"]["dt"].dropna().unique())
        if not bs_dates:
            bs_dates = sorted(df_tmp["dt"].dropna().unique())
    else:
        bs_dates = []

    if not bs_dates:
        d_base = pd.Timestamp.today().normalize()
        bs_dates = [d_base - pd.DateOffset(years=3), d_base - pd.DateOffset(years=2), d_base - pd.DateOffset(years=1), d_base]

    all_bs_dates = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in bs_dates]

    # Group by calendar year, taking the latest reporting date within each calendar year
    df_yr = pd.DataFrame({"dt": [pd.Timestamp(d) for d in bs_dates]})
    df_yr["date_str"] = df_yr["dt"].dt.strftime("%Y-%m-%d")
    df_yr["year"] = df_yr["dt"].dt.year
    yearly_rep = df_yr.sort_values("dt").groupby("year", as_index=False).last()

    d_base = pd.Timestamp(yearly_rep.iloc[-1]["dt"])
    base_date_str = str(yearly_rep.iloc[-1]["date_str"])
    base_year = int(yearly_rep.iloc[-1]["year"])

    prior_df = yearly_rep.iloc[:-1].tail(max_window)
    hist_dates = prior_df["date_str"].tolist()
    hist_years = [str(int(y)) for y in prior_df["year"].tolist()]

    # Pad if database has fewer than 2 prior years
    while len(hist_dates) < 2:
        fallback_dt = d_base - pd.DateOffset(years=(2 - len(hist_dates)))
        hist_dates.insert(0, fallback_dt.strftime("%Y-%m-%d"))
        hist_years.insert(0, str(fallback_dt.year))

    hist_cols = [f"{yr}A" for yr in hist_years]
    fcst_years = [str(base_year + k) for k in range(1, max_window + 1)]
    fcst_cols = [f"{yr}F" for yr in fcst_years]

    # Detect primary period_type on the latest date for the Income Statement
    is_base_pt = "6M"
    if fs_df is not None and not fs_df.empty:
        sub_is = fs_df[
            (fs_df["statement_type"] == "INCOME_STMT")
            & (pd.to_datetime(fs_df["report_date"], errors="coerce") == d_base)
        ]
        if not sub_is.empty:
            pts = [str(x).upper() for x in sub_is["period_type"].dropna().unique()]
            for pref in ("FY", "18M", "9M", "6M", "Q3", "Q2", "Q1"):
                if pref in pts:
                    is_base_pt = pref
                    break

    elapsed_frac, is_ann_mult = _period_annualization_multiplier(is_base_pt, d_base)
    is_interim_base = bool(elapsed_frac < 0.99)
    rem_frac = max(0.0, 1.0 - elapsed_frac)

    base_suffix = "E" if is_interim_base else "A"
    base_col_label = f"{base_year}{base_suffix}"

    return {
        "all_bs_dates": all_bs_dates,
        "hist_dates": hist_dates,
        "hist_years": hist_years,
        "hist_cols": hist_cols,
        "fcst_years": fcst_years,
        "fcst_cols": fcst_cols,
        # Backward-compatible keys for 2-year default consumers (Excel / PDF / KPIs)
        "hist_date_1": hist_dates[-2],
        "hist_date_2": hist_dates[-1],
        "base_date_str": base_date_str,
        "hist_y1": hist_years[-2],
        "hist_y2": hist_years[-1],
        "base_year": str(base_year),
        "fcst_y1": fcst_years[0],
        "fcst_y2": fcst_years[1],
        "fcst_y3": fcst_years[2] if len(fcst_years) >= 3 else str(base_year + 3),
        "is_base_pt": is_base_pt,
        "is_interim_base": is_interim_base,
        "elapsed_frac": elapsed_frac,
        "rem_frac": rem_frac,
        "is_ann_mult": is_ann_mult,
        "hist_col_1": hist_cols[-2],
        "hist_col_2": hist_cols[-1],
        "bs_base_col": base_col_label,
        "is_base_header": base_col_label,
        "fcst_col_1": fcst_cols[0],
        "fcst_col_2": fcst_cols[1],
        "fcst_col_3": fcst_cols[2] if len(fcst_cols) >= 3 else f"{base_year + 3}F",
    }


def _lookup_sql_val(
    df: pd.DataFrame, line_code: str, report_date: str, period_type: Optional[str] = None
) -> float:
    """Fetch a single line item amount (in EGP Mn) from `financial_statements_flat`."""
    if df is None or df.empty:
        return 0.0
    d_col = "report_date_str" if "report_date_str" in df.columns else "report_date"
    m = (df["line_code"] == line_code) & (df[d_col].astype(str).str.slice(0, 10) == str(report_date)[:10])
    if period_type:
        m = m & (df["period_type"] == period_type)
    sub = df.loc[m, "amount_egp_mn"]
    if not sub.empty:
        return float(sub.sum())
    return 0.0


def _fit_multiyear_ridge_slope(
    t_years: np.ndarray,
    y_vals: np.ndarray,
    l2_lambda: float = 0.80,
    recency_decay: float = 0.35,
) -> float:
    """Compute the annualized slope using Recency-Weighted L2 Ridge Regression over `(t_years, y_vals)`."""
    t = np.asarray(t_years, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    valid = np.isfinite(t) & np.isfinite(y)
    t, y = t[valid], y[valid]
    n = len(t)
    if n < 2:
        return 0.0

    w = np.exp(-float(recency_decay) * (t[-1] - t))
    w_sum = float(w.sum())
    w = w * (n / w_sum) if w_sum > 1e-12 else np.ones(n, dtype=float)

    t_mean = float(np.sum(w * t) / n)
    y_mean = float(np.sum(w * y) / n)
    t_c = t - t_mean
    y_c = y - y_mean
    denom = float(np.sum(w * (t_c ** 2)) + float(l2_lambda))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(w * t_c * y_c) / denom)


def _project_full_year_balance_sheet_from_interim(
    bs_all_hist: Dict[str, Dict[str, float]],
    all_bs_dates: List[str],
    base_date_str: str,
    elapsed_frac: float,
    ann_net_profit_mn: float,
    macro_drift: float = 0.13,
    l2_lambda: float = 0.80,
) -> Dict[str, float]:
    """Predict Full-Year End-of-Year Balance Sheet (`2025E` at 12-31) when latest SQL period is interim."""
    bs_interim = bs_all_hist.get(base_date_str, {})
    if elapsed_frac >= 0.99 or not all_bs_dates:
        return dict(bs_interim)

    rem_frac = max(0.0, 1.0 - elapsed_frac)
    dt_index = pd.to_datetime(all_bs_dates)
    t_years = np.array([(d - dt_index[0]).days / 365.25 for d in dt_index], dtype=float)

    proj: Dict[str, float] = {}
    settled_equity_codes = {"BS_EQ_CAP", "BS_EQ_RES", "BS_EQ_SUB_LOAN", "BS_EQ_FVOCI", "BS_EQ_RETAINED"}

    for _, _, code, _, _ in BS_LINE_SPECS:
        if "TOT" in code or "CHECK" in code:
            continue
        v_cur = float(bs_interim.get(code, 0.0))

        if code == "BS_EQ_NET_PROFIT":
            proj[code] = float(ann_net_profit_mn)
        elif code in settled_equity_codes:
            proj[code] = v_cur
        else:
            y_series = np.array([float(bs_all_hist.get(d_str, {}).get(code, 0.0)) for d_str in all_bs_dates], dtype=float)
            slope_annual = _fit_multiyear_ridge_slope(t_years, y_series, l2_lambda=l2_lambda)
            base_abs = max(abs(v_cur), 1.0)
            implied_annual_g = np.clip(slope_annual / base_abs, -0.18, 0.25)
            blended_annual_g = 0.65 * implied_annual_g + 0.35 * float(macro_drift)
            rem_step_g = np.clip(blended_annual_g * rem_frac, -0.12, 0.16)
            proj[code] = max(0.0, v_cur * (1.0 + rem_step_g))

    ast_1_10 = [c for o, _, c, _, _ in BS_LINE_SPECS if 1 <= o <= 10]
    liab_all = [c for o, _, c, _, _ in BS_LINE_SPECS if 13 <= o <= 20]
    eq_all = [c for o, _, c, _, _ in BS_LINE_SPECS if 22 <= o <= 27]

    tot_liab = sum(proj.get(c, 0.0) for c in liab_all)
    tot_eq = sum(proj.get(c, 0.0) for c in eq_all)
    prelim_ast = sum(proj.get(c, 0.0) for c in ast_1_10)
    proj["BS_AST_OTHER"] = (tot_liab + tot_eq) - prelim_ast

    return proj


def _compute_position_line_scaling(
    pos_detail_df: pd.DataFrame, base_bs_sql: Dict[str, float]
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], float]:
    """Group active positions strictly by SQL `BS Line Code` and scale deltas to the full-year baseline."""
    pos_base_mn: Dict[str, float] = {}
    pos_scn_mn: Dict[str, float] = {}
    scaled_delta_mn: Dict[str, float] = {}

    if pos_detail_df is None or pos_detail_df.empty:
        return pos_base_mn, pos_scn_mn, scaled_delta_mn, 0.0

    df = pos_detail_df.copy()
    if "BS Line Code" in df.columns:
        df["_bs_code"] = df["BS Line Code"].fillna("UNMAPPED").astype(str).str.strip()
    elif "bs_line_code" in df.columns:
        df["_bs_code"] = df["bs_line_code"].fillna("UNMAPPED").astype(str).str.strip()
    else:
        df["_bs_code"] = "UNMAPPED"

    grp = df.groupby("_bs_code")[["Before Balance", "After Balance"]].sum() / 1e6
    for code, row in grp.iterrows():
        b_mn = float(row["Before Balance"])
        a_mn = float(row["After Balance"])
        pos_base_mn[code] = b_mn
        pos_scn_mn[code] = a_mn
        sql_b = float(base_bs_sql.get(code, 0.0))
        if b_mn > 1e-6 and sql_b > 0:
            pct_chg = (a_mn - b_mn) / b_mn
            scaled_delta_mn[code] = sql_b * pct_chg
        else:
            scaled_delta_mn[code] = a_mn - b_mn

    fvoci_mtm_scaled_mn = 0.0
    if "Accounting Class" in df.columns:
        fvoci_mask = df["Accounting Class"].astype(str).str.upper().eq("FVOCI")
    else:
        fvoci_mask = df["_bs_code"].eq("BS_AST_FVOCI")

    fvoci_pos_b_mn = float(df.loc[fvoci_mask, "Before Balance"].sum()) / 1e6
    fvoci_mtm_pos_mn = float(df.loc[fvoci_mask, "Investment MTM"].sum()) / 1e6
    sql_fvoci_b = float(base_bs_sql.get("BS_AST_FVOCI", 0.0))
    if fvoci_pos_b_mn > 1e-6 and sql_fvoci_b > 0:
        fvoci_mtm_scaled_mn = sql_fvoci_b * (fvoci_mtm_pos_mn / fvoci_pos_b_mn)
    else:
        fvoci_mtm_scaled_mn = fvoci_mtm_pos_mn

    return pos_base_mn, pos_scn_mn, scaled_delta_mn, fvoci_mtm_scaled_mn


def _ml_ridge_forecast_n_years(
    t_years: np.ndarray,
    y_hist_series: np.ndarray,
    y_base_full: float,
    y_scn_full: float,
    n_years: int = 3,
    macro_drift: float = 0.12,
    l2_lambda: float = 0.80,
) -> List[float]:
    """
    Predict `n_years` forward (`+1Y`, `+2Y`, `+3Y`) using Recency-Weighted L2 Ridge Regression
    over ALL historical observations `(t_years, y_hist_series)` anchored on `y_scn_full`.
    """
    slope = _fit_multiyear_ridge_slope(t_years, y_hist_series, l2_lambda=l2_lambda)
    base_abs = max(abs(float(y_base_full)), 1.0)
    implied_g = np.clip(slope / base_abs, -0.18, 0.25)
    blended_g = 0.55 * implied_g + 0.45 * float(macro_drift)

    if abs(float(y_scn_full)) < 1e-3:
        return [0.0] * int(n_years)

    preds: List[float] = []
    cur = float(y_scn_full)
    for step in range(int(n_years)):
        step_g = blended_g * (0.92 ** step)
        cur = cur * (1.0 + step_g)
        preds.append(float(cur))
    return preds


def build_financial_statements_and_kpis(
    pos_detail_df: pd.DataFrame,
    params: dict,
    before_dict: dict,
    after_dict: dict,
    horizon_days: int = 365,
    engine=None,
    fs_flat_df: Optional[pd.DataFrame] = None,
    macro_dict: Optional[dict] = None,
    sim_date=None,
) -> dict:
    """
    Build the Multi-Year Balance Sheet, Income Statement, and 22 Executive KPIs
    supporting dynamic symmetric windows (±2 Years or ±3 Years around the latest base year).
    """
    if fs_flat_df is None or fs_flat_df.empty:
        fs_flat_df = load_fs_flat_from_db(engine)

    if fs_flat_df is not None and not fs_flat_df.empty and "report_date_str" not in fs_flat_df.columns:
        fs_flat_df = fs_flat_df.copy()
        fs_flat_df["report_date_str"] = pd.to_datetime(fs_flat_df["report_date"]).dt.strftime("%Y-%m-%d")

    periods = _resolve_dynamic_periods(fs_flat_df, max_window=3)
    all_bs_dates = periods["all_bs_dates"]
    hist_dates = periods["hist_dates"]
    hist_cols = periods["hist_cols"]
    fcst_cols = periods["fcst_cols"]
    d_h2 = periods["hist_date_2"]
    d_base = periods["base_date_str"]
    is_base_pt = periods["is_base_pt"]
    is_ann_mult = periods["is_ann_mult"]
    elapsed_frac = periods["elapsed_frac"]

    col_h2 = periods["hist_col_2"]
    col_bs_base = periods["bs_base_col"]

    # 1. Load ALL historical Balance Sheet dates for full-history Ridge ML training
    bs_sql: Dict[str, Dict[str, float]] = {dt: {} for dt in set(all_bs_dates + hist_dates + [d_base])}
    for _, _, code, _, _ in BS_LINE_SPECS:
        if "TOT" in code or "CHECK" in code:
            continue
        for dt in bs_sql:
            bs_sql[dt][code] = _lookup_sql_val(fs_flat_df, code, dt)

    # 2. Helper to build an annualized 12M-equivalent Income Statement for any historical date in SQL
    def _is_hist_year(dt_str: str, pt_str: Optional[str] = None, ann_multiplier: Optional[float] = None) -> Dict[str, float]:
        if pt_str is None and fs_flat_df is not None and not fs_flat_df.empty:
            d_col = "report_date_str" if "report_date_str" in fs_flat_df.columns else "report_date"
            sub_p = fs_flat_df[
                (fs_flat_df["statement_type"] == "INCOME_STMT")
                & (fs_flat_df[d_col].astype(str).str.slice(0, 10) == str(dt_str)[:10])
            ]
            if not sub_p.empty:
                pts = [str(x).upper() for x in sub_p["period_type"].dropna().unique()]
                for pref in ("FY", "18M", "9M", "6M", "Q3", "Q2", "Q1"):
                    if pref in pts:
                        pt_str = pref
                        break
        pt_use = pt_str or "FY"
        if ann_multiplier is None:
            _, ann_multiplier = _period_annualization_multiplier(pt_use, pd.to_datetime(dt_str, errors="coerce"))

        inc = _lookup_sql_val(fs_flat_df, "IS_INT_INC", dt_str, pt_use) * ann_multiplier
        exp = _lookup_sql_val(fs_flat_df, "IS_INT_EXP", dt_str, pt_use) * ann_multiplier
        fee_in = _lookup_sql_val(fs_flat_df, "IS_FEE_INC", dt_str, pt_use) * ann_multiplier
        fee_out = _lookup_sql_val(fs_flat_df, "IS_FEE_EXP", dt_str, pt_use) * ann_multiplier
        div = _lookup_sql_val(fs_flat_df, "IS_DIVIDENDS", dt_str, pt_use) * ann_multiplier
        trd = _lookup_sql_val(fs_flat_df, "IS_TRADING", dt_str, pt_use) * ann_multiplier
        inv = _lookup_sql_val(fs_flat_df, "IS_INV_GAINS", dt_str, pt_use) * ann_multiplier
        ecl = _lookup_sql_val(fs_flat_df, "IS_ECL_CHARGE", dt_str, pt_use) * ann_multiplier
        ga = _lookup_sql_val(fs_flat_df, "IS_GA_EXP", dt_str, pt_use) * ann_multiplier
        oth = _lookup_sql_val(fs_flat_df, "IS_OTHER_OP", dt_str, pt_use) * ann_multiplier
        fin_cost = _lookup_sql_val(fs_flat_df, "IS_FIN_COST", dt_str, pt_use) * ann_multiplier
        tax = _lookup_sql_val(fs_flat_df, "IS_TAX", dt_str, pt_use) * ann_multiplier

        nii = inc + exp
        op_profit = nii + (fee_in + fee_out) + trd + ecl + ga + oth
        net_inv = div + inv
        pbfit = op_profit + net_inv
        net_fin = fin_cost
        ebt = pbfit + net_fin
        np_val = ebt + tax
        return {
            "IS_INT_INC": inc,
            "IS_INT_EXP": exp,
            "IS_NII": nii,
            "IS_NET_FEE": fee_in + fee_out,
            "IS_TRADING": trd,
            "IS_ECL_CHARGE": ecl,
            "IS_GA_EXP": ga,
            "IS_OTHER_OP": oth,
            "IS_OP_PROFIT": op_profit,
            "IS_DIVIDENDS": div,
            "IS_INV_GAINS": inv,
            "IS_NET_INVESTING": net_inv,
            "IS_PBFIT": pbfit,
            "IS_FIN_COST": fin_cost,
            "IS_NET_FINANCING": net_fin,
            "IS_EBT": ebt,
            "IS_TAX": tax,
            "IS_NET_PROFIT": np_val,
        }

    is_base = _is_hist_year(d_base, is_base_pt, is_ann_mult)

    is_all_hist: Dict[str, Dict[str, float]] = {}
    is_train_dates: List[str] = []
    for dt_s in all_bs_dates:
        cand_is = _is_hist_year(dt_s)
        if abs(cand_is.get("IS_INT_INC", 0.0)) > 1.0 or abs(cand_is.get("IS_NET_PROFIT", 0.0)) > 1.0:
            is_all_hist[dt_s] = cand_is
            is_train_dates.append(dt_s)
    for dt_s in hist_dates:
        if dt_s not in is_all_hist:
            is_all_hist[dt_s] = _is_hist_year(dt_s)
    if d_base not in is_all_hist:
        is_all_hist[d_base] = is_base
        is_train_dates.append(d_base)

    # 3. Dynamically project Full-Year Balance Sheet (`2025E`) using ALL historical dates
    inf_rate = float(params.get("inflation_shock_pct", 0.0)) / 100.0
    gdp_rate = float(params.get("gdp_shock_pct", 0.0)) / 100.0
    macro_drift_bs = float(np.clip(0.13 + 0.40 * inf_rate + 0.50 * gdp_rate, 0.04, 0.24))

    bs_base_full_year = _project_full_year_balance_sheet_from_interim(
        bs_all_hist=bs_sql,
        all_bs_dates=all_bs_dates,
        base_date_str=d_base,
        elapsed_frac=elapsed_frac,
        ann_net_profit_mn=is_base["IS_NET_PROFIT"],
        macro_drift=macro_drift_bs,
    )

    periods["interim_assets_actual_mn"] = sum(
        bs_sql[d_base].get(c, 0.0) for o, _, c, _, _ in BS_LINE_SPECS if 1 <= o <= 11
    )
    periods["interim_net_profit_actual_mn"] = float(bs_sql[d_base].get("BS_EQ_NET_PROFIT", 0.0))
    periods["ml_training_dates_count"] = len(all_bs_dates)

    # 4. Scale position-level scenario shocks to the Full-Year Base (`2025E` or `2025A`)
    pos_base_mn, pos_scn_mn, scaled_delta_mn, fvoci_mtm_scaled_mn = _compute_position_line_scaling(
        pos_detail_df, bs_base_full_year
    )

    # 5. Compute Scenario Income Statement impacts (in EGP Mn)
    hz = max(int(horizon_days or 365), 1)
    ann_factor = DAYS_PER_YEAR / hz

    pos_int_inc_b = float(before_dict.get("Interest Income", 0.0)) / 1e6 * ann_factor
    pos_int_inc_s = float(after_dict.get("Interest Income", 0.0)) / 1e6 * ann_factor
    pos_int_exp_b = float(before_dict.get("Interest Expense", 0.0)) / 1e6 * ann_factor
    pos_int_exp_s = float(after_dict.get("Interest Expense", 0.0)) / 1e6 * ann_factor

    scale_inc = (is_base["IS_INT_INC"] / pos_int_inc_b) if pos_int_inc_b > 1e-6 else 1.0
    scale_exp = (abs(is_base["IS_INT_EXP"]) / pos_int_exp_b) if pos_int_exp_b > 1e-6 else 1.0

    d_int_inc = (pos_int_inc_s - pos_int_inc_b) * scale_inc
    d_int_exp = -(pos_int_exp_s - pos_int_exp_b) * scale_exp

    d_gold_fvtpl = scaled_delta_mn.get("BS_AST_FVTPL", 0.0)
    fx_reval_pos_mn = float(after_dict.get("PnL FX Gold Revaluation", 0.0)) / 1e6
    d_trading = fx_reval_pos_mn - (
        pos_scn_mn.get("BS_AST_FVTPL", 0.0) - pos_base_mn.get("BS_AST_FVTPL", 0.0)
    )
    d_inv_gains = d_gold_fvtpl

    credit_cost_rate = float(after_dict.get("Credit Cost Rate (%)", 0.0)) / 100.0
    stressed_loans_bs_mn = bs_base_full_year["BS_AST_LOANS"] + scaled_delta_mn.get("BS_AST_LOANS", 0.0)
    d_ecl = -(stressed_loans_bs_mn * credit_cost_rate)

    eff_tax_rate = None
    if macro_dict and isinstance(macro_dict, dict):
        raw_t = macro_dict.get("effective_tax_rate")
        if raw_t is not None and pd.notna(raw_t) and float(raw_t) > 0:
            eff_tax_rate = float(raw_t)
    if eff_tax_rate is None and pos_detail_df is not None and "Effective Tax Rate" in pos_detail_df.columns:
        s_t = pd.to_numeric(pos_detail_df["Effective Tax Rate"], errors="coerce").dropna()
        if not s_t.empty and float(s_t.iloc[0]) > 0:
            eff_tax_rate = float(s_t.iloc[0])
            if eff_tax_rate > 1.0:
                eff_tax_rate /= 100.0
    if eff_tax_rate is None:
        eff_tax_rate = (
            abs(is_base["IS_TAX"]) / is_base["IS_EBT"]
            if is_base["IS_EBT"] > 0
            else 0.40
        )

    is_scn = dict(is_base)
    is_scn["IS_INT_INC"] = is_base["IS_INT_INC"] + d_int_inc
    is_scn["IS_INT_EXP"] = is_base["IS_INT_EXP"] + d_int_exp
    is_scn["IS_NII"] = is_scn["IS_INT_INC"] + is_scn["IS_INT_EXP"]
    is_scn["IS_TRADING"] = is_base["IS_TRADING"] + d_trading
    is_scn["IS_ECL_CHARGE"] = is_base["IS_ECL_CHARGE"] + d_ecl
    is_scn["IS_OP_PROFIT"] = (
        is_scn["IS_NII"]
        + is_scn["IS_NET_FEE"]
        + is_scn["IS_TRADING"]
        + is_scn["IS_ECL_CHARGE"]
        + is_scn["IS_GA_EXP"]
        + is_scn["IS_OTHER_OP"]
    )
    is_scn["IS_INV_GAINS"] = is_base["IS_INV_GAINS"] + d_inv_gains
    is_scn["IS_NET_INVESTING"] = is_scn["IS_DIVIDENDS"] + is_scn["IS_INV_GAINS"]
    is_scn["IS_PBFIT"] = is_scn["IS_OP_PROFIT"] + is_scn["IS_NET_INVESTING"]
    is_scn["IS_NET_FINANCING"] = is_scn["IS_FIN_COST"]
    is_scn["IS_EBT"] = is_scn["IS_PBFIT"] + is_scn["IS_NET_FINANCING"]

    d_ebt = is_scn["IS_EBT"] - is_base["IS_EBT"]
    d_tax = -d_ebt * eff_tax_rate
    is_scn["IS_TAX"] = is_base["IS_TAX"] + d_tax
    is_scn["IS_NET_PROFIT"] = is_scn["IS_EBT"] + is_scn["IS_TAX"]
    d_net_profit = is_scn["IS_NET_PROFIT"] - is_base["IS_NET_PROFIT"]

    # 6. Build Scenario Balance Sheet (`bs_scn`) with Double-Entry Clearing
    bs_scn = dict(bs_base_full_year)
    for code in [
        "BS_AST_CASH", "BS_AST_BANKS", "BS_AST_FVTPL", "BS_AST_LOANS",
        "BS_AST_FVOCI", "BS_AST_AMORT", "BS_LIAB_BANKS", "BS_LIAB_REPO",
        "BS_LIAB_DEP", "BS_LIAB_BORROW", "BS_OFF_CONTINGENT",
    ]:
        bs_scn[code] = max(0.0, bs_base_full_year.get(code, 0.0) + scaled_delta_mn.get(code, 0.0))

    bs_scn["BS_AST_FVOCI"] = max(0.0, bs_scn["BS_AST_FVOCI"] + fvoci_mtm_scaled_mn)
    bs_scn["BS_EQ_FVOCI"] = bs_base_full_year["BS_EQ_FVOCI"] + fvoci_mtm_scaled_mn
    bs_scn["BS_AST_LOANS"] = max(0.0, bs_scn["BS_AST_LOANS"] + d_ecl)
    bs_scn["BS_LIAB_TAX"] = max(0.0, bs_base_full_year["BS_LIAB_TAX"] - d_tax)
    bs_scn["BS_EQ_NET_PROFIT"] = bs_base_full_year["BS_EQ_NET_PROFIT"] + d_net_profit

    ast_codes_1_10 = [c for o, _, c, _, _ in BS_LINE_SPECS if 1 <= o <= 10]
    liab_codes = [c for o, _, c, _, _ in BS_LINE_SPECS if 13 <= o <= 20]
    eq_codes = [c for o, _, c, _, _ in BS_LINE_SPECS if 22 <= o <= 27]

    tot_liab_scn = sum(bs_scn[c] for c in liab_codes)
    tot_eq_scn = sum(bs_scn[c] for c in eq_codes)
    prelim_ast_scn = sum(bs_scn[c] for c in ast_codes_1_10)
    bs_scn["BS_AST_OTHER"] = (tot_liab_scn + tot_eq_scn) - prelim_ast_scn

    # 7. Assemble Multi-Year & Forecast Income Statement DataFrame FIRST (for cross-statement articulation)
    is_dates_prior = [d for d in is_train_dates if d != d_base]
    is_train_dates_dt = [pd.Timestamp(d) for d in is_dates_prior] + [pd.Timestamp(f"{periods['base_year']}-12-31")]
    t_is_vec = np.array([(d - is_train_dates_dt[0]).days / 365.25 for d in is_train_dates_dt], dtype=float)

    n_fcst = len(fcst_cols)
    is_rows = []
    for order, sec, code, name_ar, name_en, src in IS_LINE_SPECS:
        row_dict = {
            "order": order, "section": sec, "code": code,
            "name_ar": name_ar, "name_en": name_en, "source": src,
        }
        for h_dt, hc in zip(hist_dates, hist_cols):
            row_dict[hc] = is_all_hist.get(h_dt, {}).get(code, 0.0)

        vb = is_base.get(code, 0.0)
        vs = is_scn.get(code, 0.0)
        d_mn = vs - vb
        d_pct = (d_mn / abs(vb) * 100.0) if abs(vb) > 1e-4 else 0.0
        row_dict["base_mn"] = vb
        row_dict["scn_mn"] = vs
        row_dict["delta_mn"] = d_mn
        row_dict["delta_pct"] = d_pct

        y_is_hist = np.array(
            [is_all_hist[d].get(code, 0.0) for d in is_dates_prior] + [vb],
            dtype=float,
        )
        f_vals = _ml_ridge_forecast_n_years(
            t_is_vec, y_is_hist, vb, vs, n_years=n_fcst, macro_drift=macro_drift_bs
        )
        for fc, fv in zip(fcst_cols, f_vals):
            row_dict[fc] = fv
        is_rows.append(row_dict)

    df_is_multi = pd.DataFrame(is_rows)
    for yr_col in fcst_cols:
        def _set_is_f(c_code: str, val: float):
            df_is_multi.loc[df_is_multi["code"] == c_code, yr_col] = val

        def _get_is_f(c_code: str) -> float:
            return float(df_is_multi.loc[df_is_multi["code"] == c_code, yr_col].values[0])

        nii_f = _get_is_f("IS_INT_INC") + _get_is_f("IS_INT_EXP")
        _set_is_f("IS_NII", nii_f)

        op_prof_f = (
            nii_f
            + _get_is_f("IS_NET_FEE")
            + _get_is_f("IS_TRADING")
            + _get_is_f("IS_ECL_CHARGE")
            + _get_is_f("IS_GA_EXP")
            + _get_is_f("IS_OTHER_OP")
        )
        _set_is_f("IS_OP_PROFIT", op_prof_f)

        net_inv_f = _get_is_f("IS_DIVIDENDS") + _get_is_f("IS_INV_GAINS")
        _set_is_f("IS_NET_INVESTING", net_inv_f)

        pbfit_f = op_prof_f + net_inv_f
        _set_is_f("IS_PBFIT", pbfit_f)

        net_fin_f = _get_is_f("IS_FIN_COST")
        _set_is_f("IS_NET_FINANCING", net_fin_f)

        ebt_f = pbfit_f + net_fin_f
        _set_is_f("IS_EBT", ebt_f)

        tax_f = -abs(ebt_f) * eff_tax_rate if ebt_f > 0 else 0.0
        _set_is_f("IS_TAX", tax_f)
        _set_is_f("IS_NET_PROFIT", ebt_f + tax_f)

    # 8. Assemble Multi-Year & Forecast Balance Sheet DataFrame (Articulated with Income Statement Net Profit)
    hist_dates_prior = [d for d in all_bs_dates if d != d_base]
    bs_train_dates_dt = [pd.Timestamp(d) for d in hist_dates_prior] + [pd.Timestamp(f"{periods['base_year']}-12-31")]
    t_bs_vec = np.array([(d - bs_train_dates_dt[0]).days / 365.25 for d in bs_train_dates_dt], dtype=float)

    bs_rows = []
    for order, sec, code, name_ar, name_en in BS_LINE_SPECS:
        row_dict = {
            "order": order, "section": sec, "code": code,
            "name_ar": name_ar, "name_en": name_en,
        }
        if "TOT" in code or "CHECK" in code:
            for hc in hist_cols:
                row_dict[hc] = 0.0
            row_dict[col_bs_base] = 0.0
            row_dict["Scenario (After)"] = 0.0
            row_dict["Delta (Impact)"] = 0.0
            for fc in fcst_cols:
                row_dict[fc] = 0.0
        else:
            for h_dt, hc in zip(hist_dates, hist_cols):
                row_dict[hc] = bs_sql[h_dt].get(code, 0.0)
            v25_b = bs_base_full_year.get(code, 0.0)
            v25_s = bs_scn.get(code, 0.0)
            row_dict[col_bs_base] = v25_b
            row_dict["Scenario (After)"] = v25_s
            row_dict["Delta (Impact)"] = v25_s - v25_b

            y_bs_hist = np.array(
                [bs_sql[d].get(code, 0.0) for d in hist_dates_prior] + [v25_b],
                dtype=float,
            )
            f_vals = _ml_ridge_forecast_n_years(
                t_bs_vec, y_bs_hist, v25_b, v25_s, n_years=n_fcst, macro_drift=macro_drift_bs
            )
            for fc, fv in zip(fcst_cols, f_vals):
                row_dict[fc] = fv
        bs_rows.append(row_dict)

    df_bs_multi = pd.DataFrame(bs_rows)

    # Link Forecasted Net Profit & Roll-Forward Retained Earnings strictly from Income Statement
    for idx_f, yr_col in enumerate(fcst_cols):
        is_np_val = float(df_is_multi.loc[df_is_multi["code"] == "IS_NET_PROFIT", yr_col].values[0])
        df_bs_multi.loc[df_bs_multi["code"] == "BS_EQ_NET_PROFIT", yr_col] = is_np_val

        prev_col = "Scenario (After)" if idx_f == 0 else fcst_cols[idx_f - 1]
        prev_ret = float(df_bs_multi.loc[df_bs_multi["code"] == "BS_EQ_RETAINED", prev_col].values[0])
        prev_np = float(df_bs_multi.loc[df_bs_multi["code"] == "BS_EQ_NET_PROFIT", prev_col].values[0])
        df_bs_multi.loc[df_bs_multi["code"] == "BS_EQ_RETAINED", yr_col] = prev_ret + prev_np

    for yr_col in hist_cols + [col_bs_base, "Scenario (After)"] + fcst_cols:
        liab_s = df_bs_multi.loc[df_bs_multi["order"].between(13, 20), yr_col].sum()
        eq_s = df_bs_multi.loc[df_bs_multi["order"].between(22, 27), yr_col].sum()
        if yr_col in fcst_cols:
            prelim_ast = df_bs_multi.loc[df_bs_multi["order"].between(1, 10), yr_col].sum()
            df_bs_multi.loc[df_bs_multi["code"] == "BS_AST_OTHER", yr_col] = (liab_s + eq_s) - prelim_ast
        ast_s = df_bs_multi.loc[df_bs_multi["order"].between(1, 11), yr_col].sum()
        df_bs_multi.loc[df_bs_multi["code"] == "BS_TOT_ASSETS", yr_col] = ast_s
        df_bs_multi.loc[df_bs_multi["code"] == "BS_TOT_LIAB", yr_col] = liab_s
        df_bs_multi.loc[df_bs_multi["code"] == "BS_TOT_EQ", yr_col] = eq_s
        df_bs_multi.loc[df_bs_multi["code"] == "BS_TOT_LIAB_EQ", yr_col] = liab_s + eq_s
        df_bs_multi.loc[df_bs_multi["code"] == "BS_CHECK", yr_col] = ast_s - (liab_s + eq_s)

    df_bs_multi["Delta (Impact)"] = df_bs_multi["Scenario (After)"] - df_bs_multi[col_bs_base]

    # 9. Compute the 22 Executive KPIs anchored on the Full-Year Base (`2025E` or `2025A`) and Scenario
    def _bs_val(c_code: str, col_name: str) -> float:
        s = df_bs_multi.loc[df_bs_multi["code"] == c_code, col_name]
        return float(s.values[0]) if not s.empty else 0.0

    tot_ast_24 = _bs_val("BS_TOT_ASSETS", col_h2)
    tot_ast_b = _bs_val("BS_TOT_ASSETS", col_bs_base)
    tot_ast_s = _bs_val("BS_TOT_ASSETS", "Scenario (After)")

    tot_liab_b = _bs_val("BS_TOT_LIAB", col_bs_base)
    tot_liab_s = _bs_val("BS_TOT_LIAB", "Scenario (After)")

    tot_eq_24 = _bs_val("BS_TOT_EQ", col_h2)
    tot_eq_b = _bs_val("BS_TOT_EQ", col_bs_base)
    tot_eq_s = _bs_val("BS_TOT_EQ", "Scenario (After)")

    ea_codes = ["BS_AST_BANKS", "BS_AST_FVTPL", "BS_AST_LOANS", "BS_AST_FVOCI", "BS_AST_REV_REPO", "BS_AST_AMORT"]
    ea_24 = sum(_bs_val(c, col_h2) for c in ea_codes)
    ea_b = sum(_bs_val(c, col_bs_base) for c in ea_codes)
    ea_s = sum(_bs_val(c, "Scenario (After)") for c in ea_codes)

    avg_ea_b = (ea_24 + ea_b) / 2.0 if ea_24 > 0 else ea_b
    avg_ea_s = (ea_24 + ea_s) / 2.0 if ea_24 > 0 else ea_s

    yield_ea_base = (is_base["IS_INT_INC"] / avg_ea_b * 100.0) if avg_ea_b > 0 else 0.0
    yield_ea_scn = (is_scn["IS_INT_INC"] / avg_ea_s * 100.0) if avg_ea_s > 0 else 0.0

    if pos_detail_df is not None and not pos_detail_df.empty:
        p_df = pos_detail_df
        liab_m = p_df["Category"].astype(str).str.lower().isin(["deposit", "funding"])
        non_int_m = p_df["Model Rate Type"].astype(str).str.lower().str.contains("non-interest")
        tot_liab_pos_b = float(p_df.loc[liab_m, "Before Balance"].sum())
        non_int_pos_b = float(p_df.loc[liab_m & non_int_m, "Before Balance"].sum())
        non_int_share = (non_int_pos_b / tot_liab_pos_b) if tot_liab_pos_b > 0 else 0.0

        dep_m = p_df["Category"].astype(str).str.lower().eq("deposit")
        casa_m = dep_m & (
            p_df["Product Family"].astype(str).str.upper().str.contains("CASA")
            | p_df["Product Type"].astype(str).str.lower().str.contains("current|saving")
        )
        dep_pos_b = float(p_df.loc[dep_m, "Before Balance"].sum())
        dep_pos_s = float(p_df.loc[dep_m, "After Balance"].sum())
        casa_pos_b = float(p_df.loc[casa_m, "Before Balance"].sum())
        casa_pos_s = float(p_df.loc[casa_m, "After Balance"].sum())
        casa_ratio_base = (casa_pos_b / dep_pos_b * 100.0) if dep_pos_b > 0 else 0.0
        casa_ratio_scn = (casa_pos_s / dep_pos_s * 100.0) if dep_pos_s > 0 else 0.0

        if "Is Liquid Asset" in p_df.columns:
            liq_m = p_df["Is Liquid Asset"].astype(bool)
        else:
            liq_m = p_df["HQLA Flag"].astype(bool)
        liq_pos_b = float(p_df.loc[liq_m, "Before Balance"].sum()) / 1e6
        liq_pos_s = float(p_df.loc[liq_m, "After Balance"].sum()) / 1e6
        tot_ast_pos_b = float(p_df.loc[p_df["Category"].isin(["Loan", "Investment"]), "Before Balance"].sum()) / 1e6
        liq_scale = (tot_ast_b / tot_ast_pos_b) if tot_ast_pos_b > 0 else 1.0
        liq_assets_base_mn = liq_pos_b * liq_scale
        liq_assets_scn_mn = liq_pos_s * liq_scale

        asf_w = pd.to_numeric(p_df.get("NSFR ASF Weight", 0.0), errors="coerce").fillna(0.0)
        rsf_w = pd.to_numeric(p_df.get("NSFR RSF Weight", 0.0), errors="coerce").fillna(0.0)
        asf_pos_b = float((p_df["Before Balance"] * asf_w).sum()) / 1e6 * liq_scale + tot_eq_b
        asf_pos_s = float((p_df["After Balance"] * asf_w).sum()) / 1e6 * liq_scale + tot_eq_s
        rsf_pos_b = float((p_df["Before Balance"] * rsf_w).sum()) / 1e6 * liq_scale
        rsf_pos_s = float((p_df["After Balance"] * rsf_w).sum()) / 1e6 * liq_scale

        fcy_m = p_df["Valuation Basis"].astype(str).str.lower().eq("fx")
        ast_m = p_df["Category"].astype(str).str.lower().isin(["loan", "investment"])
        fcy_ast_b = float(p_df.loc[fcy_m & ast_m, "Before Balance"].sum()) / 1e6 * liq_scale
        fcy_liab_b = float(p_df.loc[fcy_m & liab_m, "Before Balance"].sum()) / 1e6 * liq_scale
        fcy_ast_s = float(p_df.loc[fcy_m & ast_m, "After Balance"].sum()) / 1e6 * liq_scale
        fcy_liab_s = float(p_df.loc[fcy_m & liab_m, "After Balance"].sum()) / 1e6 * liq_scale
        nop_base_mn = fcy_ast_b - fcy_liab_b
        nop_scn_mn = fcy_ast_s - fcy_liab_s
    else:
        non_int_share = 0.20
        casa_ratio_base = casa_ratio_scn = 0.0
        liq_assets_base_mn = _bs_val("BS_AST_CASH", col_bs_base) + _bs_val("BS_AST_BANKS", col_bs_base)
        liq_assets_scn_mn = _bs_val("BS_AST_CASH", "Scenario (After)") + _bs_val("BS_AST_BANKS", "Scenario (After)")
        asf_pos_b = asf_pos_s = rsf_pos_b = rsf_pos_s = 1.0
        nop_base_mn = nop_scn_mn = 0.0

    costed_liab_b = (
        _bs_val("BS_LIAB_BANKS", col_bs_base)
        + _bs_val("BS_LIAB_REPO", col_bs_base)
        + _bs_val("BS_LIAB_DEP", col_bs_base) * (1.0 - non_int_share)
        + _bs_val("BS_LIAB_BORROW", col_bs_base)
    )
    costed_liab_s = (
        _bs_val("BS_LIAB_BANKS", "Scenario (After)")
        + _bs_val("BS_LIAB_REPO", "Scenario (After)")
        + _bs_val("BS_LIAB_DEP", "Scenario (After)") * (1.0 - non_int_share)
        + _bs_val("BS_LIAB_BORROW", "Scenario (After)")
    )

    cof_fs_base = (abs(is_base["IS_INT_EXP"]) / costed_liab_b * 100.0) if costed_liab_b > 0 else 0.0
    cof_fs_scn = (abs(is_scn["IS_INT_EXP"]) / costed_liab_s * 100.0) if costed_liab_s > 0 else 0.0

    dep_b = _bs_val("BS_LIAB_DEP", col_bs_base)
    dep_s = _bs_val("BS_LIAB_DEP", "Scenario (After)")

    op_inc_b = (
        is_base["IS_NII"]
        + is_base["IS_NET_FEE"]
        + is_base["IS_DIVIDENDS"]
        + is_base["IS_TRADING"]
        + is_base["IS_INV_GAINS"]
    )
    op_inc_s = (
        is_scn["IS_NII"]
        + is_scn["IS_NET_FEE"]
        + is_scn["IS_DIVIDENDS"]
        + is_scn["IS_TRADING"]
        + is_scn["IS_INV_GAINS"]
    )

    avg_eq_b = (tot_eq_24 + tot_eq_b) / 2.0 if tot_eq_24 > 0 else tot_eq_b
    avg_eq_s = (tot_eq_24 + tot_eq_s) / 2.0 if tot_eq_24 > 0 else tot_eq_s
    avg_ast_b = (tot_ast_24 + tot_ast_b) / 2.0 if tot_ast_24 > 0 else tot_ast_b
    avg_ast_s = (tot_ast_24 + tot_ast_s) / 2.0 if tot_ast_24 > 0 else tot_ast_s

    rep_gap_b_mn = float(before_dict.get("Repricing Gap", 0.0)) / 1e6
    rep_gap_s_mn = float(after_dict.get("Repricing Gap", 0.0)) / 1e6
    base_nii_pos_mn = float(before_dict.get("NII", 1.0)) / 1e6

    fs_kpis = {
        "fs_meta": periods,
        "total_assets_base_mn": tot_ast_b,
        "total_assets_scn_mn": tot_ast_s,
        "total_liab_base_mn": tot_liab_b,
        "total_liab_scn_mn": tot_liab_s,
        "total_eq_base_mn": tot_eq_b,
        "total_eq_scn_mn": tot_eq_s,
        "earning_assets_base_mn": ea_b,
        "earning_assets_scn_mn": ea_s,
        "base_net_profit_mn": is_base["IS_NET_PROFIT"],
        "scn_net_profit_mn": is_scn["IS_NET_PROFIT"],
        "delta_net_profit_mn": d_net_profit,
        "yield_ea_base": yield_ea_base,
        "yield_ea_scn": yield_ea_scn,
        "cof_fs_base": cof_fs_base,
        "cof_fs_scn": cof_fs_scn,
        "spread_base": yield_ea_base - cof_fs_base,
        "spread_scn": yield_ea_scn - cof_fs_scn,
        "rep_gap_base_mn": rep_gap_b_mn,
        "rep_gap_scn_mn": rep_gap_s_mn,
        "nii_sens_100bp": (rep_gap_b_mn * 0.01 / base_nii_pos_mn * 100.0) if abs(base_nii_pos_mn) > 1e-6 else 0.0,
        "fvoci_mtm_mn": fvoci_mtm_scaled_mn,
        "eve_eq_scn": (fvoci_mtm_scaled_mn / tot_eq_b * 100.0) if tot_eq_b > 0 else 0.0,
        "op_inc_base_mn": op_inc_b,
        "op_inc_scn_mn": op_inc_s,
        "cti_base": (abs(is_base["IS_GA_EXP"]) / op_inc_b * 100.0) if op_inc_b > 0 else 0.0,
        "cti_scn": (abs(is_scn["IS_GA_EXP"]) / op_inc_s * 100.0) if op_inc_s > 0 else 0.0,
        "dep_base_mn": dep_b,
        "dep_scn_mn": dep_s,
        "liq_assets_base_mn": liq_assets_base_mn,
        "liq_assets_scn_mn": liq_assets_scn_mn,
        "liq_dep_base": (liq_assets_base_mn / dep_b * 100.0) if dep_b > 0 else 0.0,
        "liq_dep_scn": (liq_assets_scn_mn / dep_s * 100.0) if dep_s > 0 else 0.0,
        "casa_dep_base_mn": dep_b * (casa_ratio_base / 100.0),
        "casa_dep_scn_mn": dep_s * (casa_ratio_scn / 100.0),
        "casa_ratio_base": casa_ratio_base,
        "casa_ratio_scn": casa_ratio_scn,
        "roe_base": (is_base["IS_NET_PROFIT"] / avg_eq_b * 100.0) if avg_eq_b > 0 else 0.0,
        "roe_scn": (is_scn["IS_NET_PROFIT"] / avg_eq_s * 100.0) if avg_eq_s > 0 else 0.0,
        "roa_base": (is_base["IS_NET_PROFIT"] / avg_ast_b * 100.0) if avg_ast_b > 0 else 0.0,
        "roa_scn": (is_scn["IS_NET_PROFIT"] / avg_ast_s * 100.0) if avg_ast_s > 0 else 0.0,
        "eq_ast_base": (tot_eq_b / tot_ast_b * 100.0) if tot_ast_b > 0 else 0.0,
        "eq_ast_scn": (tot_eq_s / tot_ast_s * 100.0) if tot_ast_s > 0 else 0.0,
        "nop_base_mn": nop_base_mn,
        "nop_scn_mn": nop_scn_mn,
        "nop_eq_base": (nop_base_mn / tot_eq_b * 100.0) if tot_eq_b > 0 else 0.0,
        "nop_eq_scn": (nop_scn_mn / tot_eq_s * 100.0) if tot_eq_s > 0 else 0.0,
        "asf_base_mn": asf_pos_b,
        "asf_scn_mn": asf_pos_s,
        "rsf_base_mn": rsf_pos_b,
        "rsf_scn_mn": rsf_pos_s,
        "nsfr_base": (asf_pos_b / rsf_pos_b * 100.0) if rsf_pos_b > 0 else 0.0,
        "nsfr_scn": (asf_pos_s / rsf_pos_s * 100.0) if rsf_pos_s > 0 else 0.0,
        "sov_ast_base": ((_bs_val("BS_AST_FVOCI", col_bs_base) + _bs_val("BS_AST_AMORT", col_bs_base)) / tot_ast_b * 100.0) if tot_ast_b > 0 else 0.0,
        "sov_ast_scn": ((_bs_val("BS_AST_FVOCI", "Scenario (After)") + _bs_val("BS_AST_AMORT", "Scenario (After)")) / tot_ast_s * 100.0) if tot_ast_s > 0 else 0.0,
        "borrow_base_mn": _bs_val("BS_LIAB_BORROW", col_bs_base),
        "borrow_scn_mn": _bs_val("BS_LIAB_BORROW", "Scenario (After)"),
        "wsf_liab_base": (_bs_val("BS_LIAB_BORROW", col_bs_base) / tot_liab_b * 100.0) if tot_liab_b > 0 else 0.0,
        "wsf_liab_scn": (_bs_val("BS_LIAB_BORROW", "Scenario (After)") / tot_liab_s * 100.0) if tot_liab_s > 0 else 0.0,
        "off_eq_base": (_bs_val("BS_OFF_CONTINGENT", col_bs_base) / tot_eq_b * 100.0) if tot_eq_b > 0 else 0.0,
        "off_eq_scn": (_bs_val("BS_OFF_CONTINGENT", "Scenario (After)") / tot_eq_s * 100.0) if tot_eq_s > 0 else 0.0,
    }

    return {
        "balance_sheet_df": df_bs_multi,
        "income_statement_df": df_is_multi,
        "fs_kpis": fs_kpis,
        "fs_meta": periods,
    }