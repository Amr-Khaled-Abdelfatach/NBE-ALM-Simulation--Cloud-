# reporting.py
"""
Custom PDF and PPTX Executive Report Generator with:
  - Full Arabic TrueType Font & BiDi Shaping Support (zero boxes in PDF)
  - NBE Header, Footer, Configured Brand Logos (Zero Hardcoding)
  - Uniform, Perfectly Aligned Scenario Parameter Cards (Chips / Shapes)
  - Rich 4-Pillar Executive ALCO Briefing (with automatic DB lookup or deterministic fallback)
  - Selected Key Performance Indicators (KPIs: Before, After, Impact/Delta)
  - Selected Simulation Charts (Plotly -> high-res PNG via Kaleido)
  - Detailed Scenario Comparison Table
  - Comparative & ML-Forecasted Financial Statements (Balance Sheet & Income Statement, +/-2Y or +/-3Y)
"""
import io
import os
import re
import zipfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    APP_NAME,
    APP_VERSION,
    CALCULATION_ENGINE_VERSION,
    FLAT_CASHFLOWS_VIEW,
    FLAT_POSITIONS_VIEW,
    LOGO_HEADER_WHITE_PATH,
    LOGO_REPORT_COLOR_PATH,
    LOGO_SIDEBAR_PATH,
    NBE_DARK_GREEN,
    NBE_GREEN,
    NBE_ORANGE,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    PARAM_SPECS,
)
from scenarios import ask_ollama

# ----------------------------------------------------------------------------
# Library Availability & Diagnostic Constants
# ----------------------------------------------------------------------------
REPORTING_AVAILABLE = True
REPORTING_MISSING_ERROR = ""
PDF_AVAILABLE = True
PDF_MISSING_ERROR = ""
PPTX_AVAILABLE = True
PPTX_MISSING_ERROR = ""

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Image as RLImage,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
except ImportError as _rl_exc:
    REPORTING_AVAILABLE = False
    REPORTING_MISSING_ERROR = str(_rl_exc)
    PDF_AVAILABLE = False
    PDF_MISSING_ERROR = str(_rl_exc)

try:
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN
    from pptx.util import Inches, Pt
except ImportError as _pptx_exc:
    REPORTING_AVAILABLE = False
    REPORTING_MISSING_ERROR = str(_pptx_exc)
    PPTX_AVAILABLE = False
    PPTX_MISSING_ERROR = str(_pptx_exc)


# ----------------------------------------------------------------------------
# Arabic Font & BiDi Shaping Support for ReportLab PDF
# ----------------------------------------------------------------------------
_AR_FONT_REGULAR = "Helvetica"
_AR_FONT_BOLD = "Helvetica-Bold"


def _register_pdf_arabic_fonts():
    """Register Windows/System TrueType fonts (Tahoma/Arial) so Arabic characters render cleanly without boxes."""
    global _AR_FONT_REGULAR, _AR_FONT_BOLD
    if not REPORTING_AVAILABLE:
        return
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont

        font_pairs = [
            ("NBE_Tahoma", "NBE_Tahoma_Bold", r"C:\Windows\Fonts\tahoma.ttf", r"C:\Windows\Fonts\tahomabd.ttf"),
            ("NBE_Arial", "NBE_Arial_Bold", r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"),
        ]
        for reg_name, bold_name, reg_path, bold_path in font_pairs:
            if os.path.exists(reg_path):
                pdfmetrics.registerFont(TTFont(reg_name, reg_path))
                _AR_FONT_REGULAR = reg_name
                if os.path.exists(bold_path):
                    pdfmetrics.registerFont(TTFont(bold_name, bold_path))
                    _AR_FONT_BOLD = bold_name
                else:
                    _AR_FONT_BOLD = reg_name
                break
    except Exception:
        pass


_register_pdf_arabic_fonts()


def _shape_arabic_text(txt: str) -> str:
    """Reshape and apply BiDi ordering strictly to Arabic segments so mixed English+Arabic lines render cleanly."""
    s = str(txt or "")
    if not s or not re.search(r"[\u0600-\u06FF]", s):
        return s
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display

        def _shape_ar_segment(match: re.Match) -> str:
            seg = match.group(0)
            reshaped = arabic_reshaper.reshape(seg)
            return get_display(reshaped)

        # Match contiguous Arabic runs (including internal spaces, numbers, colons, hyphens)
        ar_pattern = re.compile(r"[\u0600-\u06FF][\u0600-\u06FF0-9\s:،؛؟\-_/()]*[\u0600-\u06FF0-9]|[\u0600-\u06FF]")
        return ar_pattern.sub(_shape_ar_segment, s)
    except ImportError:
        return s


# ----------------------------------------------------------------------------
# Helpers: Logo Resolution, Plotly Image Export, & Executive ALCO Briefing
# ----------------------------------------------------------------------------
def _hex_to_rgb_tuple(hex_str: str) -> Tuple[int, int, int]:
    h = str(hex_str or "#005B2D").lstrip("#")
    if len(h) != 6:
        h = "005B2D"
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _get_logo_path(prefer_white: bool = False) -> Optional[str]:
    """Resolve logo path dynamically from config.py with fallback to local files."""
    candidates = (
        [LOGO_HEADER_WHITE_PATH, "nbe_logo_w.png", LOGO_REPORT_COLOR_PATH, LOGO_SIDEBAR_PATH, "nbe_logo.png"]
        if prefer_white
        else [LOGO_REPORT_COLOR_PATH, "nbe_logo.png", LOGO_SIDEBAR_PATH, LOGO_HEADER_WHITE_PATH]
    )
    for p in candidates:
        if p and os.path.exists(str(p)):
            return str(p)
    return None


def plotly_fig_to_png_bytes(fig, width: int = 900, height: int = 420) -> bytes:
    """Render a Plotly figure to high-resolution PNG bytes for PDF/PPTX embedding."""
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#173B2A"),
    )
    return fig.to_image(format="png", width=width, height=height, scale=2)


def _build_rich_deterministic_briefing(
    scenario_name: str,
    scenario_type: str,
    horizon_days: int,
    params: Optional[dict],
    selected_kpis: Optional[list],
    comparison_df: Optional[pd.DataFrame],
    risk_df: Optional[pd.DataFrame],
) -> str:
    """Build a comprehensive 4-pillar executive ALCO briefing from deterministic outputs when AI text is absent."""
    params = params or {}
    active_shocks = []
    for k, v in params.items():
        if k in PARAM_SPECS and isinstance(v, (int, float)) and abs(float(v)) > 1e-9:
            lbl, unit = PARAM_SPECS[k][0].split(" (")[0], PARAM_SPECS[k][1]
            active_shocks.append(f"{lbl} ({float(v):+g}{unit})")
    shock_str = ", ".join(active_shocks) if active_shocks else "Baseline assumptions (no active market shocks)"

    m_map = {}
    if comparison_df is not None and not comparison_df.empty and "KPI" in comparison_df.columns:
        for _, r in comparison_df.iterrows():
            m_map[str(r["KPI"]).strip()] = r.to_dict()

    def _get_m(name):
        return m_map.get(name, {})

    nii_r = _get_m("NII")
    nim_r = _get_m("NIM %")
    cof_r = _get_m("Cost of Funds %")
    lcr_r = _get_m("CF LCR %")
    gap_r = _get_m("Repricing Gap")
    ldr_r = _get_m("Loan Deposit %")
    pnl_r = _get_m("PnL Total Impact")
    cred_r = _get_m("PnL Credit Cost")
    fxg_r = _get_m("PnL FX Gold Revaluation")
    mtm_r = _get_m("PnL Investment MTM")

    high_risks, adv_risks = [], []
    if risk_df is not None and not risk_df.empty and "Risk Level" in risk_df.columns:
        for _, rr in risk_df.iterrows():
            lvl = str(rr.get("Risk Level", "")).upper()
            m_lbl = str(rr.get("Metric", ""))
            if lvl == "HIGH":
                high_risks.append(m_lbl)
            elif lvl == "ADVERSE":
                adv_risks.append(m_lbl)

    posture = "HIGH" if high_risks else ("MEDIUM / ADVERSE" if adv_risks else "CONTROLLED / LOW")

    lines = [
        f"1. Executive Verdict & Overall Risk Posture ({posture}):",
        f"Under scenario '{scenario_name}' ({scenario_type}, {horizon_days}-day horizon), the balance sheet was stressed with: {shock_str}. "
        + (f"Critical risk alerts triggered on: {', '.join(high_risks)}. " if high_risks else "")
        + (
            f"Adverse margin/funding pressure observed on: {', '.join(adv_risks)}."
            if adv_risks
            else "Core solvency and liquidity buffers remain within internal tolerance."
        ),
        "",
        "2. Net Interest Income (NII) & Margin Dynamics:",
    ]

    if nii_r:
        lines.append(
            f"- Net Interest Income (NII): Moves from {nii_r.get('Before', 0):,.2f} to {nii_r.get('After', 0):,.2f} EGP "
            f"({nii_r.get('Absolute Change', 0):+,.2f} EGP | {nii_r.get('Change %', 0):+.2f}%)."
        )
    if nim_r and cof_r:
        lines.append(
            f"- Margin & Funding Cost: NIM shifts from {nim_r.get('Before', 0):.2f}% to {nim_r.get('After', 0):.2f}% "
            f"({(nim_r.get('After', 0) - nim_r.get('Before', 0)) * 100:+.0f} bps), while Cost of Funds moves from "
            f"{cof_r.get('Before', 0):.2f}% to {cof_r.get('After', 0):.2f}% ({(cof_r.get('After', 0) - cof_r.get('Before', 0)) * 100:+.0f} bps)."
        )
    elif selected_kpis:
        for item in selected_kpis[:4]:
            lines.append(
                f"- {item.get('label', '')}: Moves from {item.get('before', '')} to {item.get('after', '')} ({item.get('delta') or '0.00'})."
            )

    lines.extend([
        "",
        "3. Liquidity, Repricing Gap & Valuation Attribution:",
    ])
    if lcr_r:
        lines.append(
            f"- Liquidity Coverage (30d LCR Proxy): Adjusts from {lcr_r.get('Before', 0):,.1f}% to {lcr_r.get('After', 0):,.1f}% "
            f"({lcr_r.get('Absolute Change', 0):+.1f} pp) after accounting for deposit runoff and HQLA stock."
        )
    if gap_r and ldr_r:
        lines.append(
            f"- Structural Repricing & LDR: 12M Repricing Gap stands at {gap_r.get('After', 0):,.2f} EGP "
            f"(vs. {gap_r.get('Before', 0):,.2f} EGP base), with Loan-to-Deposit Ratio at {ldr_r.get('After', 0):.2f}%."
        )
    if pnl_r:
        lines.append(
            f"- Total P&L & Valuation Attribution: Cumulative impact of {pnl_r.get('After', 0):+,.2f} EGP "
            f"(FX/Gold Revaluation: {fxg_r.get('After', 0):+,.2f} EGP | Investment MTM: {mtm_r.get('After', 0):+,.2f} EGP | "
            f"Macro Credit Cost: {cred_r.get('After', 0):+,.2f} EGP)."
        )

    lines.extend([
        "",
        "4. Strategic ALCO Action Plan:",
        "- Funding & Margin Protection: Optimize deposit pricing mix (CASA retention) and lock in medium-term wholesale spreads to contain Cost of Funds escalation.",
        "- Liquidity & Repricing Alignment: Maintain high-quality liquid asset (HQLA) buffers above internal stress floors and align asset repricing tenors with rate trajectory.",
        "- Market & Credit Risk Governance: Actively hedge net open FX/Gold exposures and enforce sector-level ECL provisioning reviews on rate-sensitive loan portfolios.",
    ])

    return "\n".join(lines)


def _resolve_ai_insight(
    ai_insight: Optional[str],
    scenario_name: str,
    scenario_type: str = "CUSTOM",
    horizon_days: int = 365,
    params: Optional[dict] = None,
    selected_kpis: Optional[list] = None,
    comparison_df: Optional[pd.DataFrame] = None,
    risk_df: Optional[pd.DataFrame] = None,
    engine=None,
    run_id=None,
) -> str:
    """
    Resolve the executive briefing in strict priority order:
      1. Explicit valid AI explanation passed from the UI session.
      2. Saved `ai_explanation` in MySQL `simulation_runs` for this run_id or scenario_name.
      3. Rich 4-pillar deterministic ALCO briefing built directly from simulation tables (instant, zero latency).
    """
    err_prefixes = (
        "Phi-3 error",
        "Phi-3 unavailable",
        "Phi-3 request failed",
        "Phi-3 explanation unavailable",
        "Ollama connection",
        "No response",
    )

    final_insight = str(ai_insight or "").strip()
    # Ignore short raw bullet fallbacks (< 4 lines) so the rich 4-pillar ALCO report renders instead
    if final_insight and not final_insight.startswith(err_prefixes) and len(final_insight.splitlines()) >= 5:
        return final_insight

    if engine is not None and run_id is not None:
        try:
            from sqlalchemy import text
            with engine.connect() as conn:
                row = conn.execute(
                    text("SELECT ai_explanation FROM simulation_runs WHERE run_id = :rid"),
                    {"rid": int(run_id)},
                ).fetchone()
                if row and row[0] and str(row[0]).strip():
                    val = str(row[0]).strip()
                    if not val.startswith(err_prefixes):
                        return val
        except Exception:
            pass

    if engine is not None and scenario_name:
        try:
            from sqlalchemy import text
            with engine.connect() as conn:
                row = conn.execute(
                    text("""
                        SELECT sr.ai_explanation
                        FROM simulation_runs sr
                        JOIN scenarios s ON sr.scenario_id = s.scenario_id
                        WHERE s.scenario_name = :sname AND sr.ai_explanation IS NOT NULL
                        ORDER BY sr.run_id DESC LIMIT 1
                    """),
                    {"sname": str(scenario_name)},
                ).fetchone()
                if row and row[0] and str(row[0]).strip():
                    val = str(row[0]).strip()
                    if not val.startswith(err_prefixes):
                        return val
        except Exception:
            pass

    return _build_rich_deterministic_briefing(
        scenario_name=scenario_name,
        scenario_type=scenario_type,
        horizon_days=horizon_days,
        params=params,
        selected_kpis=selected_kpis,
        comparison_df=comparison_df,
        risk_df=risk_df,
    )


# ----------------------------------------------------------------------------
# Financial Statement Dynamic Column Helpers (±2Y / ±3Y)
# ----------------------------------------------------------------------------
def _extract_year_tag(col_str: str, default_suffix: str = "") -> str:
    s = str(col_str or "").strip()
    if re.match(r"^20\d{2}[AEF]$", s):
        return s
    m = re.search(r"(20\d{2})", s)
    if m:
        yr = m.group(1)
        if s.endswith(("A", "E", "F")):
            return f"{yr}{s[-1]}"
        return f"{yr}{default_suffix}"
    return s


def _fmt_num(val, decimals: int = 1, signed: bool = False) -> str:
    if val is None or pd.isna(val):
        return "—"
    try:
        v = float(val)
    except Exception:
        return str(val)
    fmt = f"{{:+,.{decimals}f}}" if signed else f"{{:,.{decimals}f}}"
    return fmt.format(v)


def _resolve_statement_columns(
    df: pd.DataFrame,
    is_income_stmt: bool = False,
    meta: Optional[dict] = None,
    window_years: Optional[int] = None,
) -> Tuple[List[Tuple[str, str]], str]:
    """Dynamically discover historical (*A), Base (*E/*A), Scenario, Delta, and Forecast (*F) columns sliced to window_years."""
    meta = meta or {}
    win_n = int(window_years or meta.get("window_years") or 0)
    fixed_bs = {"order", "section", "code", "name_ar", "name_en", "Scenario (After)", "Delta (Impact)"}
    fixed_is = {"order", "section", "code", "name_ar", "name_en", "source", "base_mn", "scn_mn", "delta_mn", "delta_pct"}

    if not is_income_stmt:
        dyn_cols = [c for c in df.columns if c not in fixed_bs]
        all_hist = [c for c in meta.get("hist_cols", []) if c in df.columns] or [c for c in dyn_cols if str(c).endswith("A")]
        all_fcst = [c for c in meta.get("fcst_cols", []) if c in df.columns] or [c for c in dyn_cols if str(c).endswith("F")]
        rem_cols = [c for c in dyn_cols if c not in all_hist and c not in all_fcst]

        hist_cols = all_hist[-win_n:] if win_n > 0 else all_hist
        fcst_cols = all_fcst[:win_n] if win_n > 0 else all_fcst

        base_col = (
            meta.get("bs_base_col")
            if meta.get("bs_base_col") in df.columns
            else (rem_cols[0] if rem_cols else "Base")
        )
        base_suffix = "E" if meta.get("is_interim_base", True) else "A"
        base_label = _extract_year_tag(base_col, base_suffix)

        col_specs: List[Tuple[str, str]] = []
        for hc in hist_cols:
            col_specs.append((hc, _extract_year_tag(hc, "A")))
        col_specs.append((base_col, base_label))
        col_specs.append(("Scenario (After)", "Scenario"))
        col_specs.append(("Delta (Impact)", "Delta (Δ)"))
        for fc in fcst_cols:
            col_specs.append((fc, _extract_year_tag(fc, "F")))
        return col_specs, base_label
    else:
        dyn_cols = [c for c in df.columns if c not in fixed_is]
        all_hist = [c for c in meta.get("hist_cols", []) if c in df.columns] or [c for c in dyn_cols if str(c).endswith("A")]
        all_fcst = [c for c in meta.get("fcst_cols", []) if c in df.columns] or [c for c in dyn_cols if str(c).endswith("F")]

        hist_cols = all_hist[-win_n:] if win_n > 0 else all_hist
        fcst_cols = all_fcst[:win_n] if win_n > 0 else all_fcst

        base_raw = meta.get("is_base_header") or meta.get("bs_base_col") or "2025E"
        base_suffix = "E" if meta.get("is_interim_base", True) else "A"
        base_label = _extract_year_tag(base_raw, base_suffix)

        col_specs = []
        for hc in hist_cols:
            col_specs.append((hc, _extract_year_tag(hc, "A")))
        col_specs.append(("base_mn", base_label))
        col_specs.append(("scn_mn", "Scenario"))
        col_specs.append(("delta_mn", "Delta (Δ)"))
        for fc in fcst_cols:
            col_specs.append((fc, _extract_year_tag(fc, "F")))
        return col_specs, base_label


def _build_pdf_statement_table(
    df: pd.DataFrame, is_income_stmt: bool, meta: dict, avail_width: float, window_years: Optional[int] = None
):
    """Build a styled ReportLab Table for the multi-year Balance Sheet or Income Statement."""
    col_specs, _ = _resolve_statement_columns(df, is_income_stmt=is_income_stmt, meta=meta, window_years=window_years)
    n_num_cols = len(col_specs)

    label_w = max(avail_width * 0.27, 175.0)
    num_w = (avail_width - label_w) / max(n_num_cols, 1)
    col_widths = [label_w] + [num_w] * n_num_cols

    headers = ["Line Item (EGP Mn)"] + [hdr for _, hdr in col_specs]
    table_data = [headers]
    total_row_indices = []

    for r_idx, (_, r) in enumerate(df.iterrows(), start=1):
        code = str(r.get("code", ""))
        src = str(r.get("source", ""))
        is_tot = any(t in code for t in ("TOT", "CHECK", "NII", "EBT", "NET_PROFIT")) or src in ("TOTAL", "SUBTOTAL", "CHECK")
        if is_tot:
            total_row_indices.append(r_idx)

        item_lbl = str(r.get("name_en") or r.get("name_ar") or code)
        if len(item_lbl) > 42:
            item_lbl = item_lbl[:39] + "..."

        row_cells = [item_lbl]
        for c_key, _ in col_specs:
            is_delta = "delta" in c_key.lower()
            row_cells.append(_fmt_num(r.get(c_key), decimals=1, signed=is_delta))
        table_data.append(row_cells)

    font_sz = 7.0 if n_num_cols >= 9 else (7.5 if n_num_cols >= 7 else 8.2)
    t = Table(table_data, colWidths=col_widths, repeatRows=1)

    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NBE_DARK_GREEN)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), _AR_FONT_BOLD),
        ("FONTNAME", (0, 1), (-1, -1), _AR_FONT_REGULAR),
        ("FONTSIZE", (0, 0), (-1, -1), font_sz),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
    ]

    for idx in range(1, len(table_data)):
        if idx % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, idx), (-1, idx), colors.HexColor("#F8FAFC")))

    for tot_i in total_row_indices:
        style_cmds.extend([
            ("BACKGROUND", (0, tot_i), (-1, tot_i), colors.HexColor("#EAF5EF")),
            ("TEXTCOLOR", (0, tot_i), (-1, tot_i), colors.HexColor(NBE_DARK_GREEN)),
            ("FONTNAME", (0, tot_i), (-1, tot_i), _AR_FONT_BOLD),
        ])

    t.setStyle(TableStyle(style_cmds))
    return t


# ============================================================================
# 1. PDF Executive Report Generator
# ============================================================================
def _add_pdf_header_footer(canvas, doc):
    """Draw the classic NBE Header (with Logo) and Footer on every page of the PDF."""
    canvas.saveState()
    page_w, page_h = landscape(A4)

    logo_path = _get_logo_path(prefer_white=False)
    if logo_path:
        try:
            canvas.drawImage(
                logo_path,
                page_w - 120,
                page_h - 32,
                width=80,
                height=25,
                preserveAspectRatio=True,
                mask="auto",
            )
        except Exception:
            pass

    canvas.setFont(_AR_FONT_BOLD, 8)
    canvas.setFillColor(colors.HexColor(NBE_DARK_GREEN))
    canvas.drawString(36, page_h - 25, f"NATIONAL BANK OF EGYPT (NBE) — {APP_NAME.upper()}")

    canvas.setFont(_AR_FONT_REGULAR, 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawRightString(page_w - 130, page_h - 25, f"Generated: {datetime.now():%Y-%m-%d %H:%M}")

    canvas.setStrokeColor(colors.HexColor(NBE_GREEN))
    canvas.setLineWidth(1)
    canvas.line(36, page_h - 35, page_w - 36, page_h - 35)

    canvas.line(36, 35, page_w - 36, 35)
    canvas.drawString(
        36,
        22,
        f"CONFIDENTIAL — For Internal Management & ALCO Use Only | {APP_NAME} v{APP_VERSION}",
    )
    canvas.drawRightString(page_w - 36, 22, f"Page {doc.page}")
    canvas.restoreState()


def generate_pdf_report(
    scenario_name: str = "Current Scenario",
    scenario_type: str = "CUSTOM",
    simulation_date: str = "",
    horizon_days: int = 365,
    selected_kpis: Optional[list] = None,
    selected_charts: Optional[dict] = None,
    comparison_df: Optional[pd.DataFrame] = None,
    risk_df: Optional[pd.DataFrame] = None,
    ai_insight: Optional[str] = None,
    params: Optional[dict] = None,
    engine=None,
    run_id=None,
    balance_sheet_df: Optional[pd.DataFrame] = None,
    income_statement_df: Optional[pd.DataFrame] = None,
    fs_meta: Optional[dict] = None,
    **kwargs,
) -> bytes:
    """
    Generate the complete Executive PDF Report containing:
      1. Uniform, Aligned Active Scenario Parameters & Shocks Cards
      2. Rich Executive ALCO Management Insight Card (Arabic & English supported)
      3. Selected Key Performance Indicators (KPIs: Before, After, Delta)
      4. Selected Simulation Charts
      5. Detailed Scenario Comparison Table
      6. Comparative & Forecasted Balance Sheet (+/-2Y or +/-3Y)
      7. Comparative & Forecasted Income Statement (+/-2Y or +/-3Y)
    """
    if not REPORTING_AVAILABLE:
        raise RuntimeError(f"Missing reporting libraries ({REPORTING_MISSING_ERROR}). Run: pip install reportlab python-pptx kaleido")

    if hasattr(scenario_name, "before") or hasattr(scenario_name, "balance_sheet_df"):
        res_obj = scenario_name
        scenario_name = str(kwargs.get("scenario_name") or getattr(res_obj, "name", "Current Scenario"))
        scenario_type = str(getattr(res_obj, "type", "CUSTOM"))
        simulation_date = f"{res_obj.sim_date:%Y-%m-%d}" if hasattr(res_obj, "sim_date") else ""
        horizon_days = int(getattr(res_obj, "horizon", 365))
        params = params or getattr(res_obj, "params", {})
        comparison_df = comparison_df if comparison_df is not None else getattr(res_obj, "comparison", None)
        risk_df = risk_df if risk_df is not None else getattr(res_obj, "risk", None)
        balance_sheet_df = balance_sheet_df if balance_sheet_df is not None else getattr(res_obj, "balance_sheet_df", None)
        income_statement_df = income_statement_df if income_statement_df is not None else getattr(res_obj, "income_statement_df", None)
        fs_meta = fs_meta or getattr(res_obj, "fs_meta", {})

    if balance_sheet_df is None:
        balance_sheet_df = kwargs.get("df_bs")
    if income_statement_df is None:
        income_statement_df = kwargs.get("df_is")
    if ai_insight is None:
        ai_insight = kwargs.get("ai_explanation")

    buffer = io.BytesIO()
    page_w, _ = landscape(A4)
    margin = 0.5 * inch
    avail_w = page_w - 2 * margin

    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        leftMargin=margin,
        rightMargin=margin,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "NBE_Title",
        parent=styles["Heading1"],
        fontName=_AR_FONT_BOLD,
        fontSize=15,
        textColor=colors.HexColor(NBE_DARK_GREEN),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "NBE_Subtitle",
        parent=styles["Normal"],
        fontName=_AR_FONT_REGULAR,
        fontSize=9,
        textColor=colors.HexColor("#555555"),
        spaceAfter=10,
    )
    section_style = ParagraphStyle(
        "NBE_Section",
        parent=styles["Heading2"],
        fontName=_AR_FONT_BOLD,
        fontSize=11,
        textColor=colors.HexColor(NBE_GREEN),
        spaceBefore=8,
        spaceAfter=5,
    )
    body_style = ParagraphStyle(
        "NBE_Body",
        parent=styles["Normal"],
        fontName=_AR_FONT_REGULAR,
        fontSize=8.8,
        leading=13,
        textColor=colors.HexColor("#1E293B"),
    )
    chip_cell_style = ParagraphStyle(
        "NBE_ChipCell",
        parent=styles["Normal"],
        fontName=_AR_FONT_REGULAR,
        fontSize=8.0,
        leading=10.0,
        textColor=colors.HexColor(NBE_DARK_GREEN),
    )

    shaped_scn_name = _shape_arabic_text(scenario_name)
    story = [
        Paragraph(f"Executive ALM Simulation Report: {shaped_scn_name}", title_style),
        Paragraph(
            f"<b>Scenario Type:</b> {scenario_type} &nbsp;|&nbsp; "
            f"<b>Simulation Date:</b> {simulation_date} &nbsp;|&nbsp; "
            f"<b>Horizon:</b> {horizon_days} Days",
            subtitle_style,
        ),
    ]

    # 1. Uniform, Perfectly Aligned Active Scenario Parameters & Shocks Grid (4 equal columns & row heights)
    if params:
        story.append(Paragraph("Active Scenario Parameters & Shocks", section_style))
        chip_texts = []
        for k, v in params.items():
            if k in PARAM_SPECS:
                short_lbl = PARAM_SPECS[k][0].split(" (")[0]
                unit = PARAM_SPECS[k][1]
                val_str = f"{float(v):+g}{unit}" if isinstance(v, (int, float)) else f"{v}{unit}"
                chip_texts.append(f"<b>{short_lbl}:</b> {val_str}")

        if chip_texts:
            chip_table_data = []
            row = []
            for i, text_item in enumerate(chip_texts):
                row.append(Paragraph(text_item, chip_cell_style))
                if len(row) == 4 or i == len(chip_texts) - 1:
                    while len(row) < 4:
                        row.append("")
                    chip_table_data.append(row)
                    row = []

            col_w = avail_w / 4.0
            ct = Table(chip_table_data, colWidths=[col_w] * 4, rowHeights=[24] * len(chip_table_data))
            ct_cmds = [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
            for r_i, r_vals in enumerate(chip_table_data):
                for c_i, cell_val in enumerate(r_vals):
                    if cell_val != "":
                        ct_cmds.extend([
                            ("BACKGROUND", (c_i, r_i), (c_i, r_i), colors.HexColor("#EAF5EF")),
                            ("BOX", (c_i, r_i), (c_i, r_i), 0.75, colors.HexColor(NBE_GREEN)),
                        ])
            ct.setStyle(TableStyle(ct_cmds))
            story.append(ct)
            story.append(Spacer(1, 8))

    # 2. Scenario Explanation & Management Insight (Styled Institutional Callout Card)
    final_insight = _resolve_ai_insight(
        ai_insight=ai_insight,
        scenario_name=scenario_name,
        scenario_type=scenario_type,
        horizon_days=horizon_days,
        params=params,
        selected_kpis=selected_kpis,
        comparison_df=comparison_df,
        risk_df=risk_df,
        engine=engine,
        run_id=run_id,
    )
    if final_insight:
        story.append(Paragraph("Scenario Explanation & Management Insight", section_style))
        raw_lines = [ln.strip() for ln in str(final_insight).splitlines() if ln.strip()]
        insight_rows = []

        for ln in raw_lines:
            # تجاهل خطوط الفواصل (--- أو ***)
            if re.match(r"^[-*_]{3,}$", ln):
                continue

            # إزالة علامات ### وتنظيف الرموز الخاصة لـ ReportLab XML
            is_md_hdr = bool(re.match(r"^#{1,6}\s+", ln))
            clean_ln = re.sub(r"^#{1,6}\s*", "", ln).strip()
            clean_ln = _shape_arabic_text(clean_ln)
            clean_ln = clean_ln.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            # تحويل **النص** إلى Bold باللون الأخضر الداكن
            clean_ln = re.sub(r"\*\*(.+?)\*\*", r'<font color="#005B2D"><b>\1</b></font>', clean_ln)

            # تمييز عناوين الأقسام الرئيسية
            is_sec_hdr = is_md_hdr or bool(
                re.match(
                    r"^(?:\d+\.\s*)?(Executive Summary|Key Finans?cial Drivers|Risk (?:and|&) Liquidity Impact|Strategic(?: ALCO)? Recommendations|Overall Risk|Executive Verdict|Net Interest Income|Liquidity, Repricing|Strategic ALCO Action Plan)",
                    clean_ln,
                    re.IGNORECASE,
                )
            )
            if is_sec_hdr:
                clean_ln = f'<font color="#005B2D" size="9.5"><b>{clean_ln}</b></font>'

            insight_rows.append([Paragraph(clean_ln, body_style)])

        if insight_rows:
            insight_box = Table(insight_rows, colWidths=[avail_w])
            insight_box.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FBF9")),
                ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#BFE0CC")),
                ("LINEBEFORE", (0, 0), (0, -1), 3.5, colors.HexColor(NBE_GREEN)),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ("TOPPADDING", (0, 0), (-1, 0), 8),
                ("BOTTOMPADDING", (0, -1), (-1, -1), 8),
            ]))
            story.append(insight_box)
            story.append(Spacer(1, 8))

    # 3. Selected Key Performance Indicators (KPIs)
    if selected_kpis:
        story.append(Paragraph("Selected Key Performance Indicators (KPIs)", section_style))
        kpi_rows = [["KPI Metric", "Baseline (Before)", "Scenario (After)", "Impact / Delta"]]
        for item in selected_kpis:
            kpi_rows.append([
                str(item.get("label", "")),
                str(item.get("before", "")),
                str(item.get("after", "")),
                str(item.get("delta") or "-"),
            ])
        t_kpi = Table(kpi_rows, colWidths=[3.4 * inch, 2.4 * inch, 2.4 * inch, 2.4 * inch], repeatRows=1)
        t_kpi.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NBE_GREEN)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), _AR_FONT_BOLD),
            ("FONTNAME", (0, 1), (-1, -1), _AR_FONT_REGULAR),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D7E6DD")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EAF5EF")]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t_kpi)
        story.append(Spacer(1, 8))

    # 4. Selected Simulation Charts
    if selected_charts:
        story.append(Paragraph("Selected Simulation Charts", section_style))
        for chart_title, fig in selected_charts.items():
            if fig is None:
                continue
            try:
                img_bytes = plotly_fig_to_png_bytes(fig, width=900, height=380)
                story.append(Paragraph(f"<b>{chart_title}</b>", body_style))
                story.append(Spacer(1, 2))
                story.append(RLImage(io.BytesIO(img_bytes), width=7.8 * inch, height=3.2 * inch))
                story.append(Spacer(1, 10))
            except Exception as exc:
                story.append(Paragraph(f"Chart rendering note ({chart_title}): {exc}", body_style))

    # # 5. Detailed Scenario Comparison Table
    # if comparison_df is not None and not comparison_df.empty:
    #     story.append(PageBreak())
    #     story.append(Paragraph("Detailed Scenario Comparison Table", section_style))
    #     df_clean = comparison_df.drop(columns=[c for c in ["Source", "source", "Code", "code"] if c in comparison_df.columns], errors="ignore")
    #     comp_data = [list(df_clean.columns)]
    #     for _, r in df_clean.iterrows():
    #         comp_data.append([
    #             f"{v:,.2f}" if isinstance(v, (int, float, np.floating)) and pd.notna(v) else ("—" if pd.isna(v) else str(v))
    #             for v in r.values
    #         ])
    #     ct_table = Table(comp_data, repeatRows=1)
    #     ct_table.setStyle(TableStyle([
    #         ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NBE_DARK_GREEN)),
    #         ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    #         ("FONTNAME", (0, 0), (-1, 0), _AR_FONT_BOLD),
    #         ("FONTNAME", (0, 1), (-1, -1), _AR_FONT_REGULAR),
    #         ("FONTSIZE", (0, 0), (-1, -1), 8),
    #         ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
    #         ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FBF9")]),
    #         ("TOPPADDING", (0, 0), (-1, -1), 3.5),
    #         ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    #     ]))
    #     story.append(ct_table)

    # 6. Comparative & Forecasted Balance Sheet (±2Y / ±3Y)
    meta = fs_meta or {}
    win_yrs = kwargs.get("window_years")
    if balance_sheet_df is not None and not balance_sheet_df.empty:
        story.append(PageBreak())
        bs_cols, _ = _resolve_statement_columns(balance_sheet_df, is_income_stmt=False, meta=meta, window_years=win_yrs)
        horizon_bs_str = " | ".join(hdr for _, hdr in bs_cols)
        story.append(Paragraph(f"Comparative & Forecasted Balance Sheet ({horizon_bs_str})", title_style))
        story.append(
            Paragraph(
                "All figures in EGP Millions | A = Actual, E = Estimated Full-Year, F = L2 Ridge ML Forecasted (Assets = Liabilities + Equity | Δ = 0.00)",
                subtitle_style,
            )
        )
        story.append(_build_pdf_statement_table(balance_sheet_df, is_income_stmt=False, meta=meta, avail_width=avail_w, window_years=win_yrs))

    # 7. Comparative & Forecasted Income Statement (±2Y / ±3Y)
    if income_statement_df is not None and not income_statement_df.empty:
        story.append(PageBreak())
        is_cols, _ = _resolve_statement_columns(income_statement_df, is_income_stmt=True, meta=meta, window_years=win_yrs)
        horizon_is_str = " | ".join(hdr for _, hdr in is_cols)
        story.append(Paragraph(f"Comparative & Forecasted Income Statement ({horizon_is_str})", title_style))
        story.append(
            Paragraph(
                "All figures in EGP Millions | Interim periods automatically annualized to 12M Full-Year equivalent",
                subtitle_style,
            )
        )
        story.append(_build_pdf_statement_table(income_statement_df, is_income_stmt=True, meta=meta, avail_width=avail_w, window_years=win_yrs))

    doc.build(story, onFirstPage=_add_pdf_header_footer, onLaterPages=_add_pdf_header_footer)
    return buffer.getvalue()


# ============================================================================
# 2. PowerPoint (PPTX) Executive Deck Generator
# ============================================================================
def generate_pptx_report(
    scenario_name: str = "Current Scenario",
    scenario_type: str = "CUSTOM",
    simulation_date: str = "",
    horizon_days: int = 365,
    selected_kpis: Optional[list] = None,
    selected_charts: Optional[dict] = None,
    comparison_df: Optional[pd.DataFrame] = None,
    ai_insight: Optional[str] = None,
    params: Optional[dict] = None,
    engine=None,
    run_id=None,
    balance_sheet_df: Optional[pd.DataFrame] = None,
    income_statement_df: Optional[pd.DataFrame] = None,
    fs_meta: Optional[dict] = None,
    **kwargs,
) -> bytes:
    """
    Generate the complete Widescreen 16:9 NBE PowerPoint Deck containing:
      - Slide 1: Selected KPIs (Before, After, Impact / Delta)
      - Slide 2: Uniform Applied Scenario Parameters & Shocks Cards
      - Slide 3: Rich Executive ALCO Management Insight
      - Slides 4+: Selected Simulation Charts
      - Financial Statement Slides: Multi-Year & Forecast Balance Sheet and Income Statement (+/-2Y or +/-3Y)
    """
    if not PPTX_AVAILABLE:
        raise RuntimeError(f"Missing reporting libraries ({PPTX_MISSING_ERROR}). Run: pip install reportlab python-pptx kaleido")

    if hasattr(scenario_name, "before") or hasattr(scenario_name, "balance_sheet_df"):
        res_obj = scenario_name
        scenario_name = str(kwargs.get("scenario_name") or getattr(res_obj, "name", "Current Scenario"))
        scenario_type = str(getattr(res_obj, "type", "CUSTOM"))
        simulation_date = f"{res_obj.sim_date:%Y-%m-%d}" if hasattr(res_obj, "sim_date") else ""
        horizon_days = int(getattr(res_obj, "horizon", 365))
        params = params or getattr(res_obj, "params", {})
        comparison_df = comparison_df if comparison_df is not None else getattr(res_obj, "comparison", None)
        balance_sheet_df = balance_sheet_df if balance_sheet_df is not None else getattr(res_obj, "balance_sheet_df", None)
        income_statement_df = income_statement_df if income_statement_df is not None else getattr(res_obj, "income_statement_df", None)
        fs_meta = fs_meta or getattr(res_obj, "fs_meta", {})

    if balance_sheet_df is None:
        balance_sheet_df = kwargs.get("df_bs")
    if income_statement_df is None:
        income_statement_df = kwargs.get("df_is")
    if ai_insight is None:
        ai_insight = kwargs.get("ai_explanation")

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    nbe_green = RGBColor(*_hex_to_rgb_tuple(NBE_GREEN))
    nbe_dark = RGBColor(*_hex_to_rgb_tuple(NBE_DARK_GREEN))
    nbe_orange = RGBColor(*_hex_to_rgb_tuple(NBE_ORANGE))
    c_light_green = RGBColor(234, 245, 239)
    c_white = RGBColor(255, 255, 255)
    c_text = RGBColor(30, 41, 59)

    def add_nbe_slide_header_footer(slide, title: str, sub: str = ""):
        banner = slide.shapes.add_shape(1, Inches(0), Inches(0), Inches(13.333), Inches(0.9))
        banner.fill.solid()
        banner.fill.fore_color.rgb = nbe_dark
        banner.line.fill.background()
        tf = banner.text_frame
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(18)
        p.font.bold = True
        p.font.color.rgb = c_white
        if sub:
            p2 = tf.add_paragraph()
            p2.text = sub
            p2.font.size = Pt(9.5)
            p2.font.color.rgb = nbe_orange

        w_logo = _get_logo_path(prefer_white=True)
        if w_logo:
            try:
                slide.shapes.add_picture(w_logo, Inches(11.8), Inches(0.15), height=Inches(0.6))
            except Exception:
                pass

        footer = slide.shapes.add_shape(1, Inches(0), Inches(7.1), Inches(13.333), Inches(0.4))
        footer.fill.solid()
        footer.fill.fore_color.rgb = RGBColor(240, 245, 242)
        footer.line.color.rgb = RGBColor(200, 220, 210)
        ftf = footer.text_frame
        fp = ftf.paragraphs[0]
        fp.text = f"National Bank of Egypt (NBE) — {APP_NAME} v{APP_VERSION} | Confidential (ALCO Use Only)"
        fp.font.size = Pt(8)
        fp.font.color.rgb = RGBColor(100, 120, 110)

    # --- Slide 1: Selected KPIs ---
    slide1 = prs.slides.add_slide(blank_layout)
    add_nbe_slide_header_footer(
        slide1,
        f"NBE ALM Simulation Report — {scenario_name}",
        f"Type: {scenario_type} | Date: {simulation_date} | Horizon: {horizon_days} Days",
    )
    if selected_kpis:
        rows = len(selected_kpis) + 1
        ts = slide1.shapes.add_table(rows, 4, Inches(0.6), Inches(1.2), Inches(12.1), Inches(0.38 * rows))
        t = ts.table
        headers = ["KPI Metric", "Baseline (Before)", "Scenario (After)", "Impact / Delta"]
        for c, h in enumerate(headers):
            cell = t.cell(0, c)
            cell.text = h
            cell.fill.solid()
            cell.fill.fore_color.rgb = nbe_green
            for p in cell.text_frame.paragraphs:
                p.font.bold = True
                p.font.size = Pt(11)
                p.font.color.rgb = c_white
        for r, item in enumerate(selected_kpis, start=1):
            for c, val in enumerate([item.get("label", ""), item.get("before", ""), item.get("after", ""), item.get("delta") or "-"]):
                cell = t.cell(r, c)
                cell.text = str(val)
                for p in cell.text_frame.paragraphs:
                    p.font.size = Pt(10)

    # --- Slide 2: Uniform Scenario Parameters ---
    if params:
        slide_params = prs.slides.add_slide(blank_layout)
        add_nbe_slide_header_footer(
            slide_params,
            f"Applied Scenario Parameters — {scenario_name}",
            "Active macroeconomic & market stress drivers",
        )
        left_pos, top_pos = Inches(0.8), Inches(1.5)
        chip_width, chip_height = Inches(3.6), Inches(0.6)
        h_gap, v_gap = Inches(0.3), Inches(0.2)
        col_limit = 3

        idx = 0
        for k, v in params.items():
            if k in PARAM_SPECS:
                short_lbl = PARAM_SPECS[k][0].split(" (")[0]
                unit = PARAM_SPECS[k][1]
                val_str = f"{float(v):+g}{unit}" if isinstance(v, (int, float)) else f"{v}{unit}"

                c_x = left_pos + (idx % col_limit) * (chip_width + h_gap)
                c_y = top_pos + (idx // col_limit) * (chip_height + v_gap)

                shape = slide_params.shapes.add_shape(5, c_x, c_y, chip_width, chip_height)
                shape.fill.solid()
                shape.fill.fore_color.rgb = c_light_green
                shape.line.color.rgb = RGBColor(0, 102, 68)

                tf = shape.text_frame
                tf.word_wrap = True
                p = tf.paragraphs[0]
                p.text = f"{short_lbl}: {val_str}"
                p.font.size = Pt(11)
                p.font.bold = True
                p.font.color.rgb = RGBColor(23, 59, 42)
                idx += 1

    # --- Slide 3: Rich Management / AI Insight ---
    final_insight = _resolve_ai_insight(
        ai_insight=ai_insight,
        scenario_name=scenario_name,
        scenario_type=scenario_type,
        horizon_days=horizon_days,
        params=params,
        selected_kpis=selected_kpis,
        comparison_df=comparison_df,
        risk_df=None,
        engine=engine,
        run_id=run_id,
    )
    if final_insight:
        slide_ai = prs.slides.add_slide(blank_layout)
        add_nbe_slide_header_footer(slide_ai, "Scenario Explanation & Management Insight", f"Scenario: {scenario_name}")
        tb = slide_ai.shapes.add_textbox(Inches(0.7), Inches(1.15), Inches(11.9), Inches(5.6))
        tf = tb.text_frame
        tf.word_wrap = True
        clean_lines = [ln.replace("**", "").strip() for ln in str(final_insight).splitlines() if ln.strip()]
        for i_ln, ln_txt in enumerate(clean_lines):
            p = tf.paragraphs[0] if i_ln == 0 else tf.add_paragraph()
            p.text = ln_txt
            is_hdr = bool(re.match(r"^\d+\.\s*[^:]+:", ln_txt))
            p.font.size = Pt(11.5 if is_hdr else 10.5)
            p.font.bold = is_hdr
            p.font.color.rgb = nbe_dark if is_hdr else c_text

    # --- Slides 4+: Selected Charts ---
    if selected_charts:
        for chart_title, fig in selected_charts.items():
            if fig is None:
                continue
            slide_c = prs.slides.add_slide(blank_layout)
            add_nbe_slide_header_footer(slide_c, chart_title, f"Scenario: {scenario_name} ({simulation_date})")
            try:
                png_bytes = plotly_fig_to_png_bytes(fig, width=1100, height=480)
                slide_c.shapes.add_picture(io.BytesIO(png_bytes), Inches(0.8), Inches(1.15), width=Inches(11.7))
            except Exception as exc:
                tb = slide_c.shapes.add_textbox(Inches(1), Inches(2), Inches(10), Inches(2))
                tb.text_frame.text = f"Chart error ({chart_title}): {exc}"

    # --- Financial Statement Slides: Multi-Year & Forecast Balance Sheet & Income Statement ---
    meta = fs_meta or {}
    win_yrs = kwargs.get("window_years")

    def add_statement_table_slides(df: pd.DataFrame, is_income_stmt: bool, base_title: str):
        col_specs, _ = _resolve_statement_columns(df, is_income_stmt=is_income_stmt, meta=meta, window_years=win_yrs)
        n_num = len(col_specs)
        horizon_str = " — ".join(hdr for _, hdr in col_specs)

        max_rows_per_slide = 16
        rows_list = list(df.iterrows())
        chunks = [rows_list[i:i + max_rows_per_slide] for i in range(0, len(rows_list), max_rows_per_slide)]

        total_w = 12.5
        label_w = 3.2
        num_w = (total_w - label_w) / max(n_num, 1)
        font_pt = 8.5 if n_num >= 9 else (9.0 if n_num >= 7 else 9.5)

        for page_idx, chunk in enumerate(chunks, start=1):
            slide_fs = prs.slides.add_slide(blank_layout)
            part_lbl = f" (Part {page_idx}/{len(chunks)})" if len(chunks) > 1 else ""
            add_nbe_slide_header_footer(
                slide_fs,
                f"{base_title}{part_lbl} — {scenario_name}",
                f"Horizon: {horizon_str} | All figures in EGP Millions (A=Actual, E=Estimated, F=Forecasted)",
            )

            t_shape = slide_fs.shapes.add_table(
                len(chunk) + 1,
                n_num + 1,
                Inches(0.4),
                Inches(1.05),
                Inches(total_w),
                Inches(5.85),
            )
            table = t_shape.table
            table.columns[0].width = Inches(label_w)
            for c_i in range(1, n_num + 1):
                table.columns[c_i].width = Inches(num_w)

            hdr_titles = ["Line Item (EGP Mn)"] + [hdr for _, hdr in col_specs]
            for c_i, h_txt in enumerate(hdr_titles):
                cell = table.cell(0, c_i)
                cell.fill.solid()
                cell.fill.fore_color.rgb = nbe_dark
                p = cell.text_frame.paragraphs[0]
                p.text = h_txt
                p.font.size = Pt(font_pt)
                p.font.bold = True
                p.font.color.rgb = nbe_orange if h_txt == "Scenario" else c_white
                p.alignment = PP_ALIGN.LEFT if c_i == 0 else PP_ALIGN.RIGHT

            for r_i, (_, r) in enumerate(chunk, start=1):
                code = str(r.get("code", ""))
                src = str(r.get("source", ""))
                is_tot = any(t in code for t in ("TOT", "CHECK", "NII", "EBT", "NET_PROFIT")) or src in ("TOTAL", "SUBTOTAL", "CHECK")

                lbl_cell = table.cell(r_i, 0)
                if is_tot:
                    lbl_cell.fill.solid()
                    lbl_cell.fill.fore_color.rgb = c_light_green
                p0 = lbl_cell.text_frame.paragraphs[0]
                p0.text = str(r.get("name_en") or r.get("name_ar") or code)[:42]
                p0.font.size = Pt(font_pt)
                p0.font.bold = is_tot
                p0.font.color.rgb = nbe_dark if is_tot else c_text

                for c_i, (c_key, _) in enumerate(col_specs, start=1):
                    c_cell = table.cell(r_i, c_i)
                    if is_tot:
                        c_cell.fill.solid()
                        c_cell.fill.fore_color.rgb = c_light_green
                    p_c = c_cell.text_frame.paragraphs[0]
                    is_delta = "delta" in c_key.lower()
                    p_c.text = _fmt_num(r.get(c_key), signed=is_delta)
                    p_c.font.size = Pt(font_pt)
                    p_c.font.bold = is_tot or (c_key in ("Scenario (After)", "scn_mn"))
                    p_c.font.color.rgb = nbe_green if is_tot else c_text
                    p_c.alignment = PP_ALIGN.RIGHT

    if balance_sheet_df is not None and not balance_sheet_df.empty:
        add_statement_table_slides(balance_sheet_df, is_income_stmt=False, base_title="Comparative & Forecasted Balance Sheet")

    if income_statement_df is not None and not income_statement_df.empty:
        add_statement_table_slides(income_statement_df, is_income_stmt=True, base_title="Comparative & Forecasted Income Statement")

    out = io.BytesIO()
    prs.save(out)
    return out.getvalue()


# ============================================================================
# 3. ZIP Bundle Generator (Supports both pre-built bytes and on-the-fly generation)
# ============================================================================
def generate_zip_bundle(
    pdf_bytes_or_res,
    pptx_bytes: Optional[bytes] = None,
    base_filename: str = "NBE_ALM_Executive_Report",
    **kwargs,
) -> bytes:
    """
    Bundle PDF and PPTX reports into a single ZIP file.
    Supports both:
      - generate_zip_bundle(pdf_bytes, pptx_bytes, base_filename)  [used by page_dashboard.py]
      - generate_zip_bundle(res, scenario_name=...)
    """
    if isinstance(pdf_bytes_or_res, (bytes, bytearray)) and isinstance(pptx_bytes, (bytes, bytearray)):
        pdf_b = bytes(pdf_bytes_or_res)
        pptx_b = bytes(pptx_bytes)
        clean_base = str(base_filename or "NBE_ALM_Report").strip()
    else:
        scn_name = str(kwargs.get("scenario_name") or getattr(pdf_bytes_or_res, "name", "Scenario"))
        clean_base = f"NBE_ALM_Report_{re.sub(r'[^\w\s-]', '', scn_name).strip().replace(' ', '_')}"
        pdf_b = generate_pdf_report(pdf_bytes_or_res, **kwargs)
        pptx_b = generate_pptx_report(pdf_bytes_or_res, **kwargs)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{clean_base}.pdf", pdf_b)
        zf.writestr(f"{clean_base}.pptx", pptx_b)
    buf.seek(0)
    return buf.getvalue()