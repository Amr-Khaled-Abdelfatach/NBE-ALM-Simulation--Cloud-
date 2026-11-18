# ui_common.py
"""Shared UI helpers: executive theme, KPI cards, scenario execution, cached analytics, and AI panel."""
import base64
from datetime import date
import os
import time
from types import SimpleNamespace

import numpy as np
from typing import Optional, Tuple
import pandas as pd
import streamlit as st
from sqlalchemy import text

from config import *
from calculations import apply_prepared
from scenarios import ask_ollama, build_scenario_parameters, scenario_params_from_row
from database import update_run_ai_explanation
from analytics import (
    build_alm_risk_summary,
    build_data_quality_checks,
    calculation_validation_table,
    comparison_table,
    driver_attribution,
    fmt_amount,
    overall_risk_rating,
    shock_sensitivity,
)

format_large_number = fmt_amount

CUSTOM_CSS = r"""
<style>
.block-container { padding-top: 1.5rem !important; max-width: 1500px; }

.section-header { font-size: 1.25rem; font-weight: 700; margin: 1rem 0 .5rem 0; color: var(--text-color); border-bottom: 2px solid #EAF5EF; padding-bottom: .4rem; }
[data-testid="stMetric"] { background-color: var(--secondary-background-color); border: 1px solid var(--border-color); border-radius: 14px; padding: 14px; }
input, textarea, select, [data-baseweb="input"], [data-baseweb="select"], [data-baseweb="textarea"] { border-radius: 10px !important; }
.stButton > button { border-radius: 12px !important; min-height: 42px; font-weight: 600; transition: all .2s; }
.stButton > button:hover { border-color: #007A3D !important; color: #007A3D !important; }
.stButton > button[kind="primary"] { background:#007A3D !important; color:#FFFFFF !important; border-color:#007A3D !important; }
.stButton > button[kind="primary"]:hover { background:#005B2D !important; border-color:#005B2D !important; color:#FFFFFF !important; }
[data-testid="stSidebar"] { border-right: 1px solid var(--border-color); }

/* Top Executive Banner */
.app-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 18px 24px;
    margin-bottom: 16px;
    background: linear-gradient(135deg, #005B2D, #007A3D 70%, #2E8B57);
    border-radius: 16px;
    box-shadow: 0 8px 24px rgba(0,91,45,.16);
}
.app-title-container {
    display: flex;
    flex-direction: column;
    text-align: left;
}
.app-logo {
    height: 48px;
    width: auto;
    border-radius: 8px;
    background: transparent;
    padding: 2px;
}
.app-title { color: #fff; font-size: 1.55rem; font-weight: 750; }
.app-subtitle { color: #EAF5EF; font-size: .86rem; margin-top: 3px; }
.app-left-section {
    display: flex;
    align-items: center;
    gap: 15px;
}
.app-status { color: #fff; font-size: .8rem; border: 1px solid rgba(255,255,255,.25); padding: 6px 11px; border-radius: 999px; white-space: nowrap; }
.status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #F7941D; margin-right: 7px; }

/* Scenario Banner & Chips */
.scn-banner { padding: 14px 18px; border-radius: 14px; border: 1px solid var(--border-color); background: var(--secondary-background-color); margin-bottom: 12px; }
.scn-title { font-size: 1.2rem; font-weight: 750; }
.scn-meta { font-size: .82rem; opacity: .75; margin: 2px 0 8px 0; }
.chip { display: inline-block; padding: 3px 10px; margin: 2px 6px 2px 0; border-radius: 999px; background: #EAF5EF; color: #005B2D; font-size: .78rem; font-weight: 650; border: 1px solid #BFE0CC; }
.risk-badge { display: inline-block; padding: 5px 14px; border-radius: 999px; font-weight: 800; font-size: .85rem; letter-spacing: .5px; color: #fff; }
.risk-LOW { background: #007A3D; } .risk-MEDIUM { background: #F7941D; } .risk-HIGH { background: #C0392B; }

/* KPI Card Typography */
.kpi-label { font-size: .76rem; text-transform: uppercase; letter-spacing: .6px; opacity: .72; font-weight: 650; }
.kpi-value { font-size: 1.55rem; font-weight: 780; line-height: 1.25; }
.kpi-delta { font-size: .82rem; font-weight: 650; } .kpi-delta.good { color: #007A3D; } .kpi-delta.bad { color: #C0392B; } .kpi-delta.neutral { opacity: .7; }
.kpi-hint { font-size: .72rem; opacity: .6; }
.kpi-formula { font-size: .70rem; font-family: 'Consolas', 'Courier New', monospace; color: #005B2D; background: #EAF5EF; border: 1px dashed #BFE0CC; border-radius: 6px; padding: 4px 7px; margin-top: 6px; line-height: 1.35; }
.insight-box { padding: 14px 18px; border-radius: 14px; border: 1px solid #BFE0CC; background: linear-gradient(135deg,rgba(0,122,61,.07),rgba(247,148,29,.05)); margin: 8px 0 14px 0; }
.ai-card { padding: 16px; border: 1px solid var(--border-color); border-radius: 14px; background: var(--secondary-background-color); }

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #D2E3D9 !important;
    padding-top: 1.5rem;
}
[data-testid="stSidebar"] > div:first-child {
    background-color: #FFFFFF !important;
}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] h1 {
    color: #173B2A !important;
}
[data-testid="stSidebar"] .stButton > button {
    background-color: #FFFFFF !important;
    color: #173B2A !important;
    border: 1px solid #C8DCD1 !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 6px rgba(0, 91, 45, 0.03);
    transition: all 0.2s ease;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #EAF5EF !important;
    border-color: #007A3D !important;
    color: #005B2D !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #005B2D, #007A3D) !important;
    color: #FFFFFF !important;
    border: none !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 15px rgba(0, 91, 45, 0.25) !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] p,
[data-testid="stSidebar"] .stButton > button[kind="primary"] span {
    color: #FFFFFF !important;
}

/* Interactive Clickable Fintech KPI Card */
.kpi-card {
    position: relative;
    min-height: 116px;
    background: linear-gradient(145deg, #FFFFFF 0%, #F7FAF8 100%) !important;
    border: 1px solid #D5E0DA !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04), 0 1px 2px rgba(15, 23, 42, 0.03) !important;
    transition: all 0.22s cubic-bezier(0.16, 1, 0.3, 1);
    cursor: pointer;
    overflow: hidden !important;
    margin-bottom: 10px !important;
    box-sizing: border-box;
    pointer-events: none;
}
.kpi-card.good { border-left: 5px solid #007A3D !important; }
.kpi-card.bad { border-left: 5px solid #C93B2B !important; }
.kpi-card.neutral { border-left: 5px solid #5B7065 !important; }

div[data-testid="stElementContainer"]:has(.kpi-clickable-wrap):has(+ div[data-testid="stElementContainer"]:hover) .kpi-card,
.kpi-card:hover {
    transform: translateY(-3px);
    background: #FFFFFF !important;
    border-color: #007A3D !important;
    box-shadow: 0 12px 24px rgba(0, 91, 45, 0.10), 0 4px 8px rgba(0, 91, 45, 0.06) !important;
}

.kpi-calc-pill {
    position: absolute;
    top: 11px;
    right: 12px;
    display: inline-flex;
    align-items: center;
    gap: 0px;
    font-size: 0.76rem;
    font-weight: 600;
    color: #64748B;
    background: #F1F5F9;
    border: 1px solid #E2E8F0;
    border-radius: 99px;
    padding: 2px 7px;
    transition: all 0.22s ease;
}
.kpi-calc-pill .calc-text {
    max-width: 0;
    opacity: 0;
    overflow: hidden;
    white-space: nowrap;
    font-size: 0.68rem;
    font-weight: 700;
    transition: max-width 0.25s ease, opacity 0.2s ease, margin-left 0.2s ease;
}
div[data-testid="stElementContainer"]:has(.kpi-clickable-wrap):has(+ div[data-testid="stElementContainer"]:hover) .kpi-calc-pill,
.kpi-card:hover .kpi-calc-pill {
    color: #005B2D;
    background: #EAF5EF;
    border-color: #86EFAC;
    padding: 2px 10px;
    box-shadow: 0 2px 5px rgba(0, 91, 45, 0.08);
}
div[data-testid="stElementContainer"]:has(.kpi-clickable-wrap):has(+ div[data-testid="stElementContainer"]:hover) .kpi-calc-pill .calc-text,
.kpi-card:hover .kpi-calc-pill .calc-text {
    max-width: 120px;
    opacity: 1;
    margin-left: 5px;
}

div[data-testid="stElementContainer"]:has(.kpi-clickable-wrap) + div[data-testid="stElementContainer"] {
    margin-top: -126px !important;
    margin-bottom: 14px !important;
    height: 116px !important;
    position: relative !important;
    z-index: 1000 !important;
}
div[data-testid="stElementContainer"]:has(.kpi-clickable-wrap) + div[data-testid="stElementContainer"] button {
    width: 100% !important;
    height: 116px !important;
    opacity: 0 !important;
    cursor: pointer !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}
button[data-baseweb="tab"][aria-selected="true"] { color: #007A3D !important; font-weight: 700 !important; }
div[data-baseweb="tab-highlight"] { background-color: #007A3D !important; }
/* Professional NBE Toggle Switch Styling */
div[data-testid="stToggle"] {
    background: #F8FAFC;
    border: 1px solid #CBD5E1;
    border-radius: 10px;
    padding: 6px 12px;
    transition: all 0.2s ease;
}
div[data-testid="stToggle"]:hover {
    border-color: #007A3D;
    background: #EAF5EF;
}
div[data-testid="stToggle"] label[data-baseweb="checkbox"] > div:first-child {
    background-color: #007A3D !important;
}
div[data-testid="stToggle"] p {
    font-family: monospace !important;
    font-weight: 700 !important;
    color: #005B2D !important;
}
</style>
"""


# ----------------------------------------------------------------------------
# Visual components: DataFrames, Plotly themes, Header, and KPI Cards
# ----------------------------------------------------------------------------
def show_df(df, height=None):
    """Render a full-width DataFrame without index."""
    kwargs = {"use_container_width": True, "hide_index": True}
    if height:
        kwargs["height"] = height
    st.dataframe(df, **kwargs)


def doc_button(subpath: str = "", label: str = "📘 Online Documentation"):
    """Render a clean external documentation link button if DOCS_ENABLED is True."""
    if not DOCS_ENABLED:
        return
    url = DOCS_BASE_URL.rstrip("/") + "/" + str(subpath).lstrip("/")
    st.markdown(
        f'<div style="text-align:right; margin-bottom:8px;">'
        f'<a href="{url}" target="_blank" style="text-decoration:none; background:#EAF5EF; color:#005B2D; '
        f'padding:5px 12px; border-radius:8px; font-size:0.8rem; font-weight:700; border:1px solid #BFE0CC;">'
        f'{label} ↗</a></div>',
        unsafe_allow_html=True,
    )


def style_nbe_figure(fig, height=None):
    """Apply NBE corporate color palette and clean typography to any Plotly figure."""
    dark = str(st.get_option("theme.base") or "light").lower() == "dark"
    bg = "rgba(0,0,0,0)"
    text_color = "#EAF5EF" if dark else NBE_TEXT
    grid = "#35684D" if dark else "#E1ECE5"
    fig.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        colorway=NBE_PALETTE,
        font=dict(color=text_color, family=CHART_FONT),
        title_font=dict(color=NBE_ORANGE if dark else NBE_DARK_GREEN, size=16),
        legend=dict(font=dict(color=text_color), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor=grid, zerolinecolor=grid, linecolor=NBE_GREEN),
        yaxis=dict(gridcolor=grid, zerolinecolor=grid, linecolor=NBE_GREEN),
        margin=CHART_MARGIN,
        hoverlabel=dict(bgcolor=NBE_DARK_GREEN, font=dict(color="#FFFFFF")),
        height=height or 380,
    )
    return fig


def _format_ai_briefing_html(raw_text: str) -> str:
    """
    Convert raw AI ALCO report text (from either Online Gemini or Offline Ollama)
    into a clean, compact, institutional HTML memo with icons, section underlines,
    styled tables, and numbered ALCO recommendation badges.
    """
    import html
    import re

    text = str(raw_text or "").strip()
    if not text:
        return ""

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    section_icons = {
        "executive summary": "📋",
        "key financial drivers": "📊",
        "key finanscial drivers": "📊",
        "risk and liquidity impact": "🛡️",
        "risk & liquidity impact": "🛡️",
        "strategic recommendations": "🎯",
        "strategic alco recommendations": "🎯",
        "overall risk": "⚖️",
    }

    def _clean_cell_val(cell_str: str) -> str:
        s = cell_str.strip()
        if re.match(r"^-?\d+\.\d{3,}$", s):
            try:
                return f"{float(s):,.2f}"
            except Exception:
                pass
        return html.escape(s)

    def _render_html_table(table_lines: list) -> str:
        parsed_rows = []
        for t_ln in table_lines:
            if re.match(r"^\|[\s\-:|]+\|$", t_ln):
                continue
            cells = [c.strip() for c in t_ln.strip("|").split("|")]
            parsed_rows.append(cells)

        if not parsed_rows:
            return ""

        headers = parsed_rows[0]
        body_rows = parsed_rows[1:]

        t_html = [
            '<div style="overflow-x:auto; margin:12px 0; border-radius:10px; border:1px solid #CBD5E1;">'
            '<table style="width:100%; border-collapse:collapse; font-size:0.83rem; text-align:left;">'
            '<thead><tr style="background:#005B2D; color:#FFFFFF;">'
        ]
        for h in headers:
            t_html.append(
                f'<th style="padding:8px 10px; font-weight:700; white-space:nowrap; border-bottom:2px solid #004220;">'
                f'{html.escape(h)}</th>'
            )
        t_html.append("</tr></thead><tbody>")

        for r_idx, row in enumerate(body_rows):
            bg = "#FFFFFF" if r_idx % 2 == 0 else "#F8FAFC"
            t_html.append(f'<tr style="background:{bg}; border-bottom:1px solid #E2E8F0;">')
            for c_val in row:
                clean_v = _clean_cell_val(c_val)
                upper_v = clean_v.upper()
                if upper_v in ("ADVERSE", "HIGH"):
                    badge = f'<span style="background:#FEE2E2; color:#991B1B; padding:2px 7px; border-radius:6px; font-weight:700; font-size:0.76rem;">{clean_v}</span>'
                    t_html.append(f'<td style="padding:7px 10px;">{badge}</td>')
                elif upper_v in ("POSITIVE", "OK", "LOW"):
                    badge = f'<span style="background:#DCFCE7; color:#166534; padding:2px 7px; border-radius:6px; font-weight:700; font-size:0.76rem;">{clean_v}</span>'
                    t_html.append(f'<td style="padding:7px 10px;">{badge}</td>')
                else:
                    is_num = bool(re.match(r"^-?[\d,]+(?:\.\d+)?%?$", clean_v))
                    align = "right" if is_num else "left"
                    t_html.append(f'<td style="padding:7px 10px; text-align:{align}; color:#1E293B;">{clean_v}</td>')
            t_html.append("</tr>")

        t_html.append("</tbody></table></div>")
        return "".join(t_html)

    html_parts = [
        '<div style="background:#FFFFFF; border:1px solid #D2E3D9; border-radius:14px; '
        'padding:20px 24px; box-shadow:0 4px 14px rgba(0,91,45,0.05); font-family:inherit; color:#1E293B; overflow-wrap:break-word;">'
    ]

    i = 0
    while i < len(lines):
        line = lines[i]

        # 1. Markdown Tables
        if line.startswith("|") and line.endswith("|"):
            tbl_block = []
            while i < len(lines) and lines[i].startswith("|") and lines[i].endswith("|"):
                tbl_block.append(lines[i])
                i += 1
            html_parts.append(_render_html_table(tbl_block))
            continue

        # 2. Horizontal rules (--- or ***)
        if re.match(r"^[-*_]{3,}$", line):
            html_parts.append('<hr style="border:none; border-top:1px solid #E2E8F0; margin:12px 0;">')
            i += 1
            continue

        clean_ln = re.sub(r"^#{1,6}\s*", "", line).strip()
        esc_ln = html.escape(clean_ln)
        esc_ln = re.sub(r"\*\*(.+?)\*\*", r'<strong style="color:#005B2D;">\1</strong>', esc_ln)

        # 3. Document Main Title Banner (first line if it starts with ALM Executive...)
        if i == 0 and re.match(r"^(?:<strong[^>]*>)?ALM Executive", esc_ln, re.IGNORECASE):
            title_clean = re.sub(r"</?strong[^>]*>", "", esc_ln)
            html_parts.append(
                f'<div style="background:linear-gradient(135deg,#005B2D,#007A3D); color:#FFFFFF; '
                f'padding:10px 16px; border-radius:10px; font-size:1.05rem; font-weight:750; margin-bottom:12px;">'
                f'🏛️ {title_clean}</div>'
            )
            i += 1
            continue

        # 4. Section Headers (matches with OR without colon, and with optional leading number like "1. Executive Summary")
        sec_match = re.match(
            r"^(?:\d+[.)]\s*)?(?:<strong[^>]*>)?(Executive Summary|Key Finans?cial Drivers|Risk (?:and|&) Liquidity Impact|Strategic(?: ALCO)? Recommendations|Overall Risk)(?:</strong>)?(?:\s*[:\-]\s*(.*))?$",
            esc_ln,
            re.IGNORECASE,
        )

        if sec_match:
            sec_title = sec_match.group(1).strip()
            rest_text = (sec_match.group(2) or "").strip()
            icon = section_icons.get(sec_title.lower(), "📌")

            if sec_title.lower() == "overall risk":
                badge_bg = "#FEF2F2" if "HIGH" in rest_text.upper() else ("#FFFBEB" if "MEDIUM" in rest_text.upper() else "#EAF5EF")
                badge_border = "#FCA5A5" if "HIGH" in rest_text.upper() else ("#FCD34D" if "MEDIUM" in rest_text.upper() else "#86EFAC")
                badge_color = "#991B1B" if "HIGH" in rest_text.upper() else ("#92400E" if "MEDIUM" in rest_text.upper() else "#005B2D")
                html_parts.append(
                    f'<div style="margin-top:14px; padding:12px 16px; border-radius:10px; '
                    f'background:{badge_bg}; border:1px solid {badge_border}; color:{badge_color}; font-size:0.92rem; line-height:1.55;">'
                    f'<strong>{icon} {sec_title}:</strong> {rest_text}</div>'
                )
            else:
                html_parts.append(
                    f'<div style="margin-top:16px; margin-bottom:8px; padding-bottom:5px; '
                    f'border-bottom:2px solid #EAF5EF; font-size:1.04rem; font-weight:750; color:#005B2D;">'
                    f'{icon} {sec_title}</div>'
                )
                if rest_text:
                    html_parts.append(
                        f'<p style="margin:4px 0 10px 0; font-size:0.93rem; line-height:1.65; color:#1E293B;">{rest_text}</p>'
                    )
            i += 1
            continue

        # 5. Bullet items (-, •, *)
        bullet_match = re.match(r"^[-•*]\s+(.*)$", esc_ln)
        if bullet_match:
            b_body = bullet_match.group(1)
            b_body = re.sub(r"^([A-Za-z0-9\s/&()%_+-]{2,50}):\s*", r'<strong style="color:#005B2D;">\1:</strong> ', b_body)
            html_parts.append(
                f'<div style="display:flex; align-items:flex-start; gap:9px; margin:6px 0 6px 6px; font-size:0.92rem; line-height:1.62;">'
                f'<span style="color:#007A3D; font-weight:800; font-size:1.05rem; line-height:1.4;">•</span>'
                f'<div style="flex:1;">{b_body}</div></div>'
            )
            i += 1
            continue

        # 6. Numbered recommendations (1., 2., 3.)
        num_match = re.match(r"^(\d+)[.)]\s+(.*)$", esc_ln)
        if num_match:
            n_idx = num_match.group(1)
            n_body = num_match.group(2)
            n_body = re.sub(r"^([A-Za-z0-9\s/&()%_+-]{2,50}):\s*", r'<strong style="color:#005B2D;">\1:</strong> ', n_body)
            html_parts.append(
                f'<div style="display:flex; align-items:flex-start; gap:10px; margin:7px 0 7px 4px; '
                f'background:#F8FAFC; border-left:3px solid #007A3D; padding:8px 12px; border-radius:6px; font-size:0.91rem; line-height:1.6;">'
                f'<span style="background:#005B2D; color:#FFFFFF; font-weight:700; font-size:0.78rem; '
                f'min-width:20px; height:20px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center; margin-top:2px;">{n_idx}</span>'
                f'<div style="flex:1;">{n_body}</div></div>'
            )
            i += 1
            continue

        # 7. Standard paragraph
        html_parts.append(
            f'<p style="margin:6px 0; font-size:0.93rem; line-height:1.65; color:#1E293B;">{esc_ln}</p>'
        )
        i += 1

    html_parts.append("</div>")
    return "".join(html_parts)



def pct_change(before, after):
    """Compute percentage change between two values safely."""
    before = float(before or 0)
    after = float(after) if pd.notna(after) else np.nan
    return ((after - before) / abs(before) * 100.0) if before and pd.notna(after) else np.nan


def render_page_header(title, subtitle=None, status="Connected"):
    """Render the top NBE executive banner with connection status and configured bank logo."""
    sub = f'<div class="app-subtitle">{subtitle}</div>' if subtitle else ""
    logo_html = ""
    try:
        header_logo = LOGO_HEADER_WHITE_PATH if os.path.exists(LOGO_HEADER_WHITE_PATH) else LOGO_SIDEBAR_PATH
        if header_logo and os.path.exists(header_logo):
            with open(header_logo, "rb") as f:
                encoded_logo = base64.b64encode(f.read()).decode()
            logo_html = f'<img src="data:image/png;base64,{encoded_logo}" class="app-logo" alt="NBE Logo">'
    except Exception:
        pass

    st.markdown(
        f'''<div class="app-header">
            <div class="app-title-container">
                <div class="app-title">{title}</div>
                {sub}
            </div>
            <div class="app-left-section">
                <div class="app-status"><span class="status-dot"></span>{status}</div>
                {logo_html}
            </div>
        </div>''',
        unsafe_allow_html=True,
    )


def kpi_card(label, value, delta=None, tone="neutral", hint="", formula="", breakdown_rows=None):
    """Render an interactive KPI card with a hover 'View Calculation' pill."""
    arrow = {"good": "▲ ", "bad": "▼ ", "neutral": ""}[tone] if delta else ""
    delta_html = f'<div class="kpi-delta {tone}">{arrow}{delta}</div>' if delta else '<div class="kpi-delta neutral">&nbsp;</div>'
    hint_html = f'<div class="kpi-hint">{hint}</div>' if hint else '<div class="kpi-hint">&nbsp;</div>'

    pill_html = (
        '<span class="kpi-calc-pill">ⓘ<span class="calc-text">View Calculation</span></span>'
        if formula else ""
    )
    wrap_cls = "kpi-card kpi-clickable-wrap" if formula else "kpi-card"

    st.markdown(
        f'<div class="{wrap_cls} {tone}">'
        f'{pill_html}'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'{delta_html}{hint_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def tone_for(delta, higher_is_better):
    """Determine card semantic color tone ('good', 'bad', or 'neutral') from delta direction."""
    if higher_is_better is None or delta is None or pd.isna(delta) or abs(delta) < 1e-12:
        return "neutral"
    return "good" if (delta > 0) == bool(higher_is_better) else "bad"


def risk_badge(level):
    """Return HTML badge for overall risk rating."""
    return f'<span class="risk-badge risk-{level}">OVERALL RISK: {level}</span>'


def active_driver_chips(params):
    """Render pill chips for all non-zero scenario parameters."""
    chips = []
    for k, (label, unit, *_rest, _group) in PARAM_SPECS.items():
        val = float(params.get(k, SCENARIO_DEFAULTS.get(k, 0.0)))
        if abs(val - SCENARIO_DEFAULTS.get(k, 0.0)) > 1e-12:
            chips.append(f'<span class="chip">{label.split(" (")[0]}: {val:+g}{unit}</span>')
    return "".join(chips) or '<span class="chip">Base case - no stress assumptions</span>'


# ----------------------------------------------------------------------------
# Scenario execution & caching helpers
# ----------------------------------------------------------------------------
def run_scenario(ctx, params, sim_date, horizon_days, base_as_of=None):
    """Execute a complete deterministic scenario and bundle position, liquidity, and FS outputs."""
    from fs_engine import build_financial_statements_and_kpis

    params = build_scenario_parameters(params)
    before, after, detail = apply_prepared(
        ctx.prep, sim_date, int(horizon_days), params, base_as_of=base_as_of
    )
    risk = build_alm_risk_summary(before, after)
    level, reasons = overall_risk_rating(before, after, risk)

    fs_bundle = after.get("_fs_bundle")
    if not fs_bundle:
        fs_bundle = build_financial_statements_and_kpis(
            pos_detail_df=detail,
            params=params,
            before_dict=before,
            after_dict=after,
            horizon_days=int(horizon_days),
            engine=getattr(ctx, "engine", None),
            fs_flat_df=getattr(ctx, "fs_flat", None),
            macro_dict=ctx.prep.get("macro") if hasattr(ctx, "prep") and isinstance(ctx.prep, dict) else None,
            sim_date=sim_date,
        )

    return SimpleNamespace(
        params=params,
        sim_date=pd.Timestamp(sim_date),
        horizon=int(horizon_days),
        base_as_of=base_as_of,
        before=before,
        after=after,
        detail=detail,
        pos_detail=detail,
        kpis=fs_bundle["fs_kpis"],
        fs_meta=fs_bundle.get("fs_meta", {}),
        balance_sheet_df=fs_bundle["balance_sheet_df"],
        income_statement_df=fs_bundle["income_statement_df"],
        comparison=comparison_table(before, after),
        risk=risk,
        risk_level=level,
        risk_reasons=reasons,
        quality=build_data_quality_checks(
            before, ctx.positions, ctx.cashflows, ctx.market_data, after,
            raw=ctx.raw, source_notes=ctx.source_notes,
        ),
        validation=calculation_validation_table(before, after, detail, params),
        name="",
        type="",
    )


def current_session_scenario(ctx):
    """Run the scenario currently configured in Streamlit session_state."""
    res = run_scenario(
        ctx,
        st.session_state.get("current_params") or {},
        st.session_state.get("current_simulation_date", date.today()),
        st.session_state.get("current_horizon_days", DEFAULT_HORIZON_DAYS),
    )
    res.name = st.session_state.get("current_scenario_name", "Current Session")
    res.type = "Current Scenario"
    return res


def evaluate_saved_scenario(ctx, row):
    """Re-evaluate a saved database scenario row using its historical reference date and horizon."""
    sim = pd.to_datetime(row.get("simulation_date"), errors="coerce")
    sim = sim if pd.notna(sim) else pd.Timestamp(date.today())
    horizon = int(row.get("horizon_days") or DEFAULT_HORIZON_DAYS)
    base_as_of = row.get("base_as_of_datetime")
    res = run_scenario(ctx, scenario_params_from_row(row), sim, horizon, base_as_of=base_as_of)
    res.name = str(row.get("scenario_name", "Saved scenario"))
    res.type = str(row.get("scenario_type", "SAVED"))
    return res


def _items(params):
    return tuple(sorted((k, round(float(v), 10)) for k, v in params.items()))


@st.cache_data(ttl=DATA_CACHE_TTL_SECONDS, show_spinner=False, max_entries=64)
def cached_sensitivity(_prep, stamp, sim, horizon, params_items, key, values):
    return shock_sensitivity(_prep, sim, horizon, dict(params_items), key, list(values))


@st.cache_data(ttl=DATA_CACHE_TTL_SECONDS, show_spinner=False, max_entries=64)
def cached_attribution(_prep, stamp, sim, horizon, params_items):
    return driver_attribution(_prep, sim, horizon, dict(params_items))


def sensitivity(ctx, res, key, values):
    return cached_sensitivity(ctx.prep, ctx.stamp, str(res.sim_date.date()), res.horizon, _items(res.params), key, tuple(values))


def attribution(ctx, res):
    return cached_attribution(ctx.prep, ctx.stamp, str(res.sim_date.date()), res.horizon, _items(res.params))


# ----------------------------------------------------------------------------
# Local AI (Ollama / Phi-3) Executive Scenario Explanation Panel
# ----------------------------------------------------------------------------
def render_ai_scenario_panel(ctx, res, scenario_name, scenario_type, panel_key="scenario", insights=None):
    """Render the interactive Phi-3 explanation panel with single-button hard reset & MySQL persistence."""
    st.markdown('<div class="ai-card">', unsafe_allow_html=True)

    run_id = None
    if ctx.engine is not None:
        try:
            q_find = text("""
                SELECT sr.run_id
                FROM simulation_runs sr
                JOIN scenarios s ON sr.scenario_id = s.scenario_id
                WHERE s.scenario_name = :sname
                ORDER BY sr.run_id DESC LIMIT 1
            """)
            df_id = pd.read_sql(q_find, ctx.engine, params={"sname": str(scenario_name)})
            if not df_id.empty:
                run_id = int(df_id.iloc[0]["run_id"])
            else:
                q_scen = text("SELECT scenario_id FROM scenarios WHERE scenario_name = :sname")
                res_scen = pd.read_sql(q_scen, ctx.engine, params={"sname": str(scenario_name)})
                if not res_scen.empty:
                    scen_id = int(res_scen.iloc[0]["scenario_id"])
                else:
                    with ctx.engine.begin() as conn:
                        res_ins = conn.execute(
                            text("INSERT INTO scenarios (scenario_name, scenario_type, simulation_date, horizon_days) VALUES (:name, :stype, :sdate, :hz)"),
                            {"name": str(scenario_name), "stype": str(scenario_type), "sdate": res.sim_date.date(), "hz": int(res.horizon)},
                        )
                        scen_id = int(res_ins.lastrowid)

                with ctx.engine.begin() as conn:
                    res_run_ins = conn.execute(
                        text("INSERT INTO simulation_runs (scenario_id, simulation_date, horizon_days, run_status) VALUES (:sid, :sdate, :hz, 'COMPLETED')"),
                        {"sid": scen_id, "sdate": res.sim_date.date(), "hz": int(res.horizon)},
                    )
                    run_id = int(res_run_ins.lastrowid)
        except Exception as db_init_err:
            st.warning(f"Database run_id resolution warning: {db_init_err}")

    st.write(f"Current Run ID: {run_id}")
    st.markdown(f"### 🤖 AI scenario explanation — {scenario_name}")
    st.caption(f"Scenario: {scenario_name} | Type: {scenario_type} | Date: {res.sim_date:%Y-%m-%d} | Horizon: {res.horizon} days")
    st.info("AI only explains the deterministic calculation. Python remains the source of every number.")

    payload = {
        "scenario_name": scenario_name,
        "scenario_type": scenario_type,
        "simulation_date": str(res.sim_date.date()),
        "horizon_days": res.horizon,
        "inputs": res.params,
        "before": {k: v for k, v in res.before.items() if not str(k).startswith(("Macro", "_")) and not isinstance(v, (dict, pd.DataFrame))},
        "after": {k: v for k, v in res.after.items() if not str(k).startswith(("Macro", "_")) and not isinstance(v, (dict, pd.DataFrame))},
        "risk": res.risk.to_dict(orient="records"),
        "overall_risk": res.risk_level,
        "deterministic_insights": insights or [],
    }

    unique_key = f"{panel_key}_{run_id if run_id is not None else scenario_name}"
    insight_state_key = f"ai_insight_{unique_key}"
    meta_state_key = f"ai_insight_meta_{unique_key}"

    def _is_error_text(txt: str) -> bool:
        if not txt:
            return True
        s = str(txt).strip()
        return s.startswith(("Phi-3 unavailable", "Phi-3 explanation unavailable", "Phi-3 request", "Ollama connection", "AI unavailable", "AI request", "No response"))

    def _hard_delete_old_and_save_new(new_explanation: Optional[str] = None) -> Tuple[bool, str]:
        """Delete old explanation across ALL runs of this scenario in DB, and save the new one if provided."""
        if ctx.engine is None:
            return False, "No active database connection."
        try:
            with ctx.engine.begin() as conn:
                # 1. Wipe old explanation from ALL runs of this scenario_name so no stale text ever survives
                conn.execute(
                    text("""
                        UPDATE simulation_runs sr
                        JOIN scenarios s ON sr.scenario_id = s.scenario_id
                        SET sr.ai_explanation = NULL
                        WHERE s.scenario_name = :sname
                    """),
                    {"sname": str(scenario_name)},
                )
                if run_id:
                    conn.execute(
                        text("UPDATE simulation_runs SET ai_explanation = NULL WHERE run_id = :rid"),
                        {"rid": int(run_id)},
                    )
                # 2. Save the fresh explanation if provided
                if new_explanation and run_id:
                    conn.execute(
                        text("UPDATE simulation_runs SET ai_explanation = :exp WHERE run_id = :rid"),
                        {"exp": str(new_explanation).strip(), "rid": int(run_id)},
                    )
            st.cache_data.clear()
            return True, f"✅ Old explanation wiped & new AI explanation saved (Run ID: {run_id})."
        except Exception as exc:
            return False, f"Database update error: {exc}"

    # Read latest saved explanation directly from MySQL
    existing_insight = None
    if ctx.engine is not None and run_id:
        try:
            with ctx.engine.connect() as conn:
                row_exp = conn.execute(
                    text("SELECT ai_explanation FROM simulation_runs WHERE run_id = :rid"),
                    {"rid": int(run_id)},
                ).fetchone()
                if row_exp and row_exp[0]:
                    val = str(row_exp[0]).strip()
                    if val and not _is_error_text(val):
                        existing_insight = val
        except Exception:
            pass

    session_val = st.session_state.get(insight_state_key)
    if session_val and not _is_error_text(session_val):
        valid_insight = session_val
        last_error = None
    elif session_val and _is_error_text(session_val):
        valid_insight = existing_insight
        last_error = str(session_val)
    else:
        valid_insight = existing_insight
        last_error = None

    # Sync with page_dashboard.py PDF/PPTX exporter key
    st.session_state[f"ai_insight_{panel_key}"] = valid_insight

    def _trigger_fresh_generation():
        # 1. Completely purge old text from session_state & MySQL first (treat as brand new run)
        st.session_state.pop(insight_state_key, None)
        st.session_state.pop(f"ai_insight_{panel_key}", None)
        _hard_delete_old_and_save_new(new_explanation=None)

        with st.spinner("Wiping old explanation & generating fresh AI briefing from scratch..."):
            cfg_mode = str(st.session_state.get("ai_provider", AI_PROVIDER)).lower()
            host = st.session_state.get("ollama_host", OLLAMA_HOST)
            model = (
                st.session_state.get("gemini_model", GEMINI_MODEL)
                if cfg_mode == "online"
                else st.session_state.get("ollama_model", OLLAMA_MODEL)
            )
            new_text = ask_ollama(host, model, payload)

            st.session_state[insight_state_key] = new_text
            st.session_state[f"ai_insight_{panel_key}"] = new_text
            st.session_state[meta_state_key] = {"scenario_name": scenario_name, "model": model}

            if not _is_error_text(new_text) and ctx.engine is not None and run_id:
                ok, msg = _hard_delete_old_and_save_new(new_explanation=str(new_text))
                if ok:
                    st.session_state[f"ai_db_saved_msg_{unique_key}"] = msg
                else:
                    st.session_state[f"ai_db_err_msg_{unique_key}"] = msg

    if st.session_state.get(f"ai_db_saved_msg_{unique_key}"):
        st.success(st.session_state.pop(f"ai_db_saved_msg_{unique_key}"))
    if st.session_state.get(f"ai_db_err_msg_{unique_key}"):
        st.error(st.session_state.pop(f"ai_db_err_msg_{unique_key}"))

    if not valid_insight:
        if last_error:
            st.warning(last_error)
            st.session_state.pop(insight_state_key, None)
        if st.button("Explain this scenario", type="primary", key=f"ai_explain_{unique_key}"):
            _trigger_fresh_generation()
            st.rerun()
    else:
        if existing_insight and valid_insight == existing_insight:
            st.success(f"⚡ Loaded saved AI explanation from database (Run ID: {run_id}).")

        if last_error:
            st.warning(f"Last regeneration attempt failed ({last_error}). Showing previously saved report:")

        st.markdown(_format_ai_briefing_html(valid_insight), unsafe_allow_html=True)
        meta = st.session_state.get(meta_state_key, {})
        st.caption(f"Explained scenario: {meta.get('scenario_name', scenario_name)} | Model: {meta.get('model', OLLAMA_MODEL)}")

        # Single unified button: Wipes old, regenerates fresh, and saves to DB
        if st.button("🔄 Regenerate & Save AI Explanation", type="primary", key=f"ai_regen_{unique_key}"):
            _trigger_fresh_generation()
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)