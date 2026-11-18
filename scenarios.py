# scenarios.py
"""Scenario definitions, parameter parsing, templates, and local AI explanation helper."""
import json
import re
from typing import Dict, Optional, Tuple

import pandas as pd
import requests
from config import (
    INDIVIDUAL_DRIVERS,
    OLLAMA_TEMPERATURE,
    OLLAMA_TIMEOUT,
    PARAM_SPECS,
    SCENARIO_DEFAULTS,
    SCENARIO_TEMPLATES,
)


def scenario_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean summary table of saved scenarios from the database."""
    if df is None or df.empty:
        return pd.DataFrame()
    fields = [
        "scenario_id",
        "scenario_code",
        "scenario_name",
        "scenario_type",
        "base_as_of_datetime",
        "simulation_date",
        "horizon_days",
        "policy_rate_shock_bps",
        "fx_shock_pct",
        "gold_shock_pct",
        "wsf_spread_shock_bps",
        "deposit_runoff_pct",
        "loan_growth_pct",
        "deposit_growth_pct",
        "inflation_shock_pct",
        "unemployment_shock_pct",
        "gdp_shock_pct",
        "status",
    ]
    keep = [c for c in fields if c in df.columns]
    return df[keep].drop_duplicates().reset_index(drop=True)


def scenario_params_from_row(row) -> Dict[str, float]:
    """Read all 10 scenario parameters from a database row and normalize rate units."""
    if row is None:
        return dict(SCENARIO_DEFAULTS)

    def f(name, default=0.0):
        try:
            v = row.get(name, default)
            return float(v) if v is not None and not pd.isna(v) else float(default)
        except Exception:
            return float(default)

    # Smart unit check: values > 25 are stored in bps (e.g. 200 bps -> 2.0 pp); smaller values are already in pp
    raw_rate = f("policy_rate_shock_bps")
    rate_pp = raw_rate / 100.0 if abs(raw_rate) > 25.0 else raw_rate

    p = dict(SCENARIO_DEFAULTS)
    p.update({
        "rate_change_pp": rate_pp,
        "fx_shock_pct": f("fx_shock_pct"),
        "gold_shock_pct": f("gold_shock_pct"),
        "wsf_spread_shock_bps": f("wsf_spread_shock_bps"),
        "deposit_runoff_pct": f("deposit_runoff_pct"),
        "loan_growth_pct": f("loan_growth_pct"),
        "deposit_growth_pct": f("deposit_growth_pct"),
        "inflation_shock_pct": f("inflation_shock_pct"),
        "unemployment_shock_pct": f("unemployment_shock_pct"),
        "gdp_shock_pct": f("gdp_shock_pct"),
    })
    return p


def scenario_row_from_params(params: Dict[str, float]) -> Dict[str, float]:
    """Convert engine parameters into the 10 database columns of the `scenarios` table."""
    p = build_scenario_parameters(params)
    return {
        "policy_rate_shock_bps": p["rate_change_pp"] * 100.0,
        "fx_shock_pct": p["fx_shock_pct"],
        "gold_shock_pct": p["gold_shock_pct"],
        "wsf_spread_shock_bps": p.get("wsf_spread_shock_bps", 0.0),
        "deposit_runoff_pct": p["deposit_runoff_pct"],
        "loan_growth_pct": p["loan_growth_pct"],
        "deposit_growth_pct": p["deposit_growth_pct"],
        "inflation_shock_pct": p["inflation_shock_pct"],
        "unemployment_shock_pct": p["unemployment_shock_pct"],
        "gdp_shock_pct": p["gdp_shock_pct"],
    }


def build_scenario_parameters(values: Optional[dict] = None) -> Dict[str, float]:
    """Merge user-supplied parameter values with default baseline assumptions."""
    params = dict(SCENARIO_DEFAULTS)
    if values:
        for k, v in values.items():
            if k in params and v is not None:
                try:
                    params[k] = float(v)
                except (TypeError, ValueError):
                    pass
    return params


# ----------------------------------------------------------------------------
# Scenario template helpers
# ----------------------------------------------------------------------------
def template_values(template_name: str) -> dict:
    """Return initial default values for a selected UI scenario template."""
    template = SCENARIO_TEMPLATES[template_name]
    values = dict(SCENARIO_DEFAULTS)
    values.update({k: float(v) for k, v in template.get("values", {}).items()})
    if template_name == "Individual Parameter Scenario":
        first_driver = next(iter(INDIVIDUAL_DRIVERS.keys()), "Policy Rate Shock")
        values["individual_parameter"] = template.get("individual_parameter", first_driver)
        values["individual_value"] = float(template.get("individual_value", 0.0))
    return values


_TEMPLATE_BY_NAME = {name.upper(): name for name in SCENARIO_TEMPLATES}


def resolve_saved_template(scenario_type, saved_values=None) -> str:
    """Map a database scenario_type string back to its matching UI template name."""
    t = str(scenario_type).upper()
    if t in _TEMPLATE_BY_NAME:
        return _TEMPLATE_BY_NAME[t]
    if t == "BASE":
        return "Base Case"
    if t in {"COMBINED_STRESS", "COMBINED_EASING", "TWO_FACTOR", "CUSTOM"}:
        return "Combined Stress Scenario"
    if t == "FREE_TYPE":
        return "Free Type Scenario"
    if "INDIVIDUAL" in t or "SINGLE" in t or "SHOCK" in t:
        return "Individual Parameter Scenario"
    return "Combined Stress Scenario"


def individual_driver_to_params(driver_label: str, value: float) -> Dict[str, float]:
    """Build a parameter dictionary where only one selected driver is stressed."""
    params = dict(SCENARIO_DEFAULTS)
    key = INDIVIDUAL_DRIVERS.get(driver_label)
    if key:
        params[key] = float(value)
    return params


def params_to_individual_driver(params: Dict[str, float]) -> Tuple[str, float]:
    """Identify which single driver is active in an individual-driver scenario."""
    for label, key in INDIVIDUAL_DRIVERS.items():
        if abs(float(params.get(key, SCENARIO_DEFAULTS[key])) - SCENARIO_DEFAULTS[key]) > 1e-12:
            return label, float(params[key])
    first_driver = next(iter(INDIVIDUAL_DRIVERS.keys()), "Policy Rate Shock")
    return first_driver, 0.0


def build_saved_template_values(template_name: str, params: Dict[str, float]) -> dict:
    """Populate template widget state from a saved scenario's parameters."""
    values = template_values(template_name)
    if template_name == "Individual Parameter Scenario":
        label, value = params_to_individual_driver(params)
        values["individual_parameter"], values["individual_value"] = label, value
        return values
    if template_name == "Base Case":
        return values
    allowed = set(SCENARIO_TEMPLATES[template_name].get("visible", []))
    for key in PARAM_SPECS:
        if key in allowed and key in params:
            values[key] = float(params[key])
    return values


# ----------------------------------------------------------------------------
# Local AI (Ollama / Phi-3) explanation and natural-language parser
# ----------------------------------------------------------------------------
def ollama_status(host, model):
    """Check if the local Ollama server is running and the target model is installed."""
    try:
        import requests
        r = requests.get(host.rstrip("/") + "/api/tags", timeout=3)
        r.raise_for_status()
        names = [m.get("name", "") for m in r.json().get("models", [])]
        installed = any(model.lower() in n.lower() or n.lower().startswith(model.lower() + ":") for n in names)
        return True, installed, names
    except Exception as exc:
        return False, False, str(exc)



# def ask_ollama(host, model, payload):
#     try:
#         import requests
#         running, installed, info = ollama_status(host, model)
#         if not running:
#             return f"Phi-3 unavailable: Ollama is not running at {host}. Start Ollama and try again."
#         if not installed:
#             return f"Phi-3 unavailable: model '{model}' is not installed. Available: {', '.join(info) if isinstance(info, list) else info}"
#         prompt = (
#             "You are a Senior Asset and Liability Management (ALM) Analyst at the National Bank of Egypt (NBE). "
#             "Your task is to provide an executive-level analysis of the supplied deterministic ALM simulation results. "
#             "STRICT CONSTRAINTS:\n"
#             "- Do NOT change, invent, or calculate new numerical values. Python is the sole source of numbers.\n"
#             "- Base your entire analysis strictly on the provided JSON payload.\n\n"
#             "Please structure your response into the following clear sections:\n"
#             "1. Executive Summary: A concise overview of the scenario's overall effect.\n"
#             "2. Key Financial Drivers: Analyze the main drivers impacting NII, NIM, Cost of Funds, and P&L.\n"
#             "3. Risk & Liquidity Impact: Discuss changes in LCR and overall risk posture.\n"
#             "4. Strategic ALCO Recommendations: Provide actionable, professional recommendations for the ALCO committee based strictly on these results.\n\n"
#             + json.dumps(payload, ensure_ascii=False, indent=2, default=str)
#         )
#         response = requests.post(
#             host.rstrip("/") + "/api/generate",
#             json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": OLLAMA_TEMPERATURE}},
#             timeout=OLLAMA_TIMEOUT,
#         )
#         response.raise_for_status()
#         return response.json().get("response", "No response.")
#     except Exception as exc:
#         return f"Phi-3 request failed: {exc}"


def _active_ai_settings() -> dict:
    """Resolve current AI provider & credentials dynamically from Streamlit session_state or config.py."""
    from config import (
        AI_PROVIDER,
        GEMINI_API_KEY,
        GEMINI_API_URL,
        GEMINI_MODEL,
        GEMINI_TEMPERATURE,
        GEMINI_TIMEOUT,
        OLLAMA_HOST,
        OLLAMA_MODEL,
        OLLAMA_TEMPERATURE,
        OLLAMA_TIMEOUT,
    )

    provider = str(AI_PROVIDER).lower()
    gem_key = GEMINI_API_KEY
    gem_model = GEMINI_MODEL
    ol_host = OLLAMA_HOST
    ol_model = OLLAMA_MODEL

    try:
        import streamlit as st
        provider = str(st.session_state.get("ai_provider", provider)).lower().strip()
        gem_key = str(st.session_state.get("gemini_api_key", gem_key)).strip()
        gem_model = str(st.session_state.get("gemini_model", gem_model)).strip()
        ol_host = str(st.session_state.get("ollama_host", ol_host)).strip()
        ol_model = str(st.session_state.get("ollama_model", ol_model)).strip()
    except Exception:
        pass

    return {
        "provider": provider,
        "gemini_api_key": gem_key,
        "gemini_model": gem_model,
        "gemini_url": GEMINI_API_URL,
        "gemini_timeout": GEMINI_TIMEOUT,
        "gemini_temperature": GEMINI_TEMPERATURE,
        "ollama_host": ol_host,
        "ollama_model": ol_model,
        "ollama_timeout": OLLAMA_TIMEOUT,
        "ollama_temperature": OLLAMA_TEMPERATURE,
    }


def _call_gemini_rest(prompt_text: str, cfg: dict, json_mode: bool = False) -> str:
    """Internal REST client for Google Gemini API (zero external SDK dependency)."""
    key = cfg["gemini_api_key"]
    if not key:
        return "AI unavailable: Gemini API Key is not set. Configure it in Data & Settings -> AI Configurations."

    base_url = str(cfg["gemini_url"]).rstrip("/")
    model_name = cfg["gemini_model"]
    endpoint = f"{base_url}/{model_name}:generateContent?key={key}"

    gen_cfg = {"temperature": cfg["gemini_temperature"]}
    if json_mode:
        gen_cfg["responseMimeType"] = "application/json"

    resp = requests.post(
        endpoint,
        json={"contents": [{"parts": [{"text": prompt_text}]}], "generationConfig": gen_cfg},
        timeout=cfg["gemini_timeout"],
    )
    if resp.status_code != 200:
        return f"AI request failed (Gemini {resp.status_code}): {resp.text[:250]}"

    data = resp.json()
    try:
        return str(data["candidates"][0]["content"]["parts"][0]["text"]).strip()
    except Exception:
        return "AI unavailable: Empty response returned by Gemini."


def ask_ollama(host, model, payload):
    """
    Unified AI explanation entrypoint (signature unchanged for full backward compatibility).
    Automatically dispatches to Online (Gemini) or Offline (Ollama) based on AI Configurations.
    """
    try:
        cfg = _active_ai_settings()
        target_host = host or cfg["ollama_host"]
        target_model = model or cfg["ollama_model"]

        def _round_obj(obj):
            if isinstance(obj, float):
                return round(obj, 2)
            if isinstance(obj, dict):
                return {k: _round_obj(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_round_obj(v) for v in obj]
            return obj

        clean_payload = _round_obj(payload)

        prompt = (
            "You are a Senior Asset and Liability Management (ALM) Analyst at the National Bank of Egypt (NBE). "
            "Your task is to provide a fresh, executive-level analysis of the supplied deterministic ALM simulation results. "
            "STRICT CONSTRAINTS:\n"
            "- Do NOT change, invent, or calculate new numerical values. Python is the sole source of numbers.\n"
            "- Format all numbers cleanly to at most 2 decimal places (never output long decimals).\n"
            "- Base your entire analysis strictly on the provided JSON payload for THIS specific scenario.\n\n"
            "Please structure your response into the following clear sections:\n"
            "Executive Summary: A concise overview of this scenario's overall effect.\n"
            "Key Financial Drivers:\n"
            "- Analyze the main drivers impacting NII, NIM, Cost of Funds, and P&L.\n"
            "Risk and Liquidity Impact:\n"
            "- Discuss changes in LCR Proxy, Repricing Gap, and overall risk posture.\n"
            "Strategic Recommendations:\n"
            "1. Provide numbered, actionable recommendations for the ALCO committee based strictly on these results.\n"
            "Overall Risk: State the overall risk rating and concluding summary.\n\n"
            + json.dumps(clean_payload, ensure_ascii=False, indent=2, default=str)
        )

        # Online Mode (Gemini)
        if cfg["provider"] == "online":
            return _call_gemini_rest(prompt, cfg, json_mode=False)

        # Offline Mode (Ollama)
        running, installed, info = ollama_status(target_host, target_model)
        if not running:
            return f"Phi-3 unavailable: Ollama is not running at {target_host}. Start Ollama or switch to Online mode in Settings."
        if not installed:
            return f"Phi-3 unavailable: model '{target_model}' is not installed. Available: {', '.join(info) if isinstance(info, list) else info}"

        response = requests.post(
            target_host.rstrip("/") + "/api/generate",
            json={
                "model": target_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": cfg["ollama_temperature"], "num_ctx": 4096},
            },
            timeout=cfg["ollama_timeout"],
        )
        response.raise_for_status()
        return str(response.json().get("response", "No response.")).strip()
    except Exception as exc:
        return f"AI request failed: {exc}"



def clean_numeric_value(val, key_name: str = ""):
    """Extract a clean float from a raw string or number, preserving bps for spread shocks."""
    if val is None or val == "":
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().replace(",", "").lower()
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    if not m:
        return None
    f = float(m.group(0))
    if "bps" in s and key_name != "wsf_spread_shock_bps":
        return f / 100.0
    return f


_NUM = r"([+-]?\d+(?:\.\d+)?)\s*(%|pp|bps|bp|نقطة أساس|نقطة)?"
_DECREASE_WORDS = ["decrease", "reduction", "reduce", "drop", "fall", "decline", "cut", "lower", "انخفاض", "تراجع", "خفض", "سحب"]
_REGEX_PATTERNS = [
    ("deposit_runoff_pct", r"(?:run[- ]?off|outflows?|withdraw\w*|سحوبات|سحب)\D{0,25}?" + _NUM),
    ("wsf_spread_shock_bps", r"(?:wholesale|wsf|foreign borrowing|التمويل الخارجي)\D{0,25}?" + _NUM),
    ("fx_shock_pct", r"(?:\bfx\b|dollar|usd|currency|exchange rate|الدولار|العملة)\D{0,25}?" + _NUM),
    ("gold_shock_pct", r"(?:gold|ذهب|الذهب)\D{0,25}?" + _NUM),
    ("inflation_shock_pct", r"(?:inflation|تضخم|التضخم)\D{0,25}?" + _NUM),
    ("unemployment_shock_pct", r"(?:unemployment|بطالة|البطالة)\D{0,25}?" + _NUM),
    ("gdp_shock_pct", r"(?:\bgdp\b|economic growth|نمو)\D{0,25}?" + _NUM),
    ("deposit_growth_pct", r"(?:deposit\w*|ودائع|الودائع)\D{0,25}?" + _NUM),
    ("loan_growth_pct", r"(?:loan\w*|قروض|القروض)\D{0,25}?" + _NUM),
    ("rate_change_pp", r"(?:rate|rates|interest|policy|فائدة|فائده)\D{0,25}?" + _NUM),
]


def extract_parameters_regex(text_val: str) -> Dict[str, float]:
    """Fallback regex parser to extract scenario shocks from English or Arabic text."""
    res: Dict[str, float] = {}
    work = str(text_val or "").lower()
    for key, pat in _REGEX_PATTERNS:
        m = re.search(pat, work, re.I)
        if not m:
            continue
        v = float(m.group(1))
        unit = (m.group(2) or "").lower()
        if key != "wsf_spread_shock_bps" and (unit in {"bps", "bp"} or "نقطة" in unit):
            v /= 100.0
        clause = re.split(r"[,;.]| and | while | but | و", work[:m.start()])[-1] + work[m.start():m.end()]
        if key == "deposit_runoff_pct":
            v = abs(v)
        elif v > 0 and any(w in clause for w in _DECREASE_WORDS):
            v = -v
        res[key] = v
        work = work[:m.start()] + " " * (m.end() - m.start()) + work[m.end():]
    return res


def parse_free_type_scenario_with_ollama(host, model, scenario_text):
    """
    Unified natural-language scenario parser (signature unchanged).
    Dispatches automatically to Online (Gemini JSON mode) or Offline (Ollama) based on AI Configurations,
    with deterministic regex fallback for any missing parameters.
    """
    text_value = str(scenario_text or "").strip()
    if not text_value:
        raise ValueError("Scenario text is empty.")

    allowed = list(PARAM_SPECS.keys())
    result = {k: None for k in allowed}
    parser_source = "regex_fallback"

    units = ", ".join(f"{k} ({PARAM_SPECS[k][1]})" for k in allowed)
    prompt = (
        "Extract explicit ALM scenario assumptions from the text. Return ONLY a valid JSON object with numeric values (or null if not mentioned). "
        "Keys and units: " + units + ". "
        "Note: A policy/interest rate shock of +300 bps or +3% means rate_change_pp = 3.0; "
        "whereas wsf_spread_shock_bps is in basis points (e.g. +200 bps = 200.0).\n"
        "Scenario text: " + text_value
    )

    try:
        cfg = _active_ai_settings()
        target_host = host or cfg["ollama_host"]
        target_model = model or cfg["ollama_model"]
        raw_json_str = ""

        # Route 1: Online Mode (Google Gemini API with strict JSON response)
        if cfg["provider"] == "online":
            raw_json_str = _call_gemini_rest(prompt, cfg, json_mode=True)
            if not raw_json_str.startswith(("AI unavailable", "AI request failed")):
                parser_source = f"gemini ({cfg['gemini_model']})"
            else:
                raw_json_str = ""

        # Route 2: Offline Mode (Local Ollama)
        else:
            running, installed, _ = ollama_status(target_host, target_model)
            if running and installed:
                r = requests.post(
                    target_host.rstrip("/") + "/api/generate",
                    json={
                        "model": target_model,
                        "prompt": prompt,
                        "stream": False,
                        "format": "json",
                        "options": {"temperature": 0},
                    },
                    timeout=cfg["ollama_timeout"],
                )
                r.raise_for_status()
                raw_json_str = str(r.json().get("response", "")).strip()
                parser_source = f"ollama ({target_model})"

        # Clean markdown fences if any and parse JSON safely
        if raw_json_str:
            clean_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_json_str, flags=re.IGNORECASE).strip()
            parsed = json.loads(clean_json)
            if isinstance(parsed, dict):
                for k in allowed:
                    v = clean_numeric_value(parsed.get(k), key_name=k)
                    if v is not None:
                        result[k] = v
    except Exception:
        parser_source = "regex_fallback"

    # Deterministic regex safety net for any unparsed fields
    for k, v in extract_parameters_regex(text_value).items():
        if k in result and result.get(k) is None:
            result[k] = v

    return {
        "scenario_name": "Free Type Scenario",
        "parameters": result,
        "parser_source": parser_source,
    }