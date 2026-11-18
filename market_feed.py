# market_feed.py
"""
Live Market Data Feeder: Fetch live FX rates, 24K & 21K Gold (EGP/gram), and Macro indicators
from online APIs and push them directly into MySQL `market_data`.
100% Data-Driven: Initial/fallback values are read dynamically from the database (`market_data`).
"""
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import requests
import streamlit as st
from sqlalchemy import text

from config import MACRO_CODES, VALUATION_INSTRUMENTS
from calculations import macro_snapshot, valuation_rates
from ui_common import show_df

TROY_OUNCE_GRAMS = 31.1034768


def _resolve_instrument_mapping(existing_md: pd.DataFrame) -> Dict[str, dict]:
    """
    Dynamically discover the instrument_code and metadata structure used in `market_data`
    for FX currencies, 24K & 21K Gold, and Macroeconomic indicators.
    """
    mapping = {}

    # 1. Standard instrument code structure aligned with `vw_alm_flat_table`
    fx_gold_specs = {
        "USD": ("FX_USD_EGP", "FX", "USD", "EGP", "EGP"),
        "EUR": ("FX_EUR_EGP", "FX", "EUR", "EGP", "EGP"),
        "GBP": ("FX_GBP_EGP", "FX", "GBP", "EGP", "EGP"),
        "SAR": ("FX_SAR_EGP", "FX", "SAR", "EGP", "EGP"),
        "AED": ("FX_AED_EGP", "FX", "AED", "EGP", "EGP"),
        "KWD": ("FX_KWD_EGP", "FX", "KWD", "EGP", "EGP"),
        "GOLD": ("GOLD_24K_EGP", "GOLD", "XAU", "EGP", "EGP/Gram"),
        "GOLD_21K": ("GOLD_21K_EGP", "GOLD", "XAU", "EGP", "EGP/Gram"),
    }
    for key, (code, itype, b_ccy, q_ccy, unit) in fx_gold_specs.items():
        mapping[key] = {
            "instrument_code": code,
            "data_type": "Commodity" if itype == "GOLD" else "FX",
            "instrument_type": itype,
            "base_currency": b_ccy,
            "quote_currency": q_ccy,
            "unit": unit,
            "source_name": "Live Market API",
        }

    # Override metadata from existing `market_data` rows if present in the database
    if existing_md is not None and not existing_md.empty and "instrument_code" in existing_md.columns:
        val_rows = existing_md[existing_md["instrument_code"].isin(VALUATION_INSTRUMENTS)]
        if val_rows.empty:
            val_rows = existing_md
        for _, r in val_rows.drop_duplicates(subset=["instrument_code"], keep="last").iterrows():
            code = str(r.get("instrument_code", "")).strip()
            code_u = code.upper()
            b_ccy = str(r.get("base_currency", "")).strip().upper()

            if code_u == "GOLD_21K_EGP" or "21K" in code_u:
                key = "GOLD_21K"
            elif code_u == "GOLD_24K_EGP" or "GOLD" in code_u or b_ccy in ("GOLD", "XAU"):
                key = "GOLD"
            elif b_ccy in mapping:
                key = b_ccy
            elif code_u.startswith("FX_") and "_" in code_u:
                key = code_u.split("_")[1]
            else:
                continue

            if key in mapping:
                mapping[key].update({
                    "instrument_code": code,
                    "data_type": r.get("data_type", mapping[key]["data_type"]),
                    "instrument_type": r.get("instrument_type", mapping[key]["instrument_type"]),
                    "base_currency": r.get("base_currency", mapping[key]["base_currency"]),
                    "quote_currency": r.get("quote_currency", "EGP"),
                    "unit": r.get("unit", mapping[key]["unit"]),
                })

    # 2. Macroeconomic indicators dynamically mapped from `MACRO_CODES` in config.py
    macro_meta = {
        "policy_rate": ("INTEREST_RATE", "%", "CBE / Live Feed"),
        "inflation": ("MACRO", "%", "World Bank / CAPMAS"),
        "unemployment": ("MACRO", "%", "World Bank / CAPMAS"),
        "gdp_growth": ("MACRO", "%", "World Bank / MoP"),
        "effective_tax_rate": ("MACRO", "%", "NBE Financial Statements"),
    }
    for m_key, code in (MACRO_CODES.items() if isinstance(MACRO_CODES, dict) else []):
        m_type, m_unit, m_src = macro_meta.get(m_key, ("MACRO", "%", "Live Feed"))
        mapping[m_key] = {
            "instrument_code": code,
            "data_type": "Macroeconomic",
            "instrument_type": m_type,
            "base_currency": "EGP",
            "quote_currency": "EGP",
            "unit": m_unit,
            "source_name": m_src,
        }

    return mapping


def _extract_db_baseline(existing_md: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Read the latest FX, Gold (24K & 21K), and Macro values dynamically from `market_data`
    so the UI and API fallbacks never rely on hardcoded numbers.
    """
    vr_df = valuation_rates(existing_md)
    inst_map = (
        dict(zip(vr_df["instrument_code"].astype(str).str.upper(), vr_df["market_rate"]))
        if not vr_df.empty and "instrument_code" in vr_df.columns
        else {}
    )
    ccy_map = (
        dict(zip(vr_df["base_currency"].astype(str).str.upper(), vr_df["market_rate"]))
        if not vr_df.empty
        else {}
    )

    usd_db = float(inst_map.get("FX_USD_EGP", ccy_map.get("USD", 0.0)))
    g24_db = float(inst_map.get("GOLD_24K_EGP", ccy_map.get("GOLD", 0.0)))
    g21_db = float(inst_map.get("GOLD_21K_EGP", ccy_map.get("GOLD_21K", round(g24_db * (21.0 / 24.0), 2) if g24_db > 0 else 0.0)))

    fx_gold_db = {
        "USD": usd_db,
        "EUR": float(inst_map.get("FX_EUR_EGP", ccy_map.get("EUR", 0.0))),
        "GBP": float(inst_map.get("FX_GBP_EGP", ccy_map.get("GBP", 0.0))),
        "SAR": float(inst_map.get("FX_SAR_EGP", ccy_map.get("SAR", 0.0))),
        "AED": float(inst_map.get("FX_AED_EGP", ccy_map.get("AED", 0.0))),
        "KWD": float(inst_map.get("FX_KWD_EGP", ccy_map.get("KWD", 0.0))),
        "GOLD": g24_db,
        "GOLD_21K": g21_db,
    }

    mac_snap = macro_snapshot(existing_md)
    macro_db = {}
    for k in MACRO_CODES:
        raw_v = mac_snap.get(k, np.nan)
        macro_db[k] = round(float(raw_v) * 100.0, 2) if pd.notna(raw_v) else 0.0

    return fx_gold_db, macro_db


def fetch_live_fx_and_gold(existing_md: pd.DataFrame = None, timeout: int = 8) -> Tuple[Dict[str, float], List[str]]:
    """
    Fetch live exchange rates against EGP and international Gold price (XAU/USD),
    converting Gold into both 24K and 21K EGP per gram. Falls back to database values if offline.
    """
    db_fx_gold, _ = _extract_db_baseline(existing_md)
    rates = dict(db_fx_gold)
    logs = []

    # 1. Fetch live FX rates against EGP
    try:
        resp = requests.get("https://open.er-api.com/v6/latest/USD", timeout=timeout)
        resp.raise_for_status()
        data = resp.json().get("rates", {})
        if "EGP" in data and float(data["EGP"]) > 0:
            usd_egp = float(data["EGP"])
            rates["USD"] = round(usd_egp, 4)
            for ccy in ["EUR", "GBP", "SAR", "AED", "KWD"]:
                if ccy in data and float(data[ccy]) > 0:
                    rates[ccy] = round(usd_egp / float(data[ccy]), 4)
            logs.append(f"✅ Live FX rates fetched successfully (1 USD = {rates['USD']:,.4f} EGP).")
    except Exception as exc:
        logs.append(f"⚠️ Live FX API unreachable ({exc}); retained latest database FX rates.")

    # 2. Fetch live Gold (XAU/USD) and compute both 24K and 21K EGP per gram
    gold_oz_usd = None
    gold_endpoints = [
        ("https://api.gold-api.com/price/XAU", lambda j: float(j["price"])),
        ("https://data-asg.goldprice.org/dbXRates/USD", lambda j: float(j["items"][0]["xauPrice"])),
    ]
    for url, parser in gold_endpoints:
        try:
            r_g = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            if r_g.status_code == 200:
                gold_oz_usd = parser(r_g.json())
                if gold_oz_usd and gold_oz_usd > 0:
                    break
        except Exception:
            continue

    if gold_oz_usd and rates.get("USD", 0.0) > 0:
        gram_24k_egp = (gold_oz_usd / TROY_OUNCE_GRAMS) * rates["USD"]
        gram_21k_egp = gram_24k_egp * (21.0 / 24.0)
        rates["GOLD"] = round(gram_24k_egp, 2)
        rates["GOLD_21K"] = round(gram_21k_egp, 2)
        logs.append(
            f"✅ Live Gold fetched: {gold_oz_usd:,.1f} USD/oz ➔ "
            f"24K = {rates['GOLD']:,.2f} EGP/g | 21K = {rates['GOLD_21K']:,.2f} EGP/g."
        )
    else:
        logs.append("ℹ️ Retained latest database Gold prices (editable below).")

    return rates, logs


def fetch_live_macro_indicators(existing_md: pd.DataFrame, timeout: int = 6) -> Tuple[Dict[str, float], List[str]]:
    """
    Sync latest macroeconomic indicators from World Bank API for Egypt (EGY),
    starting from the latest values stored in `market_data`.
    """
    _, macro_vals = _extract_db_baseline(existing_md)
    logs = []

    wb_indicators = {
        "inflation": "FP.CPI.TOTL.ZG",
        "unemployment": "SL.UEM.TOTL.ZS",
        "gdp_growth": "NY.GDP.MKTP.KD.ZG",
    }
    fetched_wb = []
    for key, ind_code in wb_indicators.items():
        try:
            url = f"https://api.worldbank.org/v2/country/EGY/indicator/{ind_code}?format=json&per_page=5"
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                payload = r.json()
                if isinstance(payload, list) and len(payload) > 1 and payload[1]:
                    for entry in payload[1]:
                        if entry.get("value") is not None:
                            macro_vals[key] = round(float(entry["value"]), 2)
                            fetched_wb.append(f"{key}={macro_vals[key]}%")
                            break
        except Exception:
            continue

    if fetched_wb:
        logs.append(f"✅ Macro indicators synced from World Bank API ({', '.join(fetched_wb)}).")
    else:
        logs.append("ℹ️ Loaded latest Macroeconomic rates from database (`market_data`).")

    return macro_vals, logs


def insert_market_data_rows(engine, rows_to_insert: List[dict], existing_md: pd.DataFrame) -> Tuple[bool, str]:
    """
    Insert updated market records into MySQL `market_data`, dynamically matching the table's schema
    and rate storage convention (decimal fraction vs. percentage).
    """
    if engine is None:
        return False, "Database engine is not connected."

    try:
        with engine.begin() as conn:
            col_res = conn.execute(text("SHOW COLUMNS FROM market_data")).fetchall()
            db_cols = [str(r[0]) for r in col_res]

            macro_as_decimal = False
            if existing_md is not None and not existing_md.empty and "instrument_code" in existing_md.columns:
                m_codes = list(MACRO_CODES.values()) if isinstance(MACRO_CODES, dict) else []
                sub_m = existing_md[existing_md["instrument_code"].isin(m_codes)]
                if not sub_m.empty:
                    vals = pd.to_numeric(sub_m["value"], errors="coerce").dropna()
                    if not vals.empty and vals.abs().max() <= 1.5:
                        macro_as_decimal = True

            now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            inserted_count = 0

            for item in rows_to_insert:
                val = float(item["value"])
                if val <= 0 and item.get("category") in ("FX", "GOLD"):
                    continue
                if item.get("category") == "MACRO" and macro_as_decimal and abs(val) > 1.5:
                    val = val / 100.0

                candidate_data = {
                    "instrument_code": item["instrument_code"],
                    "instrument_name": item.get("instrument_name", item["instrument_code"]),
                    "data_type": item.get("data_type", "FX"),
                    "instrument_type": item.get("instrument_type", "FX"),
                    "base_currency": item.get("base_currency", "USD"),
                    "quote_currency": item.get("quote_currency", "EGP"),
                    "value": val,
                    "unit": item.get("unit", "EGP"),
                    "source_name": item.get("source_name", "Live Market Feed"),
                    "as_of_datetime": now_ts,
                }

                valid_payload = {k: v for k, v in candidate_data.items() if k in db_cols}
                cols_sql = ", ".join(f"`{k}`" for k in valid_payload.keys())
                params_sql = ", ".join(f":{k}" for k in valid_payload.keys())

                sql = text(f"INSERT INTO market_data ({cols_sql}) VALUES ({params_sql})")
                conn.execute(sql, valid_payload)
                inserted_count += 1

        return True, f"✅ Successfully inserted {inserted_count} live market records into `market_data` (Timestamp: {now_ts})."
    except Exception as exc:
        return False, f"❌ Failed to insert into `market_data`: {exc}"


def render_live_market_feed_panel(ctx):
    """Render the interactive Live Market Data Feeder page."""
    st.markdown(
        '<div class="section-header">🌐 Live Market Data Feeder (FX, 24K/21K Gold & Macro Indicators)</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Fetch real-time exchange rates, 24K & 21K gold prices per gram in EGP, and macroeconomic indicators "
        "from the internet and push them directly into MySQL `market_data`."
    )

    mapping = _resolve_instrument_mapping(ctx.market_data)
    db_fx_gold, db_macro = _extract_db_baseline(ctx.market_data)

    c_btn1, _ = st.columns([1, 2])
    if c_btn1.button("🔄 Fetch Latest Live Prices from Internet", type="primary", use_container_width=True):
        with st.spinner("Connecting to live FX, Gold & Macro APIs..."):
            fx_gold_live, logs1 = fetch_live_fx_and_gold(ctx.market_data)
            macro_live, logs2 = fetch_live_macro_indicators(ctx.market_data)
            st.session_state["live_fetched_fx_gold"] = fx_gold_live
            st.session_state["live_fetched_macro"] = macro_live
            st.session_state["live_fetch_logs"] = logs1 + logs2
            st.session_state["live_fetch_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for log_msg in st.session_state.get("live_fetch_logs", []):
        st.info(log_msg)

    fx_gold = st.session_state.get("live_fetched_fx_gold", db_fx_gold)
    macro_vals = st.session_state.get("live_fetched_macro", db_macro)

    with st.form("push_live_market_data_form"):
        st.markdown("##### 1️⃣ Live Foreign Exchange & Gold (24K / 21K) Prices (in EGP)")
        g1, g2, g3 = st.columns(3)
        usd_val = g1.number_input(
            f"💱 USD / EGP ({mapping['USD']['instrument_code']})",
            value=float(fx_gold.get("USD", 0.0)),
            step=0.05,
            format="%.4f",
        )
        gold24_val = g2.number_input(
            f"🪙 24K Gold Gram / EGP ({mapping['GOLD']['instrument_code']})",
            value=float(fx_gold.get("GOLD", 0.0)),
            step=5.0,
            format="%.2f",
        )
        gold21_default = float(fx_gold.get("GOLD_21K", 0.0)) or round(gold24_val * (21.0 / 24.0), 2)
        gold21_val = g3.number_input(
            f"🪙 21K Gold Gram / EGP ({mapping['GOLD_21K']['instrument_code']})",
            value=gold21_default,
            step=5.0,
            format="%.2f",
        )

        g4, g5, g6 = st.columns(3)
        eur_val = g4.number_input(
            f"💶 EUR / EGP ({mapping['EUR']['instrument_code']})",
            value=float(fx_gold.get("EUR", 0.0)),
            step=0.05,
            format="%.4f",
        )
        gbp_val = g5.number_input(
            f"💷 GBP / EGP ({mapping['GBP']['instrument_code']})",
            value=float(fx_gold.get("GBP", 0.0)),
            step=0.05,
            format="%.4f",
        )
        sar_val = g6.number_input(
            f"🇸🇦 SAR / EGP ({mapping['SAR']['instrument_code']})",
            value=float(fx_gold.get("SAR", 0.0)),
            step=0.01,
            format="%.4f",
        )

        st.markdown("##### 2️⃣ Macroeconomic & CBE Policy Indicators (%)")
        m1, m2, m3, m4 = st.columns(4)
        pol_val = m1.number_input(
            f"🏛️ CBE Policy Rate % ({mapping['policy_rate']['instrument_code']})",
            value=float(macro_vals.get("policy_rate", 0.0)),
            step=0.25,
            format="%.2f",
        )
        inf_val = m2.number_input(
            f"📈 Inflation Rate % ({mapping['inflation']['instrument_code']})",
            value=float(macro_vals.get("inflation", 0.0)),
            step=0.10,
            format="%.2f",
        )
        une_val = m3.number_input(
            f"👥 Unemployment % ({mapping['unemployment']['instrument_code']})",
            value=float(macro_vals.get("unemployment", 0.0)),
            step=0.10,
            format="%.2f",
        )
        gdp_val = m4.number_input(
            f"🏗️ GDP Growth % ({mapping['gdp_growth']['instrument_code']})",
            value=float(macro_vals.get("gdp_growth", 0.0)),
            step=0.10,
            format="%.2f",
        )

        inc_extra_fx = st.checkbox("Also insert EUR, GBP, SAR alongside USD & Gold (24K/21K)", value=True)
        submitted = st.form_submit_button("💾 Save & Push directly to `market_data` Table", use_container_width=True)

    if submitted:
        rows_payload = [
            {**mapping["USD"], "value": usd_val, "category": "FX"},
            {**mapping["GOLD"], "value": gold24_val, "category": "GOLD"},
            {**mapping["GOLD_21K"], "value": gold21_val, "category": "GOLD"},
            {**mapping["policy_rate"], "value": pol_val, "category": "MACRO"},
            {**mapping["inflation"], "value": inf_val, "category": "MACRO"},
            {**mapping["unemployment"], "value": une_val, "category": "MACRO"},
            {**mapping["gdp_growth"], "value": gdp_val, "category": "MACRO"},
        ]
        if inc_extra_fx:
            for ccy_key, ccy_v in [("EUR", eur_val), ("GBP", gbp_val), ("SAR", sar_val)]:
                if ccy_key in mapping and ccy_v > 0:
                    rows_payload.append({**mapping[ccy_key], "value": ccy_v, "category": "FX"})

        ok, msg = insert_market_data_rows(ctx.engine, rows_payload, ctx.market_data)
        if ok:
            st.success(msg)
            st.cache_data.clear()
            st.rerun()
        else:
            st.error(msg)

    st.markdown("##### 📋 Latest Records in `market_data` Table")
    if ctx.market_data is not None and not ctx.market_data.empty:
        df_sorted = (
            ctx.market_data.sort_values("market_data_id", ascending=False).head(15)
            if "market_data_id" in ctx.market_data.columns
            else ctx.market_data.tail(15)
        )
        show_df(df_sorted, height=280)