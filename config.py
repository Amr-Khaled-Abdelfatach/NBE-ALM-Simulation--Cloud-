# config.py
"""Global configuration, database settings, and scenario parameters for NBE ALM."""
import os

APP_NAME = "NBE ALM Simulation Center"
APP_VERSION = "1.0.0"
CALCULATION_ENGINE_VERSION = 1

# ----------------------------------------------------------------------------
# Core financial model conventions
# ----------------------------------------------------------------------------
MODEL_AMOUNT_UNIT = "EGP"
RATE_STORAGE_CONVENTION = "decimal_fraction"   # e.g., 0.2000 represents 20.00%
DAYS_PER_YEAR = 365.0
DEFAULT_HORIZON_DAYS = 365
DEFAULT_CURRENCY = "EGP"
DATABASE_NAME = "nbe_alm_simulation"
RATE_FLOOR = 0.0

# ----------------------------------------------------------------------------
# MySQL database connection settings
# ----------------------------------------------------------------------------
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_NAME = os.getenv("DB_NAME", DATABASE_NAME)
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_DRIVER = os.getenv("DB_DRIVER", "mysql+pymysql")
DB_POOL_PRE_PING = True
DB_POOL_RECYCLE = 1800
DB_READ_LIMIT = 100000

REQUIRED_TABLES = [
    "products",
    "positions",
    "cash_flows",
    "market_data",
    "financial_statements_flat",
    "scenarios",
    "simulation_runs",
    "simulation_results",
]
FLAT_POSITIONS_VIEW = "vw_alm_flat_table"
FLAT_CASHFLOWS_VIEW = "vw_alm_cashflows_flat"
UNIFIED_BS_VIEW = "vw_nbe_balance_sheet_unified"
OPTIONAL_VIEWS = [FLAT_POSITIONS_VIEW, FLAT_CASHFLOWS_VIEW, UNIFIED_BS_VIEW]

# ----------------------------------------------------------------------------
# Market valuation and macroeconomic instruments (mapped to `market_data`)
# ----------------------------------------------------------------------------
VALUATION_INSTRUMENTS = [
    "FX_USD_EGP",
    "FX_EUR_EGP",
    "FX_GBP_EGP",
    "FX_SAR_EGP",
    "GOLD_21K_EGP",
    "GOLD_24K_EGP",
]
MACRO_CODES = {
    "policy_rate": "CBE_POLICY_RATE",
    "inflation": "MACRO_INFLATION",
    "unemployment": "MACRO_UNEMPLOYMENT",
    "gdp_growth": "MACRO_GDP_GROWTH",
    "effective_tax_rate": "EFFECTIVE_TAX_RATE",
}
FUNDING_SEGMENTS = ["Treasury"]
EXCLUDE_MATURED_POSITIONS = True
ROLLOVER_AT_MATURITY = True

# Credit cost sensitivity: basis points of loans per 1 pp adverse macro shock
CREDIT_COST_BPS_PER_UNEMP_PP = 25.0
CREDIT_COST_BPS_PER_GDP_DROP_PP = 20.0

# ----------------------------------------------------------------------------
# Repricing frequency and tenor mapping (in days)
# ----------------------------------------------------------------------------
TENOR_DAY_FACTORS = {"D": 1.0, "W": 7.0, "M": 30.4375, "Y": 365.0}
FREQUENCY_WORD_DAYS = {
    "DAILY": 1,
    "DAILY SPOT": 1,
    "WEEKLY": 7,
    "MONTHLY": 30,
    "QUARTERLY": 91,
    "SEMI-ANNUAL": 182,
    "SEMI ANNUAL": 182,
    "SEMIANNUAL": 182,
    "ANNUAL": 365,
    "ANNUALLY": 365,
    "YEARLY": 365,
    "AT MATURITY": None,
    "SPOT": None,
    "OPEN-ENDED": None,
    "NONE": None,
}

# ----------------------------------------------------------------------------
# Liquidity & LCR Proxy settings
# ----------------------------------------------------------------------------
LCR_WINDOW_DAYS = 30
LCR_INFLOW_CAP = 0.75
LCR_DEPOSIT_STRESS_FLOOR_PCT = 0.15
MATURITY_BUCKET_ORDER = [
    "0-7 Days",
    "8-30 Days",
    "31-90 Days",
    "91-180 Days",
    "181-365 Days",
    "Over 1 Year",
]

# ----------------------------------------------------------------------------
# Scenario parameters (1-to-1 match with the 10 columns in `scenarios` table)
# Format: key -> (label, unit, min, max, default, step, group)
# ----------------------------------------------------------------------------
PARAM_SPECS = {
    # Market shocks
    "rate_change_pp":          ("Policy / market rate shock", "pp", -1000.0, 1000.0, 0.0, 0.25, "Market"),
    "fx_shock_pct":            ("FX shock (USD/EGP & FCY)", "%", -500.0, 500.0, 0.0, 1.0, "Market"),
    "gold_shock_pct":          ("Gold price shock", "%", -500.0, 500.0, 0.0, 1.0, "Market"),
    "wsf_spread_shock_bps":    ("FX wholesale funding spread shock", "bps", -500.0, 1000.0, 0.0, 25.0, "Market"),
    # Balance sheet & liquidity
    "deposit_runoff_pct":      ("Deposit run-off (outflow)", "%", 0.0, 100.0, 0.0, 1.0, "Balance sheet & liquidity"),
    "loan_growth_pct":         ("Loan growth", "%", -100.0, 100.0, 0.0, 1.0, "Balance sheet & liquidity"),
    "deposit_growth_pct":      ("Deposit growth", "%", -100.0, 100.0, 0.0, 1.0, "Balance sheet & liquidity"),
    # Macroeconomic shocks
    "inflation_shock_pct":     ("Inflation shock", "pp", -200.0, 200.0, 0.0, 0.5, "Macro"),
    "unemployment_shock_pct":  ("Unemployment shock", "pp", -100.0, 100.0, 0.0, 0.5, "Macro"),
    "gdp_shock_pct":           ("GDP growth shock", "pp", -200.0, 200.0, 0.0, 0.5, "Macro"),
}

SCENARIO_DEFAULTS = {k: v[4] for k, v in PARAM_SPECS.items()}
SCENARIO_LIMITS = {k: (v[2], v[3]) for k, v in PARAM_SPECS.items()}
PARAMETER_DEFAULTS = SCENARIO_DEFAULTS

INDIVIDUAL_DRIVERS = {
    "Policy Rate Shock": "rate_change_pp",
    "FX Shock (USD/EGP)": "fx_shock_pct",
    "Gold Price Shock": "gold_shock_pct",
    "FX Wholesale Spread Shock": "wsf_spread_shock_bps",
    "Deposit Run-off": "deposit_runoff_pct",
    "Loan Growth": "loan_growth_pct",
    "Deposit Growth": "deposit_growth_pct",
    "Inflation Shock": "inflation_shock_pct",
    "Unemployment Shock": "unemployment_shock_pct",
    "GDP Growth Shock": "gdp_shock_pct",
}

SCENARIO_TEMPLATES = {
    "Base Case": {
        "scenario_type": "BASE",
        "description": "Baseline scenario with no stress assumptions.",
        "visible": [],
        "values": {},
    },
    "Combined Stress Scenario": {
        "scenario_type": "COMBINED_STRESS",
        "description": "Combined multi-factor stress testing across all 10 database scenario drivers.",
        "visible": list(PARAM_SPECS.keys()),
        "values": {k: 0.0 for k in PARAM_SPECS},
    },
    "Individual Parameter Scenario": {
        "scenario_type": "INDIVIDUAL_PARAMETER",
        "description": "Choose ONE primary scenario driver and set its value while keeping all other assumptions neutral.",
        "visible": ["individual_parameter", "individual_value"],
        "values": {},
        "individual_parameter": "Policy Rate Shock",
        "individual_value": 0.0,
    },
    "Free Type Scenario": {
        "scenario_type": "FREE_TYPE",
        "description": "Describe the scenario in natural language.",
        "visible": ["text_prompt"],
        "values": {},
    },
}
PREDEFINED_TEMPLATE_NAMES = list(SCENARIO_TEMPLATES.keys())

# ----------------------------------------------------------------------------
# Navigation and UI layout
# ----------------------------------------------------------------------------
PAGES = {
    "Dashboard": "dashboard",
    "Scenarios": "scenarios",
    "Documentation": "documentation",
    "Live Market Feed": "market_feed",
    "Data & Settings": "system",
}

PAGE_ICONS = {
    "Dashboard": "",
    "Scenarios": "",
    "Documentation": "",
    "Data & Settings": "",
}

HORIZON_PRESETS = {
    "30 days": 30,
    "90 days (quarter)": 90,
    "180 days": 180,
    "365 days (1 year - standard)": 365,
    "730 days (2 years)": 730,
}

# Risk rating thresholds
RISK_NII_HIGH_DROP_PCT = -10.0
RISK_NII_MEDIUM_DROP_PCT = -3.0
RISK_COF_MEDIUM_RISE_BPS = 50.0
RISK_LCR_MEDIUM_RELATIVE_DROP_PCT = -25.0

# Local AI (Ollama / Phi-3) settings
AI_ENABLED = os.getenv("AI_ENABLED", "true").lower() == "true"
AI_EXPLANATION_ONLY = True
AI_PROVIDER = os.getenv("AI_PROVIDER", "online").lower()

# Offline AI (Local Ollama)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "600"))
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.0"))

# Online AI (Google Gemini API)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AQ.Ab8RN6Knxx5FtyHZ_ndMVv5VDsAoyDFCVoLs7A1OzlqTxMF7PQ")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_API_URL = os.getenv("GEMINI_API_URL", "https://generativelanguage.googleapis.com/v1beta/models")
GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "600"))
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.0"))

# Audit & validation thresholds
CALCULATION_TOLERANCE = float(os.getenv("CALCULATION_TOLERANCE", "1e-6"))
RATE_WARNING_THRESHOLD = float(os.getenv("RATE_WARNING_THRESHOLD", "2.0"))
LCR_WARNING_THRESHOLD = float(os.getenv("LCR_WARNING_THRESHOLD", "100.0"))
MARKET_DATA_STALE_DAYS = int(os.getenv("MARKET_DATA_STALE_DAYS", "7"))

# Visual theme & styling
PAGE_TITLE = APP_NAME
PAGE_ICON = "🏦"
PAGE_LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "auto"
CHART_HEIGHT = 400
CHART_MARGIN = dict(l=20, r=20, t=50, b=40)
CHART_FONT = "Arial"
UI_BORDER_RADIUS = 14
UI_BUTTON_HEIGHT = 42
DEFAULT_TABLE_HEIGHT = 420
DATA_CACHE_TTL_SECONDS = 120


# Brand Assets & Logos (Configurable via environment variables or default filenames)
LOGO_SIDEBAR_PATH = os.getenv("LOGO_SIDEBAR_PATH", "nbe_logo_only.png")
LOGO_HEADER_WHITE_PATH = os.getenv("LOGO_HEADER_WHITE_PATH", "nbe_logo_w.png")
LOGO_REPORT_COLOR_PATH = os.getenv("LOGO_REPORT_COLOR_PATH", "nbe_logo.png")


NBE_GREEN = "#007A3D"
NBE_DARK_GREEN = "#005B2D"
NBE_ORANGE = "#F7941D"
NBE_LIGHT_GREEN = "#EAF5EF"
NBE_PALE_ORANGE = "#FFF3E0"
NBE_TEXT = "#173B2A"
NBE_MUTED = "#66786D"
NBE_PALETTE = [NBE_GREEN, NBE_ORANGE, NBE_DARK_GREEN, "#2E8B57", "#F6A64B", "#73B88D"]

# External documentation links
DOCS_ENABLED = True
DOCS_BASE_URL = "https://ahmed01hashem.github.io/nbe-alm-simulation/"

PAGE_DOCS = {
    "Dashboard": "",
    "Scenarios": "scenarios/",
    "Documentation": "",
    "Data & Settings": "database/",
}

DOCS_LINKS = {
    "Market-rate assumption": "scenarios/#market-rate-assumption",
    "Scenario assumptions": "scenarios/#scenario-assumptions",
    "Live calculation preview": "scenarios/#live-calculation-preview",
    "ALM Analytics": "analytics/",
    "Calculation Audit": "calculation-engine/",
}