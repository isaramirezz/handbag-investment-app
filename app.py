import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd

# ── Load model & feature list ────────────────────────────────────────────────
with open('xgb_final.pkl', 'rb') as f:
    model = pickle.load(f)
with open('feature_columns.json', 'r') as f:
    feature_cols = json.load(f)



MODEL_OPTIONS = {
    "HERMES":        ["BIRKIN", "KELLY"],
    "CHANEL":        ["CLASSIC_FLAP", "2_55"],
    "DIOR":          ["LADY_DIOR", "SADDLE"],
    "LOUIS_VUITTON": ["NEVERFULL", "SPEEDY"],
}

# Materials keyed by (brand, model_family) — only combinations present in dataset
MATERIAL_OPTIONS = {
    ("CHANEL",        "CLASSIC_FLAP"): ["Caviar Leather", "Calfskin", "Lambskin", "Sheepskin",
                                        "Satin", "Patent Leather", "Denim", "Alligator", "Canvas"],
    ("CHANEL",        "2_55"):         ["Calfskin", "Lambskin", "Sheepskin",
                                        "Patent Leather", "Denim", "Canvas"],
    ("DIOR",          "LADY_DIOR"):    ["Calfskin", "Lambskin", "Patent Leather",
                                        "Denim", "Sheepskin", "Alligator", "Crocodile"],
    ("DIOR",          "SADDLE"):       ["Calfskin", "Canvas", "Denim", "Satin"],
    ("HERMES",        "BIRKIN"):       ["Togo Leather", "Epsom Leather", "Clemence Leather",
                                        "Swift Leather", "Calfskin", "Alligator", "Crocodile"],
    ("HERMES",        "KELLY"):        ["Togo Leather", "Epsom Leather", "Clemence Leather",
                                        "Swift Leather", "Alligator", "Crocodile"],
    ("LOUIS_VUITTON", "NEVERFULL"):    ["Canvas", "Calfskin"],
    ("LOUIS_VUITTON", "SPEEDY"):       ["Canvas"],
}

EXOTIC_MATERIALS = {"Crocodile", "Alligator", "Lizard"}

COLOR_OPTIONS = ["Black", "Beige", "Brown", "White", "Grey", "Blue", "Navy Blue",
                 "Indigo Blue", "Red", "Pink", "Orange", "Yellow", "Green", "Multicolor", "Other"]

HARDWARE_OPTIONS = ["Gold", "Silver", "Palladium", "Ruthenium", "Mixed"]

CONDITION_MAP = {
    "As Is": 1, "Fair": 2, "Good": 3,
    "Very Good": 4, "Excellent": 5, "Pristine": 6
}

MAIN_MATERIALS = [
    "Canvas", "Sheepskin", "Caviar Leather", "Togo Leather", "Epsom Leather",
    "Calfskin", "Denim", "Patent Leather", "Clemence Leather", "Alligator",
    "Satin", "Swift Leather", "Lambskin", "Crocodile"
]

def normalize_color(color):
    corrections = {
        "Bubblegum Pink": "Pink", "Fuchsia Pink": "Pink",
        "Rose Pink": "Pink", "Salmon Pink": "Pink",
        "Peach Orange": "Orange", "Mustard Yellow": "Yellow",
        "Chalk White": "White", "Light Beige": "Beige", "Grey Mist": "Grey",
    }
    return corrections.get(color, color)

def build_input(brand, model_family, material, color, hardware,
                condition_score, year, includes_box, includes_dustbag, includes_card):
    age      = 2026 - year
    age_sq   = age ** 2
    is_exotic = 1 if material in EXOTIC_MATERIALS else 0
    mat_clean = material if material in MAIN_MATERIALS else "Other"
    color_clean = normalize_color(color)

    inp = pd.DataFrame(0, index=[0], columns=feature_cols)

    for col, val in [
        ("age", age), ("age_sq", age_sq),
        ("condition_score", condition_score),
        ("is_exotic", is_exotic),
        ("includes_box", int(includes_box)),
        ("includes_dustbag", int(includes_dustbag)),
        ("includes_card_of_certificate", int(includes_card)),
        ("discount_pct", 0.0),
    ]:
        if col in inp.columns:
            inp[col] = val

    for b in ["DIOR", "HERMES", "LOUIS_VUITTON"]:
        col = f"brand_{b}"
        if col in inp.columns:
            inp[col] = 1 if brand == b else 0

    for m in ["BIRKIN", "CLASSIC_FLAP", "KELLY", "LADY_DIOR", "NEVERFULL", "SADDLE", "SPEEDY"]:
        col = f"model_family_{m}"
        if col in inp.columns:
            inp[col] = 1 if model_family == m else 0

    # Calfskin is the reference (dropped), all others get dummies
    for mat in ["Alligator", "Canvas", "Caviar Leather", "Clemence Leather",
                "Crocodile", "Denim", "Epsom Leather", "Lambskin", "Other",
                "Patent Leather", "Satin", "Sheepskin", "Swift Leather", "Togo Leather"]:
        col = f"material_group_{mat}"
        if col in inp.columns:
            inp[col] = 1 if mat_clean == mat else 0

    for c in ["Black", "Blue", "Brown", "Green", "Grey", "Indigo Blue",
              "Multicolor", "Navy Blue", "Orange", "Other", "Pink",
              "Red", "White", "Yellow"]:
        col = f"color_clean_{c}"
        if col in inp.columns:
            inp[col] = 1 if color_clean == c else 0

    # Gold is reference; only Silver dummy exists in model
    if "hardware_raw_Silver" in inp.columns:
        inp["hardware_raw_Silver"] = 1 if hardware == "Silver" else 0

    if "platform_The RealReal" in inp.columns:
        inp["platform_The RealReal"] = 0
    # model_variant: BASE is reference (all zeros)
    for v in ["DOUBLE_FLAP", "MONOGRAM", "Oblique", "REISSUE", "RETOURNE", "SELLIER"]:
        col = f"model_variant_{v}"
        if col in inp.columns:
            inp[col] = 0  # always BASE for standard bags

    return inp, age, is_exotic

# ── Comparable market stats by brand + model (from resale_listings.csv) ─────
MARKET_STATS = {
    ("CHANEL",        "2_55"):         {"n": 21,  "median": 3745,  "low": 2312,  "high": 9800},
    ("CHANEL",        "CLASSIC_FLAP"): {"n": 66,  "median": 10045, "low": 3369,  "high": 22707},
    ("DIOR",          "LADY_DIOR"):    {"n": 33,  "median": 2971,  "low": 1028,  "high": 17846},
    ("DIOR",          "SADDLE"):       {"n": 31,  "median": 2879,  "low": 912,   "high": 7108},
    ("HERMES",        "BIRKIN"):       {"n": 43,  "median": 22880, "low": 8602,  "high": 60231},
    ("HERMES",        "KELLY"):        {"n": 44,  "median": 25514, "low": 9397,  "high": 62892},
    ("LOUIS_VUITTON", "NEVERFULL"):    {"n": 27,  "median": 1319,  "low": 1076,  "high": 7310},
    ("LOUIS_VUITTON", "SPEEDY"):       {"n": 32,  "median": 1952,  "low": 770,   "high": 15152},
}

# ── Retail reference prices 2026 (from Retail_dataset.xlsx) ─────────────────
# Used only for resale/retail ratio — size does not affect XGBoost prediction
RETAIL_2026 = {
    "CHANEL_2_55_224":                4950,
    "CHANEL_2_55_225":               10300,
    "CHANEL_2_55_226":               11100,
    "CHANEL_2_55_227":               13100,
    "CHANEL_CLASSIC_FLAP_MINI_SQUARE":   4750,
    "CHANEL_CLASSIC_FLAP_MINI_RECTANGULAR": 4950,
    "CHANEL_CLASSIC_FLAP_SMALL":      9900,
    "CHANEL_CLASSIC_FLAP_MEDIUM":    10300,
    "CHANEL_CLASSIC_FLAP_JUMBO":     11100,
    "CHANEL_CLASSIC_FLAP_MAXI":      11700,
    "DIOR_LADY_DIOR_MINI":            4700,
    "DIOR_LADY_DIOR_MEDIUM":          5900,
    "DIOR_SADDLE_MINI":               3450,
    "DIOR_SADDLE_MEDIUM":             3900,
    "LOUIS_VUITTON_NEVERFULL_PM":     1500,
    "LOUIS_VUITTON_NEVERFULL_MM":     1550,
    "LOUIS_VUITTON_NEVERFULL_GM":     1600,
    "LOUIS_VUITTON_SPEEDY_NANO":      1600,
    "LOUIS_VUITTON_SPEEDY_20":        1900,
    "LOUIS_VUITTON_SPEEDY_25":        1600,
    "LOUIS_VUITTON_SPEEDY_30":        1650,
    "LOUIS_VUITTON_SPEEDY_35":        1700,
    "HERMES_BIRKIN_25":               9600,
    "HERMES_BIRKIN_30":              10600,
    "HERMES_BIRKIN_35":              11600,
    "HERMES_KELLY_20":                8000,
    "HERMES_KELLY_25":                9600,
    "HERMES_KELLY_28":               10100,
}

# Sizes with retail reference available
SIZE_OPTIONS = {
    ("HERMES",        "BIRKIN"):       ["25", "30", "35"],
    ("HERMES",        "KELLY"):        ["20", "25", "28"],
    ("CHANEL",        "CLASSIC_FLAP"): ["Mini Square", "Mini Rectangular", "Small", "Medium", "Jumbo", "Maxi"],
    ("CHANEL",        "2_55"):         ["224", "225", "226", "227"],
    ("DIOR",          "LADY_DIOR"):    ["Mini", "Medium"],
    ("DIOR",          "SADDLE"):       ["Mini", "Medium"],
    ("LOUIS_VUITTON", "NEVERFULL"):    ["PM", "MM", "GM"],
    ("LOUIS_VUITTON", "SPEEDY"):       ["Nano", "20", "25", "30", "35"],
}

# ── Brand-level price percentiles from dataset (n=296) ───────────────────────
# Used to normalise predicted price within brand space (Component 1)

BRAND_PERCENTILES = {
    "HERMES":        [12913, 15757, 23693, 32472, 46332, 49572, 60604],
    "CHANEL":        [3461,  5056,  8094,  11632, 15496, 17002, 22466],
    "DIOR":          [2296,  2608,  2945,  4281,  5186,  6355,  11662],
    "LOUIS_VUITTON": [975,   1228,  1632,  2775,  4578,  5357,  10604],
}
# Percentile breakpoints: p10, p25, p50, p75, p90, p95, p99

def price_to_percentile_score(price, brand):
    """
    Maps predicted price to 0-10 score within brand distribution.
    Compares the bag against others of the SAME brand in the dataset.
    """
    thresholds = BRAND_PERCENTILES[brand]
    # thresholds = [p10, p25, p50, p75, p90, p95, p99]
    if   price >= thresholds[6]: return 10.0   # top 1%
    elif price >= thresholds[5]: return 9.0    # top 5%
    elif price >= thresholds[4]: return 8.0    # top 10%
    elif price >= thresholds[3]: return 7.0    # top 25%
    elif price >= thresholds[2]: return 5.5    # above median
    elif price >= thresholds[1]: return 4.0    # p25-p50
    elif price >= thresholds[0]: return 2.5    # p10-p25
    else:                        return 1.5    # bottom 10%

def compute_score(price_pred, brand, model_family,
                  condition_score, age, is_exotic,
                  includes_box, includes_dustbag, includes_card, material):
    """
    4-component investment score grounded in TFG findings:

    1. Predicted Resale Strength (40%)
       — XGBoost price normalised within brand space (same-brand percentile)
       — Removes brand bias: LV is scored vs LV, Hermès vs Hermès

    2. Condition & Preservation (25%)
       — SHAP: sharp non-linear premium above score 4 (condition is the only
         post-purchase lever)

    3. Lifecycle Timing (20%)
       — SHAP age dependence plot: contemporaneity <5yr, depreciation zone
         5-15yr, vintage recovery 20+yr

    4. Premium Attribute Modifiers (15%)
       — Exotic leather (is_exotic SHAP=0.082)
       — Structured premium leather (Epsom, Caviar — durability signal)
       — Original box (+21% OLS coefficient)
       — Full set (box + dustbag + card)
       — High-performing model families (Kelly, Birkin)
    """

    # ── Component 1: Predicted Resale Strength (40%) ──────────────────────────
    r_score = price_to_percentile_score(price_pred, brand)

    # ── Component 2: Lifecycle Timing (25%) ───────────────────────────────────
    # SHAP age rank 5th (mean |SHAP|=0.103) — non-linear lifecycle confirmed
    if   age < 3:         t_score = 7.5   # contemporaneity premium
    elif age < 5:         t_score = 6.5
    elif age <= 10:       t_score = 3.5   # early depreciation zone
    elif age <= 15:       t_score = 4.0   # mid depreciation zone
    elif age <= 20:       t_score = 5.5   # transitional
    elif age <= 30:       t_score = 7.0   # vintage recovery
    else:                 t_score = 7.5   # established vintage

    # ── Component 3: Condition & Preservation (20%) ───────────────────────────
    # SHAP condition rank 8th (mean |SHAP|=0.056) — non-linear inflection at score 4
    cond_map = {6: 10.0, 5: 8.5, 4: 6.0, 3: 3.5, 2: 1.5, 1: 1.0}
    c_score = cond_map.get(condition_score, 5.0)

    # ── Component 4: Premium Attribute Modifiers (15%) ────────────────────────
    # is_exotic SHAP rank 7th (mean |SHAP|=0.082)
    mod = 5.0  # neutral baseline
    if is_exotic:
        mod = 10.0
    elif material in ["Caviar Leather", "Epsom Leather", "Togo Leather"]:
        mod = 7.5
    elif material in ["Calfskin", "Clemence Leather", "Swift Leather", "Lambskin"]:
        mod = 5.5
    elif material in ["Canvas", "Denim", "Satin"]:
        mod = 3.5

    # ── Weighted composite (sums to 1.0) ──────────────────────────────────────
    composite = (
        0.40 * r_score +
        0.25 * t_score +
        0.20 * c_score +
        0.15 * mod
    )

    return round(min(10.0, max(1.0, composite)), 1)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & CSS
# ════════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Investment Score — Luxury Handbags", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,500;1,400&family=EB+Garamond:ital,wght@0,400;0,500;1,400&family=Jost:wght@300;400;500&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Jost', sans-serif;
    background-color: #FAF8F5;
    color: #1A1614;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2.5rem; padding-bottom: 3rem; max-width: 1200px; }

/* ── Divider ── */
hr { border: none; border-top: 1px solid #E2DDD8; margin: 2rem 0; }

/* ── Label style for all inputs ── */
.stSelectbox label,
.stSlider label,
.stCheckbox label,
[data-testid="stWidgetLabel"] p {
    font-family: 'Jost', sans-serif !important;
    font-size: 0.68rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: #8A7F76 !important;
}

/* ── Select boxes ── */
.stSelectbox > div > div {
    border: 1px solid #D9D4CE !important;
    border-radius: 2px !important;
    background: #FFFFFF !important;
    font-family: 'Jost', sans-serif !important;
    font-size: 0.9rem !important;
    color: #1A1614 !important;
    box-shadow: none !important;
}
.stSelectbox > div > div:focus-within {
    border-color: #1A1614 !important;
    box-shadow: none !important;
}

/* ── Slider ── */
.stSlider [data-baseweb="slider"] {
    padding-top: 0.4rem;
}
.stSlider [data-testid="stThumbValue"] {
    font-family: 'Jost', sans-serif;
    font-size: 0.78rem;
    color: #1A1614;
}

/* ── Checkboxes ── */
.stCheckbox > label > div:first-child {
    border: 1px solid #C8C3BD !important;
    border-radius: 2px !important;
    background: white !important;
}

/* ── Section label ── */
.section-label {
    font-family: 'Jost', sans-serif;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #B0A89F;
    margin-bottom: 0.2rem;
}

/* ── Thin rule ── */
.thin-rule {
    border: none;
    border-top: 1px solid #E8E3DE;
    margin: 1.2rem 0;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='padding-bottom: 0.5rem;'>
    <div style='font-family:"Jost",sans-serif; font-size:0.65rem; letter-spacing:0.25em;
                text-transform:uppercase; color:#B0A89F; margin-bottom:0.6rem;'>
        IE University · TFG 2026
    </div>
    <div style='font-family:"Playfair Display",serif; font-size:2.6rem; font-weight:400;
                color:#1A1614; line-height:1.1; letter-spacing:-0.01em;'>
        Luxury Handbag<br><em>Investment Score</em>
    </div>
    <div style='font-family:"EB Garamond",serif; font-size:1rem; color:#8A7F76;
                margin-top:0.6rem; font-style:italic;'>
        Which bag should you buy if you plan to resell? Powered by XGBoost &amp; SHAP
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ════════════════════════════════════════════════════════════════════════════════
left, spacer, right = st.columns([10, 1, 11])

# ── LEFT: Inputs ──────────────────────────────────────────────────────────────
with left:
    st.markdown('<div class="section-label">Configure your bag</div>', unsafe_allow_html=True)
    st.markdown('<div class="thin-rule"></div>', unsafe_allow_html=True)

    brand = st.selectbox("Brand",
        ["HERMES", "CHANEL", "DIOR", "LOUIS_VUITTON"],
        format_func=lambda x: x.replace("_", " ").title())

    model_family = st.selectbox("Model Family",
        MODEL_OPTIONS[brand],
        format_func=lambda x: x.replace("_", " ").title())

    # Size — only used for retail ratio lookup, not for XGBoost prediction
    ALL_SIZE_OPTIONS = {
        ("HERMES",        "BIRKIN"):       ["25", "30", "35", "40"],
        ("HERMES",        "KELLY"):        ["20", "25", "28", "32", "35"],
        ("CHANEL",        "CLASSIC_FLAP"): ["Mini Square", "Mini Rectangular", "Small", "Medium", "Jumbo", "Maxi"],
        ("CHANEL",        "2_55"):         ["224", "225", "226", "227"],
        ("DIOR",          "LADY_DIOR"):    ["Micro", "Mini", "Small", "Medium", "Large"],
        ("DIOR",          "SADDLE"):       ["Mini", "Medium"],
        ("LOUIS_VUITTON", "NEVERFULL"):    ["PM", "MM", "GM"],
        ("LOUIS_VUITTON", "SPEEDY"):       ["Nano", "20", "25", "30", "35", "40", "45"],
    }
    size = st.selectbox("Size", ALL_SIZE_OPTIONS.get((brand, model_family), ["Standard"]))

    material = st.selectbox("Material", MATERIAL_OPTIONS[(brand, model_family)])

    col_color, col_hw = st.columns(2)
    with col_color:
        color = st.selectbox("Color", COLOR_OPTIONS)
    with col_hw:
        hardware = st.selectbox("Hardware", HARDWARE_OPTIONS)

    condition_label = st.select_slider("Condition",
        options=["As Is", "Fair", "Good", "Very Good", "Excellent", "Pristine"],
        value="Very Good")
    condition_score = CONDITION_MAP[condition_label]

    year = st.slider("Year of manufacture", 1975, 2026, 2015)

    st.markdown('<div style="margin-top:1.2rem;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Accessories included</div>', unsafe_allow_html=True)
    ca, cb, cc = st.columns(3)
    with ca: includes_box      = st.checkbox("Original box")
    with cb: includes_dustbag  = st.checkbox("Dustbag")
    with cc: includes_card     = st.checkbox("Auth. card")

# ── RIGHT: Outputs ────────────────────────────────────────────────────────────
with right:

    inp, age, is_exotic = build_input(
        brand, model_family, material, color, hardware,
        condition_score, year, includes_box, includes_dustbag, includes_card
    )

    log_pred   = model.predict(inp)[0]
    price_pred = np.exp(log_pred)

    # ── Retail ratio lookup (size-specific, does not affect XGBoost) ────────────
    retail_price = None
    ratio = None
    if size:
        size_key = str(size).replace(" ", "_").upper()
        product_key = f"{brand}_{model_family}_{size_key}"
        retail_price = RETAIL_2026.get(product_key, None)
        if retail_price:
            ratio = price_pred / retail_price

    score = compute_score(
        price_pred, brand, model_family,
        condition_score, age, is_exotic,
        includes_box, includes_dustbag, includes_card, material
    )

    # Score thresholds → label + accent colour
    if score >= 7.5:
        verdict   = "Strong Investment"
        accent    = "#2D5016"   # deep forest
        accent_bg = "#F0F5EB"
        bar_color = "#4A7C2E"
    elif score >= 5.5:
        verdict   = "Moderate Potential"
        accent    = "#6B4A1E"
        accent_bg = "#FBF5EC"
        bar_color = "#B8872A"
    elif score >= 3.5:
        verdict   = "Limited Upside"
        accent    = "#7A4520"
        accent_bg = "#FBF0E8"
        bar_color = "#C4692A"
    else:
        verdict   = "High Depreciation Risk"
        accent    = "#7A1F1F"
        accent_bg = "#FBE8E8"
        bar_color = "#B02A2A"

    score_pct = int((score / 10) * 100)

    # BUYER signal — based on SHAP age lifecycle (age is 5th most important feature, mean |SHAP|=0.103)
    if age < 5:
        buyer_label, buyer_color, buyer_sub = "BUY", "#4A7C2E", "Contemporaneity premium, strong entry point"
    elif 5 <= age <= 15:
        buyer_label, buyer_color, buyer_sub = "AVOID", "#9B2226", "Depreciation zone, negotiate hard or wait"
    elif 16 <= age <= 20:
        buyer_label, buyer_color, buyer_sub = "BUY", "#4A7C2E", "Approaching vintage premium, good entry"
    else:
        buyer_label, buyer_color, buyer_sub = "BUY", "#4A7C2E", "Vintage premium consolidated, strong value"

    # SELLER signal
    if age < 5:
        seller_label, seller_color, seller_sub = "SELL", "#2A5BAD", f"Contemporary premium active, sell within the next {5 - age} years to avoid the depreciation zone"
    elif 5 <= age <= 15:
        seller_label, seller_color, seller_sub = "HOLD", "#B8872A", "Weakest resale window, wait for vintage status (~20+ years)"
    elif 16 <= age <= 20:
        seller_label, seller_color, seller_sub = "HOLD", "#B8872A", f"Almost there, hold {21 - age} more years to capture vintage premium"
    else:
        seller_label, seller_color, seller_sub = "SELL", "#2A5BAD", "Vintage premium window, strong sell moment"

    # Keep signal_color/label for score card bar (use buyer as reference)
    signal_color = buyer_color
    signal_label = buyer_label

    # ── Global Market Tier — based on SHAP brand hierarchy ───────────────────
    # HERMES=0.466, CHANEL=ref, DIOR=-0.140, LV=-0.147
    TIER_CONFIG = {
        "HERMES":        ("TIER I",   "#1B3A4B", "Premium",   "Strongest resale retention globally · SHAP=+0.466"),
        "CHANEL":        ("TIER II",  "#2D5016", "Strong",    "Second-tier resale leader · SHAP reference category"),
        "DIOR":          ("TIER III", "#B8872A", "Selective", "Resale value highly attribute-dependent · SHAP=−0.140"),
        "LOUIS_VUITTON": ("TIER IV",  "#9B2226", "Lifestyle", "Limited financial investment case · SHAP=−0.147"),
    }
    tier_label, tier_color, tier_name, tier_sub = TIER_CONFIG[brand]
    brand_display = brand.replace("_", " ").title()

    # ── Three-column layout: Score | Global Tier | Buyer/Seller ─────────────────
    col_score, col_tier, col_signal = st.columns([5, 4, 4])

    with col_score:
        st.markdown(f"""
        <div style='background:#1A1614; padding:28px 24px; border-radius:3px;
                    height:260px; box-sizing:border-box; display:flex; flex-direction:column;'>
            <div style='font-family:"Jost",sans-serif; font-size:0.55rem; letter-spacing:0.22em;
                        text-transform:uppercase; color:#FFFFFF; margin-bottom:4px;'>
                Investment Score
            </div>
            <div style='font-family:"Jost",sans-serif; font-size:0.55rem; color:#C8C3BD;
                        letter-spacing:0.08em; margin-bottom:8px;'>
                Within {brand_display} benchmark
            </div>
            <div style='display:flex; align-items:flex-end; gap:8px; margin-bottom:12px;'>
                <div style='font-family:"Playfair Display",serif; font-size:4.2rem;
                            font-weight:400; color:#FAF8F5; line-height:1;'>{score}</div>
                <div style='font-family:"Jost",sans-serif; font-size:0.65rem; color:#6B6460;
                            padding-bottom:10px;'>/ 10</div>
            </div>
            <div style='background:#2E2926; border-radius:1px; height:3px; margin-bottom:14px;'>
                <div style='background:{bar_color}; width:{score_pct}%; height:3px; border-radius:1px;'></div>
            </div>
            <div style='margin-top:auto; display:inline-block; background:{bar_color}22;
                        border:1px solid {bar_color}66; padding:5px 12px; border-radius:1px;'>
                <span style='font-family:"Jost",sans-serif; font-size:0.65rem; letter-spacing:0.12em;
                             text-transform:uppercase; color:{bar_color};'>{verdict}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_tier:
        st.markdown(f"""
        <div style='background:{tier_color}; padding:28px 24px; border-radius:3px;
                    height:260px; box-sizing:border-box; display:flex; flex-direction:column;
                    align-items:center; text-align:center;'>
            <div style='font-family:"Jost",sans-serif; font-size:0.55rem; letter-spacing:0.22em;
                        text-transform:uppercase; color:rgba(255,255,255,0.55); margin-bottom:8px;
                        width:100%; text-align:center;'>
                Global Market Tier
            </div>
            <div style='display:flex; align-items:flex-end; gap:8px; margin-bottom:12px;'>
                <div style='font-family:"Jost",sans-serif; font-size:0.7rem; color:rgba(255,255,255,0.5);
                            letter-spacing:0.1em; padding-bottom:10px;'>TIER</div>
                <div style='font-family:"Playfair Display",serif; font-size:4.2rem;
                            font-weight:400; color:white; line-height:1;'>
                    {tier_label.replace("TIER ", "")}</div>
            </div>
            <div style='background:rgba(255,255,255,0.15); border-radius:1px; height:3px;
                        margin-bottom:14px; width:100%;'>
                <div style='background:rgba(255,255,255,0.5); width:100%; height:3px; border-radius:1px;'></div>
            </div>
            <div style='font-family:"Jost",sans-serif; font-size:0.65rem; font-weight:400;
                        color:white; letter-spacing:0.14em; text-transform:uppercase; margin-bottom:6px;'>
                {tier_name}
            </div>
            <div style='font-family:"Jost",sans-serif; font-size:0.6rem;
                        color:rgba(255,255,255,0.6); line-height:1.6; margin-top:auto;'>
                {tier_sub}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_signal:
        st.markdown(f"""
        <div style='display:flex; flex-direction:column; gap:8px; height:260px;'>
            <div style='background:{buyer_color}; padding:14px 18px; border-radius:3px; flex:1; overflow:hidden;'>
                <div style='font-family:"Jost",sans-serif; font-size:0.5rem; letter-spacing:0.18em;
                            text-transform:uppercase; color:rgba(255,255,255,0.6); margin-bottom:4px;'>
                    Buying
                </div>
                <div style='font-family:"Playfair Display",serif; font-size:2rem;
                            font-weight:400; color:white; line-height:1; margin-bottom:5px;'>
                    {buyer_label}
                </div>
                <div style='font-family:"Jost",sans-serif; font-size:0.6rem;
                            color:rgba(255,255,255,0.75); line-height:1.4;'>
                    {buyer_sub}
                </div>
            </div>
            <div style='background:{seller_color}; padding:14px 18px; border-radius:3px; flex:1; overflow:hidden;'>
                <div style='font-family:"Jost",sans-serif; font-size:0.5rem; letter-spacing:0.18em;
                            text-transform:uppercase; color:rgba(255,255,255,0.6); margin-bottom:4px;'>
                    Selling
                </div>
                <div style='font-family:"Playfair Display",serif; font-size:2rem;
                            font-weight:400; color:white; line-height:1; margin-bottom:5px;'>
                    {seller_label}
                </div>
                <div style='font-family:"Jost",sans-serif; font-size:0.6rem;
                            color:rgba(255,255,255,0.75); line-height:1.4;'>
                    {seller_sub}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Price card ────────────────────────────────────
    st.markdown(f"""
    <div style='background:#FFFFFF; border:1px solid #E2DDD8; padding:22px 24px;
                border-radius:3px; margin-bottom:14px;'>
        <div style='font-family:"Jost",sans-serif; font-size:0.6rem; letter-spacing:0.18em;
                    text-transform:uppercase; color:#B0A89F; margin-bottom:8px;'>
            Estimated Resale Price
        </div>
        <div style='font-family:"Playfair Display",serif; font-size:2.4rem;
                    color:#1A1614; font-weight:400;'>€{price_pred:,.0f}</div>
        <div style='font-family:"Jost",sans-serif; font-size:0.65rem; color:#C8C3BD;
                    margin-top:6px;'>XGBoost · R²=0.802 · Pearson r=0.897</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Resale / Retail Ratio card ───────────────────────
    if retail_price and ratio:
        ratio_color = "#2D5016" if ratio >= 1.0 else "#9B2226"
        ratio_label = "above" if ratio >= 1.0 else "below"
        col_ratio_l, col_ratio_r = st.columns([1, 1])
        with col_ratio_l:
            st.markdown(f"""
            <div style='background:{ratio_color}; padding:22px 24px; border-radius:3px; margin-bottom:14px;'>
                <div style='font-family:"Jost",sans-serif; font-size:0.55rem; letter-spacing:0.22em;
                            text-transform:uppercase; color:rgba(255,255,255,0.6); margin-bottom:8px;'>
                    Resale / Retail Ratio
                </div>
                <div style='font-family:"Playfair Display",serif; font-size:3.5rem;
                            color:white; line-height:1; margin-bottom:6px;'>
                    {ratio:.2f}×
                </div>
                <div style='font-family:"Jost",sans-serif; font-size:0.7rem; color:rgba(255,255,255,0.85);'>
                    {ratio_label} boutique retail
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_ratio_r:
            st.markdown(f"""
            <div style='background:#F5F0E8; border:1px solid #E2DDD8; padding:22px 24px;
                        border-radius:3px; margin-bottom:14px; height:100%; box-sizing:border-box;'>
                <div style='font-family:"Jost",sans-serif; font-size:0.55rem; letter-spacing:0.18em;
                            text-transform:uppercase; color:#B0A89F; margin-bottom:10px;'>
                    Retail Reference 2026
                </div>
                <div style='font-family:"Playfair Display",serif; font-size:1.8rem; color:#1A1614;'>
                    €{retail_price:,}
                </div>
                <div style='font-family:"Jost",sans-serif; font-size:0.65rem; color:#9B8B7A; margin-top:8px;'>
                    Boutique price for this size
                </div>
                <div style='font-family:"EB Garamond",serif; font-size:0.75rem; color:#B0A89F;
                            font-style:italic; margin-top:8px; line-height:1.5;'>
                    Indicative · retail reference may vary by region and year
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Comparable Market Evidence ───────────────────────
    mkt = MARKET_STATS.get((brand, model_family))
    if mkt:
        brand_label = brand.replace("_", " ").title()
        model_label = model_family.replace("_", " ").title()
        # Position of predicted price within range (for bar)
        pos_pct = int(max(0, min(100, (price_pred - mkt['low']) / max(1, mkt['high'] - mkt['low']) * 100)))
        st.markdown(f"""
        <div style='background:#FFFFFF; border:1px solid #E2DDD8; padding:20px 24px;
                    border-radius:3px; margin-bottom:14px;'>
            <div style='font-family:"Jost",sans-serif; font-size:0.6rem; letter-spacing:0.18em;
                        text-transform:uppercase; color:#B0A89F; margin-bottom:12px;'>
                Comparable Market Evidence
            </div>
            <div style='font-family:"Jost",sans-serif; font-size:0.78rem; color:#1A1614;
                        margin-bottom:4px; font-weight:500;'>
                {brand_label} {model_label}
            </div>
            <div style='font-family:"EB Garamond",serif; font-size:0.85rem; color:#6B6460;
                        margin-bottom:12px;'>
                Market range · €{mkt['low']:,} — €{mkt['high']:,} · {mkt['n']} listings
            </div>
            <!-- Range bar with predicted price marker -->
            <div style='position:relative; margin-bottom:6px;'>
                <div style='background:#F0EDE8; border-radius:2px; height:6px;'>
                    <div style='background:#D9D0C3; width:100%; height:6px; border-radius:2px;'></div>
                </div>
                <div style='position:absolute; top:-3px; left:{pos_pct}%;
                            width:2px; height:12px; background:#1A1614; border-radius:1px;'></div>
            </div>
            <div style='display:flex; justify-content:space-between; margin-top:4px;'>
                <span style='font-family:"Jost",sans-serif; font-size:0.62rem; color:#B0A89F;'>€{mkt['low']:,}</span>
                <span style='font-family:"Jost",sans-serif; font-size:0.62rem; color:#1A1614; font-weight:500;'>
                    Your bag: €{price_pred:,.0f}
                </span>
                <span style='font-family:"Jost",sans-serif; font-size:0.62rem; color:#B0A89F;'>€{mkt['high']:,}</span>
            </div>
            <div style='font-family:"EB Garamond",serif; font-size:0.82rem; color:#9B8B7A;
                        margin-top:10px; font-style:italic;'>
                Your configuration (material, condition, age, accessories) determines where
                within this range your bag falls. The model predicts €{price_pred:,.0f}.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Key Value Drivers ─────────────────────────────
    st.markdown("""
    <div style='font-family:"Jost",sans-serif; font-size:0.6rem; letter-spacing:0.2em;
                text-transform:uppercase; color:#B0A89F; margin-top:22px; margin-bottom:10px;'>
        Key Value Drivers
    </div>
    """, unsafe_allow_html=True)

    drivers = []

    if brand == "HERMES":
        drivers.append(("+", "Brand, Model Premium",
            "Hermès commands the highest SHAP contribution, with median resale more than 10× above Louis Vuitton."))
    elif brand == "CHANEL" and model_family == "CLASSIC_FLAP":
        drivers.append(("+", "Brand Premium",
            "Classic Flap is the strongest second-tier segment, with resale/retail ratio up to 2.12×."))
    elif brand == "CHANEL":
        drivers.append(("n", "Brand Premium, Moderate",
            "2.55 shows lower resale activity than the Classic Flap in this dataset."))
    else:
        drivers.append(("-", "Brand Depreciation Risk",
            "Dior and LV show negative SHAP versus Hermès/Chanel, most listings trade below retail."))

    if is_exotic:
        drivers.append(("+", "Exotic Leather Premium",
            "Crocodile (median €32,344) and Alligator (€29,200) dominate the top price deciles."))
    elif material in ["Caviar Leather", "Togo Leather", "Epsom Leather"]:
        drivers.append(("n", f"{material}",
            "Durable premium leather, a positive but moderate signal relative to exotic skins."))

    if condition_score == 6:
        drivers.append(("+", "Pristine Condition",
            "Condition is the only post-purchase lever, with a sharp non-linear SHAP premium above score 4."))
    elif condition_score == 5:
        drivers.append(("+", "Excellent Condition",
            "Strong condition signal with meaningful value premium, just below the Pristine threshold."))
    elif condition_score <= 3:
        drivers.append(("-", f"Condition Risk, {condition_label}",
            "Significant negative SHAP impact below Very Good. Condition is the biggest controllable driver."))

    if age >= 25:
        drivers.append(("+", "Vintage Premium",
            "Non-linear age effect confirmed in dataset: 1990s bags show meaningful price recovery."))
    elif 5 <= age <= 15:
        drivers.append(("-", "Mid-Age Depreciation Zone",
            "5–15 year bags show the weakest resale, neither contemporary nor collectible vintage."))
    elif age < 5:
        drivers.append(("n", "Contemporary",
            "Recent production carries a current-season premium, but no vintage upside yet."))

    if includes_box:
        drivers.append(("+", "Original Box",
            "includes_box ranks 14th in SHAP global importance (mean |SHAP|=0.027), the most impactful accessory decision."))
    if includes_dustbag and not includes_box:
        drivers.append(("n", "Dustbag Included",
            "Positive accessory signal, though smaller in magnitude than the original box."))

    dir_styles = {
        "+": ("#2D5016", "#4A7C2E", "+"),
        "-": ("#7A1F1F", "#B02A2A", "—"),
        "n": ("#5C4A1E", "#9A7A30", "~"),
    }

    for direction, name, desc in drivers:
        text_c, border_c, symbol = dir_styles[direction]
        st.markdown(f"""
        <div style='border-left:2px solid {border_c}; padding:12px 16px 12px 18px;
                    margin:6px 0; background:#FFFFFF;'>
            <div style='display:flex; align-items:baseline; gap:8px; margin-bottom:4px;'>
                <span style='font-family:"Jost",sans-serif; font-size:0.65rem; font-weight:500;
                             letter-spacing:0.06em; color:{border_c};'>{symbol}</span>
                <span style='font-family:"Jost",sans-serif; font-size:0.78rem; font-weight:500;
                             letter-spacing:0.04em; color:{text_c};'>{name}</span>
            </div>
            <div style='font-family:"EB Garamond",serif; font-size:0.88rem; color:#6B6460;
                        line-height:1.55;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Warnings ──────────────────────────────────────


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style='font-family:"Jost",sans-serif; font-size:0.62rem; letter-spacing:0.08em;
            color:#C8C3BD; text-align:center; line-height:2;'>
    XGBoost · R²=0.802 · CV R²=0.861 ± 0.036 · Pearson r=0.897 · MAE=0.362 (log scale)
    &nbsp;&nbsp;·&nbsp;&nbsp;
    296 resale listings · Farfetch Pre-Owned &amp; The RealReal · Feb–Mar 2026
    &nbsp;&nbsp;·&nbsp;&nbsp;
    IE University TFG 2026
</div>
""", unsafe_allow_html=True)
