import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from datetime import datetime
import json
import random
import streamlit.components.v1 as components

# ─────────────────────────────────────────────────────────────────────────────
# 1. PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smartfarm AI",
    page_icon="🌿",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. STYLING
# ─────────────────────────────────────────────────────────────────────────────
def inject_ui():
    # Force scroll to top — overrides st.chat_input's built-in auto-scroll
    # st.markdown strips <script> tags, so we use components.html which runs real JS
    components.html("""
    <script>
        window.parent.scrollTo(0, 0);
        setTimeout(function() { window.parent.scrollTo(0, 0); }, 80);
        setTimeout(function() { window.parent.scrollTo(0, 0); }, 200);
    </script>
    """, height=0)
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

        html, body, .stApp {
            font-family: 'Inter', sans-serif;
            background: #0b0e14;
            color: #e0e0e0;
        }

        /* ── Shrink Streamlit's massive default top padding ── */
        .stApp > div:first-child > div:first-child > div:first-child {
            padding-top: 0.5rem !important;
        }
        /* Target the main block container top padding */
        div[data-testid="stAppViewBlockContainer"] {
            padding-top: 0.75rem !important;
        }
        /* Also covers older Streamlit versions */
        .main .block-container {
            padding-top: 0.75rem !important;
        }

        /* ── Brand header ── */
        .brand-header {
            display: flex;
            align-items: center;
            gap: 14px;
            margin-bottom: 4px;
        }
        .brand-icon {
            width: 52px; height: 52px;
            background: linear-gradient(135deg, #1db954, #4CAF50);
            border-radius: 14px;
            display: flex; align-items: center; justify-content: center;
            font-size: 28px;
            box-shadow: 0 4px 20px rgba(76,175,80,0.4);
            flex-shrink: 0;
        }
        .brand-title {
            font-size: 2.4rem;
            font-weight: 800;
            background: linear-gradient(90deg, #4CAF50, #81C784);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0; line-height: 1;
        }
        .brand-subtitle {
            font-size: 0.85rem;
            color: #6b7280;
            margin: 2px 0 0;
        }

        /* ── Metric cards ── */
        .metric-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            padding: 18px 14px;
            border-radius: 14px;
            text-align: center;
            transition: border-color 0.2s;
        }
        .metric-card:hover { border-color: rgba(76,175,80,0.4); }
        .metric-card h3 { margin: 4px 0 0; font-size: 1.5rem; color: #4CAF50; }

        /* ── AI card ── */
        .ai-card {
            background: linear-gradient(135deg, rgba(76,175,80,0.12), rgba(0,0,0,0.3));
            border: 1px solid #4CAF50;
            border-radius: 14px;
            padding: 16px;
            margin-bottom: 16px;
        }

        /* ── Hourly item ── */
        .hourly-item {
            text-align: center; padding: 10px; min-width: 80px;
            background: rgba(255,255,255,0.04);
            border-radius: 12px; margin: 4px;
            border: 1px solid rgba(255,255,255,0.06);
        }

        /* ── Suggestion pills ── */
        div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
            background: rgba(255,255,255,0.04) !important;
            border: 1px solid rgba(255,255,255,0.12) !important;
            border-radius: 12px !important;
            color: #b0b8c8 !important;
            font-size: 0.80rem !important;
            padding: 8px 12px !important;
            line-height: 1.4 !important;
            white-space: normal !important;
            word-break: break-word !important;
            text-align: left !important;
            height: auto !important;
            min-height: 48px !important;
            transition: all 0.18s ease !important;
        }
        div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover {
            background: rgba(76,175,80,0.14) !important;
            border-color: rgba(76,175,80,0.6) !important;
            color: #e8ffe8 !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 18px rgba(76,175,80,0.2) !important;
        }

        /* ── Chat area ── */
        .stChatMessage { border-radius: 12px; }
        section[data-testid="stSidebar"] { z-index: 999; }

        /* ── AI section header ── */
        .ai-section-header {
            display: flex; align-items: center; gap: 10px;
            margin: 8px 0 4px;
        }
        .ai-badge {
            background: linear-gradient(135deg, #1db954, #4CAF50);
            color: #fff; font-size: 0.72rem; font-weight: 700;
            padding: 2px 10px; border-radius: 999px;
            letter-spacing: 0.5px;
        }

        /* ── Pill buttons: clean, readable, no clipping ── */
        div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
            background: rgba(255,255,255,0.05) !important;
            border: 1px solid rgba(255,255,255,0.15) !important;
            border-radius: 10px !important;
            color: #e0e0e0 !important;
            font-size: 0.80rem !important;
            font-weight: 500 !important;
            padding: 10px 10px !important;
            white-space: normal !important;
            word-break: break-word !important;
            text-align: center !important;
            line-height: 1.4 !important;
            height: auto !important;
            min-height: 56px !important;
            transition: all 0.18s ease !important;
        }
        div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover {
            background: rgba(76,175,80,0.15) !important;
            border-color: rgba(76,175,80,0.6) !important;
            color: #c8ffc8 !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 18px rgba(76,175,80,0.2) !important;
        }
    </style>
    """, unsafe_allow_html=True)

inject_ui()

# ── Prevent Streamlit chat input auto-scroll ──────────────────────────────
components.html(
    """
    <script>
    const scrollPosition = sessionStorage.getItem('scrollPosition');
    if (scrollPosition !== null) {
        window.parent.scrollTo(0, parseInt(scrollPosition, 10));
    }
    window.parent.addEventListener('scroll', function() {
        sessionStorage.setItem('scrollPosition', window.parent.scrollY);
    });
    </script>
    """,
    height=0,
)


# ─────────────────────────────────────────────────────────────────────────────
# 3. SECRETS & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
WEATHER_API_KEY = "e0ac0df776637a4ccbff5a43d8ed12ad"
GEMINI_API_KEY  = st.secrets.get("GEMINI_API_KEY", "")

INDIAN_CITIES = [
    "", "Agra", "Ahmedabad", "Ajmer", "Aligarh", "Amritsar", "Aurangabad",
    "Bangalore", "Bareilly", "Bathinda", "Bhavnagar", "Bhopal", "Bhubaneswar",
    "Bikaner", "Chandigarh", "Chennai", "Coimbatore", "Cuttack", "Dehradun",
    "Delhi", "Dhanbad", "Faridabad", "Gandhinagar", "Ghaziabad", "Gorakhpur",
    "Gurgaon", "Guwahati", "Gwalior", "Hubli", "Hyderabad", "Indore",
    "Jabalpur", "Jaipur", "Jalandhar", "Jammu", "Jamnagar", "Jamshedpur",
    "Jodhpur", "Kanpur", "Kochi", "Kolkata", "Kota", "Kozhikode", "Lucknow",
    "Ludhiana", "Madurai", "Mangalore", "Meerut", "Moradabad", "Mumbai",
    "Mysore", "Nagpur", "Nashik", "Noida", "Patna", "Pune", "Raipur",
    "Rajkot", "Ranchi", "Rohtak", "Rourkela", "Salem", "Siliguri",
    "Solapur", "Srinagar", "Surat", "Thiruvananthapuram", "Tiruchirappalli",
    "Udaipur", "Ujjain", "Vadodara", "Varanasi", "Vijayawada", "Visakhapatnam", "Warangal"
]

CROP_KB = {
  "Wheat":     {"ideal_temp": "15°C to 25°C",  "water_needs": "Moderate (450–650mm/season). Water at crown root initiation (21 DAS) and grain filling stages.", "rain_warning": "Heavy rain during ripening causes lodging and fungal rust. Avoid irrigation 10 days before harvest."},
  "Rice":      {"ideal_temp": "25°C to 35°C",  "water_needs": "High (1000–2000mm/season). Maintain 5–10cm water level in paddy fields throughout growing season.", "rain_warning": "Resilient, but extreme flooding (>30cm for >7 days) submerges seedlings. Ensure drainage channels are clear."},
  "Cotton":    {"ideal_temp": "21°C to 37°C",  "water_needs": "Low to moderate (700–1300mm/season). Critical periods: flowering and boll development. Drip irrigation preferred.", "rain_warning": "Rain during boll-opening ruins fiber quality through staining and rotting. Avoid wetting bolls."},
  "Maize":     {"ideal_temp": "18°C to 32°C",  "water_needs": "Moderate (500–800mm/season). Most critical at tasseling and silking stages. Water stress at these stages cuts yield 50%.", "rain_warning": "Waterlogging for >48 hours damages roots. Ensure proper field drainage. High humidity promotes leaf blight."},
  "Sugarcane": {"ideal_temp": "25°C to 38°C",  "water_needs": "Very high (1500–2500mm/season). Regular irrigation every 7–10 days during grand growth period.", "rain_warning": "Excess rain during ratoon stage increases smut disease. Ensure drainage to avoid root rot in low-lying areas."},
  "Mustard":   {"ideal_temp": "10°C to 25°C",  "water_needs": "Low (250–400mm/season). 2–3 irrigations sufficient. Avoid waterlogging — roots rot easily.", "rain_warning": "Rain at flowering causes pollen washing, reducing pod set. Rain at pod maturity splits pods causing seed loss."}
}

SUGGESTIONS = [
    "🌾 Wheat water needs today?",
    "🌧️ Rain impact on Cotton?",
    "💊 Safe to spray pesticide?",
    "🦠 Disease risk today?",
]

# ─────────────────────────────────────────────────────────────────────────────
# 4. SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""
if "weather_ctx" not in st.session_state:
    st.session_state.weather_ctx = None

# ─────────────────────────────────────────────────────────────────────────────
# 5. HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def get_wind_direction(deg):
    dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    return dirs[int((deg + 22.5) / 45) % 8]

def get_crop_recommendation(temp, rain_prob):
    if rain_prob > 70 and temp > 25:
        return "RICE, JUTE, or SOYBEAN", "These thrive in high water availability and warm, humid temperatures."
    elif temp > 35:
        return "COTTON, BAJRA, or GUAR", "Highly heat tolerant. These require dry, arid conditions to prevent rot."
    elif 25 <= temp <= 35:
        return "SUGARCANE, GROUNDNUT, or MOONG", "The 25–35°C range is the optimal window for the rapid growth of these crops."
    elif 15 <= temp < 25:
        return "WHEAT, CHICKPEA, or PEAS", "Perfect cooler temperatures for the tillering and flowering stages."
    elif temp < 15:
        return "MUSTARD or BARLEY", "Winter (Rabi) crops that require a distinctly cool climate to yield properly."
    else:
        return "MAIZE or SORGHUM", "Versatile crops suited for moderate, transitional climates."

def _in_ideal(temp, ideal_str):
    try:
        parts = ideal_str.replace("°C", "").split(" to ")
        lo, hi = float(parts[0].strip()), float(parts[1].strip())
        return lo <= temp <= hi
    except Exception:
        return False

def call_gemini(prompt: str):
    if not GEMINI_API_KEY:
        return None
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    )
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 512},
    }
    try:
        r = requests.post(url, json=body, timeout=15)
        data = r.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return None

def rule_based_response(question, city, avg_temp, rain_prob, max_wind, avg_hum, current_crop):
    q = question.lower()
    detected_crop = None
    for crop in CROP_KB:
        if crop.lower() in q:
            detected_crop = crop
            break
    if detected_crop is None and current_crop != "None" and current_crop in CROP_KB:
        detected_crop = current_crop
    crop_info = CROP_KB.get(detected_crop, {}) if detected_crop else {}

    if any(w in q for w in ["water", "irrigat", "moisture", "rain"]):
        if detected_crop and crop_info:
            rain_advice = ("Skip irrigation today — rain probability is high." if rain_prob > 50
                           else "Proceed with scheduled irrigation as rain probability is low.")
            return (
                f"**💧 Water Advisory for {detected_crop} in {city}**\n\n"
                f"- **Water needs:** {crop_info['water_needs']}\n"
                f"- **Current rain probability:** {int(rain_prob)}% → {rain_advice}\n"
                f"- **Current humidity:** {avg_hum}%\n"
                f"- **Rainfall caution:** {crop_info['rain_warning']}"
            )
        rain_label = "High" if rain_prob > 50 else "Low"
        return (
            f"**💧 Irrigation Overview for {city}**\n\n"
            f"- Rain probability today: **{int(rain_prob)}%** ({rain_label})\n"
            f"- {'Consider skipping irrigation — natural rainfall expected.' if rain_prob > 50 else 'Standard irrigation is recommended today.'}\n"
            f"- Humidity is at **{avg_hum}%**.\n\n"
            f"Select a specific crop in the sidebar or mention a crop name for detailed advice."
        )

    elif any(w in q for w in ["spray", "pesticide", "fertilizer", "chemical", "apply"]):
        safe = max_wind < 10 and rain_prob < 20
        wind_ok = "✅ Wind speed is safe" if max_wind < 10 else f"❌ Wind speed is **{max_wind} km/h** (ideal: <10 km/h)"
        rain_ok = "✅ Low rain probability" if rain_prob < 20 else f"❌ Rain probability is **{int(rain_prob)}%** (ideal: <20%)"
        verdict = "✅ **Safe to spray today.**" if safe else "❌ **Do NOT spray today** — conditions are unfavourable."
        return (
            f"**🌱 Spraying Conditions in {city}**\n\n"
            f"{verdict}\n\n"
            f"- {wind_ok}\n"
            f"- {rain_ok}\n"
            f"- Temperature: **{avg_temp}°C**\n\n"
            f"Best time to spray: Early morning (6–9 AM) when winds are calmest."
        )

    elif any(w in q for w in ["temp", "heat", "sow", "plant", "grow", "safe", "suitable"]):
        if detected_crop and crop_info:
            return (
                f"**🌡️ Temperature Assessment for {detected_crop} in {city}**\n\n"
                f"- **Current temperature:** {avg_temp}°C\n"
                f"- **Ideal range for {detected_crop}:** {crop_info['ideal_temp']}\n"
                f"- **Assessment:** {'✅ Within ideal range.' if _in_ideal(avg_temp, crop_info['ideal_temp']) else '⚠️ Outside ideal range — monitor crop stress carefully.'}\n"
                f"- **Rain warning:** {crop_info['rain_warning']}"
            )
        rec_crop, rec_reason = get_crop_recommendation(avg_temp, rain_prob)
        return (
            f"**🌡️ Temperature Overview for {city}**\n\n"
            f"- Current temperature: **{avg_temp}°C**\n"
            f"- Best suited crop for today's conditions: **{rec_crop}**\n"
            f"- Reason: {rec_reason}\n\n"
            f"Mention a specific crop to get a tailored temperature assessment."
        )

    elif any(w in q for w in ["wind", "storm", "breeze"]):
        return (
            f"**💨 Wind Advisory for {city}**\n\n"
            f"- Current wind: **{max_wind} km/h**\n"
            f"- {'⚠️ High winds — delay spraying and secure young plants.' if max_wind > 15 else '✅ Wind is within safe limits for field operations.'}\n"
            f"- Pollination is {'aided by gentle breeze.' if 5 < max_wind < 15 else 'potentially disrupted at this wind speed.'}"
        )

    elif any(w in q for w in ["disease", "fungal", "pest", "blight", "rust", "mold"]):
        risk = "HIGH" if avg_hum > 75 and avg_temp > 20 else "MODERATE" if avg_hum > 60 else "LOW"
        color = "🔴" if risk == "HIGH" else "🟡" if risk == "MODERATE" else "🟢"
        return (
            f"**🦠 Disease Risk Assessment for {city}**\n\n"
            f"- {color} Disease pressure today: **{risk}**\n"
            f"- Humidity: **{avg_hum}%** | Temperature: **{avg_temp}°C**\n"
            f"- {'Apply a preventive fungicide before rain.' if risk in ['HIGH', 'MODERATE'] else 'Conditions are low-risk today.'}"
            + (f"\n- **{detected_crop} specific:** {crop_info.get('rain_warning', '')}" if detected_crop and crop_info else "")
        )

    elif any(w in q for w in ["harvest", "pick", "cut", "reap"]):
        safe_harvest = rain_prob < 20 and max_wind < 15
        return (
            f"**🌾 Harvest Conditions for {city}**\n\n"
            f"- {'✅ Good conditions for harvesting today.' if safe_harvest else '❌ Avoid harvesting — rain or high winds can damage quality.'}\n"
            f"- Rain probability: **{int(rain_prob)}%** | Wind: **{max_wind} km/h**\n"
            f"- Harvest early in the day when dew has dried."
        )

    else:
        rec_crop, rec_reason = get_crop_recommendation(avg_temp, rain_prob)
        base = (
            f"**🤖 Smartfarm AI — {city} Today**\n\n"
            f"| Parameter | Value |\n|---|---|\n"
            f"| 🌡️ Temperature | {avg_temp}°C |\n"
            f"| 💧 Humidity | {avg_hum}% |\n"
            f"| ☔ Rain Probability | {int(rain_prob)}% |\n"
            f"| 💨 Wind Speed | {max_wind} km/h |\n\n"
            f"**Recommended crop for today:** {rec_crop} — {rec_reason}\n\n"
        )
        if detected_crop and crop_info:
            base += f"**{detected_crop} info:**\n- Water needs: {crop_info['water_needs']}\n- Caution: {crop_info['rain_warning']}"
        else:
            base += "Ask me about *irrigation, spraying, disease risk, temperature suitability, or harvest timing*."
        return base

def generate_ai_response(question, city, avg_temp, rain_prob, max_wind, avg_hum, current_crop):
    if GEMINI_API_KEY:
        kb_str = json.dumps(CROP_KB, indent=2)
        system_prompt = f"""You are Smartfarm AI, an expert agronomist assistant for Indian farmers.
You have access to real-time weather data for {city}:
- Average Temperature: {avg_temp}°C
- Humidity: {avg_hum}%
- Rain Probability: {int(rain_prob)}%
- Max Wind Speed: {max_wind} km/h
- Farmer's current crop: {current_crop}

Crop knowledge base:
{kb_str}

Answer the farmer's question in a helpful, concise manner. Use markdown formatting.
Use emojis where appropriate. Give specific, actionable advice based on the weather data above.
Farmer's question: {question}"""
        reply = call_gemini(system_prompt)
        if reply:
            return reply
    return rule_based_response(question, city, avg_temp, rain_prob, max_wind, avg_hum, current_crop)

# ─────────────────────────────────────────────────────────────────────────────
# 6. SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚜 Farm Profile")
    current_crop = st.selectbox(
        "🌱 Growing Currently",
        ["None", "Wheat", "Rice", "Maize", "Cotton", "Sugarcane", "Mustard"]
    )
    target_date = st.date_input("📅 Forecast View", datetime.now())
    st.markdown("---")
    st.markdown("### 🤖 About Smartfarm AI")
    st.caption(
        "Powered by Gemini 1.5 + agronomy knowledge base. "
        "Ask me anything about your crops, irrigation, disease risks, and more."
    )

# ─────────────────────────────────────────────────────────────────────────────
# 7. BRAND HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="brand-header">
    <div class="brand-icon">🌿</div>
    <div>
        <div class="brand-title">Smartfarm AI</div>
        <div class="brand-subtitle">Digital Twin · Predictive Farming · AI Agronomist</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 8. CITY SELECTOR (full-width)
# ─────────────────────────────────────────────────────────────────────────────
city = st.selectbox(
    "🔍 Search City / District",
    INDIAN_CITIES,
    format_func=lambda x: "Type or select an Indian city..." if x == "" else x,
    key="city_select",
)

# ─────────────────────────────────────────────────────────────────────────────
# 9. WEATHER SECTION (full-width)
# ─────────────────────────────────────────────────────────────────────────────
if city:
    try:
        url = (
            f"http://api.openweathermap.org/data/2.5/forecast"
            f"?q={city},IN&appid={WEATHER_API_KEY}&units=metric"
        )
        res = requests.get(url, timeout=10).json()

        if res.get("cod") == "200":
            forecast = res["list"]
            target_date_str = target_date.strftime("%Y-%m-%d")
            day_forecast = [i for i in forecast if target_date_str in i["dt_txt"]]

            if not day_forecast:
                first_date_str = datetime.fromtimestamp(forecast[0]["dt"]).strftime("%Y-%m-%d")
                day_forecast = [i for i in forecast if first_date_str in i["dt_txt"]]
                st.warning(
                    f"⚠️ No data for **{target_date_str}**. "
                    f"Showing **{first_date_str}** (free API: next 5 days only)."
                )

            if day_forecast:
                avg_temp  = round(sum(i["main"]["temp"]     for i in day_forecast) / len(day_forecast), 1)
                avg_hum   = round(sum(i["main"]["humidity"] for i in day_forecast) / len(day_forecast), 1)
                max_wind  = round(max(i["wind"]["speed"]    for i in day_forecast), 1)
                rain_prob = max(i.get("pop", 0)             for i in day_forecast) * 100
                wind_deg  = day_forecast[len(day_forecast) // 2]["wind"]["deg"]

                st.session_state.weather_ctx = {
                    "city": city, "avg_temp": avg_temp, "avg_hum": avg_hum,
                    "max_wind": max_wind, "rain_prob": rain_prob, "current_crop": current_crop,
                }

                # ── Alerts ──
                if avg_temp > 40:
                    st.error(f"🔥 **Extreme Heatwave Alert:** Avg {avg_temp}°C. Increase irrigation.")
                if max_wind > 15:
                    st.warning(f"💨 **High Wind Warning:** {max_wind} km/h. Avoid pesticide spraying.")
                if rain_prob > 70:
                    st.info(f"🌧️ **Heavy Rain Expected:** {int(rain_prob)}% probability. Skip fertilizer.")
                elif avg_temp <= 40 and max_wind <= 15 and rain_prob <= 70:
                    st.success("✅ **Optimal Conditions:** Weather is stable for standard field operations.")

                # ── Key metrics (4-column) ──
                st.markdown("#### 🌦️ Today's Parameters")
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(f"<div class='metric-card'>🌡️ Temp<br><h3>{avg_temp}°C</h3></div>", unsafe_allow_html=True)
                with m2:
                    st.markdown(f"<div class='metric-card'>💧 Humidity<br><h3>{avg_hum}%</h3></div>", unsafe_allow_html=True)
                with m3:
                    st.markdown(f"<div class='metric-card'>☔ Rain Prob.<br><h3>{int(rain_prob)}%</h3></div>", unsafe_allow_html=True)
                with m4:
                    st.markdown(
                        f"<div class='metric-card'>💨 Wind<br>"
                        f"<h3>{max_wind} km/h ({get_wind_direction(wind_deg)})</h3></div>",
                        unsafe_allow_html=True,
                    )

                # ── Hourly forecast ──
                st.markdown(f"### 🕒 Hourly Forecast for {target_date.strftime('%b %d')}")
                cols = st.columns(len(day_forecast))
                for idx, hour in enumerate(day_forecast):
                    with cols[idx]:
                        time_str  = datetime.strptime(hour["dt_txt"], "%Y-%m-%d %H:%M:%S").strftime("%I %p")
                        icon_code = hour["weather"][0]["icon"]
                        st.markdown(f"""
                            <div class='hourly-item'>
                                <p style='font-size:0.78rem; margin:0;'>{time_str}</p>
                                <img src="http://openweathermap.org/img/wn/{icon_code}.png" width="38">
                                <p style='font-weight:bold; margin:0;'>{int(hour['main']['temp'])}°</p>
                            </div>
                        """, unsafe_allow_html=True)

                st.markdown("---")

                # ── Crop recommendation + Activity guide ──
                rec_col, advice_col = st.columns([1, 1.5])
                with rec_col:
                    crop_name, crop_reason = get_crop_recommendation(avg_temp, rain_prob)
                    st.markdown(f"""
                    <div class='ai-card'>
                        <p style='margin:0; font-size:0.88rem;'>💡 <b>AI Recommendation</b></p>
                        <h2 style='color:#4CAF50; margin:6px 0 4px;'>{crop_name}</h2>
                        <p style='font-size:0.83rem; color:#aaa; margin:0;'>{crop_reason}</p>
                    </div>
                    """, unsafe_allow_html=True)

                with advice_col:
                    if current_crop != "None":
                        st.subheader(f"🛠️ Activity Guide for {current_crop}")
                        if max_wind < 10 and rain_prob < 20:
                            st.success("✅ Perfect window for **Pesticide Spraying** (low wind).")
                        else:
                            st.error("❌ Delay **Spraying**; weather is too volatile.")
                        if rain_prob > 50:
                            st.warning("🚱 **Skip Irrigation:** Rain will provide sufficient moisture.")
                        else:
                            st.info("💧 Routine irrigation recommended based on soil dryness.")
                    else:
                        st.info("Select your current crop in the sidebar to get specific field activity advice.")

                # ── 5-day trend ──
                st.markdown("### 📈 5-Day Ecosystem Trend")
                df = pd.DataFrame([{
                    "Time": datetime.fromtimestamp(i["dt"]),
                    "Temperature (°C)": i["main"]["temp"],
                    "Humidity (%)":     i["main"]["humidity"],
                } for i in forecast])
                fig = px.line(
                    df, x="Time", y=["Temperature (°C)", "Humidity (%)"],
                    color_discrete_sequence=["#4CAF50", "#2196F3"],
                    template="plotly_dark",
                )
                fig.update_xaxes(tickformat="%a, %d %b", title=None, nticks=10)
                fig.update_layout(
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=0, r=0, t=30, b=0), height=350,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(255,255,255,0.02)",
                )
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.error(
                f"❌ City **'{city}'** not found or API error: "
                f"{res.get('message', 'Unknown error')}. Try a different city name."
            )

    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. Please check your internet connection.")
    except Exception as e:
        st.error(f"Failed to fetch data. Error: {e}")

else:
    st.info("👆 Select a city above to load weather data and generate the digital twin.")

# ─────────────────────────────────────────────────────────────────────────────
# 10. SMARTFARM AI CHAT — full-width at the bottom
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="ai-section-header">
    <span style="font-size:1.35rem; font-weight:800;">🌿 Smartfarm AI</span>
    <span class="ai-badge">LIVE</span>
</div>
""", unsafe_allow_html=True)

ctx = st.session_state.get("weather_ctx")
if ctx:
    st.caption(
        f"Answering based on live data for **{ctx['city']}** — "
        f"{ctx['avg_temp']}°C · {int(ctx['rain_prob'])}% rain · {ctx['max_wind']} km/h wind"
    )
else:
    st.caption("Select a city above to unlock location-specific farming advice.")

# ── Suggestion pills — 4 pills in a single row ──
st.markdown("<p style='font-size:0.8rem;color:#9ca3af;margin:6px 0 10px;'>💡 Quick questions — click to ask:</p>", unsafe_allow_html=True)
pill_cols = st.columns(4)
for idx, suggestion in enumerate(SUGGESTIONS):
    with pill_cols[idx]:
        if st.button(suggestion, key=f"pill_{idx}", use_container_width=True):
            st.session_state.pending_question = suggestion

# ── Chat history ──
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar="🌿" if msg["role"] == "assistant" else None):
        st.markdown(msg["content"])

# ── Chat input ──
user_input = st.chat_input(
    placeholder="e.g. Should I irrigate today?",
    key="main_chat_input",
)

# Handle pill injection
if st.session_state.pending_question and not user_input:
    user_input = st.session_state.pending_question
    st.session_state.pending_question = ""

if user_input:
    if not ctx:
        st.warning("⚠️ Please select a city above first so I can give you location-specific advice!")
    else:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("assistant", avatar="🌿"):
            with st.spinner("Smartfarm AI is analysing field parameters…"):
                answer = generate_ai_response(
                    question=user_input,
                    city=ctx["city"],
                    avg_temp=ctx["avg_temp"],
                    rain_prob=ctx["rain_prob"],
                    max_wind=ctx["max_wind"],
                    avg_hum=ctx["avg_hum"],
                    current_crop=ctx["current_crop"],
                )
            st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})


# ─────────────────────────────────────────────────────────────────────────────
# 9. DETAILED WEATHER  (full-width below, only when city is loaded)
# ─────────────────────────────────────────────────────────────────────────────
_forecast     = st.session_state.get("_forecast")
_day_forecast = st.session_state.get("_day_forecast")

if city and _forecast and _day_forecast:
    avg_temp  = st.session_state.weather_ctx["avg_temp"]
    avg_hum   = st.session_state.weather_ctx["avg_hum"]
    max_wind  = st.session_state.weather_ctx["max_wind"]
    rain_prob = st.session_state.weather_ctx["rain_prob"]

    st.markdown("---")

    # ── Hourly forecast ──
    st.markdown(f"### 🕒 Hourly Forecast for {target_date.strftime('%b %d')}")
    cols = st.columns(len(_day_forecast))
    for idx, hour in enumerate(_day_forecast):
        with cols[idx]:
            time_str  = datetime.strptime(hour["dt_txt"], "%Y-%m-%d %H:%M:%S").strftime("%I %p")
            icon_code = hour["weather"][0]["icon"]
            st.markdown(f"""
                <div class='hourly-item'>
                    <p style='font-size:0.78rem; margin:0;'>{time_str}</p>
                    <img src="http://openweathermap.org/img/wn/{icon_code}.png" width="38">
                    <p style='font-weight:bold; margin:0;'>{int(hour['main']['temp'])}°</p>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Crop recommendation + Activity guide ──
    rec_col, advice_col = st.columns([1, 1.5])
    with rec_col:
        crop_name, crop_reason = get_crop_recommendation(avg_temp, rain_prob)
        st.markdown(f"""
        <div class='ai-card'>
            <p style='margin:0; font-size:0.88rem;'>💡 <b>AI Recommendation</b></p>
            <h2 style='color:#4CAF50; margin:6px 0 4px;'>{crop_name}</h2>
            <p style='font-size:0.83rem; color:#aaa; margin:0;'>{crop_reason}</p>
        </div>
        """, unsafe_allow_html=True)

    with advice_col:
        if current_crop != "None":
            st.subheader(f"🛠️ Activity Guide for {current_crop}")
            if max_wind < 10 and rain_prob < 20:
                st.success("✅ Perfect window for **Pesticide Spraying** (low wind).")
            else:
                st.error("❌ Delay **Spraying** — weather is too volatile.")
            if rain_prob > 50:
                st.warning("🚱 **Skip Irrigation:** Rain will provide sufficient moisture.")
            else:
                st.info("💧 Routine irrigation recommended based on soil dryness.")
        else:
            st.info("Select your current crop in the sidebar for specific field activity advice.")

    # ── 5-day trend ──
    st.markdown("### 📈 5-Day Ecosystem Trend")
    df = pd.DataFrame([{
        "Time": datetime.fromtimestamp(i["dt"]),
        "Temperature (°C)": i["main"]["temp"],
        "Humidity (%)":     i["main"]["humidity"],
    } for i in _forecast])
    fig = px.line(
        df, x="Time", y=["Temperature (°C)", "Humidity (%)"],
        color_discrete_sequence=["#4CAF50", "#2196F3"],
        template="plotly_dark",
    )
    fig.update_xaxes(tickformat="%a, %d %b", title=None, nticks=10)
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=30, b=0), height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
    )
    st.plotly_chart(fig, use_container_width=True)