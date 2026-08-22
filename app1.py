import streamlit as st
import joblib
import pandas as pd

# ==========================================================================
# PAGE CONFIGURATION
# ==========================================================================
st.set_page_config(
    page_title="Social Media Well-being Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================================
# MODEL & DATA LOADING  (unchanged)
# ==========================================================================
model = joblib.load("social_media_model.pkl")
df = pd.read_csv("social_media_screentime_mental_health_2026.csv")

# ==========================================================================
# CATEGORICAL MAPPINGS  (unchanged — same keys, same values, same order)
# ==========================================================================
gender_map = {'Male': 0, 'Female': 1}
df['gender'] = df['gender'].map(gender_map)

occupation_map = {
    'Student': 0,
    'Full-time employed': 1,
    'Part-time employed': 2,
    'Self-employed': 3,
    'Unemployed': 4,
    'Retired': 5}
df['occupation'] = df['occupation'].map(occupation_map)

most_used_platform_map = {
    'TikTok': 0,
    'Instagram': 1,
    'YouTube': 2,
    'LinkedIn': 3,
    'X/Twitter': 4,
    'Facebook': 5,
    'Snapchat': 6,
    'Reddit': 7}
df['most_used_platform'] = df['most_used_platform'].map(most_used_platform_map)

night_time_use_map = {
    'Never': 0,
    'Often': 1,
    'Every night': 2,
    'Sometimes': 3}
df['night_time_use'] = df['night_time_use'].map(night_time_use_map)

primary_purpose_map = {
    'Entertainment': 0,
    'News/information': 1,
    'Work/career': 2,
    'Connection with friends': 3,
    'Passing time/boredom': 4,
    'Content creation': 5}
df['primary_purpose'] = df['primary_purpose'].map(primary_purpose_map)

uses_screen_time_limits_map = {'Yes': 1, 'No': 0}
df['uses_screen_time_limits'] = df['uses_screen_time_limits'].map(uses_screen_time_limits_map)

attempted_digital_detox_map = {
    'No': 0,
    'Yes, failed': 1,
    'Yes, succeeded': 2}
df['attempted_digital_detox'] = df['attempted_digital_detox'].map(attempted_digital_detox_map)

seeks_mental_health_support_map = {
    'Yes': 0,
    'No': 1,
    'Considering it': 2}
df['seeks_mental_health_support'] = df['seeks_mental_health_support'].map(seeks_mental_health_support_map)

# ==========================================================================
# SIDEBAR — project information
# ==========================================================================
with st.sidebar:
    st.header("📱 About This Project")
    st.write(
        "A machine learning app that estimates a person's overall "
        "well-being band based on their social media habits, lifestyle, "
        "and self-reported mental health indicators."
    )

    st.divider()

    st.subheader("🧠 Model")
    st.write("Scikit-learn classifier trained on survey data, predicting one of three well-being bands: **Good**, **Moderate**, or **At Risk**.")

    st.subheader("🛠️ Technologies")
    st.markdown(
        "- Python\n"
        "- Streamlit\n"
        "- Scikit-learn\n"
        "- Pandas\n"
        "- Joblib"
    )

    st.subheader("📊 Dataset")
    st.write("Social Media Screentime & Mental Health survey (2026), covering usage patterns, mood indicators, and lifestyle factors.")

    st.divider()

    with st.expander("ℹ️ How to use this app"):
        st.write(
            "1. Fill in each section on the main page.\n"
            "2. Move through personal info, usage habits, mental health, "
            "and lifestyle questions.\n"
            "3. Click **Predict My Well-being** to see your result."
        )

    st.caption("Built as a portfolio project · Not a substitute for professional mental health advice.")
    st.divider()
    st.caption("👩‍💻 Developed by Das Purnima")
# ==========================================================================
# HEADER
# ==========================================================================
st.title("📱 Social Media Well-being Predictor")
st.write(
    "Answer a few questions about your social media habits, mindset, and "
    "lifestyle to get an estimate of your overall well-being band."
)
st.divider()

# ==========================================================================
# SECTION 1 — PERSONAL INFORMATION
# ==========================================================================
st.subheader("👤 Personal Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=13, max_value=71, value=20)

with col2:
    gender_choice = st.selectbox("Gender", ["Male", "Female"])
    gender = gender_map[gender_choice]

with col3:
    occupation_choice = st.selectbox("Occupation", occupation_map.keys())
    occupation = occupation_map[occupation_choice]

st.divider()

# ==========================================================================
# SECTION 2 — SOCIAL MEDIA USAGE
# ==========================================================================
st.subheader("📱 Social Media Usage")

col1, col2 = st.columns(2)

with col1:
    most_used_platform_choice = st.selectbox("Most Used Platform", most_used_platform_map.keys())
    most_used_platform = most_used_platform_map[most_used_platform_choice]

    platforms_used_count = st.number_input(
        "Number of Platforms Used", min_value=1, max_value=20, value=1, step=1
    )

    daily_screen_hours = st.number_input(
        "Daily Screen Time (hours)", value=1, step=1
    )

with col2:
    night_time_use_choice = st.selectbox("Night-Time Use", night_time_use_map.keys())
    night_time_use = night_time_use_map[night_time_use_choice]

    primary_purpose_choice = st.selectbox("Primary Purpose of Usage", primary_purpose_map.keys())
    primary_purpose = primary_purpose_map[primary_purpose_choice]

st.divider()

# ==========================================================================
# SECTION 3 — MENTAL HEALTH
# ==========================================================================
st.subheader("🧠 Mental Health")

col1, col2 = st.columns(2)

with col1:
    anxiety_score_0to27 = st.slider("Anxiety Score", min_value=1, max_value=27, value=1)
    low_mood_score_0to27 = st.slider("Low Mood Score", min_value=1, max_value=27, value=1)
    life_satisfaction_1to10 = st.slider("Life Satisfaction", min_value=1, max_value=10, value=1)

with col2:
    loneliness_1to10 = st.slider("Loneliness Level", min_value=1, max_value=10, value=1)
    self_esteem_1to10 = st.slider("Self-Esteem Score", min_value=1, max_value=10, value=1)
    fomo_1to10 = st.slider("FOMO Score", min_value=1, max_value=10, value=1)

social_comparison_1to10 = st.slider("Social Comparison Score", min_value=1, max_value=10, value=1)

st.divider()

# ==========================================================================
# SECTION 4 — LIFESTYLE
# ==========================================================================
st.subheader("🏃 Lifestyle")

col1, col2 = st.columns(2)

with col1:
    avg_sleep_hours = st.number_input("Average Sleep Hours", value=1, step=1)

    physical_activity_days_per_week = st.number_input(
        "Physical Activity Days per Week", min_value=1, max_value=10, value=1, step=1
    )

with col2:
    uses_screen_time_limits_choice = st.selectbox("Use Screen Time Limits?", uses_screen_time_limits_map.keys())
    uses_screen_time_limits = uses_screen_time_limits_map[uses_screen_time_limits_choice]

    attempted_digital_detox_choice = st.selectbox("Attempted Digital Detox?", attempted_digital_detox_map.keys())
    attempted_digital_detox = attempted_digital_detox_map[attempted_digital_detox_choice]

seeks_mental_health_support_choice = st.selectbox(
    "Seeking Mental Health Support?", seeks_mental_health_support_map.keys()
)
seeks_mental_health_support = seeks_mental_health_support_map[seeks_mental_health_support_choice]

st.divider()

# ==========================================================================
# PREDICTION SECTION
# ==========================================================================
st.subheader("🔮 Prediction")

predict_clicked = st.button("Predict My Well-being", type="primary", use_container_width=True)

if predict_clicked:

    # Feature order preserved exactly as in the original application
    input_data = [[
        age,
        gender,
        occupation,
        most_used_platform,
        platforms_used_count,
        daily_screen_hours,
        night_time_use,
        primary_purpose,
        avg_sleep_hours,
        anxiety_score_0to27,
        low_mood_score_0to27,
        life_satisfaction_1to10,
        loneliness_1to10,
        self_esteem_1to10,
        fomo_1to10,
        social_comparison_1to10,
        physical_activity_days_per_week,
        uses_screen_time_limits,
        attempted_digital_detox,
        seeks_mental_health_support
    ]]

    prediction = model.predict(input_data)[0]

    st.divider()
    st.subheader("📋 Result")

    if prediction == "Good":
        st.success("🟢 Well-being Band: GOOD")
        st.balloons()
        st.write("Your overall well-being appears to be good. Keep maintaining healthy social media habits!")

    elif prediction == "Moderate":
        st.warning("🟡 Well-being Band: MODERATE")
        st.write("Your well-being is moderate. Small improvements in sleep, screen time and daily routine may help.")

    elif prediction == "At Risk":
        st.error("🔴 Well-being Band: AT RISK")
        st.write(
            "Your responses indicate that your well-being may be at risk. "
            "Consider reducing excessive screen time, improving sleep and "
            "reaching out to trusted people or a mental health professional "
            "if you're struggling."
        )

    else:
        st.info(f"Prediction: {prediction}")
