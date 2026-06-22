import streamlit as st

st.set_page_config(
    page_title="ParkSense AI",
    page_icon="🏎",
    layout="wide",
    initial_sidebar_state="expanded"
)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
import folium

from folium.plugins import HeatMap
from streamlit_folium import st_folium
from streamlit_option_menu import option_menu


# =======================
# MERCEDES AMG THEME
# =======================

st.markdown("""
<style>

div[data-testid="metric-container"] {

    background: rgba(20,27,45,0.75);

    backdrop-filter: blur(15px);

    border: 1px solid rgba(0,229,255,0.3);

    border-radius: 25px;

    padding: 25px;

    box-shadow:

    0 0 15px rgba(0,229,255,0.15),

    0 0 40px rgba(0,229,255,0.05);

}

div[data-testid="metric-container"]:hover{

    transform: scale(1.03);

    transition:0.3s;

}

</style>
""",unsafe_allow_html=True)


# =======================
# LOAD MODEL
# =======================

model = joblib.load("xgb_model.pkl")
encoders = joblib.load("encoders.pkl")

explainer = shap.TreeExplainer(model)


# =======================
# LOAD DATA (OPTIMIZED WITH CACHING)
# =======================

@st.cache_data
def load_and_process_data():
    # Read the data once and save it in memory cache
    data = pd.read_csv("parking_clustered.csv")
    
    data['created_datetime'] = pd.to_datetime(
        data['created_datetime'],
        format='mixed',
        errors='coerce'
    )
    
    data['hour'] = data['created_datetime'].dt.hour
    data['month'] = data['created_datetime'].dt.month
    
    from sklearn.cluster import MiniBatchKMeans
    coords_df = data[['latitude','longitude']].dropna()
    
    kmeans_obj = MiniBatchKMeans(
        n_clusters=30,
        random_state=42,
        batch_size=10000
    )
    
    data.loc[coords_df.index, 'cluster'] = kmeans_obj.fit_predict(coords_df)
    data['cluster'] = data['cluster'].astype(int)
    
    # Clean string objects
    data['location'] = data['location'].fillna('Unknown').astype(str)
    data['vehicle_type'] = data['vehicle_type'].fillna('Unknown').astype(str)
    data['junction_name'] = data['junction_name'].fillna('Unknown').astype(str)
    data['police_station'] = data['police_station'].fillna('Unknown').astype(str)
    
    return data

# Call the cached function to populate your global dataframe
df = load_and_process_data()


# =======================
# RECOMMENDATIONS TABLE
# =======================

recommendations = (
    df.groupby('location')
    .size()
    .reset_index(name='violations')
)

recommendations['officers_required'] = (
    recommendations['violations']//20 + 1
)

recommendations = recommendations.sort_values(
    'violations',
    ascending=False
)

# 1. LANGUAGE CONFIGURATION
if 'lang' not in st.session_state:
    st.session_state['lang'] = 'en' # Default to English
if 'selected_station' not in st.session_state:
    st.session_state['selected_station'] = 'All Bengaluru'

def toggle_language():
    if st.session_state['lang'] == 'en':
        st.session_state['lang'] = 'kn'
    else:
        st.session_state['lang'] = 'en'

# 2. TRANSLATION DICTIONARY
# 2. TRANSLATION DICTIONARY (Master Version)
TEXT = {
    'en': {
        # --- App Header & Menus ---
        'app_title': '🏎 ParkSense AI',
        'subtitle': 'Intelligent Parking Hotspot Detection and Congestion Risk Prediction System',
        'menu_dashboard': 'Dashboard',
        'menu_prediction': 'Prediction',
        'menu_heatmap': 'Heatmap',
        'menu_analytics': 'Analytics',
        'menu_feature': 'Feature Importance',
        'menu_explain': 'Explainability',
        'menu_recommendations': 'Recommendations',
        'menu_alerts': 'Alerts',
        'metric_peak_hour': '⚡ Peak Risk Hour',
        'dashboard_map_title': '🌐 Live Spatial Jurisdiction Vector',
        'metric_revenue': '💰 Projected Fine Recovery',
        'alerts_title': '⚠️ Live Congestion Chokepoints',
        
        # --- Dashboard Metrics & Charts ---
        'metric_violations': '🚨 Violations',
        'metric_locations': '📍 Locations',
        'metric_stations': '🚔 Stations',
        'metric_hotspots': '🔥 Hotspots',
        'gauge_title': 'Congestion Risk Score',
        'bar_title': 'Top Parking Hotspots',
        
        # --- Prediction Page Inputs ---
        'prediction_title': '🤖 Congestion Risk Prediction',
        'vehicle_type': 'Vehicle Type',
        'location': 'Location',
        'junction': 'Junction',
        'police_station': 'Police Station',
        'hour': 'Hour',
        'month': 'Month',
        'predict_btn': 'Predict Risk',
        
        # --- Other Page Titles ---
        'heatmap_title': '🗺 Heatmap',
        'recommendations_title': '🚔 Deployment Recommendations',
        'feature_title': '📈 Feature Importance',
        'explain_title': '🔍 SHAP Explainability',
        'alerts_title': '🚨 Critical Alerts Dashboard',
        'analytics_chart_title': 'Violation Telemetry',

        'footer_subtitle': 'Mercedes AMG Telemetry Inspired Dashboard',
        'footer_tech': 'XGBoost • SHAP • KMeans • Folium • Streamlit',
        'footer_built_by': 'Built by Amal Joji & Amrisha Srivastava'
    },
    'kn': {
        # --- App Header & Menus ---
        'app_title': '🏎 ಪಾರ್ಕ್‌ಸೆನ್ಸ್ ಎಐ',
        'subtitle': 'ಬುದ್ಧಿವಂತ ಪಾರ್ಕಿಂಗ್ ಹಾಟ್‌ಸ್ಪಾಟ್ ಪತ್ತೆ ಮತ್ತು ದಟ್ಟಣೆ ಅಪಾಯದ ಮುನ್ಸೂಚನೆ ವ್ಯವಸ್ಥೆ',
        'menu_dashboard': 'ಡ್ಯಾಶ್‌ಬೋರ್ಡ್',
        'menu_prediction': 'ಮುನ್ಸೂಚನೆ',
        'menu_heatmap': 'ಹೀಟ್‌ಮ್ಯಾಪ್',
        'menu_analytics': 'ಅನಾಲಿಟಿಕ್ಸ್',
        'menu_feature': 'ವೈಶಿಷ್ಟ್ಯದ ಪ್ರಾಮುಖ್ಯತೆ',
        'menu_explain': 'ವಿವರಿಸುವಿಕೆ',
        'menu_recommendations': 'ಶಿಫಾರಸುಗಳು',
        'menu_alerts': 'ಎಚ್ಚರಿಕೆಗಳು',
        'metric_peak_hour': '⚡ ಗರಿಷ್ಠ ದಟ್ಟಣೆ ಸಮಯ',
        'dashboard_map_title': '🌐 ಲೈವ್ ಪ್ರಾದೇಶಿಕ ನಕ್ಷೆ',
        'metric_revenue': '💰 ಅಂದಾಜು ದಂಡ ವಸೂಲಿ',
        'alerts_title': '⚠️ ಲೈವ್ ಸಂಚಾರ ದಟ್ಟಣೆ ಕೇಂದ್ರಗಳು',
        
        # --- Dashboard Metrics & Charts ---
        'metric_violations': '🚨 ಉಲ್ಲಂಘನೆಗಳು',
        'metric_locations': '📍 ಸ್ಥಳಗಳು',
        'metric_stations': '🚔 ನಿಲ್ದಾಣಗಳು',
        'metric_hotspots': '🔥 ಹಾಟ್‌ಸ್ಪಾಟ್‌ಗಳು',
        'gauge_title': 'ದಟ್ಟಣೆ ಅಪಾಯದ ಸ್ಕೋರ್',
        'bar_title': 'ಪ್ರಮುಖ ಪಾರ್ಕಿಂಗ್ ಹಾಟ್‌ಸ್ಪಾಟ್‌ಗಳು',
        
        # --- Prediction Page Inputs ---
        'prediction_title': '🤖 ದಟ್ಟಣೆ ಅಪಾಯದ ಮುನ್ಸೂಚನೆ',
        'vehicle_type': 'ವಾಹನದ ಪ್ರಕಾರ',
        'location': 'ಸ್ಥಳ',
        'junction': 'ಜಂಕ್ಷನ್',
        'police_station': 'ಪೊಲೀಸ್ ಠಾಣೆ',
        'hour': 'ಗಂಟೆ',
        'month': 'ತಿಂಗಳು',
        'predict_btn': 'ಅಪಾಯವನ್ನು ಊಹಿಸಿ',
        
        # --- Other Page Titles ---
        'heatmap_title': '🗺 ಹೀಟ್‌ಮ್ಯಾಪ್',
        'recommendations_title': '🚔 ನಿಯೋಜನೆ ಶಿಫಾರಸುಗಳು',
        'feature_title': '📈 ವೈಶಿಷ್ಟ್ಯದ ಪ್ರಾಮುಖ್ಯತೆ',
        'explain_title': '🔍 SHAP ವಿವರಿಸುವಿಕೆ',
        'alerts_title': '🚨 ನಿರ್ಣಾಯಕ ಎಚ್ಚರಿಕೆಗಳ ಡ್ಯಾಶ್‌ಬೋರ್ಡ್',
        'analytics_chart_title': 'ಉಲ್ಲಂಘನೆ ಟೆಲಿಮೆಟ್ರಿ',

        'footer_subtitle': 'ಮರ್ಸಿಡಿಸ್ ಎಎಂಜಿ ಟೆಲಿಮೆಟ್ರಿ ಪ್ರೇರಿತ ಡ್ಯಾಶ್‌ಬೋರ್ಡ್',
        'footer_tech': 'XGBoost • SHAP • KMeans • Folium • Streamlit',
        'footer_built_by': 'ಅಮಲ್ ಜೋಜಿ ಮತ್ತು ಅಮ್ರೀಶಾ ಶ್ರೀವಾಸ್ತವ ಅವರಿಂದ ನಿರ್ಮಿಸಲಾಗಿದೆ'
    }
}
# Helper variable
lang = st.session_state['lang']

st.markdown(
f"""
<div style='position: fixed; top: 15px; right: 15px; opacity: 0.4; color: #ffffff; font-size: 14px; font-weight: 300; z-index: 9999;'>
    Amal Joji
</div>

<h1 style='text-align:center; color:#00E5FF;'>
    {TEXT[lang]['app_title']}
</h1>

<h4 style='text-align:center; color:white;'>
    {TEXT[lang]['subtitle']}
</h4>
""",
unsafe_allow_html=True
)

# =======================
# SIDEBAR
# =======================
# =======================
# SIDEBAR
# =======================
with st.sidebar:
    # Language Toggle Button
    button_text = "ಕನ್ನಡದಲ್ಲಿ ವೀಕ್ಷಿಸಿ (View in Kannada)" if lang == 'en' else "View in English"
    st.button(button_text, on_click=toggle_language, use_container_width=True)

# Now pass the translated list to your option_menu
selected = option_menu(
    menu_title=None,
    options=[
        TEXT[lang]['menu_dashboard'], 
        TEXT[lang]['menu_prediction'], 
        TEXT[lang]['menu_heatmap'],
        TEXT[lang]['menu_analytics'],
        TEXT[lang]['menu_feature'],
        TEXT[lang]['menu_explain'],
        TEXT[lang]['menu_recommendations'],
        TEXT[lang]['menu_alerts']
    ],


    icons=[
        "speedometer2",
        "cpu",
        "geo-alt",
        "bar-chart",
        "graph-up",
        "eye",
        "shield-check",
        "exclamation-triangle"
    ],

    default_index=0,

    orientation="horizontal",

    styles={
        "container": {
            "padding": "0!important",
            "background-color": "#0B0F19"
        },

        "icon": {
            "color": "#00E5FF",
            "font-size": "18px"
        },

        "nav-link": {
            "font-size": "16px",
            "text-align": "center",
            "margin": "2px",
            "color": "white",
            "--hover-color": "#111827"
        },

        "nav-link-selected": {
            "background-color": "#00E5FF",
            "color": "black",
            "font-weight": "bold"
        }
    }
)


# ======================================================
# DASHBOARD
# ======================================================

if selected == TEXT[lang]['menu_dashboard']:

    st.title(TEXT[lang]['app_title'])

    st.markdown(
        f"""
        <h4 style='text-align:center; color:white;'>
            {TEXT[lang]['subtitle']}
        </h4>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # 1. INTERACTIVE FILTER: Bound cleanly to global session state
    station_options = ["All Bengaluru"] + sorted(list(df['police_station'].dropna().unique()))
    
    # Track the current state's array index to prevent visual resetting on click
    try:
        current_index = station_options.index(st.session_state['selected_station'])
    except ValueError:
        current_index = 0

    selected_station = st.selectbox(
        "🔍 Filter Dashboard by Police Jurisdiction / ಪೊಲೀಸ್ ಠಾಣೆ", 
        options=station_options,
        index=current_index
    )
    
    # Save the choice globally across tabs
    st.session_state['selected_station'] = selected_station

    # 2. DATA FILTERING LOGIC
    if st.session_state['selected_station'] != "All Bengaluru":
        dashboard_df = df[df['police_station'] == st.session_state['selected_station']]
    else:
        dashboard_df = df

    # 3. SUGGESTION 3 CALCULATION: Calculate Peak Enforcement Hours dynamically
    if len(dashboard_df) > 0:
        peak_hour_raw = dashboard_df['hour'].mode()[0]
        # Format the hour nicely to a 12-hour AM/PM string representation
        peak_hour_str = f"{peak_hour_raw}:00 AM" if peak_hour_raw < 12 else f"{peak_hour_raw-12 if peak_hour_raw > 12 else 12}:00 PM"
    else:
        peak_hour_str = "N/A"

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. METRIC CARDS (Now showing the Peak Risk Hour instead of redundant station counts)
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(TEXT[lang]['metric_violations'], f"{len(dashboard_df):,}")

    with col2:
        st.metric(
            TEXT[lang]['metric_locations'],
            dashboard_df['location'].nunique()
        )

    with col3:
        # IMPLEMENTED SUGGESTION 3: Replaced redundant station counter with live peak hour text
        st.metric(TEXT[lang]['metric_peak_hour'], peak_hour_str)

    with col4:
        st.metric(
            TEXT[lang]['metric_hotspots'],
            dashboard_df['cluster'].nunique()
        )
    with col5:
        # NEW: Compute dynamic financial forecasting based on ₹500 base fine rate
        projected_revenue = len(dashboard_df) * 500
        st.metric(TEXT[lang]['metric_revenue'], f"₹{projected_revenue:,}")

    st.markdown("---")

    col1, col2 = st.columns([1, 1])
    
    # 5. DYNAMIC CONGESTION GAUGE CALCULATION (Filtered by selection)
    with col1:
        local_recs = dashboard_df.groupby('location').size().reset_index(name='violations')
        avg_violations = local_recs['violations'].mean() if len(local_recs) > 0 else 0
        critical_zones = local_recs[local_recs['violations'] > (avg_violations * 1.5)] if avg_violations > 0 else []
        
        if len(local_recs) > 0:
            calculated_score = (len(critical_zones) / len(local_recs)) * 100
            risk_score = min(100, max(15, int(calculated_score * 4)))  
        else:
            risk_score = 0

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=risk_score,
                title={'text': TEXT[lang]['gauge_title']},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': 'cyan'},
                    'steps': [
                        {'range': [0, 40], 'color': 'green'},
                        {'range': [40, 70], 'color': 'orange'},
                        {'range': [70, 100], 'color': 'red'}
                    ]
                }
            )
        )

        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='#0B0F19',
            font_color='white',
            margin=dict(t=40, b=10, l=30, r=30)
        )

        st.plotly_chart(fig, use_container_width=True)


    # 6. DYNAMIC BAR CHART
    with col2:
        top_locations = (
            dashboard_df.groupby('location')
            .size()
            .sort_values(ascending=False)
            .head(10)
        )

        fig2 = px.bar(
            x=top_locations.values,
            y=top_locations.index,
            orientation='h',
            labels={
                'x': 'Violations',
                'y': 'Location'
            },
            title=f"{TEXT[lang]['bar_title']} ({selected_station})"
        )

        fig2.update_layout(
            template='plotly_dark',
            paper_bgcolor='#0B0F19',
            plot_bgcolor='#0B0F19',
            yaxis={'autorange': 'reversed'} 
        )

        st.plotly_chart(fig2, use_container_width=True)

    # 7. IMPLEMENTED SUGGESTION 2: Live Heatmap Preview below the main data charts
    st.markdown("---")
    st.markdown(f"### {TEXT[lang]['dashboard_map_title']}")

    # Compute dynamic geographical coordinate focal centers based on filtered data
    if len(dashboard_df) > 0:
        center = [dashboard_df['latitude'].mean(), dashboard_df['longitude'].mean()]
        zoom_level = 13 if selected_station != "All Bengaluru" else 11
    else:
        center = [12.9716, 77.5946] # Fallback default center for Bengaluru
        zoom_level = 11

    # Render an interactive, beautifully synced spatial heatmap preview canvas
    m_mini = folium.Map(location=center, zoom_start=zoom_level, tiles="cartodbpositron")

    heat_data_mini = dashboard_df[['latitude', 'longitude']].dropna().values.tolist()
    if heat_data_mini:
        HeatMap(heat_data_mini, radius=8, blur=12).add_to(m_mini)

    st_folium(m_mini, width=1400, height=350, key="dashboard_map_preview")

# ======================================================
# PREDICTION PAGE
# ======================================================

elif selected == TEXT[lang]['menu_prediction']:

    st.title(TEXT[lang]['app_title'])

    col1, col2 = st.columns(2)

    with col1:
        # 1. READ GLOBAL FILTER STATE: Check if a station was already selected on the dashboard
        active_station = st.session_state.get('selected_station', 'All Bengaluru')
        
        station_options = sorted(list(df['police_station'].dropna().astype(str).unique()))
        
        # Determine the correct pre-selection index for the selectbox
        if active_station in station_options:
            default_station_idx = station_options.index(active_station)
        else:
            default_station_idx = 0 # Fallback to first option if "All Bengaluru" is active

        # Police Station acts as the master controller dropdown
        station = st.selectbox(
            TEXT[lang]['police_station'],
            options=station_options,
            index=default_station_idx
        )

        # 2. DYNAMIC GEOGRAPHIC FILTERING: Extract rows belonging strictly to this station
        prediction_filtered_df = df[df['police_station'] == station]

        # 3. BOUND DROPDOWNS: Populate options strictly from the filtered slice
        location = st.selectbox(
            TEXT[lang]['location'],
            options=sorted(prediction_filtered_df['location'].dropna().astype(str).unique())
        )

        junction = st.selectbox(
            TEXT[lang]['junction'],
            options=sorted(prediction_filtered_df['junction_name'].dropna().astype(str).unique())
        )

        vehicle = st.selectbox(
            TEXT[lang]['vehicle_type'],
            options=sorted(prediction_filtered_df['vehicle_type'].dropna().astype(str).unique())
        )

        hour = st.slider(
            TEXT[lang]['hour'],
            0, 23, 10
        )

        month = st.slider(
            TEXT[lang]['month'],
            1, 12, 3
        )

    with col2:
        st.image(
            "image.png",
            width=250
        )
    

    if st.button(TEXT[lang]['predict_btn']):


        vehicle_encoded = encoders['vehicle_type'].transform([vehicle])[0]

        location_encoded = encoders['location'].transform([location])[0]

        junction_encoded = encoders['junction_name'].transform([junction])[0]

        station_encoded = encoders['police_station'].transform([station])[0]



        input_data = pd.DataFrame(

            [[

                vehicle_encoded,

                location_encoded,

                junction_encoded,

                station_encoded,

                hour,

                month

            ]],

            columns=[

                'vehicle_type',

                'location',

                'junction_name',

                'police_station',

                'hour',

                'month'

            ]

        )


        # Get the probabilities for [Low, Medium, High]
        probabilities = model.predict_proba(input_data)[0]
        low_prob, med_prob, high_prob = probabilities[0], probabilities[1], probabilities[2]

        # Get the standard prediction code (0, 1, or 2)
        prediction = model.predict(input_data)[0]

        # Determine base parameters and localized status subheadings
        if prediction == 2: # High Risk
            color = "#FF3366" # Premium neon crimson
            risk = "🔴 HIGH RISK"
            officers = int(high_prob * 5) + 3 
            status_text = "CRITICAL ACTION REQUIRED / ಗರಿಷ್ಠ ಅಪಾಯ"
            bg_glow = "rgba(255, 51, 102, 0.2)"
        elif prediction == 1: # Medium Risk
            color = "#FFA500" # Safety neon amber
            risk = "🟠 MEDIUM RISK"
            officers = int(med_prob * 2) + 1 
            status_text = "MODERATE MONITORING REQUIRED / ಮಧ್ಯಮ ಅಪಾಯ"
            bg_glow = "rgba(255, 165, 0, 0.2)"
        else: # Low Risk
            color = "#00FF66" # Premium neon emerald
            risk = "🟢 LOW RISK"
            officers = 1
            status_text = "NORMAL COMPLIANCE DETECTED / ಕಡಿಮೆ ಅಪಾಯ"
            bg_glow = "rgba(0, 255, 102, 0.15)"

        # FIXED: Built using clean continuous string concatenation to avoid markdown indentation errors
        prediction_card = (
            f'<div style="background: rgba(15, 22, 38, 0.95); padding: 35px; border: 2px solid {color}; '
            f'border-radius: 20px; box-shadow: 0 0 25px {bg_glow}; text-align: center; margin-top: 25px; width: 100%;">'
            f'<span style="color: #aaaaaa; font-size: 13px; text-transform: uppercase; letter-spacing: 2px; font-weight: bold;">'
            f'AI Risk Engine Evaluation Matrix</span>'
            f'<h1 style="color: white; margin: 15px 0 5px 0; font-weight: 800; font-size: 32px;">'
            f'{risk} <span style="font-size: 22px; color: #aaaaaa; font-weight: 400;">({max(probabilities)*100:.1f}%)</span>'
            f'</h1>'
            f'<p style="color: {color}; margin: 0 0 20px 0; font-size: 15px; font-weight: bold; letter-spacing: 0.5px;">⚠️ {status_text}</p>'
            f'<hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.1); margin: 20px 0;">'
            f'<h2 style="color: white; margin: 0; font-size: 24px; font-weight: 700;">'
            f'🚔 Recommended Deployment: <span style="color: {color};">{officers} Officer(s)</span>'
            f'</h2>'
            f'</div>'
        )
        
        st.markdown(prediction_card, unsafe_allow_html=True)

elif selected == TEXT[lang]['menu_analytics']:
    hourly = df.groupby(
    'hour'
    ).size().reset_index(
    name='violations'
    )

    fig = px.line(

    hourly,

    x='hour',

    y='violations',

    markers=True

    )

    fig.update_traces(

    line_color='#00E5FF',

    marker_color='#00E5FF'

    )

    fig.update_layout(

    template='plotly_dark',

    paper_bgcolor='#0B0F19',

    plot_bgcolor='#0B0F19',

    font_color='white',

    title=TEXT[lang]['analytics_chart_title']

    )

    st.plotly_chart(

    fig,

    use_container_width=True
    )


    vehicle_counts = (
    df['vehicle_type']
    .value_counts()
    .head(5)
    )

    fig = px.pie(

    values=vehicle_counts.values,

    names=vehicle_counts.index,

    hole=0.7

    )

    fig.update_layout(

    template='plotly_dark',

    paper_bgcolor='#0B0F19'

    )

    st.plotly_chart(

    fig,

    use_container_width=True
    )

elif selected == TEXT[lang]['menu_heatmap']:

    st.title(TEXT[lang]['heatmap_title'])

    center = [
        df['latitude'].mean(),
        df['longitude'].mean()
    ]

    m = folium.Map(
        location=center,
        zoom_start=11
    )

    heat_data = (
        df[['latitude','longitude']]
        .dropna()
        .values
        .tolist()
    )

    HeatMap(
        heat_data,
        radius=10,
        blur=15
    ).add_to(m)

    hotspots = (
        df.groupby('cluster')
        [['latitude','longitude']]
        .mean()
    )

    for idx,row in hotspots.iterrows():

        folium.Marker(
            [row.latitude,row.longitude],
            popup=f"Hotspot {idx}",
            icon=folium.Icon(color='red')
        ).add_to(m)

    st_folium(
        m,
        width=1200,
        height=600
    )

elif selected == TEXT[lang]['menu_recommendations']:

    # Retrieve the global filter setting from session state
    active_station = st.session_state.get('selected_station', 'All Bengaluru')

    # Update the title dynamically based on selection scope
    st.title(f"{TEXT[lang]['recommendations_title']} ({active_station})")

    # Re-compile recommendations strictly for this active jurisdiction scope
    if active_station != "All Bengaluru":
        filtered_rec_df = df[df['police_station'] == active_station]
    else:
        filtered_rec_df = df

    local_recommendations = (
        filtered_rec_df.groupby('location')
        .size()
        .reset_index(name='violations')
    )
    local_recommendations['officers_required'] = (local_recommendations['violations'] // 20) + 1
    local_recommendations = local_recommendations.sort_values('violations', ascending=False)

    # Prepare the CSV data payload based on the dynamically compiled table view
    csv_data = local_recommendations.head(10).to_csv(index=False).encode('utf-8')

    # Download Button
    st.download_button(
        label=f"📥 Export Deployment Schedule for {active_station} (CSV)",
        data=csv_data,
        file_name=f"deployment_schedule_{active_station.lower().replace(' ', '_')}.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

# Render cards in a perfectly stable, un-squishable single horizontal line per entry
    if len(local_recommendations) > 0:
        for _, row in local_recommendations.head(10).iterrows():
            # FIXED: Built as a continuous string with zero leading whitespace blocks to bypass Markdown code rules
            html_card = (
                f'<div style="display: flex; justify-content: space-between; align-items: center; '
                f'background: rgba(20,27,45,0.85); padding: 18px 30px; border-left: 7px solid #00E5FF; '
                f'border-radius: 16px; margin-bottom: 12px; box-shadow: 0 0 15px rgba(0,229,255,0.15); width: 100%;">'
                f'<div style="flex: 1; min-width: 0; padding-right: 20px;">'
                f'<h4 style="margin: 0; color: white; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="📍 {row["location"]}">'
                f'📍 {row["location"]}'
                f'</h4>'
                f'</div>'
                f'<div style="flex-shrink: 0; padding: 0 20px; white-space: nowrap; text-align: center;">'
                f'<span style="color: #aaaaaa; font-size: 16px;">🚨 Violations:</span>'
                f'<b style="color: white; font-size: 18px; margin-left: 5px;">{row["violations"]}</b>'
                f'</div>'
                f'<div style="flex-shrink: 0; padding-left: 20px; white-space: nowrap; text-align: right;">'
                f'<span style="color: #00E5FF; font-size: 16px;">Automated Deployment:</span>'
                f'<b style="color: #00E5FF; font-size: 18px; margin-left: 5px;">{row["officers_required"]} Officers</b>'
                f'</div>'
                f'</div>'
            )
            
            st.markdown(html_card, unsafe_allow_html=True)
    else:
        st.info("No active parking violations data captured within this precinct zone.")


elif selected == TEXT[lang]['menu_feature']:

    st.title(TEXT[lang]['feature_title'])

    # Get the raw importance scores from the model
    importance = model.feature_importances_
    features = ['vehicle_type', 'location', 'junction_name', 'police_station', 'hour', 'month']

    # Apply sleek dark mode styling to the Matplotlib canvas
    plt.style.use('dark_background')
    
    # Expanded layout width to give features maximum horizontal breathing room
    fig, ax = plt.subplots(figsize=(12, 5))
    
    fig.patch.set_facecolor('#0B0F19')
    ax.set_facecolor('#0B0F19')

    # Draw the horizontal bars using your theme's cyan color
    ax.barh(features, importance, color='#00E5FF', height=0.5)

    ax.set_xlabel("Importance Weight", color='white', fontsize=12, labelpad=10)
    ax.set_title("XGBoost Feature Importance Telemetry", color='#00E5FF', fontsize=15, fontweight='bold', pad=20)
    
    # Adjust feature y-axis labels size to ensure clean single-line reading
    ax.tick_params(axis='y', labelsize=12, colors='white')
    ax.tick_params(axis='x', colors='white')
    
    # Clean up the chart borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color((1.0, 1.0, 1.0, 0.2))
    ax.spines['bottom'].set_color((1.0, 1.0, 1.0, 0.2))

    # CRITICAL FORCE: Tells Matplotlib to cleanly space margins out to fit all text elements on one line
    plt.tight_layout()

    st.pyplot(fig)

elif selected == TEXT[lang]['menu_explain']:

    # 1. Retrieve the global filter setting from session state
    active_station = st.session_state.get('selected_station', 'All Bengaluru')

    st.title(f"🔍 SHAP Explainability Engine ({active_station})")

    # 2. DYNAMIC DATA SLICE: Filter the matrix before passing it to the explainer
    if active_station != "All Bengaluru":
        explain_filtered_df = df[df['police_station'] == active_station]
    else:
        explain_filtered_df = df

    # Prepare features matching the original training structure
    X_explain = explain_filtered_df[['vehicle_type', 'location', 'junction_name', 'police_station', 'hour', 'month']]

    # 3. SAFE SAMPLING: Handle scenarios where a precinct has fewer rows than the standard sample size
    sample_size = min(100, len(X_explain))
    
    if sample_size > 0:
        # Pull a clean random sample strictly from the active jurisdiction scope
        sample_shap = X_explain.sample(n=sample_size, random_state=42)
        
        # 4. ENCODER TRACKING: Gracefully intercept and neutralize any 'Unknown' or unseen labels
        for col in ['vehicle_type', 'location', 'junction_name', 'police_station']:
            valid_classes = set(encoders[col].classes_)
            
            # Fall back to a recognized baseline class if 'Unknown' appears
            sample_shap[col] = sample_shap[col].astype(str).apply(
                lambda x: x if x in valid_classes else encoders[col].classes_[0]
            )
            
            # Safe transform mapping execution
            sample_shap[col] = encoders[col].transform(sample_shap[col])

        # 5. SHAP ENGINE CORRELATION MATRICES (Initializes and Runs Calculations)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample_shap)  # <-- Restored calculation line!

        # 6. VISUAL RENDERING: Match your AMG dark telemetry theme
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(10, 6))
        fig.patch.set_facecolor('#0B0F19')

        shap.summary_plot(
            shap_values,
            sample_shap,
            plot_type="bar",
            show=False
        )
        
        # Customize chart boundaries and axes styling
        ax = plt.gca()
        ax.set_facecolor('#0B0F19')
        ax.set_title(f"SHAP Feature Weight Impacts for {active_station}", color='#00E5FF', fontsize=14, fontweight='bold', pad=20)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        st.pyplot(plt.gcf(), clear_figure=True)
    else:
        st.info("Insufficient data elements captured within this precinct zone to generate an XAI breakdown matrix.")

elif selected == TEXT[lang]['menu_alerts']:

    # 1. Retrieve the global filter setting from session state
    active_station = st.session_state.get('selected_station', 'All Bengaluru')

    # 2. Update the title dynamically based on selection scope
    st.title(f"{TEXT[lang]['alerts_title']} ({active_station})")

    # 3. Filter data slice dynamically based on selection
    if active_station != "All Bengaluru":
        filtered_alerts_df = df[df['police_station'] == active_station]
    else:
        filtered_alerts_df = df

    # Group and extract the top 6 absolute worst bottlenecks in this zone
    top_alerts = (
        filtered_alerts_df.groupby('location')
        .size()
        .sort_values(ascending=False)
        .head(6)
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. Render alerts using a sleek, single-line horizontal Neon Red layout
    if len(top_alerts) > 0:
        for loc, count in top_alerts.items():
            # Dynamic operational severity labeling based on count thresholds
            severity_label = "CRITICAL CHOKEPOINT" if count > 40 else "HIGH CONGESTION"
            
            # Built as a clean continuous string to guarantee perfect markdown execution
            html_alert = (
                f'<div style="display: flex; justify-content: space-between; align-items: center; '
                f'background: rgba(30, 16, 22, 0.85); padding: 18px 30px; border-left: 7px solid #FF3366; '
                f'border-radius: 16px; margin-bottom: 12px; box-shadow: 0 0 15px rgba(255, 51, 102, 0.15); width: 100%;">'
                
                f''
                f'<div style="flex: 1; min-width: 0; padding-right: 20px;">'
                f'<h4 style="margin: 0; color: white; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="🚨 {loc}">'
                f'🚨 {loc}'
                f'</h4>'
                f'</div>'
                
                f''
                f'<div style="flex-shrink: 0; padding: 0 20px; white-space: nowrap; text-align: center;">'
                f'<span style="color: #FF3366; font-size: 13px; font-weight: bold; letter-spacing: 1px;">⚠️ {severity_label}</span>'
                f'</div>'
                
                f''
                f'<div style="flex-shrink: 0; padding-left: 20px; white-space: nowrap; text-align: right;">'
                f'<span style="color: #aaaaaa; font-size: 16px;">Active Infractions:</span>'
                f'<b style="color: #FF3366; font-size: 18px; margin-left: 5px;">{count}</b>'
                f'</div>'
                f'</div>'
            )
            
            st.markdown(html_alert, unsafe_allow_html=True)
    else:
        st.info("No critical parking congestion alerts active within this jurisdiction.")

st.markdown(
f"""<div style='text-align: center; padding: 20px 10px; margin-top: 40px; border-top: 1px solid rgba(255,255,255,0.1);'>
<div style='color: #00E5FF; font-size: 18px; font-weight: bold; margin-bottom: 4px;'>
{TEXT[lang]['app_title']}
</div>
<div style='color: #aaaaaa; font-size: 14px; margin-bottom: 8px;'>
{TEXT[lang]['footer_subtitle']}
</div>
<div style='color: #888888; font-size: 12px; margin-bottom: 12px; letter-spacing: 1px;'>
{TEXT[lang]['footer_tech']}
</div>
<div style='color: #00E5FF; font-size: 13px;'>
{TEXT[lang]['footer_built_by']}
</div>
</div>""",
unsafe_allow_html=True
)
