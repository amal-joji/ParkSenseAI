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
# LOAD DATA
# =======================

df = pd.read_csv(
    "parking_clustered.csv"
)

df['created_datetime'] = pd.to_datetime(
    df['created_datetime'],
    format='mixed',
    errors='coerce'
)

df['hour'] = df['created_datetime'].dt.hour
df['month'] = df['created_datetime'].dt.month
from sklearn.cluster import MiniBatchKMeans

coords = df[['latitude','longitude']].dropna()

kmeans = MiniBatchKMeans(
    n_clusters=30,
    random_state=42,
    batch_size=10000
)

df.loc[coords.index, 'cluster'] = kmeans.fit_predict(coords)

df['cluster'] = df['cluster'].astype(int)

# Fill missing values

df['location'] = df['location'].fillna('Unknown').astype(str)
df['vehicle_type'] = df['vehicle_type'].fillna('Unknown').astype(str)
df['junction_name'] = df['junction_name'].fillna('Unknown').astype(str)
df['police_station'] = df['police_station'].fillna('Unknown').astype(str)


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

    col1,col2,col3,col4 = st.columns(4)

    with col1:
        st.metric(TEXT[lang]['metric_violations'], f"{len(df):,}")

    with col2:

        st.metric(
            TEXT[lang]['metric_locations'],
            df['location'].nunique()
        )

    with col3:

        st.metric(
            TEXT[lang]['metric_stations'],
            df['police_station'].nunique()
        )

    with col4:

        st.metric(
            TEXT[lang]['metric_hotspots'],
            df['cluster'].nunique()
        )



    st.markdown("---")

    col1,col2 = st.columns([1,1])
    
    # ======================
    # CONGESTION GAUGE
    # ======================

    with col1:

        risk_score = 82

        fig = go.Figure(
            go.Indicator(

                mode="gauge+number",

                value=risk_score,

                title={'text': TEXT[lang]['gauge_title']},

                gauge={

                    'axis':{'range':[0,100]},

                    'bar':{'color':'cyan'},

                    'steps':[

                        {'range':[0,40],'color':'green'},

                        {'range':[40,70],'color':'orange'},

                        {'range':[70,100],'color':'red'}

                    ]

                }

            )
        )

        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='#0B0F19',
            font_color='white'
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # ======================
    # TOP HOTSPOTS
    # ======================

    with col2:

        top_locations = (
            df.groupby('location')
            .size()
            .sort_values(ascending=False)
            .head(10)
        )

        fig2 = px.bar(

            x=top_locations.values,

            y=top_locations.index,

            orientation='h',

            labels={
                'x':'Violations',
                'y':'Location'
            },

            title=TEXT[lang]['bar_title']
        )

        fig2.update_layout(
            template='plotly_dark',
            paper_bgcolor='#0B0F19',
            plot_bgcolor='#0B0F19'
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )



# ======================================================
# PREDICTION PAGE
# ======================================================

elif selected == TEXT[lang]['menu_prediction']:

    st.title(TEXT[lang]['app_title'])

    col1,col2 = st.columns(2)

    with col1:

        vehicle = st.selectbox(

            TEXT[lang]['vehicle_type'],

            sorted(
                df['vehicle_type']
                .dropna()
                .astype(str)
                .unique()
            )

        )



        location = st.selectbox(

            TEXT[lang]['location'],

            sorted(
                df['location']
                .dropna()
                .astype(str)
                .unique()
            )

        )



        junction = st.selectbox(

            TEXT[lang]['junction'],

            sorted(
                df['junction_name']
                .dropna()
                .astype(str)
                .unique()
            )

        )



        station = st.selectbox(

            TEXT[lang]['police_station'],

            sorted(
                df['police_station']
                .dropna()
                .astype(str)
                .unique()
            )

        )



        hour = st.slider(

            TEXT[lang]['hour'],

            0,

            23,

            10

        )


        month = st.slider(

            TEXT[lang]['month'],

            1,

            12,

            3

        )



    with col2:

        st.image(
            "https://cdn-icons-png.flaticon.com/512/3202/3202926.png",
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


        prediction = model.predict(input_data)[0]


        if prediction == "High":
            color = "#FF0000"
            risk = "🔴 HIGH RISK"
            officers = 3
        elif prediction == "Medium":
            color = "#FFA500"
            risk = "🟠 MEDIUM RISK"
            officers = 2
        else:
            color = "#00cc66"
            risk = "🟢 LOW RISK"
            officers = 1

        st.markdown(
            f"""
            <div style="
            background:rgba(20,27,45,0.85);
            border:2px solid {color};
            border-radius:25px;
            padding:40px;
            box-shadow:0 0 30px {color};
            text-align:center;
            ">

            <h1 style="color:{color};">{risk}</h1>

            <h2 style="color:white;">
            🚔 Deploy {officers} Officer
            </h2>
            </div>
            """,
            unsafe_allow_html=True
        )

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

    title='Violation Telemetry'

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

    st.title(TEXT[lang]['recommendations_title'])

    for _, row in recommendations.head(10).iterrows():

        st.markdown(
        f"""
        <div style="
        background:rgba(20,27,45,0.85);
        padding:20px;
        border-left:7px solid #00E5FF;
        border-radius:20px;
        margin-bottom:15px;
        box-shadow:0 0 15px rgba(0,229,255,0.2);
        ">

        <h3>📍 {row['location']}</h3>

        <p style="font-size:18px;">
        🚨 Violations : {row['violations']}
        </p>

        <p style="font-size:18px;">
        🚔 Officers Required : {row['officers_required']}
        </p>

        </div>
        """,
        unsafe_allow_html=True
        )

elif selected == TEXT[lang]['menu_feature']:

    st.title(TEXT[lang]['feature_title'])

    importance = model.feature_importances_

    features = model.feature_names_in_

    fig, ax = plt.subplots(figsize=(10,6))

    ax.barh(
        features,
        importance
    )

    ax.set_xlabel("Importance")

    ax.set_title("XGBoost Feature Importance")

    st.pyplot(fig)

elif selected == TEXT[lang]['menu_explain']:

    st.title(TEXT[lang]['explain_title'])

    sample = df[
    [
        'vehicle_type',
        'location',
        'junction_name',
        'police_station',
        'hour',
        'month'
    ]
    ].dropna().copy()

    cat_cols = [
        'vehicle_type',
        'location',
        'junction_name',
        'police_station'
    ]

    for col in cat_cols:

        sample = sample[
            sample[col].astype(str).isin(encoders[col].classes_)
        ]

        sample[col] = encoders[col].transform(
            sample[col].astype(str)
        )

    sample_shap = sample.sample(
        min(500, len(sample)),
        random_state=42
    )

    shap_values = explainer.shap_values(sample_shap)

    plt.figure(figsize=(10,6))

    shap.summary_plot(
        shap_values,
        sample_shap,
        plot_type="bar",
        show=False
    )

    st.pyplot(
        plt.gcf(),
        clear_figure=True
    )

elif selected == TEXT[lang]['menu_alerts']:

    st.title(TEXT[lang]['alerts_title'])

    top_locations = (
        df.groupby('location')
        .size()
        .sort_values(ascending=False)
        .head(6)
    )

    cols = st.columns(2)

    for i, (loc, cnt) in enumerate(top_locations.items()):

        with cols[i % 2]:

            st.markdown(
            f"""
            <div style="
            background:rgba(20,27,45,0.85);
            border:2px solid red;
            border-radius:25px;
            padding:25px;
            box-shadow:0 0 25px red;
            margin-bottom:20px;
            ">

            <h2 style="color:red;">
            ⚠ ALERT
            </h2>

            <h3>
            📍 {loc}
            </h3>

            <h4>
            🚨 {cnt} Violations
            </h4>

            <h4>
            🚔 Deploy Extra Officers
            </h4>

            </div>
            """,
            unsafe_allow_html=True
            )

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
