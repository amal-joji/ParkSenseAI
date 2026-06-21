# 🏎 ParkSense AI

> **Intelligent Parking Hotspot Detection and Congestion Risk Prediction System**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green.svg)
![Status](https://img.shields.io/badge/Status-Hackathon_Ready-success.svg)

## 📌 Overview
Urban congestion and illegal parking are escalating challenges for city traffic management. **ParkSense AI** is an intelligent, data-driven dashboard designed to help traffic authorities and city planners predict congestion risks, identify chronic parking hotspots, and optimize the deployment of traffic officers. 

Featuring a sleek, Mercedes AMG Telemetry-inspired dark mode interface, the platform turns complex traffic datasets into actionable, localized insights.

## ✨ Key Features
* **🤖 Predictive Risk Modeling:** Utilizes a trained XGBoost classifier to predict congestion risk (High, Medium, Low) based on variables like location, time, and vehicle type.
* **🗺️ Dynamic Hotspot Heatmaps:** Employs MiniBatch KMeans clustering and Folium to generate interactive spatial heatmaps of parking violations.
* **🚔 Smart Deployment Recommendations:** Automatically calculates and suggests the exact number of officers required at specific junctions based on historical violation volume.
* **🔍 Model Explainability:** Integrates SHAP (SHapley Additive exPlanations) to provide transparent insights into which features are driving the AI's risk predictions.
* **🌐 Native Bilingual Support:** Fully localized in both **English and Kannada (ಕನ್ನಡ)** via a custom internal session-state architecture, ensuring accessibility for regional traffic authorities.
* **🏎️ AMG-Inspired UI:** A high-performance, glass-morphism aesthetic featuring live metric gauges and responsive visual telemetry.

## 🛠️ Tech Stack
* **Frontend/UI:** Streamlit, Streamlit-Option-Menu
* **Machine Learning:** Scikit-Learn (KMeans), XGBoost
* **Data Visualization:** Plotly Express, Plotly Graph Objects, Folium, Streamlit-Folium
* **Explainable AI:** SHAP
* **Data Manipulation:** Pandas, NumPy, Joblib

## 🚀 Running Locally

**1. Clone the repository**
```bash
git clone [https://github.com/yourusername/ParkSenseAI.git](https://github.com/yourusername/ParkSenseAI.git)
cd ParkSenseAI
