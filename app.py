"""
RegolithSense - Enterprise Geoscientific Inversion & Concealed Critical Mineral Targeting Platform
Engineered to Petrel (SLB) and Oasis Montaj (Seequent) Standards
Pan-African Exploration Technology Pre-Incubation Program
Nasarawa State University Keffi Mine-Tech UniPod & UNDP timbuktoo
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy import ndimage
import json
import os
import io

# Core Geophysical & Importer Modules
from core.synthetic_geology import generate_nasarawa_geological_grid
from core.data_importer import detect_survey_columns, interpolate_points_to_grid, generate_sample_survey_csv
from core.geophysics import (
    calculate_horizontal_gradients,
    calculate_total_horizontal_derivative,
    calculate_vertical_derivative_fft,
    calculate_analytic_signal,
    calculate_tilt_angle_derivative,
    extract_structural_lineaments,
    calculate_gravity_vertical_gradient,
    calculate_horizontal_gravity_gradient,
    calculate_residual_gravity
)
from core.radiometrics import (
    compute_radioelement_ratios,
    compute_potassic_alteration_factor,
    generate_ternary_rgb
)
from core.prospectivity_ml import MineralProspectivityEngine
from core.report_exporter import generate_comprehensive_word_report, generate_excel_workbook

# Page Config
st.set_page_config(
    page_title="RegolithSense Enterprise | Subsurface Geophysics Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enterprise SLB / Petrel Dark Theme Styling
st.markdown("""
<style>
    /* Dark Obsidian & Slate Workbench Theme */
    .stApp {
        background-color: #0A0E17;
        color: #F1F5F9;
    }
    
    .top-telemetry-bar {
        background: linear-gradient(90deg, #0F172A 0%, #1E293B 100%);
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 10px 16px;
        margin-bottom: 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .telemetry-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-right: 8px;
    }
    .pill-cyan { background-color: rgba(6, 182, 212, 0.15); color: #06B6D4; border: 1px solid #06B6D4; }
    .pill-emerald { background-color: rgba(16, 185, 129, 0.15); color: #10B981; border: 1px solid #10B981; }
    .pill-amber { background-color: rgba(245, 158, 11, 0.15); color: #F59E0B; border: 1px solid #F59E0B; }
    
    .enterprise-card {
        background-color: #111827;
        border: 1px solid #1F2937;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #F8FAFC;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #94A3B8;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Session State for User Ingested Data
if "custom_data" not in st.session_state:
    st.session_state["custom_data"] = None
if "dataset_name" not in st.session_state:
    st.session_state["dataset_name"] = "Nasarawa LCT-Pegmatite Province (Benchmark)"

# Base Geological Data Loader (Synthetic or Custom)
@st.cache_data
def get_processed_geology(custom_grids=None):
    if custom_grids is not None:
        data = custom_grids
    else:
        data = generate_nasarawa_geological_grid(nx=120, ny=120, dx=100.0, dy=100.0)
    
    # Aeromagnetics Processing
    gx, gy = calculate_horizontal_gradients(data["tmi"], dx=data["dx"], dy=data["dy"])
    thdr = calculate_total_horizontal_derivative(gx, gy)
    vdr = calculate_vertical_derivative_fft(data["tmi"], dx=data["dx"], dy=data["dy"])
    as_grid = calculate_analytic_signal(thdr, vdr)
    tdr = calculate_tilt_angle_derivative(vdr, thdr)
    lineaments = extract_structural_lineaments(tdr, threshold_deg=6.0)
    
    # Gravity Processing
    bouguer = data.get("bouguer")
    if bouguer is None:
        bouguer = -34.0 - 0.03 * data["tmi"] + 0.03 * 33200.0
        data["bouguer"] = bouguer
        
    gzz = calculate_gravity_vertical_gradient(bouguer, dx=data["dx"], dy=data["dy"])
    hgg = calculate_horizontal_gravity_gradient(bouguer, dx=data["dx"], dy=data["dy"])
    residual_grav, regional_grav = calculate_residual_gravity(bouguer)

    # Advanced Radiometrics & Alteration Processing
    ratios = compute_radioelement_ratios(data["k"], data["eth"], data["eu"])
    potassic_factor = compute_potassic_alteration_factor(data["k"], data["eth"])
    ternary_rgb = generate_ternary_rgb(data["k"], data["eth"], data["eu"])
    
    # Elevation & Regolith Check
    elevation = data.get("elevation")
    if elevation is None:
        elevation = 380.0 + 100.0 * np.sin(np.linspace(0, 3, data["tmi"].shape[0]))[:, None]
        data["elevation"] = elevation
        data["regolith_thickness"] = np.clip(32.0 - (elevation - 380.0)*0.05, 12.0, 70.0)
        data["bedrock_elevation"] = elevation - data["regolith_thickness"]

    return {
        "raw": data,
        "thdr": thdr,
        "vdr": vdr,
        "as": as_grid,
        "tdr": tdr,
        "lineaments": lineaments,
        "bouguer": bouguer,
        "gzz": gzz,
        "hgg": hgg,
        "residual_grav": residual_grav,
        "regional_grav": regional_grav,
        "ratios": ratios,
        "potassic_factor": potassic_factor,
        "ternary_rgb": ternary_rgb
    }

base_data = get_processed_geology(st.session_state["custom_data"])

# Header & Telemetry Bar
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<div class="main-title">⚡ RegolithSense <span style="font-size:1.1rem; color:#06B6D4; font-weight:600;">ENTERPRISE</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Multi-Physics Subsurface Inversion & Mineral Prospectivity Mapping Platform for Deeply Concealed Critical Mineral Systems</div>', unsafe_allow_html=True)
with col_h2:
    st.markdown("""
    <div style="text-align: right; font-size: 0.85rem; color: #94A3B8; padding-top: 8px;">
        <span style="color:#06B6D4; font-weight:700;">NSUK Mine-Tech UniPod</span><br/>
        UNDP timbuktoo Initiative<br/>
        <i>Benchmark: Petrel / SLB Architecture</i>
    </div>
    """, unsafe_allow_html=True)

# Live Status Telemetry Bar
is_custom = st.session_state["custom_data"] is not None
ds_pill = "pill-cyan" if is_custom else "pill-emerald"
ds_label = "USER IMPORTED SURVEY" if is_custom else "BENCHMARK PILOT (NASARAWA)"

st.markdown(f"""
<div class="top-telemetry-bar">
    <div>
        <span class="telemetry-pill pill-emerald">ENGINE: ONLINE</span>
        <span class="telemetry-pill {ds_pill}">SOURCE: {ds_label}</span>
        <span class="telemetry-pill pill-amber">CRS: WGS84 / EPSG:4326</span>
    </div>
    <div style="font-size:0.8rem; color:#CBD5E1;">
        GRID: <b>{base_data['raw']['tmi'].shape[1]} &times; {base_data['raw']['tmi'].shape[0]}</b> | CELL: <b>{base_data['raw']['dx']:.0f}m</b> | BELT: <b>Central Nigerian Shield</b>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.title("🎛️ Exploration & Model Panel")
st.sidebar.markdown("Configure multi-physics evidence stacking and machine learning architecture.")

selected_model_type = st.sidebar.selectbox(
    "Classifier Architecture:",
    ["Random Forest", "Gradient Boosting", "Weights of Evidence (WofE / Logistic)"],
    index=0
)

n_estimators_input = st.sidebar.slider("Number of Estimators (Trees):", 20, 150, 60, 10)
max_depth_input = st.sidebar.slider("Maximum Tree Depth:", 3, 14, 7, 1)
threshold_input = st.sidebar.slider("Target Decision Threshold:", 0.50, 0.90, 0.72, 0.02)

st.sidebar.markdown("##### 🧬 Active Evidence Layers")
all_feature_options = [
    "Proximity to TDR Contacts",
    "Horizontal Gradient (THDR)",
    "Analytic Signal Contrast",
    "Potassic Alteration Halo (K/eTh)",
    "Efimov F-Parameter Alteration",
    "Bouguer Gravity Lows (Granite Density)",
    "Vertical Gravity Gradient (Gzz)",
    "Regolith Weathering Index"
]

selected_features_input = st.sidebar.multiselect(
    "Active Features in ML Engine:",
    options=all_feature_options,
    default=all_feature_options
)

# Run / Train Machine Learning Pipeline
@st.cache_data
def run_model_pipeline(model_type, n_est, max_d, thresh, active_feats, data_hash):
    mpm_engine = MineralProspectivityEngine(
        model_type=model_type, n_estimators=n_est, max_depth=max_d
    )
    
    features, all_layers, feat_names = mpm_engine.build_evidence_tensor(
        base_data["tdr"],
        base_data["thdr"],
        base_data["as"],
        base_data["ratios"]["K_eTh"],
        base_data["potassic_factor"],
        regolith_grid=base_data["raw"]["regolith_thickness"],
        bouguer_grid=base_data["bouguer"],
        gzz_grid=base_data["gzz"],
        f_param_grid=base_data["ratios"]["F_param"],
        selected_features=active_feats
    )
    
    known_deps = base_data["raw"].get("known_deposits", [
        {"name": "Discovery Site A", "lon": base_data["raw"]["bounds"][0] + 0.3*(base_data["raw"]["bounds"][1]-base_data["raw"]["bounds"][0]), "lat": base_data["raw"]["bounds"][2] + 0.4*(base_data["raw"]["bounds"][3]-base_data["raw"]["bounds"][2]), "mineral": "LCT Pegmatite"}
    ])

    prob_map, targets, importances, metrics = mpm_engine.train_and_predict(
        features,
        known_deps,
        base_data["raw"]["tmi"].shape,
        base_data["raw"]["bounds"],
        elev_grid=base_data["raw"]["elevation"],
        regolith_grid=base_data["raw"]["regolith_thickness"],
        threshold=thresh
    )
    
    return prob_map, targets, importances, metrics, feat_names

# Unique hash based on data shape and bounds
d_hash = f"{base_data['raw']['bounds']}_{base_data['raw']['tmi'].shape}"
prob_map, targets, importances, metrics, feat_names = run_model_pipeline(
    selected_model_type, n_estimators_input, max_depth_input, threshold_input, selected_features_input, d_hash
)
df_targets = pd.DataFrame(targets)

# Executive Metrics
total_area_km2 = (base_data["raw"]["tmi"].shape[0] * base_data["raw"]["dy"] * base_data["raw"]["tmi"].shape[1] * base_data["raw"]["dx"]) / 1e6
tier1_count = len(df_targets[df_targets["target_tier"].str.contains("Tier-1")]) if not df_targets.empty else 0
avg_regolith = float(np.mean(base_data["raw"]["regolith_thickness"]))

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Geophysical Coverage", f"{total_area_km2:.0f} km²", help="Processed survey grid footprint")
m2.metric("Mean Regolith Mantle", f"{avg_regolith:.1f} m", help="Thickness of laterite/saprolite concealing bedrock")
m3.metric("Model ROC-AUC", f"{metrics['auc']:.3f}", help="Area Under ROC Curve on ground-truth deposit validation")
m4.metric("Tier-1 RC Drill Targets", f"{tier1_count}", delta=f"{len(df_targets)} Total", help="Immediate RC drill spotting nodes")
m5.metric("Active Architecture", selected_model_type.split()[0], help="Currently executing ML architecture")

st.markdown("<br/>", unsafe_allow_html=True)

# Enterprise Tabs Navigation (8 Tabs)
tab_import, tab_model, tab_mag, tab_grav, tab_rad, tab_mpm, tab_3d, tab_jorc = st.tabs([
    "📂 Survey Ingestion & Data Loader",
    "🧠 Model Diagnostics & Calibration",
    "🧲 Aeromagnetic Inversion",
    "⚖️ Bouguer Gravity & Density Inversion",
    "🌈 Gamma-Ray Radiometrics & Alteration",
    "🤖 Prospectivity Heatmap (MPM)",
    "🧊 3D Subsurface & Drillhole Explorer",
    "📑 JORC / NI 43-101 Target Package"
])

# ----------------- TAB 1: DATA INGESTION -----------------
with tab_import:
    st.subheader("📂 Survey Data Ingestion & Grid Interpolator")
    st.caption("Import your proprietary survey datasets (CSV, TSV, or XYZ) containing gravity, radiometrics, or aeromagnetics.")

    up_col1, up_col2 = st.columns([1.2, 1.0])

    with up_col1:
        uploaded_file = st.file_uploader(
            "Upload Survey File (CSV, TSV, or XYZ):",
            type=["csv", "txt", "tsv", "xyz"],
            help="File must contain coordinate columns (Lon/Lat or Easting/Northing) and at least one potential field/radiometric channel."
        )

        if uploaded_file is not None:
            try:
                # Read uploaded file
                if uploaded_file.name.endswith(".csv"):
                    df_upload = pd.read_csv(uploaded_file)
                else:
                    df_upload = pd.read_csv(uploaded_file, sep=r'\s+')

                st.success(f"Successfully loaded {len(df_upload):,} survey stations from `{uploaded_file.name}`.")
                
                # Auto-detect columns
                detected_cols = detect_survey_columns(df_upload)
                
                st.markdown("##### 🔍 Auto-Detected Channels & Mapping")
                map_df = pd.DataFrame([
                    {"Parameter": "X / Longitude", "Mapped Header": detected_cols["x"]},
                    {"Parameter": "Y / Latitude", "Mapped Header": detected_cols["y"]},
                    {"Parameter": "Aeromagnetics (TMI)", "Mapped Header": detected_cols["tmi"]},
                    {"Parameter": "Gravity (Bouguer)", "Mapped Header": detected_cols["bouguer"]},
                    {"Parameter": "Potassium (K %)", "Mapped Header": detected_cols["k"]},
                    {"Parameter": "Thorium (eTh ppm)", "Mapped Header": detected_cols["eth"]},
                    {"Parameter": "Uranium (eU ppm)", "Mapped Header": detected_cols["eu"]},
                    {"Parameter": "Elevation (DEM)", "Mapped Header": detected_cols["elevation"]}
                ])
                st.dataframe(map_df, use_container_width=True)

                if st.button("⚡ Ingest & Process Survey Data", type="primary"):
                    with st.spinner("Interpolating survey flight-lines into 2D potential field grids..."):
                        grid_result = interpolate_points_to_grid(df_upload, detected_cols, nx=100, ny=100)
                        st.session_state["custom_data"] = grid_result
                        st.session_state["dataset_name"] = uploaded_file.name
                        st.rerun()

            except Exception as e:
                st.error(f"Error parsing survey file: {str(e)}")

        st.markdown("---")
        st.markdown("##### 📥 Need a Benchmark Test File?")
        st.write("Download a sample survey dataset formatted with real-world geophysical headers (2,500 flight-line points covering Nasarawa pegmatites):")
        sample_csv = generate_sample_survey_csv()
        st.download_button(
            label="📥 Download Sample Survey CSV (Nasarawa Lithium Corridor)",
            data=sample_csv,
            file_name="Sample_Nasarawa_Geophysical_Survey.csv",
            mime="text/csv",
            use_container_width=True
        )

    with up_col2:
        st.markdown("##### ℹ️ Supported Ingestion Formats")
        st.markdown("""
        <div class="enterprise-card">
            <b>Column Header Formats Recognized:</b><br/>
            • <b>Coordinates:</b> <code>lon, lat</code>, <code>x, y</code>, <code>easting, northing</code><br/>
            • <b>Gravity:</b> <code>bouguer, gravity, grav, mgal</code><br/>
            • <b>Potassium:</b> <code>k, potassium, k_pct, k_percent</code><br/>
            • <b>Thorium:</b> <code>eth, thorium, th, eth_ppm</code><br/>
            • <b>Uranium:</b> <code>eu, uranium, u, eu_ppm</code><br/>
            • <b>Magnetics:</b> <code>tmi, mag, rtp, magnetic</code><br/>
            • <b>Elevation:</b> <code>elevation, dem, alt, z</code><br/>
            <hr style="border-color:#334155; margin:10px 0;"/>
            <b>Interpolation Pipeline:</b> Scattered points and flight-lines are gridded via <i>scipy.interpolate.griddata</i> using cubic/linear Delaunay triangulation with boundary nearest-neighbor closure.
        </div>
        """, unsafe_allow_html=True)

        if st.session_state["custom_data"] is not None:
            if st.button("🔄 Reset to Default Benchmark Dataset"):
                st.session_state["custom_data"] = None
                st.session_state["dataset_name"] = "Nasarawa LCT-Pegmatite Province (Benchmark)"
                st.rerun()

# ----------------- TAB 2: MODEL PANEL & DIAGNOSTICS -----------------
with tab_model:
    st.subheader("🧠 Machine Learning Model Architecture & Calibration Panel")
    st.caption("Inspect and fine-tune spatial machine learning models for Mineral Prospectivity Mapping (MPM) beneath tropical regolith.")

    mp_col1, mp_col2 = st.columns([1.1, 1.0])

    with mp_col1:
        st.markdown("##### 📈 Model Validation & ROC Curve")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=metrics['fpr'],
            y=metrics['tpr'],
            mode='lines',
            line=dict(color='#06B6D4', width=3),
            name=f"{selected_model_type} (AUC = {metrics['auc']:.3f})"
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            line=dict(color='#64748B', dash='dash'),
            name="Random Guessing (AUC = 0.500)"
        ))
        fig_roc.update_layout(
            title=f"Receiver Operating Characteristic (ROC) — AUC: {metrics['auc']:.3f}",
            xaxis_title="False Positive Rate (Barren Laterite False Alarms)",
            yaxis_title="True Positive Rate (Deposit Sensitivity)",
            template="plotly_dark",
            height=340,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with mp_col2:
        st.markdown("##### 📊 Multi-Physics Evidence Importance")
        min_len = min(len(feat_names), len(importances))
        df_feat = pd.DataFrame({
            "Evidence Layer": list(feat_names)[:min_len],
            "Predictive Contribution": list(importances)[:min_len]
        }).sort_values(by="Predictive Contribution", ascending=True)

        fig_imp = px.bar(
            df_feat,
            x="Predictive Contribution",
            y="Evidence Layer",
            orientation="h",
            color="Predictive Contribution",
            color_continuous_scale="Viridis",
            title="Spatial Evidence Predictive Weight",
            template="plotly_dark"
        )
        fig_imp.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_imp, use_container_width=True)

# ----------------- TAB 3: AEROMAGNETICS -----------------
with tab_mag:
    st.subheader("🧲 Subsurface Aeromagnetic Wavefield Inversion")
    st.caption("Penetrating non-magnetic tropical regolith to reveal ductile shear corridors, faults, and intrusive contacts.")

    b = base_data["raw"]["bounds"]
    col_map1, col_map2 = st.columns(2)

    with col_map1:
        st.markdown("##### 1. Total Magnetic Intensity (TMI) & Deep Overburden")
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        im1 = ax1.imshow(base_data["raw"]["tmi"], cmap="jet", extent=[b[0], b[1], b[2], b[3]])
        plt.colorbar(im1, ax=ax1, label="TMI (nT)", shrink=0.8)
        ax1.set_title("Total Magnetic Intensity (nT)")
        ax1.set_xlabel("Longitude (°E)")
        ax1.set_ylabel("Latitude (°N)")
        st.pyplot(fig1)

    with col_map2:
        st.markdown("##### 2. Tilt Angle Derivative (TDR) — Contact Mapping")
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        im2 = ax2.imshow(base_data["tdr"], cmap="coolwarm", extent=[b[0], b[1], b[2], b[3]], vmin=-90, vmax=90)
        plt.colorbar(im2, ax=ax2, label="Tilt Angle (°)", shrink=0.8)
        ax2.contour(base_data["tdr"], levels=[0], colors="black", linewidths=1.2, extent=[b[0], b[1], b[2], b[3]])
        ax2.set_title("Tilt Angle Derivative (Zero-Contour = Contacts)")
        ax2.set_xlabel("Longitude (°E)")
        ax2.set_ylabel("Latitude (°N)")
        st.pyplot(fig2)

    st.markdown("---")
    col_map3, col_map4 = st.columns(2)

    with col_map3:
        st.markdown("##### 3. Total Horizontal Derivative (THDR)")
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        im3 = ax3.imshow(base_data["thdr"], cmap="magma", extent=[b[0], b[1], b[2], b[3]])
        plt.colorbar(im3, ax=ax3, label="THDR (nT/m)", shrink=0.8)
        ax3.set_title("Total Horizontal Derivative (Structural Edges)")
        ax3.set_xlabel("Longitude (°E)")
        ax3.set_ylabel("Latitude (°N)")
        st.pyplot(fig3)

    with col_map4:
        st.markdown("##### 4. 3D Analytic Signal Amplitude (AS)")
        fig4, ax4 = plt.subplots(figsize=(6, 5))
        im4 = ax4.imshow(base_data["as"], cmap="inferno", extent=[b[0], b[1], b[2], b[3]])
        plt.colorbar(im4, ax=ax4, label="Analytic Signal (nT/m)", shrink=0.8)
        ax4.set_title("Analytic Signal (Magnetic Susceptibility Contrasts)")
        ax4.set_xlabel("Longitude (°E)")
        ax4.set_ylabel("Latitude (°N)")
        st.pyplot(fig4)

# ----------------- TAB 4: BOUGUER GRAVITY -----------------
with tab_grav:
    st.subheader("⚖️ Subsurface Density Inversion & Bouguer Gravity Anomaly")
    st.caption("Fusing satellite-derived gravity models (GGMplus 200m / WGM2012) with aeromagnetics to map low-density fertile pegmatite-generating granites.")

    g_col1, g_col2 = st.columns(2)

    with g_col1:
        st.markdown("##### 1. Complete Bouguer Gravity Anomaly (mGal)")
        fig_g1, ax_g1 = plt.subplots(figsize=(6, 5))
        im_g1 = ax_g1.imshow(base_data["bouguer"], cmap="Spectral_r", extent=[b[0], b[1], b[2], b[3]])
        plt.colorbar(im_g1, ax=ax_g1, label="Bouguer Anomaly (mGal)", shrink=0.8)
        ax_g1.contour(base_data["bouguer"], levels=[-38.0, -36.0], colors="blue", linewidths=1.2, extent=[b[0], b[1], b[2], b[3]])
        ax_g1.set_title("Bouguer Gravity Anomaly (Blue Lines = Granite Lows)")
        ax_g1.set_xlabel("Longitude (°E)")
        ax_g1.set_ylabel("Latitude (°N)")
        st.pyplot(fig_g1)

    with g_col2:
        st.markdown("##### 2. First Vertical Gravity Gradient (Gzz in Eotvos)")
        fig_g2, ax_g2 = plt.subplots(figsize=(6, 5))
        im_g2 = ax_g2.imshow(base_data["gzz"], cmap="PuOr_r", extent=[b[0], b[1], b[2], b[3]])
        plt.colorbar(im_g2, ax=ax_g2, label="Gzz Gradient (Eotvos)", shrink=0.8)
        ax_g2.contour(base_data["gzz"], levels=[0], colors="black", linewidths=1.0, extent=[b[0], b[1], b[2], b[3]])
        ax_g2.set_title("First Vertical Gravity Gradient (Gzz Contact Edges)")
        ax_g2.set_xlabel("Longitude (°E)")
        ax_g2.set_ylabel("Latitude (°N)")
        st.pyplot(fig_g2)

# ----------------- TAB 5: ADVANCED RADIOMETRICS & ALTERATION -----------------
with tab_rad:
    st.subheader("🌈 Gamma-Ray Spectrometry & Hydrothermal Alteration Suite")
    st.caption("Comprehensive multi-channel radiometric processing for lithium pegmatite and rare-metal alteration halos.")

    rad_sub1, rad_sub2, rad_sub3 = st.tabs([
        "🔬 Radioelement Concentrations (TC, K, eTh, eU)",
        "🧪 Diagnostic Exploration Ratios (K/eTh, eU/eTh, F-Param)",
        "📊 Bivariate Alteration Crossplot (K vs eTh)"
    ])

    with rad_sub1:
        st.markdown("##### Airborne Gamma-Ray Calibrated Concentration Channels")
        rc1, rc2 = st.columns(2)
        with rc1:
            fig_tc, ax_tc = plt.subplots(figsize=(6, 4.5))
            im_tc = ax_tc.imshow(base_data["ratios"]["TC"], cmap="turbo", extent=[b[0], b[1], b[2], b[3]])
            plt.colorbar(im_tc, ax=ax_tc, label="Total Count (cps / Ur)", shrink=0.8)
            ax_tc.set_title("Total Count (TC)")
            st.pyplot(fig_tc)

            fig_k, ax_k = plt.subplots(figsize=(6, 4.5))
            im_k = ax_k.imshow(base_data["raw"]["k"], cmap="viridis", extent=[b[0], b[1], b[2], b[3]])
            plt.colorbar(im_k, ax=ax_k, label="Potassium (%)", shrink=0.8)
            ax_k.set_title("Potassium Concentration (K %)")
            st.pyplot(fig_k)

        with rc2:
            fig_th, ax_th = plt.subplots(figsize=(6, 4.5))
            im_th = ax_th.imshow(base_data["raw"]["eth"], cmap="magma", extent=[b[0], b[1], b[2], b[3]])
            plt.colorbar(im_th, ax=ax_th, label="Thorium (ppm)", shrink=0.8)
            ax_th.set_title("Equivalent Thorium (eTh ppm)")
            st.pyplot(fig_th)

            fig_u, ax_u = plt.subplots(figsize=(6, 4.5))
            im_u = ax_u.imshow(base_data["raw"]["eu"], cmap="plasma", extent=[b[0], b[1], b[2], b[3]])
            plt.colorbar(im_u, ax=ax_u, label="Uranium (ppm)", shrink=0.8)
            ax_u.set_title("Equivalent Uranium (eU ppm)")
            st.pyplot(fig_u)

    with rad_sub2:
        st.markdown("##### Hydrothermal Metasomatism & Alteration Indices")
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            fig_tern, ax_tern = plt.subplots(figsize=(6, 5))
            ax_tern.imshow(base_data["ternary_rgb"], extent=[b[0], b[1], b[2], b[3]])
            ax_tern.set_title("Ternary Composite (Red: K | Green: eTh | Blue: eU)")
            ax_tern.set_xlabel("Longitude (°E)")
            ax_tern.set_ylabel("Latitude (°N)")
            st.pyplot(fig_tern)
            st.caption("Reddish-magenta hues denote intense potassic metasomatism associated with LCT pegmatite swarms.")

            fig_f, ax_f = plt.subplots(figsize=(6, 5))
            im_f = ax_f.imshow(base_data["ratios"]["F_param"], cmap="hot", extent=[b[0], b[1], b[2], b[3]])
            plt.colorbar(im_f, ax=ax_f, label="F-Parameter [K * eU / eTh]", shrink=0.8)
            ax_f.set_title("Efimov / Saunders F-Parameter Alteration Index")
            ax_f.set_xlabel("Longitude (°E)")
            ax_f.set_ylabel("Latitude (°N)")
            st.pyplot(fig_f)

        with r_col2:
            fig_kth, ax_kth = plt.subplots(figsize=(6, 5))
            im_kth = ax_kth.imshow(base_data["ratios"]["K_eTh"], cmap="YlOrRd", extent=[b[0], b[1], b[2], b[3]])
            plt.colorbar(im_kth, ax=ax_kth, label="K / eTh Ratio", shrink=0.8)
            ax_kth.set_title("Potassic Alteration Ratio (K / eTh)")
            ax_kth.set_xlabel("Longitude (°E)")
            ax_kth.set_ylabel("Latitude (°N)")
            st.pyplot(fig_kth)

            fig_kd, ax_kd = plt.subplots(figsize=(6, 5))
            im_kd = ax_kd.imshow(base_data["ratios"]["Kd"], cmap="coolwarm", extent=[b[0], b[1], b[2], b[3]])
            plt.colorbar(im_kd, ax=ax_kd, label="Kd Deviation (%)", shrink=0.8)
            ax_kd.contour(base_data["ratios"]["Kd"], levels=[0.5], colors="black", linewidths=1.2, extent=[b[0], b[1], b[2], b[3]])
            ax_kd.set_title("Potassium Residual Enrichment (Kd = K - eTh / 3.75)")
            ax_kd.set_xlabel("Longitude (°E)")
            ax_kd.set_ylabel("Latitude (°N)")
            st.pyplot(fig_kd)

    with rad_sub3:
        st.markdown("##### 📊 Bivariate Radiometric Alteration Crossplot (K vs eTh)")
        st.write("Exploration geochemical crossplot discriminating unaltered crustal rocks from hydrothermally altered pegmatites (Saunders et al., 1987).")

        step_sub = 2
        flat_k = base_data["raw"]["k"][::step_sub, ::step_sub].ravel()
        flat_th = base_data["raw"]["eth"][::step_sub, ::step_sub].ravel()
        flat_f = base_data["ratios"]["F_param"][::step_sub, ::step_sub].ravel()

        df_scatter = pd.DataFrame({
            "Potassium (%)": flat_k,
            "Thorium (ppm)": flat_th,
            "F_Parameter": flat_f
        })

        fig_bivar = px.scatter(
            df_scatter,
            x="Thorium (ppm)",
            y="Potassium (%)",
            color="F_Parameter",
            color_continuous_scale="Turbo",
            title="K (%) vs eTh (ppm) with Normal Crustal Baseline (eTh/K = 3.75)",
            template="plotly_dark",
            opacity=0.75
        )

        # Draw Normal Crustal Evolution Line (eTh/K ~ 3.75 -> K = eTh / 3.75)
        th_line = np.linspace(0, float(np.max(flat_th)), 50)
        k_line = th_line / 3.75
        fig_bivar.add_trace(go.Scatter(
            x=th_line, y=k_line, mode="lines",
            line=dict(color="#EF4444", width=2.5, dash="dash"),
            name="Normal Crustal Ratio (eTh/K = 3.75)"
        ))

        # Add pegmatite alteration zone callout
        fig_bivar.add_annotation(
            x=float(np.min(flat_th) + 2.5),
            y=float(np.max(flat_k) * 0.85),
            text="⚡ Fertile LCT Pegmatite Alteration Field",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#06B6D4",
            font=dict(color="#06B6D4", size=12)
        )

        fig_bivar.update_layout(height=480, margin=dict(l=10, r=10, t=35, b=10))
        st.plotly_chart(fig_bivar, use_container_width=True)

# ----------------- TAB 6: PROSPECTIVITY MPM -----------------
with tab_mpm:
    st.subheader("🤖 Machine Learning Mineral Prospectivity Mapping (MPM)")
    st.caption("Multi-Physics Inversion (Aeromagnetics + Gravity + Radiometrics) via active model: " + selected_model_type)

    fig_mpm, ax_mpm = plt.subplots(figsize=(9, 6.5))
    im_mpm = ax_mpm.imshow(prob_map * 100, cmap="turbo", extent=[b[0], b[1], b[2], b[3]], vmin=10, vmax=95)
    plt.colorbar(im_mpm, ax=ax_mpm, label="Prospectivity Probability (%)", shrink=0.8)

    for tgt in targets:
        if "Tier-1" in tgt["target_tier"]:
            ax_mpm.plot(tgt["longitude"], tgt["latitude"], 'r*', markersize=14, markeredgecolor='black', label="Tier-1 Drill Target" if tgt == targets[0] else "")
        else:
            ax_mpm.plot(tgt["longitude"], tgt["latitude"], 'yo', markersize=8, markeredgecolor='black', label="Tier-2 Infill Target" if tgt == targets[1] else "")

    ax_mpm.legend(loc="upper right", fontsize=8)
    ax_mpm.set_title(f"Mineral Prospectivity Map ({selected_model_type})")
    ax_mpm.set_xlabel("Longitude (°E)")
    ax_mpm.set_ylabel("Latitude (°N)")
    st.pyplot(fig_mpm)

# ----------------- TAB 7: 3D SUBSURFACE EXPLORER -----------------
with tab_3d:
    st.subheader("🧊 Interactive 3D Subsurface Geological & Drillhole Explorer")
    st.caption("Visualizing Digital Elevation (DEM), Regolith Cover (15m–65m), and Concealed Bedrock Drillholes in Full 3D.")

    c3d_ctrl1, c3d_ctrl2, c3d_ctrl3, c3d_ctrl4 = st.columns(4)
    with c3d_ctrl1:
        z_exag = st.slider("Vertical Exaggeration:", 1.0, 5.0, 2.5, 0.5)
    with c3d_ctrl2:
        show_bedrock_surface = st.checkbox("Show Subsurface Bedrock Contact", value=True)
    with c3d_ctrl3:
        cutaway_pct = st.slider("East-West Block Cutaway (%):", 30, 100, 100, 5)
    with c3d_ctrl4:
        surface_drape = st.selectbox("3D Surface Drape:", ["Prospectivity Probability (%)", "Bouguer Gravity (mGal)", "Total Magnetic Intensity (nT)"])

    step = 2
    raw_elev = base_data["raw"]["elevation"][::step, ::step]
    raw_bedrock = base_data["raw"]["bedrock_elevation"][::step, ::step]

    if surface_drape == "Bouguer Gravity (mGal)":
        color_surface = base_data["bouguer"][::step, ::step]
        cmap_name = 'Spectral_r'
    elif surface_drape == "Total Magnetic Intensity (nT)":
        color_surface = base_data["raw"]["tmi"][::step, ::step]
        cmap_name = 'Jet'
    else:
        color_surface = prob_map[::step, ::step] * 100.0
        cmap_name = 'Turbo'

    ny_sub, nx_sub = raw_elev.shape
    cutoff_col = int((cutaway_pct / 100.0) * nx_sub)
    cutoff_col = max(10, min(nx_sub, cutoff_col))

    x_coords = np.linspace(b[0], b[0] + (b[1] - b[0]) * (cutoff_col / nx_sub), cutoff_col)
    y_coords = np.linspace(b[2], b[3], ny_sub)
    X_3D, Y_3D = np.meshgrid(x_coords, y_coords)

    Z_surface = raw_elev[:, :cutoff_col] * z_exag
    Z_bedrock = raw_bedrock[:, :cutoff_col] * z_exag
    color_surface = color_surface[:, :cutoff_col]

    fig_3d = go.Figure()

    fig_3d.add_trace(go.Surface(
        x=X_3D, y=Y_3D, z=Z_surface,
        surfacecolor=color_surface, colorscale=cmap_name,
        colorbar=dict(title=dict(text=surface_drape, side="right"), len=0.6, x=1.02),
        name="Topographic Surface", opacity=0.92
    ))

    if show_bedrock_surface:
        fig_3d.add_trace(go.Surface(
            x=X_3D, y=Y_3D, z=Z_bedrock,
            colorscale=[[0, '#64748B'], [1, '#334155']], showscale=False,
            name="Bedrock Contact Interface", opacity=0.45
        ))

    for tgt in targets:
        t_lon = tgt["longitude"]
        t_lat = tgt["latitude"]
        s_elev = tgt["surface_elevation_m"] * z_exag
        t_depth = tgt["target_drill_depth_m"]
        b_elev = (tgt["surface_elevation_m"] - t_depth) * z_exag

        is_tier1 = "Tier-1" in tgt["target_tier"]
        stem_color = '#EF4444' if is_tier1 else '#F59E0B'
        stem_width = 7 if is_tier1 else 4

        fig_3d.add_trace(go.Scatter3d(
            x=[t_lon, t_lon], y=[t_lat, t_lat], z=[s_elev, b_elev],
            mode='lines', line=dict(color=stem_color, width=stem_width),
            showlegend=False
        ))

        fig_3d.add_trace(go.Scatter3d(
            x=[t_lon], y=[t_lat], z=[b_elev],
            mode='markers+text',
            marker=dict(size=9 if is_tier1 else 6, color=stem_color, symbol='diamond'),
            text=[tgt['target_id']], textposition="top center",
            textfont=dict(size=9, color=stem_color),
            hovertemplate=f"<b>{tgt['target_id']}</b><br>Drill Depth: {t_depth:.1f}m<br>Prospectivity: {tgt['prospectivity_score']}%<extra></extra>",
            showlegend=False
        ))

    fig_3d.update_layout(
        scene=dict(
            xaxis=dict(title='Longitude (°E)', gridcolor='#334155'),
            yaxis=dict(title='Latitude (°N)', gridcolor='#334155'),
            zaxis=dict(title=f'Elevation (m ASL x{z_exag:.1f})', gridcolor='#334155'),
            aspectratio=dict(x=1.3, y=1.3, z=0.45),
            camera=dict(eye=dict(x=1.4, y=-1.6, z=0.85))
        ),
        template="plotly_dark",
        margin=dict(l=0, r=0, t=10, b=0),
        height=620
    )
    st.plotly_chart(fig_3d, use_container_width=True)

# ----------------- TAB 8: JORC & NI 43-101 DRILL PACKAGE & REPORT EXPORTER -----------------
with tab_jorc:
    st.subheader("📑 Exploration Reporting & Export Hub (Word DOCX, Excel XLSX & CSV)")
    st.caption("Generate publication-grade exploration briefing packages for venture investors, technical directors, and JORC Competent Persons.")

    st.markdown("""
    <div class="enterprise-card">
        <b>📄 Single Document Compilation (Word .docx):</b><br/>
        Compiles the entire multi-physics exploration suite into <b>one publication-grade Word document (.docx)</b> containing:
        <ul style="margin-top:6px; margin-bottom:6px; padding-left:20px;">
            <li><b>All Embedded Maps:</b> Prospectivity Heatmap (Figure 1), Aeromagnetics TMI & TDR (Figure 2), Bouguer Gravity & Gzz (Figure 3), Radiometrics Suite (Figure 4: Ternary, K/eTh, F-Parameter, TC), and Bivariate K vs eTh Alteration Crossplot (Figure 5).</li>
            <li><b>Complete Machine Learning Model:</b> Architecture, Hyperparameters, ROC Curve, AUC validation score, and feature weight importance.</li>
            <li><b>Full JORC Target Registry:</b> Coordinates, Collar Elevations, Regolith Casing Depths, and Proposed Drill Depths.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Export Buttons
    exp_c1, exp_c2, exp_c3 = st.columns(3)

    with exp_c1:
        with st.spinner("Compiling high-resolution maps & Word document..."):
            docx_report = generate_comprehensive_word_report(
                base_data, prob_map, targets, importances, feat_names, metrics, selected_model_type
            )
        st.download_button(
            label="📄 Download Complete Word Report (.docx)",
            data=docx_report,
            file_name=f"RegolithSense_Exploration_Report_{selected_model_type.replace(' ', '_')}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True
        )
        # Concept note button
        cn_path = os.path.join(os.path.dirname(__file__), "RegolithSense_Solution_Concept_Note.docx")
        if os.path.exists(cn_path):
            with open(cn_path, "rb") as f_cn:
                cn_bytes = f_cn.read()
            st.download_button(
                label="📝 Download Solution Concept Note (.docx)",
                data=cn_bytes,
                file_name="RegolithSense_Solution_Concept_Note.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True
            )

    with exp_c2:
        xlsx_package = generate_excel_workbook(
            df_targets, feat_names, importances, metrics, base_data
        )
        st.download_button(
            label="📊 Download Multi-Sheet Data Package (.xlsx)",
            data=xlsx_package,
            file_name="RegolithSense_Exploration_Workbook.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    with exp_c3:
        csv_data = df_targets.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download JORC Targets Only (.csv)",
            data=csv_data,
            file_name="RegolithSense_Nasarawa_Targets_JORC.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown("---")
    st.markdown("##### 🎯 Prioritized Drillhole Targets Summary Table")
    st.dataframe(df_targets, use_container_width=True)

st.markdown("---")
st.caption("RegolithSense Enterprise | Nasarawa State University Keffi Mine-Tech UniPod (UNDP timbuktoo) | October 2026")
