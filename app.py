"""
RegolithSense: targeting concealed pegmatites beneath weathering cover from
airborne magnetic, radiometric and gravity surveys.

Run with:  streamlit run app.py
"""
import hashlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import gridio, pipeline, prospectivity, report
from core.regolith import REGIMES

st.set_page_config(page_title="RegolithSense", page_icon="🜨", layout="wide", initial_sidebar_state="expanded")

STEPS = ["Overview", "1. Data", "2. Magnetics", "3. Radiometrics", "4. Regolith and gravity", "5. Prospectivity", "6. Targets and export", "7. 3D view",
         "8. Decision brief", "Methods and limits"]
DATA, MAG, RAD, REG, PROS, TARG, VIEW3D, BRIEF, METHODS = STEPS[1:]
REGIME_COLOURS = ["#b5651d", "#e8dcc0", "#5b8fb0"]
GREEN = "#2f5d50"

st.markdown("""
<style>
div[data-testid="stMetric"] {background: #f4f7f5; border: 1px solid #e1e8e4; border-radius: 10px; padding: 10px 14px;}
div[data-testid="stMetric"] label {color: #5c6b65;}
section[data-testid="stSidebar"] {border-right: 1px solid #e1e8e4;}
.rs-note {background:#fff6e5; border-left:3px solid #c98a1b; padding:8px 12px; border-radius:6px; font-size:0.9rem;}
</style>
""", unsafe_allow_html=True)

state = st.session_state
state.setdefault("uploads", {})      # file name -> {kind or None: raw layer}
state.setdefault("kinds", {})        # file name -> chosen layer kind
state.setdefault("project", None)
state.setdefault("source", "")
state.setdefault("occurrences", None)
state.setdefault("simulated", False)
state.setdefault("booted", False)
if "goto" in state:                      # a "next step" button was pressed on the previous run
    state["step"] = state.pop("goto")


# ------------------------------------------------------------------ helpers
def project_token(project):
    h = hashlib.md5()
    for k in sorted(project["layers"]):
        a = project["layers"][k]
        h.update(k.encode()); h.update(np.ascontiguousarray(a[::7, ::7]).tobytes())
    h.update(repr((project["x0"], project["y0"], project["dx"], project["nx"], project["ny"])).encode())
    return h.hexdigest()


@st.cache_data(show_spinner="Processing magnetic data")
def cached_magnetics(token, _project, inclination, declination, reduce, sensor_height, min_length):
    return pipeline.run_magnetics(_project, inclination, declination, reduce, sensor_height, min_length)


@st.cache_data(show_spinner="Processing radiometric data")
def cached_radiometrics(token, _project):
    return pipeline.run_radiometrics(_project)


@st.cache_data(show_spinner="Classifying regolith")
def cached_regolith(token, _project):
    return pipeline.run_regolith(_project)


@st.cache_data(show_spinner="Processing gravity data")
def cached_gravity(token, _project):
    return pipeline.run_gravity(_project)


@st.cache_data(show_spinner="Training and cross-validating the model")
def cached_model(token, _evidence, _occurrences, cell, buffer_m, names, occ_key, model_kind, n_trees, max_depth):
    return prospectivity.train_forest(_evidence, _occurrences, cell, buffer_m=buffer_m, n_trees=n_trees,
                                      model_kind=model_kind, max_depth=max_depth)


def load_project(layers, cell, source, simulated):
    state["project"] = gridio.harmonise(layers, cell=cell)
    state["source"] = source; state["simulated"] = simulated; state["occurrences"] = None
    state["score"] = None; state["docx"] = None


def axes_km(project, step):
    x = (project["x0"] + np.arange(0, project["nx"], step) * project["dx"]) / 1000.0
    y = (project["y0"] + np.arange(0, project["ny"], step) * project["dx"]) / 1000.0
    return x, y


def map_figure(project, grid, title, colours="Viridis", symmetric=False, unit="", mask=None, points=None, occurrences=None,
               discrete=None, height=520, cols=2, contour=None):
    """One interactive map. Every map in the app goes through here so they look and behave the same.
    `cols` is how many maps share the row; the figure height follows the grid's aspect so the map fills the frame."""
    width_px = {1: 820, 2: 440, 3: 290, 4: 215}.get(cols, 440)
    height = min(height, int(width_px * project["ny"] / project["nx"]) + 90)
    step = max(int(np.ceil(max(project["nx"], project["ny"]) / 450.0)), 1)
    z = np.asarray(grid, float)[::step, ::step]
    x, y = axes_km(project, step)
    fig = go.Figure()
    if discrete:
        n = len(discrete)
        scale = []
        for i, colour in enumerate(discrete):
            scale += [[i / n, colour], [(i + 1) / n, colour]]
        fig.add_trace(go.Heatmap(z=z, x=x, y=y, colorscale=scale, zmin=0.5, zmax=n + 0.5, showscale=False,
                                 hovertemplate="E %{x:.1f} km<br>N %{y:.1f} km<br>class %{z}<extra></extra>"))
    else:
        finite = z[np.isfinite(z)]
        lo, hi = (np.percentile(finite, (2, 98)) if finite.size else (0.0, 1.0))
        if symmetric:
            m = max(abs(lo), abs(hi)); lo, hi = -m, m
        fig.add_trace(go.Heatmap(z=z, x=x, y=y, colorscale=colours, zmin=float(lo), zmax=float(hi), colorbar=dict(title=unit, thickness=14),
                                 hovertemplate="E %{x:.1f} km<br>N %{y:.1f} km<br>%{z:.3g} " + unit + "<extra></extra>"))
    if contour is not None:
        fig.add_trace(go.Contour(z=z, x=x, y=y, contours=dict(start=contour, end=contour, size=1, coloring="none"),
                                 line=dict(color="black", width=1), showscale=False, hoverinfo="skip"))
    if mask is not None:
        rr, cc = np.nonzero(mask)
        if rr.size > 20000:
            keep = np.linspace(0, rr.size - 1, 20000).astype(int); rr, cc = rr[keep], cc[keep]
        fig.add_trace(go.Scattergl(x=(project["x0"] + cc * project["dx"]) / 1000.0, y=(project["y0"] + rr * project["dx"]) / 1000.0, mode="markers",
                                   marker=dict(size=2, color="black"), hoverinfo="skip", showlegend=False))
    if occurrences is not None and len(occurrences):
        fig.add_trace(go.Scatter(x=occurrences["easting"] / 1000.0, y=occurrences["northing"] / 1000.0, mode="markers", text=occurrences["name"],
                                 marker=dict(size=9, symbol="triangle-up", color="#d62728", line=dict(color="white", width=1)),
                                 hovertemplate="%{text}<extra>known occurrence</extra>", showlegend=False))
    if points is not None and len(points):
        fig.add_trace(go.Scatter(x=points["Easting (m)"] / 1000.0, y=points["Northing (m)"] / 1000.0, mode="markers+text", text=points["Target"],
                                 textposition="top center", textfont=dict(size=10, color="white"),
                                 marker=dict(size=10, symbol="circle-open", color="white", line=dict(width=2)),
                                 hovertemplate="%{text}<extra>target</extra>", showlegend=False))
    fig.update_layout(title=dict(text=title, font=dict(size=15)), height=height, margin=dict(l=10, r=10, t=44, b=10),
                      xaxis=dict(title="Easting (km)", constrain="domain"), yaxis=dict(title="Northing (km)", scaleanchor="x", scaleratio=1, constrain="domain"),
                      plot_bgcolor="white")
    return fig


def show(fig):
    st.plotly_chart(fig, use_container_width=True)


def need_project():
    if state["project"] is None:
        st.info("Load survey data in step 1 first.")
        st.stop()
    return state["project"]


def mag_settings():
    return (state.get("inclination", -7.0), state.get("declination", -2.0), state.get("reduce", True),
            state.get("sensor_height", 80.0), state.get("min_length", 1500.0))


def all_results(project):
    token = project_token(project)
    mag = cached_magnetics(token, project, *mag_settings())
    rad = cached_radiometrics(token, project)
    reg = cached_regolith(token, project)
    grav = cached_gravity(token, project)
    return token, mag, rad, reg, grav


def kpi_band(P, mag, rad, reg):
    """The headline numbers, shown at the top of every processing step."""
    area = P["nx"] * P["ny"] * P["dx"] ** 2 / 1e6
    cols = st.columns(5)
    cols[0].metric("Survey area", "%.0f km²" % area, help="%d x %d cells of %.0f m" % (P["nx"], P["ny"], P["dx"]))
    if mag is not None:
        t = mag["lineament_table"]
        cols[1].metric("Magnetic lineaments", len(t), help="Traced from tilt-angle zero crossings")
        d = mag["depth"][np.isfinite(mag["depth"])]
        cols[2].metric("Median depth to source", "%.0f m" % np.median(d) if d.size else "n/a", help="Tilt-depth along magnetic contacts")
    if rad is not None:
        cols[3].metric("Potassic alteration", "%.1f%% of area" % (100 * rad["alteration"].mean()))
    if reg is not None:
        cols[4].metric("Transported cover", "%.0f%% of area" % (100 * float((reg["regime"] == 3).mean())), help="Depositional regolith regime")
    info = state.get("model_info")
    if info and info.get("auc") is not None and reg is None:
        cols[4].metric("Model AUC", "%.2f" % info["auc"])


def nav_buttons(current):
    """Previous / next buttons at the foot of every step, so a visitor can walk the workflow without the sidebar."""
    i = STEPS.index(current)
    st.divider()
    a, _, b = st.columns([1, 4, 1])
    if i > 0 and a.button("← " + STEPS[i - 1].split(". ", 1)[-1], use_container_width=True, key="prev_" + current):
        state["goto"] = STEPS[i - 1]; st.rerun()
    if i < len(STEPS) - 1 and b.button(STEPS[i + 1].split(". ", 1)[-1] + " →", use_container_width=True, type="primary", key="next_" + current):
        state["goto"] = STEPS[i + 1]; st.rerun()


def explain_targets(targets, evidence, project):
    """Evidence values at each target cell, so every pick can be justified in one row."""
    if not len(targets):
        return pd.DataFrame()
    rr = np.clip(((targets["Northing (m)"].to_numpy() - project["y0"]) / project["dx"]).round().astype(int), 0, project["ny"] - 1)
    cc = np.clip(((targets["Easting (m)"].to_numpy() - project["x0"]) / project["dx"]).round().astype(int), 0, project["nx"] - 1)
    out = pd.DataFrame({"Target": targets["Target"]})
    for name, layer in evidence.items():
        out[name] = np.round(np.asarray(layer)[rr, cc], 2)
    lead = out.drop(columns="Target").idxmax(axis=1)
    out.insert(1, "Strongest evidence", lead)
    return out


def source_note():
    if state["simulated"]:
        st.markdown('<div class="rs-note">Simulated example data. The results do not describe any real ground.</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ first run
if not state["booted"]:
    state["booted"] = True
    try:
        load_project(pipeline.demo_layers(), 200.0, "Wamba sheet 210, Nasarawa State (NGSA airborne survey, 200 m demo grid)", False)
    except Exception:
        pass

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("RegolithSense")
    st.caption("Targeting concealed pegmatites beneath weathering cover")
    step = st.radio("Workflow", STEPS, label_visibility="collapsed", key="step")
    st.divider()
    if state["project"] is not None:
        P = state["project"]
        st.markdown("**Current dataset**")
        st.write(state["source"])
        st.caption("%d x %d cells of %.0f m, UTM zone %dN" % (P["nx"], P["ny"], P["dx"], P["zone"]))
        st.caption("Layers: " + ", ".join(gridio.LAYER_NAMES[k].split(" (")[0] for k in P["layers"]))
        if state["occurrences"] is not None:
            st.caption("%d known occurrences loaded" % len(state["occurrences"]))
        if state["simulated"]:
            st.warning("Simulated example data. Results are not real.")
        have = P["layers"]
        checks = [("Magnetics", "tmi" in have), ("Radiometrics", all(k in have for k in ("k", "eth", "eu"))),
                  ("Regolith (needs elevation)", "dem" in have), ("Gravity", "grav" in have),
                  ("Trained model (needs 10+ occurrences)", state["occurrences"] is not None and len(state["occurrences"]) >= 10)]
        st.caption("\n".join(("✅ " if ok else "▫️ ") + name for name, ok in checks))
    else:
        st.caption("No dataset loaded.")
    st.divider()
    st.caption("Federal University of Lafia · UniPod\nRegolithSense prototype v2")


# ------------------------------------------------------------------ overview
if step == STEPS[0]:
    st.title("RegolithSense")
    st.subheader("Finding the pegmatites that the weathering cover hides")
    a, b = st.columns([3, 2])
    with a:
        st.markdown("""
**The problem.** Across the Nigerian Basement Complex, lithium–tantalum pegmatites and the granites that feed them lie under
10 to 60 m of laterite and saprolite. Surface mapping sees the cover, not the rock, so most exploration is still done by
walking, pitting and guesswork. Artisanal miners find deposits by accident; companies drill blind.

**What RegolithSense does.** It takes the airborne geophysics that already exists for the whole country (NGSA magnetic and
radiometric surveys, free SRTM elevation, satellite gravity) and turns it into a ranked list of places worth visiting:

1. **Magnetics** see through the cover and map the structures and contacts that host pegmatites.
2. **Radiometrics** detect potassium added by mineralising fluids, where the surface is in-situ rock.
3. **Regolith mapping** says where that surface signal can be trusted and where it is transported sediment.
4. **Gravity** points to the low-density fertile granites.
5. A **prospectivity model** combines them, either with expert weights or trained on known occurrences and tested on ground it has not seen.
6. The output is a **target package**: ranked coordinates, maps, a 3D view and a Word report, ready for a field team.

**Why it matters.** On the Wamba sheet loaded now, the model reduces 3,058 km² to the best 5% and ranks it. A field team can
check that in weeks instead of seasons. The same workflow runs on any NGSA sheet.
""")
    with b:
        with st.container(border=True):
            st.markdown("**Try it in two minutes**")
            st.write("Wamba sheet 210 (Nasarawa State) is already loaded from real NGSA grids.")
            if st.button("Start with the data →", type="primary", use_container_width=True):
                state["goto"] = DATA; st.rerun()
            if st.button("Jump to the ranked targets →", use_container_width=True):
                state["goto"] = TARG; st.rerun()
            if st.button("Read the decision brief →", use_container_width=True):
                state["goto"] = BRIEF; st.rerun()
        with st.container(border=True):
            st.markdown("**What this is, and is not**")
            st.caption("A screening tool that ranks ground for follow-up using public geophysics. Scores are rankings, not probabilities "
                       "of a deposit, and nothing here replaces field mapping, sampling or drilling. It is not a JORC or NI 43-101 report.")
        with st.container(border=True):
            st.markdown("**Built by**")
            st.caption("Abubakar Sadiq Musa, Exploration Geophysics, Federal University of Lafia · UniPod prototype, November 2026. Open source on GitHub.")
    nav_buttons(step)

elif step == METHODS:
    st.header("Methods and limits")
    st.write("Every number in the app comes from a published, reproducible method. This page says which, and where each one can mislead.")
    rows = [
        ("Grid reading and harmonisation", "Geosoft .grd reader (compressed and plain); WGS84 UTM projection; resampling with anti-alias averaging; nearest-cell gap fill; median/MAD despiking gated by global percentiles.",
         "Sheets in degrees are reprojected, so cell sizes change slightly across a sheet. Despiking can remove real narrow anomalies if set too aggressively."),
        ("Reduction to the equator", "FFT-domain operator with amplitude floor, for low magnetic latitudes (Nigeria, inclination about −7°).",
         "Amplifies noise along the magnetic declination; north–south structures remain weak at low latitude."),
        ("Edges and lineaments", "Tilt angle (Miller & Singh 1994), analytic signal (Roest et al. 1992), total horizontal derivative; lineaments from connected tilt zero-crossings gated by analytic-signal amplitude, strike by SVD.",
         "Lineaments are automatic and unverified; flight-line noise can produce spurious east–west or north–south trends."),
        ("Depth to source", "Tilt-depth (Salem et al. 2007): horizontal distance between the ±45° tilt contours, minus sensor height.",
         "Assumes vertical contacts; a 100 m grid cannot resolve sources shallower than about one cell. Treat as an upper bound."),
        ("Radiometric alteration", "K/eTh, eU/eTh, F-parameter (Efimov 1978), Kd and Ud deviation from the regional K–eTh trend (Pires 1995); ternary image (Dickson & Scott 1997).",
         "Gamma rays come from the top 30–40 cm. Over transported cover the signal is the sediment, not the bedrock."),
        ("Regolith regime", "Relict / erosional / depositional classes from SRTM slope, topographic position index and the eTh/K ratio (after Anand & Paine 2002).",
         "Rule-based and uncalibrated for Nasarawa. Field checking is needed before it is used for anything beyond weighting."),
        ("Prospectivity: weighted overlay", "Evidence layers scaled 0–1 (2nd–98th percentile) and combined with user weights (Carranza 2008). Radiometric evidence multiplied by the regolith weight.",
         "The result is a ranking. Weights encode expert opinion, not measured likelihoods."),
        ("Prospectivity: trained models", "Random Forest, Gradient Boosting or logistic regression on cells within a buffer of known occurrences versus distant background; accuracy by spatial block cross-validation (Roberts et al. 2017); uncertainty as the spread between trees.",
         "Known occurrences are biased to where people have looked. Fewer than about 25 occurrences gives unstable, optimistic accuracy."),
        ("Targets", "Local maxima of the score above a percentile threshold, at least a chosen distance apart, ranked and tagged with regime and depth.",
         "Targets are places to check on the ground, not drill collars."),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Stage", "Method", "Where it can mislead"]), use_container_width=True, hide_index=True, height=420)
    st.markdown("""
**References.** Anand & Paine 2002, *Aust. J. Earth Sci.* 49; Carranza 2008, *Geochemical Anomaly and Mineral Prospectivity Mapping in GIS*;
Dickson & Scott 1997, *AGSO J.* 17; Efimov 1978; Miller & Singh 1994, *J. Appl. Geophys.* 32; Pires 1995, *Rev. Bras. Geociências* 25;
Roberts et al. 2017, *Ecography* 40; Roest, Verhoef & Pilkington 1992, *Geophysics* 57; Salem et al. 2007, *The Leading Edge* 26.

**Validation so far.** No published list of pegmatite or mine coordinates exists for sheet 210 in the open literature we could find, so the trained models
have not been tested on real occurrences and the Wamba ranking uses expert weights. One independent cross-check was made against a 2026 preprint
(Research Square, not yet peer reviewed) that interpreted airborne magnetic and radiometric data over the same sheet and flagged two zones.
In its north-western zone (8°48′–8°52′N, 8°32′–8°36′E) the score is above the sheet's top 5% over 29% of the area and five of the 40 targets fall inside it.
Around its second zone (8°40′N, 8°48′E) the score is low and no target falls within 6 km. Both studies use the same NGSA grids, so this shows the methods
agree in part, not that mineralisation is confirmed.

**Data.** Nigerian Geological Survey Agency airborne magnetic and radiometric grids (2005–2009 national survey, 500 m line spacing, 80 m clearance);
SRTM 1-arc-second elevation. Satellite gravity (WGM2012 or GGMplus) can be added as a Bouguer grid.
""")
    nav_buttons(step)

# ------------------------------------------------------------------- 1 data
elif step == DATA:
    st.header("Survey data")
    st.write("Upload the grids for one survey area. RegolithSense reads Geosoft grids (.grd) as supplied by NGSA and Oasis montaj, "
             "GeoTIFF, and CSV or XYZ point files. Layers in different cell sizes or coordinate systems are brought onto one grid.")
    left, right = st.columns([3, 2])
    with left:
        files = st.file_uploader("Survey files", type=["grd", "tif", "tiff", "csv", "xyz", "txt"], accept_multiple_files=True,
                                 help="Magnetic field, potassium, thorium, uranium, total count, and optionally elevation and Bouguer gravity.")
        for f in files or []:
            if f.name not in state["uploads"]:
                try:
                    state["uploads"][f.name] = gridio.read_file(f.name, f.getvalue())
                except Exception as exc:  # show the reason and carry on with the other files
                    st.error("%s could not be read: %s" % (f.name, exc))
        names = {f.name for f in files or []}
        for gone in [n for n in state["uploads"] if n not in names]:
            state["uploads"].pop(gone); state["kinds"].pop(gone, None)

        kinds = list(gridio.LAYER_NAMES)
        chosen = {}
        for name, layers in state["uploads"].items():
            for kind, layer in layers.items():
                if kind is None or len(layers) == 1:
                    label = "%s (%d x %d)" % (name, layer["grid"].shape[1], layer["grid"].shape[0])
                    default = kinds.index(kind) if kind in kinds else 0
                    pick = st.selectbox(label, kinds, index=default, format_func=lambda k: gridio.LAYER_NAMES[k], key="kind_" + name)
                    chosen[pick] = layer
                else:
                    chosen[kind] = layer
        cell = st.select_slider("Working cell size (m)", options=[50, 100, 125, 200, 250, 500], value=100,
                                help="Smaller cells keep more detail but take longer. 100 m suits NGSA 500 m line-spaced surveys.")
        if st.button("Prepare data", type="primary", disabled=not chosen):
            try:
                with st.spinner("Bringing layers onto one grid"):
                    load_project(chosen, float(cell), ", ".join(sorted(state["uploads"])), False)
            except Exception as exc:
                st.error(str(exc))
            else:
                st.rerun()
    with right:
        st.subheader("Ready-made datasets")
        st.write("**Wamba sheet 210, Nasarawa State.** NGSA magnetic, K, eTh, eU and SRTM grids, resampled to 200 m so it loads in seconds.")
        if st.button("Load Wamba 210 (NGSA)", type="primary", use_container_width=True):
            with st.spinner("Loading Wamba 210"):
                load_project(pipeline.demo_layers(), 200.0, "Wamba sheet 210, Nasarawa State (NGSA airborne survey, 200 m demo grid)", False)
            st.rerun()
        st.write("**Simulated survey.** Computer-generated, includes gravity, useful for showing the full workflow. It does not describe any real ground.")
        if st.button("Load simulated example", use_container_width=True):
            load_project(pipeline.simulated_layers(), 100.0, "Simulated example", True)
            st.rerun()
        st.subheader("Known occurrences (optional)")
        st.write("A CSV of known pegmatite or mine locations with easting/northing or longitude/latitude columns. "
                 "With ten or more, the prospectivity model can be trained and tested on them.")
        st.download_button("Download a CSV template", open(pipeline.TEMPLATE_PATH, "rb").read(), "occurrences_template.csv", "text/csv",
                           help="Columns: name, longitude, latitude (or easting, northing). Use surveyed locations of confirmed pegmatites or mines only.")
        occ_file = st.file_uploader("Occurrence file", type=["csv"], key="occ_file")
        if occ_file is not None and state["project"] is None:
            st.info("Prepare the survey data first, then the occurrences will be placed on it.")
        if occ_file is not None and state["project"] is not None:
            try:
                occ, outside = gridio.read_occurrences(occ_file.getvalue(), occ_file.name, state["project"])
                state["occurrences"] = occ
                st.success("%d occurrences inside the survey area%s." % (len(occ), " (%d outside were ignored)" % outside if outside else ""))
            except Exception as exc:
                st.error(str(exc))

    if state["project"] is not None:
        P = state["project"]
        st.divider()
        source_note()
        st.subheader("Prepared layers")
        st.caption("Source: " + state["source"])
        palette = {"tmi": ("RdBu_r", True), "k": ("YlOrRd", False), "eth": ("YlGn", False), "eu": ("PuBu", False), "tc": ("Viridis", False),
                   "dem": ("Earth", False), "grav": ("Spectral_r", True)}
        cols = st.columns(3)
        for i, (kind, grid) in enumerate(P["layers"].items()):
            with cols[i % 3]:
                c, sym = palette.get(kind, ("Viridis", False))
                show(map_figure(P, grid - (np.median(grid) if sym else 0), gridio.LAYER_NAMES[kind], c, sym, height=380, cols=3,
                                occurrences=state["occurrences"]))
        with st.expander("Data preparation report"):
            st.dataframe(pd.DataFrame(P["log"]), use_container_width=True, hide_index=True)
            st.caption("Blank cells are filled from the nearest valid cell. Radiometric spikes are replaced by the local median and negative concentrations set to zero.")
            st.download_button("Download prepared grids (CSV, easting/northing/values)", report.all_layers_xyz(P), "regolithsense_prepared_grids.csv", "text/csv")

# -------------------------------------------------------------- 2 magnetics
elif step == MAG:
    P = need_project()
    st.header("Magnetic interpretation")
    if "tmi" not in P["layers"]:
        st.info("No magnetic grid in this dataset."); st.stop()
    with st.expander("Processing settings"):
        a, b, c, d = st.columns(4)
        state["reduce"] = a.checkbox("Reduce to the equator", value=state.get("reduce", True),
                                     help="Recommended at low magnetic latitudes such as Nigeria. Centres anomalies over their sources.")
        state["inclination"] = b.number_input("Inclination (degrees)", value=float(state.get("inclination", -7.0)), step=0.5,
                                              help="Geomagnetic inclination at the survey date and location (from the IGRF).")
        state["declination"] = c.number_input("Declination (degrees)", value=float(state.get("declination", -2.0)), step=0.5)
        state["sensor_height"] = d.number_input("Survey terrain clearance (m)", value=float(state.get("sensor_height", 80.0)), step=10.0)
        state["min_length"] = st.slider("Shortest lineament to keep (m)", 500, 5000, int(state.get("min_length", 1500)), 250)
    token, mag, rad, reg, grav = all_results(P)
    kpi_band(P, mag, rad, reg); source_note()
    st.write("Magnetic fields pass through non-magnetic regolith, so these maps show bedrock structure beneath the cover.")
    tabs = st.tabs(["Field", "Edges and lineaments", "Depth to source"])
    with tabs[0]:
        a, b = st.columns(2)
        with a:
            show(map_figure(P, mag["field"] - np.median(mag["field"]), "Magnetic field" + (" reduced to the equator" if state.get("reduce", True) else ""), "RdBu_r", True, "nT"))
        with b:
            show(map_figure(P, mag["residual"], "Residual field (regional removed)", "RdBu_r", True, "nT"))
        st.caption("The residual removes wavelengths longer than about 2 km (upward continuation) so that local sources stand out.")
    with tabs[1]:
        a, b = st.columns(2)
        with a:
            show(map_figure(P, mag["tilt"], "Tilt angle with traced lineaments", "PuOr_r", True, "deg", mask=mag["lineaments"]))
        with b:
            show(map_figure(P, mag["asig"], "Analytic signal", "Magma", False, "nT/m"))
        a, b = st.columns(2)
        with a:
            show(map_figure(P, mag["hg"], "Total horizontal derivative (structural edges)", "Magma", False, "nT/m"))
        with b:
            show(map_figure(P, mag["vd"], "First vertical derivative", "RdBu_r", True, "nT/m"))
        table = mag["lineament_table"]
        if len(table):
            counts, edges = np.histogram(table[:, 1], bins=np.arange(0, 181, 15), weights=table[:, 0] / 1000.0)
            fig = go.Figure(go.Bar(x=["%d-%d" % (edges[i], edges[i + 1]) for i in range(len(counts))], y=counts, marker_color=GREEN))
            fig.update_layout(title="Lineament length by strike", xaxis_title="Strike (degrees from north)", yaxis_title="Total length (km)", height=320,
                              margin=dict(l=10, r=10, t=44, b=10), plot_bgcolor="white")
            show(fig)
            st.caption("%d lineaments traced, %.0f km in total." % (len(table), table[:, 0].sum() / 1000.0))
        else:
            st.info("No lineaments at the current settings.")
        st.caption("At low magnetic latitude, structures striking north-south produce weak anomalies and are under-represented. "
                   "The analytic signal is less affected than the tilt angle.")
    with tabs[2]:
        depth = mag["depth"]
        a, b = st.columns([2, 1])
        with a:
            show(map_figure(P, depth, "Tilt-depth estimate to the top of magnetic sources", "Viridis_r", False, "m"))
        with b:
            d = depth[np.isfinite(depth)]
            if d.size:
                st.metric("Median depth below ground", "%.0f m" % np.median(d))
                st.metric("Middle 80% of estimates", "%.0f to %.0f m" % (np.percentile(d, 10), np.percentile(d, 90)))
                fig = go.Figure(go.Histogram(x=d[d < np.percentile(d, 99)], nbinsx=40, marker_color=GREEN))
                fig.update_layout(title="Distribution of depth estimates", xaxis_title="Depth (m)", height=260, margin=dict(l=10, r=10, t=44, b=10), plot_bgcolor="white")
                show(fig)
            st.caption("Estimates exist only along magnetic contacts. A %.0f m grid cannot resolve sources much shallower than one cell, "
                       "so treat these as upper bounds on cover thickness, to be checked by drilling or ground geophysics." % P["dx"])

# ----------------------------------------------------------- 3 radiometrics
elif step == RAD:
    P = need_project()
    st.header("Radiometric alteration")
    token, mag, rad, reg, grav = all_results(P)
    if rad is None:
        st.info("Potassium, thorium and uranium grids are all needed for this step."); st.stop()
    kpi_band(P, mag, rad, reg); source_note()
    st.write("Potassium added by hydrothermal fluids shows as potassium enrichment relative to thorium, which stays largely immobile. "
             "The signal comes from the top 30 to 40 cm of the ground, so it is most reliable where the surface is in-situ rock or saprolite.")
    tabs = st.tabs(["Ternary image", "Alteration indices", "Channels", "K vs eTh crossplot"])
    with tabs[0]:
        a, b = st.columns([3, 2])
        with a:
            st.image((rad["ternary"][::-1] * 255).astype(np.uint8), caption="Red = potassium, green = thorium, blue = uranium. North is up.", use_container_width=True)
        with b:
            st.markdown("**Reading the image**\n\n- Red and pink: potassium-rich rock, such as granite and potassic alteration.\n"
                        "- Green: thorium retained in weathered or lateritic surfaces.\n- Blue: uranium relatively enriched.\n"
                        "- White: all three high. Dark: all three low, as over water or thick sand.")
    with tabs[1]:
        a, b = st.columns(2)
        with a:
            show(map_figure(P, rad["kd"], "Potassium enrichment relative to thorium (Kd)", "RdYlBu_r", True, "", mask=rad["alteration"]))
            st.caption("Black marks cells flagged as potassic alteration: %.1f%% of the area." % (100 * rad["alteration"].mean()))
        with b:
            show(map_figure(P, rad["f_param"], "F-parameter (K x eU / eTh)", "YlOrRd", False, ""))
        a, b = st.columns(2)
        with a:
            show(map_figure(P, rad["k_eth"], "K / eTh ratio", "YlOrRd", False, ""))
        with b:
            show(map_figure(P, rad["eu_eth"], "eU / eTh ratio", "PuBu", False, ""))
    with tabs[2]:
        chans = [("k", "Potassium", "YlOrRd", "%"), ("eth", "Thorium", "YlGn", "ppm"), ("eu", "Uranium", "PuBu", "ppm")]
        if "tc" in P["layers"]:
            chans.append(("tc", "Total count", "Viridis", ""))
        cols = st.columns(len(chans))
        for col, (k, title, cmap, unit) in zip(cols, chans):
            with col:
                show(map_figure(P, P["layers"][k], title, cmap, False, unit, height=420, cols=len(chans)))
    with tabs[3]:
        k = P["layers"]["k"].ravel(); th = P["layers"]["eth"].ravel(); kd = rad["kd"].ravel(); alt = rad["alteration"].ravel()
        rng = np.random.default_rng(0); pick = rng.choice(k.size, size=min(15000, k.size), replace=False)
        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=th[pick], y=k[pick], mode="markers", marker=dict(size=3, color=kd[pick], colorscale="RdYlBu_r", cmin=-2, cmax=2,
                                   colorbar=dict(title="Kd", thickness=12)), name="cells", hoverinfo="skip"))
        sel = pick[alt[pick]]
        fig.add_trace(go.Scattergl(x=th[sel], y=k[sel], mode="markers", marker=dict(size=4, color="black"), name="flagged alteration"))
        xs = np.linspace(np.nanpercentile(th, 1), np.nanpercentile(th, 99), 50)
        slope = np.nanmean(k) / max(np.nanmean(th), 1e-9)
        fig.add_trace(go.Scatter(x=xs, y=slope * xs, mode="lines", line=dict(color=GREEN, dash="dash"), name="regional K:eTh trend"))
        fig.update_layout(title="Potassium against thorium (random sample of cells)", xaxis_title="eTh (ppm)", yaxis_title="K (%)", height=480,
                          margin=dict(l=10, r=10, t=44, b=10), plot_bgcolor="white", legend=dict(orientation="h", y=-0.15))
        show(fig)
        st.caption("Cells above the regional trend carry more potassium than their thorium content predicts: the signature of potassic alteration "
                   "or K-feldspar-rich rock. Cells below it are leached.")

# --------------------------------------------------------------- 4 regolith
elif step == REG:
    P = need_project()
    st.header("Regolith and gravity")
    token, mag, rad, reg, grav = all_results(P)
    kpi_band(P, mag, rad, reg); source_note()
    tabs = st.tabs(["Regolith regime", "Gravity"])
    with tabs[0]:
        if reg is None:
            st.info("Add an elevation grid (for example SRTM) in step 1 to classify the regolith. Without it, radiometric evidence is used at full weight everywhere.")
        else:
            st.write("Transported cover hides the bedrock signal and produces false surface anomalies. This first-pass classification, from elevation "
                     "and radiometrics, marks where surface evidence can be trusted.")
            a, b = st.columns([3, 2])
            with a:
                show(map_figure(P, reg["regime"], "Regolith regime", discrete=REGIME_COLOURS))
            with b:
                share = pd.DataFrame({"Regime": [REGIMES[i] for i in (1, 2, 3)], "Share of area (%)": [round(100 * float((reg["regime"] == i).mean()), 1) for i in (1, 2, 3)],
                                      "Weight given to surface radiometrics": [0.6, 1.0, 0.2]})
                st.dataframe(share, use_container_width=True, hide_index=True)
                st.markdown("Brown: relict lateritic surfaces. Cream: erosional ground with in-situ material. Blue: depositional, transported cover.")
                st.caption("Rule-based and uncalibrated. Check against field observation before relying on it.")
                show(map_figure(P, reg["slope"], "Slope", "Greys", False, "deg", height=320))
            show(map_figure(P, P["layers"]["dem"], "Elevation", "Earth", False, "m", height=460, cols=1))
    with tabs[1]:
        if grav is None:
            st.info("No gravity grid in this dataset. Add a Bouguer gravity grid in step 1 (ground survey, or satellite models such as WGM2012 or GGMplus) "
                    "to include low-density granite as evidence.")
        else:
            st.write("Fertile pegmatite-bearing granites are less dense than their host rocks, so they appear as gravity lows.")
            a, b = st.columns(2)
            with a:
                show(map_figure(P, P["layers"]["grav"] - np.median(P["layers"]["grav"]), "Bouguer gravity", "Spectral_r", True, "mGal"))
            with b:
                show(map_figure(P, grav["residual"], "Residual gravity (lows may mark granite)", "Spectral_r", True, "mGal", contour=0.0))
            a, b = st.columns(2)
            with a:
                show(map_figure(P, grav["vd"] * 1e4, "Vertical gravity gradient (Gzz)", "PuOr_r", True, "Eotvos", contour=0.0))
            with b:
                show(map_figure(P, grav["hg"], "Horizontal gravity gradient (density contacts)", "Magma", False, "mGal/m"))

# ---------------------------------------------- 5, 6, 7 share the same score
else:
    P = need_project()
    token, mag, rad, reg, grav = all_results(P)
    all_evidence = pipeline.evidence_layers(mag, rad, reg, grav)
    if not all_evidence:
        st.header("Prospectivity"); st.info("Magnetic or radiometric data is needed to build evidence layers."); st.stop()
    occ = state["occurrences"]
    can_train = occ is not None and len(occ) >= 10
    score_token = token + repr(mag_settings())
    if state.get("score_token") != score_token:  # data or settings changed: drop the old score
        state["score"] = None; state["docx"] = None
        state["score_token"] = score_token

    def default_score():
        if state.get("score") is None:
            state["score"] = prospectivity.knowledge_score(all_evidence)
            state["uncertainty"] = None; state["model_info"] = None; state["score_label"] = "Prospectivity score (weighted overlay)"
            state["evidence_used"] = list(all_evidence)
        return state["score"]

    def current_targets():
        score = default_score()
        keep = state.get("keep_pct", 5); spacing = state.get("spacing_m", 1500)
        threshold = float(np.quantile(score, 1 - keep / 100.0))
        targets = prospectivity.pick_targets(score, P, threshold, float(spacing), regime=reg["regime"] if reg else None,
                                             depth=mag["depth"] if mag else None, uncertainty=state.get("uncertainty"))
        return score, threshold, targets

    # ................................................................ 5
    if step == PROS:
        st.header("Prospectivity")
        kpi_band(P, mag, rad, reg); source_note()
        with st.container(border=True):
            st.markdown("**Model settings**")
            a, b = st.columns([2, 3])
            modes = ["Weighted overlay (no training data needed)"] + (["Trained on known occurrences"] if can_train else [])
            mode = a.radio("Method", modes, key="mode")
            use = b.multiselect("Active evidence layers", list(all_evidence), default=list(all_evidence), key="evidence_pick")
            if not can_train:
                a.caption("Upload ten or more known occurrences in step 1 to enable the trained models.")
            if not use:
                st.error("Keep at least one evidence layer active."); st.stop()
            evidence = {k: all_evidence[k] for k in use}
            if mode.startswith("Weighted"):
                st.caption("Each layer is scaled from 0 to 1 and combined with the weights below. The result ranks ground for follow-up. It is not a probability.")
                cols = st.columns(len(evidence))
                weights = {}
                for col, name in zip(cols, evidence):
                    weights[name] = col.slider(name, 0.0, 3.0, float(prospectivity.DEFAULT_WEIGHTS.get(name, 1.0)), 0.25, key="w_" + name)
                if sum(weights.values()) == 0:
                    st.error("Give at least one layer a weight above zero."); st.stop()
                state["score"] = prospectivity.knowledge_score(evidence, weights)
                state["uncertainty"] = None; state["model_info"] = None; state["score_label"] = "Prospectivity score (weighted overlay)"
            else:
                c1, c2, c3, c4 = st.columns(4)
                model_kind = c1.selectbox("Classifier", prospectivity.MODELS, key="model_kind")
                n_trees = c2.slider("Trees / boosting rounds", 50, 500, 200, 50, disabled=model_kind == "Logistic regression")
                max_depth = c3.slider("Maximum tree depth", 2, 16, 8, 1, disabled=model_kind == "Logistic regression")
                buffer_m = c4.slider("Radius treated as mineralised (m)", 100, 1000, 300, 50)
                occ_key = hashlib.md5(occ[["row", "col"]].to_numpy().tobytes()).hexdigest()
                prob, unc, info = cached_model(score_token, evidence, occ, P["dx"], float(buffer_m), tuple(evidence), occ_key, model_kind, int(n_trees), int(max_depth))
                state["score"] = prob; state["uncertainty"] = unc; state["model_info"] = info; state["score_label"] = "Prospectivity (%s)" % model_kind
            state["evidence_used"] = list(evidence)
        score = state["score"]
        info = state.get("model_info")
        if info:
            a, b, c, d = st.columns(4)
            a.metric("Known occurrences used", info["n_occurrences"])
            b.metric("Cross-validated AUC", "n/a" if info["auc"] is None else "%.2f" % info["auc"],
                     help="Area under the ROC curve, scored on spatial blocks the model did not see during training.")
            c.metric("Spatial blocks scored", info["blocks_scored"])
            d.metric("Training cells", "%d + / %d -" % (info["n_positive_cells"], info["n_negative_cells"]))
            if info["auc"] is None:
                st.warning("The occurrences are too clustered to test the model on separate areas. Treat the map as untested.")
            elif info["n_occurrences"] < 25:
                st.warning("Fewer than 25 occurrences: the accuracy figure is unstable and likely optimistic.")
        a, b = st.columns([3, 2])
        with a:
            show(map_figure(P, score, state["score_label"], "Viridis", False, "", occurrences=occ, height=560))
        with b:
            if info and info.get("importance"):
                imp = pd.Series(info["importance"]).sort_values()
                fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=GREEN))
                fig.update_layout(title="What the model relied on", height=280, margin=dict(l=10, r=10, t=44, b=10), plot_bgcolor="white")
                show(fig)
                if info.get("fpr"):
                    fig = go.Figure([go.Scatter(x=info["fpr"], y=info["tpr"], mode="lines", line=dict(color=GREEN, width=3), name="Model"),
                                     go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#999999", dash="dash"), name="Chance")])
                    fig.update_layout(title="ROC curve (spatial cross-validation)", xaxis_title="False positive rate", yaxis_title="True positive rate",
                                      height=300, margin=dict(l=10, r=10, t=44, b=10), plot_bgcolor="white")
                    show(fig)
            else:
                weights_now = {n: state.get("w_" + n, prospectivity.DEFAULT_WEIGHTS.get(n, 1.0)) for n in evidence}
                fig = go.Figure(go.Bar(x=list(weights_now.values()), y=list(weights_now.keys()), orientation="h", marker_color=GREEN))
                fig.update_layout(title="Evidence weights in use", height=280, margin=dict(l=10, r=10, t=44, b=10), plot_bgcolor="white")
                show(fig)
                hist = go.Figure(go.Histogram(x=score.ravel(), nbinsx=50, marker_color=GREEN))
                hist.update_layout(title="Score distribution", xaxis_title="Score", height=260, margin=dict(l=10, r=10, t=44, b=10), plot_bgcolor="white")
                show(hist)
            if state.get("uncertainty") is not None:
                show(map_figure(P, state["uncertainty"], "Uncertainty (spread between trees)", "Greys", False, "", height=360))
            if reg is not None and rad is not None:
                st.caption("Radiometric evidence is down-weighted over relict and transported cover.")
        with st.expander("Show each evidence layer"):
            cols = st.columns(3)
            for i, (name, layer) in enumerate(evidence.items()):
                with cols[i % 3]:
                    show(map_figure(P, layer, name, "Viridis", False, "", height=360, cols=3))

    # ................................................................ 6
    elif step == TARG:
        st.header("Targets and export")
        kpi_band(P, mag, rad, reg); source_note()
        a, b = st.columns(2)
        state["keep_pct"] = a.slider("Keep the top share of the area (%)", 1, 20, int(state.get("keep_pct", 5)), 1,
                                     help="The decision threshold. 5% keeps only the best twentieth of the survey.")
        state["spacing_m"] = b.slider("Minimum spacing between targets (m)", 500, 5000, int(state.get("spacing_m", 1500)), 250)
        score, threshold, targets = current_targets()
        a, b, c, d = st.columns(4)
        a.metric("Score threshold", "%.2f" % threshold)
        b.metric("Search area reduced by", "%.0f%%" % prospectivity.area_reduction(score, threshold))
        c.metric("Targets", len(targets))
        d.metric("High priority", int((targets["Priority"] == "High").sum()) if len(targets) else 0)
        show(map_figure(P, score, state["score_label"] + " with ranked targets", "Viridis", False, "", points=targets, occurrences=occ, height=600))
        st.dataframe(targets, use_container_width=True, hide_index=True)
        with st.expander("Why these targets? Evidence at each one"):
            used = {k: all_evidence[k] for k in state.get("evidence_used", list(all_evidence)) if k in all_evidence}
            st.dataframe(explain_targets(targets, used, P), use_container_width=True, hide_index=True)
            st.caption("Each evidence layer is scaled 0 to 1; higher means more favourable. 'Strongest evidence' is the layer that contributes most at that cell.")
        st.caption("Targets are the highest points of the score, ranked best first. They are places to check on the ground, not drill-ready collars, "
                   "and this is not a JORC or NI 43-101 report.")
        st.subheader("Target package")
        notes = ["Simulated example data: the results do not describe real ground."] if state["simulated"] else []
        r1 = st.columns(4)
        r1[0].download_button("Targets (CSV)", targets.to_csv(index=False).encode(), "regolithsense_targets.csv", "text/csv", use_container_width=True)
        r1[1].download_button("Targets (GeoJSON, for QGIS)", report.targets_geojson(targets), "regolithsense_targets.geojson", "application/geo+json", use_container_width=True)
        r1[2].download_button("Targets (KML, Google Earth)", report.targets_kml(targets), "regolithsense_targets.kml", "application/vnd.google-earth.kml+xml", use_container_width=True)
        r1[3].download_button("Workbook (Excel)", report.excel_workbook(P, targets, state.get("model_info")), "regolithsense_workbook.xlsx",
                              "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        r2 = st.columns(4)
        r2[0].download_button("Score grid (XYZ)", report.grid_xyz(P, score, "score"), "regolithsense_score.csv", "text/csv", use_container_width=True)
        r2[1].download_button("All grids (XYZ)", report.all_layers_xyz(P, {"score": score}), "regolithsense_grids.csv", "text/csv", use_container_width=True)
        if r2[2].button("Build Word report", use_container_width=True, type="primary"):
            with st.spinner("Drawing maps and writing the report"):
                state["docx"] = report.word_report(P, state["source"], mag, rad, reg, score, state["score_label"], targets, state.get("model_info"), notes)
        if state.get("docx"):
            r2[3].download_button("Download Word report", state["docx"], "regolithsense_report.docx",
                                  "application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True)

    # ................................................................ 7
    elif step == VIEW3D:
        st.header("3D view")
        kpi_band(P, mag, rad, reg); source_note()
        score, threshold, targets = current_targets()
        drapes = {"Prospectivity score": (score, "Viridis"), "Magnetic field": (mag["field"] if mag else None, "RdBu_r")}
        if rad is not None:
            drapes["Potassium enrichment (Kd)"] = (rad["kd"], "RdYlBu_r")
        if reg is not None:
            drapes["Regolith regime"] = (reg["regime"].astype(float), [[0, REGIME_COLOURS[0]], [0.5, REGIME_COLOURS[1]], [1, REGIME_COLOURS[2]]])
        if grav is not None:
            drapes["Bouguer gravity"] = (P["layers"]["grav"], "Spectral_r")
        drapes = {k: v for k, v in drapes.items() if v[0] is not None}
        a, b, c = st.columns(3)
        drape = a.selectbox("Surface colour", list(drapes))
        exag = b.slider("Vertical exaggeration", 1.0, 10.0, 4.0, 0.5)
        show_targets = c.checkbox("Show targets with estimated depth", value=True)
        stepn = max(int(np.ceil(max(P["nx"], P["ny"]) / 220.0)), 1)
        x, y = axes_km(P, stepn)
        dem = P["layers"]["dem"] if "dem" in P["layers"] else np.zeros((P["ny"], P["nx"]))
        zs = dem[::stepn, ::stepn]
        col, cmap = drapes[drape]
        colz = np.asarray(col, float)[::stepn, ::stepn]
        lo, hi = np.nanpercentile(colz, (2, 98))
        fig = go.Figure(go.Surface(x=x, y=y, z=zs, surfacecolor=colz, colorscale=cmap, cmin=lo, cmax=hi, colorbar=dict(title=drape, thickness=14),
                                   hovertemplate="E %{x:.1f} km<br>N %{y:.1f} km<br>elev %{z:.0f} m<extra></extra>"))
        if show_targets and len(targets):
            for _, t in targets.iterrows():
                ex, ny_ = t["Easting (m)"] / 1000.0, t["Northing (m)"] / 1000.0
                r = min(int((t["Northing (m)"] - P["y0"]) / P["dx"]), P["ny"] - 1); c_ = min(int((t["Easting (m)"] - P["x0"]) / P["dx"]), P["nx"] - 1)
                top = float(dem[r, c_]); d = t.get("Depth to magnetic source (m)")
                bottom = top - (float(d) if d is not None and not pd.isna(d) else 150.0)
                fig.add_trace(go.Scatter3d(x=[ex, ex], y=[ny_, ny_], z=[top + 20, bottom], mode="lines+markers+text", text=[t["Target"], ""],
                                           textposition="top center", line=dict(color="#d62728" if t["Priority"] == "High" else "#f4a261", width=5),
                                           marker=dict(size=[4, 2], color="#d62728" if t["Priority"] == "High" else "#f4a261"), showlegend=False,
                                           hovertemplate="%s<br>score %.2f<br>%s<extra></extra>" % (t["Target"], t["Score"], t.get("Regolith regime", ""))))
        zr = float(np.nanmax(zs) - np.nanmin(zs)) or 1.0
        xr = (x[-1] - x[0]) * 1000.0; yr = (y[-1] - y[0]) * 1000.0
        fig.update_layout(height=680, margin=dict(l=0, r=0, t=10, b=0),
                          scene=dict(xaxis_title="Easting (km)", yaxis_title="Northing (km)", zaxis_title="Elevation (m)",
                                     aspectmode="manual", aspectratio=dict(x=1, y=yr / xr, z=exag * zr / xr), camera=dict(eye=dict(x=1.3, y=-1.4, z=0.9))))
        show(fig)
        st.caption("Terrain from the elevation grid, coloured by the chosen layer. Target pins reach down to the tilt-depth estimate of the nearest magnetic "
                   "source, a rough guide to how deep the cover is at each target, not a measured thickness." if "dem" in P["layers"] else
                   "No elevation grid in this dataset, so the surface is flat. Add SRTM in step 1 for terrain.")

    # ................................................................ 8
    else:
        st.header("Decision brief")
        source_note()
        score, threshold, targets = current_targets()
        area = P["nx"] * P["ny"] * P["dx"] ** 2 / 1e6
        kept = area * float(np.mean(score >= threshold))
        info = state.get("model_info")
        st.write("A one-page summary of what the data say about **%s**, written for someone deciding where to send a field team." % state["source"].split(" (")[0])
        c = st.columns(4)
        c[0].metric("Survey area", "%.0f km²" % area)
        c[1].metric("Ground worth visiting", "%.0f km²" % kept, "-%.0f%%" % prospectivity.area_reduction(score, threshold), delta_color="inverse")
        c[2].metric("Ranked targets", len(targets), "%d high priority" % int((targets["Priority"] == "High").sum()) if len(targets) else None)
        c[3].metric("Model check", "AUC %.2f" % info["auc"] if info and info.get("auc") is not None else "Expert weights", help="Spatially cross-validated where occurrences were supplied.")
        a, b = st.columns([3, 2])
        with a:
            show(map_figure(P, score, "Prospectivity with ranked targets", "Viridis", False, "", points=targets, occurrences=occ, height=520))
        with b:
            st.markdown("**Top targets**")
            used = {k: all_evidence[k] for k in state.get("evidence_used", list(all_evidence)) if k in all_evidence}
            why = explain_targets(targets.head(5), used, P)
            for (_, t), (_, w) in zip(targets.head(5).iterrows(), why.iterrows()):
                with st.container(border=True):
                    st.markdown("**%s** · %s priority · score %.2f" % (t["Target"], t["Priority"], t["Score"]))
                    bits = ["%.5f°N %.5f°E" % (t["Latitude"], t["Longitude"])]
                    if "Regolith regime" in t: bits.append("%s regolith" % t["Regolith regime"])
                    if t.get("Depth to magnetic source (m)") is not None and not pd.isna(t.get("Depth to magnetic source (m)")):
                        bits.append("source about %d m down" % int(t["Depth to magnetic source (m)"]))
                    bits.append("led by " + str(w["Strongest evidence"]).lower())
                    st.caption(" · ".join(bits))
        st.subheader("What the field team should do")
        notes = []
        if reg is not None:
            dep = int((targets.get("Regolith regime", pd.Series(dtype=str)) == "depositional").sum()) if len(targets) else 0
            notes.append("%d of %d targets sit on transported cover, where surface sampling will not see bedrock; plan auger or pit sampling there." % (dep, len(targets)))
        if mag is not None:
            d = mag["depth"][np.isfinite(mag["depth"])]
            if d.size:
                notes.append("Magnetic sources are typically %.0f m below surface (median tilt-depth), so expect cover of that order or less at the targets." % np.median(d))
        if rad is not None:
            notes.append("Potassic alteration is flagged over %.1f%% of the sheet; check targets led by potassium evidence with a hand-held spectrometer and K-feldspar logging." % (100 * rad["alteration"].mean()))
        if info is None:
            notes.append("No known occurrences were supplied, so the ranking uses expert weights. Add the known pegmatites of the area to test and retrain the model.")
        elif info.get("auc") is not None and info["n_occurrences"] < 25:
            notes.append("The trained model has only %d occurrences; the AUC of %.2f is optimistic. Treat the ranking as a guide, not a verdict." % (info["n_occurrences"], info["auc"]))
        notes.append("Visit targets in rank order; log outcrop, float, artisanal workings and take soil or stream samples. Feed confirmed and empty sites back into the occurrence file.")
        for n in notes:
            st.markdown("- " + n)
        with st.expander("Illustrative cost framing"):
            cost = st.number_input("Reconnaissance cost per km² (₦ thousand, your estimate)", 10, 5000, 150, 10,
                                   help="Field mapping, sampling and logistics per square kilometre. Change it to your own budget figures.")
            st.write("Covering the whole sheet at that rate: **₦%.0f million**. Covering only the ranked ground: **₦%.0f million**." % (area * cost / 1000, kept * cost / 1000))
            st.caption("Arithmetic on your own assumption, shown to make the trade-off concrete. It is not a quotation.")
        st.subheader("Export")
        e = st.columns(3)
        e[0].download_button("Targets (CSV)", targets.to_csv(index=False).encode(), "regolithsense_targets.csv", "text/csv", use_container_width=True)
        e[1].download_button("Targets (KML)", report.targets_kml(targets), "regolithsense_targets.kml", "application/vnd.google-earth.kml+xml", use_container_width=True)
        if e[2].button("Build Word report", use_container_width=True, type="primary", key="brief_docx"):
            with st.spinner("Drawing maps and writing the report"):
                state["docx"] = report.word_report(P, state["source"], mag, rad, reg, score, state["score_label"], targets, info,
                                                   ["Simulated example data: the results do not describe real ground."] if state["simulated"] else [])
        if state.get("docx"):
            st.download_button("Download Word report", state["docx"], "regolithsense_report.docx",
                               "application/vnd.openxmlformats-officedocument.wordprocessingml.document", key="brief_docx_dl")

if step not in (STEPS[0], METHODS):
    nav_buttons(step)
