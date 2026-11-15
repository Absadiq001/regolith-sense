"""
RegolithSense: targeting concealed pegmatites beneath weathering cover from
airborne magnetic and radiometric surveys.

Run with:  streamlit run app.py
"""
import hashlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import gridio, pipeline, prospectivity, report
from core.regolith import REGIMES

st.set_page_config(page_title="RegolithSense", layout="wide", initial_sidebar_state="expanded")

STEPS = ["1. Data", "2. Magnetics", "3. Radiometrics", "4. Regolith", "5. Prospectivity", "6. Targets and export"]
REGIME_COLOURS = ["#b5651d", "#e8dcc0", "#5b8fb0"]

state = st.session_state
state.setdefault("uploads", {})      # file name -> {kind or None: raw layer}
state.setdefault("kinds", {})        # file name -> chosen layer kind
state.setdefault("project", None)
state.setdefault("source", "")
state.setdefault("occurrences", None)
state.setdefault("simulated", False)


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


@st.cache_data(show_spinner="Training the model")
def cached_forest(token, _evidence, _occurrences, cell, buffer_m, names, occ_key):
    return prospectivity.train_forest(_evidence, _occurrences, cell, buffer_m=buffer_m)


def axes_km(project, step):
    x = (project["x0"] + np.arange(0, project["nx"], step) * project["dx"]) / 1000.0
    y = (project["y0"] + np.arange(0, project["ny"], step) * project["dx"]) / 1000.0
    return x, y


def map_figure(project, grid, title, colours="Viridis", symmetric=False, unit="", mask=None, points=None, occurrences=None, discrete=None, height=520, cols=2):
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


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("RegolithSense")
    st.caption("Targeting concealed pegmatites beneath weathering cover")
    step = st.radio("Workflow", STEPS, label_visibility="collapsed")
    st.divider()
    if state["project"] is not None:
        P = state["project"]
        st.markdown("**Current dataset**")
        st.write(state["source"])
        st.caption("%d x %d cells of %.0f m, UTM zone %dN" % (P["nx"], P["ny"], P["dx"], P["zone"]))
        st.caption("Layers: " + ", ".join(gridio.LAYER_NAMES[k].split(" (")[0] for k in P["layers"]))
        if state["simulated"]:
            st.warning("Simulated example data. Results are not real.")
    else:
        st.caption("No dataset loaded.")


# ------------------------------------------------------------------- 1 data
if step == STEPS[0]:
    st.header("Survey data")
    st.write("Upload the grids for one survey area. RegolithSense reads Geosoft grids (.grd) as supplied by NGSA and Oasis montaj, "
             "GeoTIFF, and CSV or XYZ point files. Layers in different cell sizes or coordinate systems are brought onto one grid.")
    left, right = st.columns([3, 2])
    with left:
        files = st.file_uploader("Survey files", type=["grd", "tif", "tiff", "csv", "xyz", "txt"], accept_multiple_files=True,
                                 help="Magnetic field, potassium, thorium, uranium, and optionally elevation and gravity.")
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
        cell = st.select_slider("Working cell size (m)", options=[50, 100, 125, 200, 250, 500], value=100)
        if st.button("Prepare data", type="primary", disabled=not chosen):
            try:
                with st.spinner("Bringing layers onto one grid"):
                    state["project"] = gridio.harmonise(chosen, cell=float(cell))
                state["source"] = ", ".join(sorted(state["uploads"]))
                state["simulated"] = False; state["occurrences"] = None
            except Exception as exc:
                st.error(str(exc))
            else:
                st.rerun()
    with right:
        st.subheader("No data to hand?")
        st.write("Load a simulated survey to see how the workflow runs. It is computer-generated and does not describe any real ground.")
        if st.button("Load simulated example"):
            state["project"] = gridio.harmonise(pipeline.simulated_layers(), cell=100.0)
            state["source"] = "Simulated example"; state["simulated"] = True; state["occurrences"] = None
            st.rerun()
        st.subheader("Known occurrences (optional)")
        st.write("A CSV of known pegmatite or mine locations with easting/northing or longitude/latitude columns. "
                 "With ten or more, the prospectivity model can be trained and tested on them.")
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
        st.subheader("Data preparation report")
        st.dataframe(pd.DataFrame(P["log"]), use_container_width=True, hide_index=True)
        st.caption("Blank cells are filled from the nearest valid cell. Radiometric spikes are replaced by the local median and negative concentrations set to zero.")
        palette = {"tmi": ("RdBu_r", True), "k": ("YlOrRd", False), "eth": ("YlGn", False), "eu": ("PuBu", False), "tc": ("Viridis", False),
                   "dem": ("Earth", False), "grav": ("Spectral_r", True)}
        cols = st.columns(3)
        for i, (kind, grid) in enumerate(P["layers"].items()):
            with cols[i % 3]:
                c, sym = palette.get(kind, ("Viridis", False))
                show(map_figure(P, grid - (np.median(grid) if sym else 0), gridio.LAYER_NAMES[kind], c, sym, height=380, cols=3,
                                occurrences=state["occurrences"]))

# -------------------------------------------------------------- 2 magnetics
elif step == STEPS[1]:
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
    st.write("Magnetic fields pass through non-magnetic regolith, so these maps show bedrock structure beneath the cover.")
    tabs = st.tabs(["Field", "Edges and lineaments", "Depth to source"])
    with tabs[0]:
        a, b = st.columns(2)
        with a:
            show(map_figure(P, mag["field"] - np.median(mag["field"]), "Magnetic field" + (" reduced to the equator" if state.get("reduce", True) else ""), "RdBu_r", True, "nT"))
        with b:
            show(map_figure(P, mag["residual"], "Residual field (regional removed)", "RdBu_r", True, "nT"))
    with tabs[1]:
        a, b = st.columns(2)
        with a:
            show(map_figure(P, mag["tilt"], "Tilt angle with traced lineaments", "PuOr_r", True, "deg", mask=mag["lineaments"]))
        with b:
            show(map_figure(P, mag["asig"], "Analytic signal", "Magma", False, "nT/m"))
        a, b = st.columns(2)
        with a:
            show(map_figure(P, mag["vd"], "First vertical derivative", "RdBu_r", True, "nT/m"))
        with b:
            table = mag["lineament_table"]
            if len(table):
                counts, edges = np.histogram(table[:, 1], bins=np.arange(0, 181, 15), weights=table[:, 0] / 1000.0)
                fig = go.Figure(go.Bar(x=["%d-%d" % (edges[i], edges[i + 1]) for i in range(len(counts))], y=counts, marker_color="#3d6b5c"))
                fig.update_layout(title="Lineament length by strike", xaxis_title="Strike (degrees from north)", yaxis_title="Total length (km)", height=520,
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
            st.caption("Estimates exist only along magnetic contacts. A %.0f m grid cannot resolve sources much shallower than one cell, "
                       "so treat these as upper bounds on cover thickness, to be checked by drilling or ground geophysics." % P["dx"])

# ----------------------------------------------------------- 3 radiometrics
elif step == STEPS[2]:
    P = need_project()
    st.header("Radiometric alteration")
    token, mag, rad, reg, grav = all_results(P)
    if rad is None:
        st.info("Potassium, thorium and uranium grids are all needed for this step."); st.stop()
    st.write("Potassium added by hydrothermal fluids shows as potassium enrichment relative to thorium, which stays largely immobile. "
             "The signal comes from the top 30 to 40 cm of the ground, so it is most reliable where the surface is in-situ rock or saprolite.")
    tabs = st.tabs(["Ternary image", "Alteration indices", "Channels"])
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
        a, b, c = st.columns(3)
        with a:
            show(map_figure(P, P["layers"]["k"], "Potassium", "YlOrRd", False, "%", height=420, cols=3))
        with b:
            show(map_figure(P, P["layers"]["eth"], "Thorium", "YlGn", False, "ppm", height=420, cols=3))
        with c:
            show(map_figure(P, P["layers"]["eu"], "Uranium", "PuBu", False, "ppm", height=420, cols=3))

# --------------------------------------------------------------- 4 regolith
elif step == STEPS[3]:
    P = need_project()
    st.header("Regolith and gravity")
    token, mag, rad, reg, grav = all_results(P)
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
        show(map_figure(P, P["layers"]["dem"], "Elevation", "Earth", False, "m", height=420, cols=1))
    st.subheader("Gravity")
    if grav is None:
        st.info("No gravity grid in this dataset. Add a Bouguer gravity grid in step 1 to include low-density granite as evidence.")
    else:
        a, b = st.columns(2)
        with a:
            show(map_figure(P, P["layers"]["grav"] - np.median(P["layers"]["grav"]), "Bouguer gravity", "Spectral_r", True, "mGal"))
        with b:
            show(map_figure(P, grav["residual"], "Residual gravity (lows may mark granite)", "Spectral_r", True, "mGal"))

# ---------------------------------------------- 5 and 6 share the same score
else:
    P = need_project()
    token, mag, rad, reg, grav = all_results(P)
    evidence = pipeline.evidence_layers(mag, rad, reg, grav)
    if not evidence:
        st.header("Prospectivity"); st.info("Magnetic or radiometric data is needed to build evidence layers."); st.stop()
    occ = state["occurrences"]
    can_train = occ is not None and len(occ) >= 10
    score_token = token + repr(mag_settings())
    if state.get("score_token") != score_token:  # data or settings changed: drop the old score
        state["score"] = None; state["docx"] = None
        state["score_token"] = score_token

    if step == STEPS[4]:
        st.header("Prospectivity")
        modes = ["Weighted overlay (no training data needed)"] + (["Random Forest trained on known occurrences"] if can_train else [])
        mode = st.radio("Method", modes, key="mode")
        if not can_train:
            st.caption("Upload ten or more known occurrences in step 1 to enable the trained model.")
        if mode.startswith("Weighted"):
            st.write("Each evidence layer is scaled from 0 to 1 and combined with the weights below. The result ranks ground for follow-up. It is not a probability.")
            cols = st.columns(len(evidence))
            weights = {}
            for col, name in zip(cols, evidence):
                weights[name] = col.slider(name, 0.0, 3.0, float(prospectivity.DEFAULT_WEIGHTS.get(name, 1.0)), 0.25, key="w_" + name)
            if sum(weights.values()) == 0:
                st.error("Give at least one layer a weight above zero."); st.stop()
            state["score"] = prospectivity.knowledge_score(evidence, weights)
            state["uncertainty"] = None; state["model_info"] = None; state["score_label"] = "Prospectivity score (weighted overlay)"
        else:
            buffer_m = st.slider("Distance around each occurrence treated as mineralised (m)", 100, 1000, 300, 50)
            occ_key = hashlib.md5(occ[["row", "col"]].to_numpy().tobytes()).hexdigest()
            prob, unc, info = cached_forest(token + repr(mag_settings()), evidence, occ, P["dx"], float(buffer_m), tuple(evidence), occ_key)
            state["score"] = prob; state["uncertainty"] = unc; state["model_info"] = info; state["score_label"] = "Prospectivity (Random Forest)"
            a, b, c = st.columns(3)
            a.metric("Known occurrences used", info["n_occurrences"])
            b.metric("Cross-validated AUC", "n/a" if info["auc"] is None else "%.2f" % info["auc"],
                     help="Area under the ROC curve, scored on spatial blocks the model did not see during training.")
            c.metric("Blocks scored", info["blocks_scored"])
            if info["auc"] is None:
                st.warning("The occurrences are too clustered to test the model on separate areas. Treat the map as untested.")
            elif info["n_occurrences"] < 25:
                st.warning("Fewer than 25 occurrences: the accuracy figure is unstable and likely optimistic.")
        score = state["score"]
        a, b = st.columns([3, 2])
        with a:
            show(map_figure(P, score, state["score_label"], "Viridis", False, "", occurrences=occ, height=560))
        with b:
            info = state.get("model_info")
            if info and info.get("importance"):
                imp = pd.Series(info["importance"]).sort_values()
                fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color="#3d6b5c"))
                fig.update_layout(title="What the model relied on", height=300, margin=dict(l=10, r=10, t=44, b=10), plot_bgcolor="white")
                show(fig)
                if info.get("fpr"):
                    fig = go.Figure([go.Scatter(x=info["fpr"], y=info["tpr"], mode="lines", line=dict(color="#3d6b5c", width=3), name="Model"),
                                     go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#999999", dash="dash"), name="Chance")])
                    fig.update_layout(title="ROC curve (spatial cross-validation)", xaxis_title="False positive rate", yaxis_title="True positive rate",
                                      height=300, margin=dict(l=10, r=10, t=44, b=10), plot_bgcolor="white")
                    show(fig)
            if state.get("uncertainty") is not None:
                show(map_figure(P, state["uncertainty"], "Uncertainty (spread between trees)", "Greys", False, "", height=360))
            st.markdown("**Evidence layers in use**")
            for name in evidence:
                st.write("- " + name)
            if reg is not None and rad is not None:
                st.caption("Radiometric evidence is down-weighted over relict and transported cover.")
        with st.expander("Show each evidence layer"):
            cols = st.columns(3)
            for i, (name, layer) in enumerate(evidence.items()):
                with cols[i % 3]:
                    show(map_figure(P, layer, name, "Viridis", False, "", height=360, cols=3))

    else:
        st.header("Targets and export")
        if state.get("score") is None:
            state["score"] = prospectivity.knowledge_score(evidence)
            state["uncertainty"] = None; state["model_info"] = None; state["score_label"] = "Prospectivity score (weighted overlay)"
        score = state["score"]
        a, b = st.columns(2)
        keep = a.slider("Keep the top share of the area (%)", 1, 20, 5, 1)
        spacing = b.slider("Minimum spacing between targets (m)", 500, 5000, 1500, 250)
        threshold = float(np.quantile(score, 1 - keep / 100.0))
        targets = prospectivity.pick_targets(score, P, threshold, float(spacing), regime=reg["regime"] if reg else None,
                                             depth=mag["depth"] if mag else None, uncertainty=state.get("uncertainty"))
        a, b, c = st.columns(3)
        area = P["nx"] * P["ny"] * P["dx"] ** 2 / 1e6
        a.metric("Survey area", "%.0f sq km" % area)
        b.metric("Search area reduced by", "%.0f%%" % prospectivity.area_reduction(score, threshold))
        c.metric("Targets", len(targets))
        show(map_figure(P, score, state["score_label"] + " with ranked targets", "Viridis", False, "", points=targets, occurrences=occ, height=600))
        st.dataframe(targets, use_container_width=True, hide_index=True)
        st.caption("Targets are the highest points of the score, ranked best first. They are places to check on the ground, not drill-ready collars, "
                   "and this is not a JORC or NI 43-101 report.")
        st.subheader("Export")
        notes = ["Simulated example data: the results do not describe real ground."] if state["simulated"] else []
        a, b, c, d = st.columns(4)
        a.download_button("Targets (CSV)", targets.to_csv(index=False).encode(), "regolithsense_targets.csv", "text/csv", use_container_width=True)
        b.download_button("Workbook (Excel)", report.excel_workbook(P, targets, state.get("model_info")), "regolithsense_workbook.xlsx",
                          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        c.download_button("Score grid (XYZ)", report.grid_xyz(P, score, "score"), "regolithsense_score.csv", "text/csv", use_container_width=True)
        if d.button("Build Word report", use_container_width=True):
            with st.spinner("Drawing maps and writing the report"):
                state["docx"] = report.word_report(P, state["source"], mag, rad, reg, score, state["score_label"], targets, state.get("model_info"), notes)
        if state.get("docx"):
            st.download_button("Download Word report", state["docx"], "regolithsense_report.docx",
                               "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
