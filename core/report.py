"""Word and Excel exports. Figures are drawn with matplotlib so they print cleanly."""
import io
from datetime import date

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import docx
from docx.shared import Inches, Pt

REGIME_COLOURS = ["#b5651d", "#e8dcc0", "#5b8fb0"]


def _extent(project):
    x0, y0, c = project["x0"], project["y0"], project["dx"]
    return [x0 / 1000, (x0 + project["nx"] * c) / 1000, y0 / 1000, (y0 + project["ny"] * c) / 1000]


def map_png(project, grid, title, cmap="viridis", symmetric=False, label="", points=None, mask=None, rgb=False, classes=False):
    """Render one map to PNG bytes."""
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=150)
    ext = _extent(project)
    if rgb:
        ax.imshow(grid, origin="lower", extent=ext)
    elif classes:
        ax.imshow(grid, origin="lower", extent=ext, cmap=ListedColormap(REGIME_COLOURS), vmin=1, vmax=3)
    else:
        finite = grid[np.isfinite(grid)]
        lo, hi = np.percentile(finite, (2, 98)) if finite.size else (0, 1)
        if symmetric:
            m = max(abs(lo), abs(hi)); lo, hi = -m, m
        im = ax.imshow(grid, origin="lower", extent=ext, cmap=cmap, vmin=lo, vmax=hi)
        fig.colorbar(im, ax=ax, shrink=0.8, label=label)
    if mask is not None:
        rr, cc = np.nonzero(mask)
        ax.plot(ext[0] + cc * project["dx"] / 1000, ext[2] + rr * project["dx"] / 1000, ",", color="black")
    if points is not None and len(points):
        ax.plot(points["Easting (m)"] / 1000, points["Northing (m)"] / 1000, "o", mfc="none", mec="white", ms=6, mew=1.3)
    ax.set_title(title, fontsize=11); ax.set_xlabel("Easting (km)"); ax.set_ylabel("Northing (km)")
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
    return buf.getvalue()


def word_report(project, source, mag, rad, reg, score, score_label, targets, model_info=None, notes=None):
    doc = docx.Document()
    doc.styles["Normal"].font.size = Pt(10.5)
    doc.add_heading("RegolithSense exploration targeting report", 0)
    doc.add_paragraph("Dataset: %s" % source)
    doc.add_paragraph("Generated: %s" % date.today().isoformat())
    c = project["dx"]
    doc.add_paragraph("Grid: %d by %d cells of %.0f m, UTM zone %dN (WGS 84). Area: %.0f square km." % (
        project["nx"], project["ny"], c, project["zone"], project["nx"] * project["ny"] * c * c / 1e6))
    doc.add_paragraph(
        "This report was produced automatically from the uploaded survey grids. It is a targeting aid for planning "
        "follow-up work. It is not a resource estimate and has not been reviewed by a Competent Person.")
    for n in notes or []:
        doc.add_paragraph(n, style="List Bullet")

    doc.add_heading("Data preparation", 1)
    log = pd.DataFrame(project["log"])
    _table(doc, log)

    def figure(png, caption):
        doc.add_picture(io.BytesIO(png), width=Inches(5.6))
        doc.add_paragraph(caption).runs[0].italic = True

    if mag:
        doc.add_heading("Magnetic interpretation", 1)
        figure(map_png(project, mag["field"], "Magnetic field", "RdBu_r", True, "nT"), "Magnetic field used for interpretation.")
        figure(map_png(project, mag["tilt"], "Tilt angle and traced lineaments", "PuOr_r", True, "degrees", mask=mag["lineaments"]),
               "Tilt angle. Black traces mark magnetic contacts and lineaments.")
        figure(map_png(project, mag["asig"], "Analytic signal", "magma", False, "nT/m"), "Analytic signal amplitude.")
        d = mag["depth"][np.isfinite(mag["depth"])]
        if d.size:
            doc.add_paragraph("Tilt-depth estimates to the top of magnetic sources: median %.0f m, middle 80%% between %.0f and %.0f m below ground. "
                              "The %.0f m grid cell limits how shallow a source can be resolved." % (np.median(d), np.percentile(d, 10), np.percentile(d, 90), c))
    if rad:
        doc.add_heading("Radiometric alteration", 1)
        figure(map_png(project, rad["ternary"], "Ternary image: K red, eTh green, eU blue", rgb=True), "Radiometric ternary image.")
        figure(map_png(project, rad["kd"], "Potassium enrichment relative to thorium (Kd)", "RdYlBu_r", True, "relative deviation", mask=rad["alteration"]),
               "Potassium enrichment. Black marks cells flagged as potassic alteration.")
    if reg:
        doc.add_heading("Regolith regime", 1)
        figure(map_png(project, reg["regime"], "Relict (brown), erosional (cream), depositional (blue)", classes=True), "First-pass regolith-landform classes.")
    doc.add_heading("Prospectivity and targets", 1)
    figure(map_png(project, score, score_label, "viridis", False, "score", points=targets), "%s. Circles mark ranked targets." % score_label)
    if model_info and model_info.get("auc") is not None:
        doc.add_paragraph("Model trained on %d known occurrences. Area under the ROC curve from spatial block cross-validation: %.2f." % (
            model_info["n_occurrences"], model_info["auc"]))
    elif model_info is None:
        doc.add_paragraph("The score is a weighted overlay of the evidence layers. It ranks ground for follow-up and is not a probability.")
    if targets is not None and len(targets):
        _table(doc, targets)
    buf = io.BytesIO(); doc.save(buf)
    return buf.getvalue()


def _table(doc, df):
    t = doc.add_table(rows=1, cols=len(df.columns)); t.style = "Light Grid Accent 1"
    for i, col in enumerate(df.columns):
        t.rows[0].cells[i].text = str(col)
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
    for r in t.rows:
        for cell in r.cells:
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(7.5)


def excel_workbook(project, targets, model_info=None):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        (targets if targets is not None else pd.DataFrame()).to_excel(xl, sheet_name="Targets", index=False)
        pd.DataFrame(project["log"]).to_excel(xl, sheet_name="Data preparation", index=False)
        if model_info and model_info.get("importance"):
            pd.DataFrame({"Evidence layer": list(model_info["importance"]), "Importance": list(model_info["importance"].values())}).to_excel(
                xl, sheet_name="Model", index=False)
    return buf.getvalue()


def grid_xyz(project, grid, name):
    """A grid as x, y, value text so it opens in Oasis montaj, QGIS or Surfer."""
    rr, cc = np.mgrid[0:project["ny"], 0:project["nx"]]
    df = pd.DataFrame({"easting": project["x0"] + cc.ravel() * project["dx"], "northing": project["y0"] + rr.ravel() * project["dx"], name: np.round(grid.ravel(), 5)})
    return df.to_csv(index=False).encode()


def targets_geojson(targets):
    """Targets as a GeoJSON FeatureCollection (WGS84) for QGIS, Google Earth or a web map."""
    import json
    feats = []
    for _, t in targets.iterrows():
        props = {k: (None if (isinstance(v, float) and np.isnan(v)) else (v.item() if hasattr(v, "item") else v)) for k, v in t.items()}
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [float(t["Longitude"]), float(t["Latitude"])]}, "properties": props})
    return json.dumps({"type": "FeatureCollection", "features": feats}, indent=1).encode()


def targets_kml(targets, name="RegolithSense targets"):
    """Targets as KML so they open directly in Google Earth."""
    from xml.sax.saxutils import escape
    colour = {"High": "ff1c54b4", "Medium": "ff1a6d8a", "Low": "ff8c8c8c"}
    rows = []
    for _, t in targets.iterrows():
        desc = "<br/>".join("%s: %s" % (escape(str(k)), escape(str(v))) for k, v in t.items())
        rows.append("<Placemark><name>%s</name><description><![CDATA[%s]]></description><Style><IconStyle><color>%s</color></IconStyle></Style>"
                    "<Point><coordinates>%.6f,%.6f,0</coordinates></Point></Placemark>" % (escape(str(t["Target"])), desc, colour.get(t.get("Priority"), "ff8c8c8c"),
                                                                                           t["Longitude"], t["Latitude"]))
    return ('<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>%s</name>%s</Document></kml>'
            % (escape(name), "".join(rows))).encode()


def all_layers_xyz(project, extra=None):
    """Every prepared layer (and any extra grids) as one CSV of easting, northing, values."""
    import io
    cols = dict(project["layers"]); cols.update(extra or {})
    step = max(int(np.ceil(max(project["nx"], project["ny"]) / 600.0)), 1)
    rr, cc = np.mgrid[0:project["ny"]:step, 0:project["nx"]:step]
    df = pd.DataFrame({"easting": project["x0"] + cc.ravel() * project["dx"], "northing": project["y0"] + rr.ravel() * project["dx"]})
    for k, g in cols.items():
        df[k] = np.asarray(g, float)[::step, ::step].ravel()
    buf = io.StringIO(); df.to_csv(buf, index=False, float_format="%.4f"); return buf.getvalue().encode()


# --------------------------------------------------------------------- figure pack
def _png(fig, dpi=150):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight"); plt.close(fig); return buf.getvalue()


def _bar_png(names, values, title, xlabel=""):
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(names) + 1.4))
    ax.barh(np.array(names)[order], np.array(values)[order], color="#2f5d50")
    ax.set_title(title, fontsize=11); ax.set_xlabel(xlabel)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return _png(fig)


def _hist_png(values, title, xlabel, bins=40):
    v = np.asarray(values, float); v = v[np.isfinite(v)]
    fig, ax = plt.subplots(figsize=(6.4, 3.6)); ax.hist(v, bins=bins, color="#2f5d50")
    ax.set_title(title, fontsize=11); ax.set_xlabel(xlabel); ax.set_ylabel("Cells")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return _png(fig)


def _rose_png(table):
    counts, edges = np.histogram(table[:, 1], bins=np.arange(0, 181, 15), weights=table[:, 0] / 1000.0)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(["%d-%d" % (edges[i], edges[i + 1]) for i in range(len(counts))], counts, color="#2f5d50")
    ax.set_title("Lineament length by strike (%d lineaments, %.0f km)" % (len(table), table[:, 0].sum() / 1000.0), fontsize=11)
    ax.set_xlabel("Strike (degrees from north)"); ax.set_ylabel("Total length (km)"); plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return _png(fig)


def _roc_png(info):
    fig, ax = plt.subplots(figsize=(5, 5)); ax.plot(info["fpr"], info["tpr"], color="#2f5d50", lw=2.5, label="Model (AUC %.2f)" % info["auc"])
    ax.plot([0, 1], [0, 1], "--", color="#999999", label="Chance"); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (spatial cross-validation)", fontsize=11); ax.legend(loc="lower right")
    return _png(fig)


def _crossplot_png(k, th, kd, alt):
    rng = np.random.default_rng(0); k = k.ravel(); th = th.ravel(); kd = kd.ravel(); alt = alt.ravel()
    pick = rng.choice(k.size, size=min(15000, k.size), replace=False)
    fig, ax = plt.subplots(figsize=(6.4, 5)); sc = ax.scatter(th[pick], k[pick], c=kd[pick], s=3, cmap="RdYlBu_r", vmin=-2, vmax=2)
    sel = pick[alt[pick]]; ax.scatter(th[sel], k[sel], s=5, c="black", label="flagged alteration")
    xs = np.linspace(np.nanpercentile(th, 1), np.nanpercentile(th, 99), 50); ax.plot(xs, np.nanmean(k) / max(np.nanmean(th), 1e-9) * xs, "--", color="#2f5d50", label="regional K:eTh trend")
    fig.colorbar(sc, ax=ax, label="Kd"); ax.set_xlabel("eTh (ppm)"); ax.set_ylabel("K (%)"); ax.set_title("Potassium against thorium", fontsize=11); ax.legend(loc="upper left", fontsize=8)
    return _png(fig)


def surface_png(project, dem, drape, cmap, title, label, targets=None, exag=4.0, azim=-60, elev=32, symmetric=False):
    """Terrain with a layer draped over it, as a static image (matplotlib 3D)."""
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    step = max(int(np.ceil(max(project["nx"], project["ny"]) / 140.0)), 1)
    z = np.asarray(dem, float)[::step, ::step]; c = np.asarray(drape, float)[::step, ::step]
    x = (project["x0"] + np.arange(0, project["nx"], step) * project["dx"]) / 1000.0
    y = (project["y0"] + np.arange(0, project["ny"], step) * project["dx"]) / 1000.0
    X, Y = np.meshgrid(x, y)
    lo, hi = np.nanpercentile(c, (2, 98))
    if symmetric:
        m = max(abs(lo), abs(hi)); lo, hi = -m, m
    norm = matplotlib.colors.Normalize(lo, hi); cmap_obj = plt.get_cmap(cmap)
    fig = plt.figure(figsize=(9, 7)); ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, z, facecolors=cmap_obj(norm(np.nan_to_num(c, nan=lo))), rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    if targets is not None and len(targets):
        for _, t in targets.iterrows():
            r = int(np.clip((t["Northing (m)"] - project["y0"]) / project["dx"] / step, 0, z.shape[0] - 1)); cc = int(np.clip((t["Easting (m)"] - project["x0"]) / project["dx"] / step, 0, z.shape[1] - 1))
            top = float(z[r, cc]); d = t.get("Depth to magnetic source (m)")
            bottom = top - (float(d) if d is not None and not pd.isna(d) else 150.0)
            col = "#d62728" if t["Priority"] == "High" else "#f4a261"
            ax.plot([t["Easting (m)"] / 1000] * 2, [t["Northing (m)"] / 1000] * 2, [top + 20, bottom], color=col, lw=2.2)
            if t["Target"] in set(targets["Target"].head(10)):
                ax.text(t["Easting (m)"] / 1000, t["Northing (m)"] / 1000, top + 60, t["Target"], fontsize=7, color="black", ha="center", fontweight="bold")
    xr = (x[-1] - x[0]) or 1.0; yr = (y[-1] - y[0]) or 1.0; zr = float(np.nanmax(z) - np.nanmin(z)) / 1000.0 or 0.1
    ax.set_box_aspect((1, yr / xr, max(exag * 2.0 * zr / xr, 0.12)))
    from matplotlib.ticker import MaxNLocator
    ax.zaxis.set_major_locator(MaxNLocator(3))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("Easting (km)"); ax.set_ylabel("Northing (km)"); ax.set_zlabel("Elevation (m)"); ax.set_title(title, fontsize=12)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap_obj), ax=ax, shrink=0.55, pad=0.08, label=label)
    return _png(fig, dpi=160)


def figure_pack(project, source, mag, rad, reg, grav, score, score_label, targets, evidence=None, uncertainty=None, model_info=None,
                notes=None, interactive_3d_html=None, drape3d=None, exag=4.0):
    """Every map, chart and 3D view as images in one ZIP, with the targets and a README listing what is inside."""
    import zipfile
    L = project["layers"]; made, skipped = [], []
    zf_buf = io.BytesIO(); zf = zipfile.ZipFile(zf_buf, "w", zipfile.ZIP_DEFLATED)

    def add(path, producer):
        try:
            data = producer()
            if data:
                zf.writestr(path, data); made.append(path)
        except Exception as exc:  # one failed figure must not stop the pack
            skipped.append("%s (%s)" % (path, str(exc)[:80]))

    pal = {"tmi": ("RdBu_r", True, "nT"), "k": ("YlOrRd", False, "%"), "eth": ("YlGn", False, "ppm"), "eu": ("PuBu", False, "ppm"), "tc": ("viridis", False, ""),
           "dem": ("terrain", False, "m"), "grav": ("Spectral_r", True, "mGal")}
    names = {"tmi": "Total magnetic intensity", "k": "Potassium", "eth": "Thorium", "eu": "Uranium", "tc": "Total count", "dem": "Elevation", "grav": "Bouguer gravity"}
    for kind, grid in L.items():
        c, sym, unit = pal.get(kind, ("viridis", False, ""))
        add("01_prepared_layers/%s.png" % kind, lambda grid=grid, c=c, sym=sym, unit=unit, kind=kind: map_png(project, grid - (np.median(grid) if sym else 0), names.get(kind, kind), c, sym, unit))
    if mag is not None:
        add("02_magnetics/field_reduced_to_equator.png", lambda: map_png(project, mag["field"] - np.median(mag["field"]), "Magnetic field (reduced to the equator)", "RdBu_r", True, "nT"))
        add("02_magnetics/residual_field.png", lambda: map_png(project, mag["residual"], "Residual field (regional removed)", "RdBu_r", True, "nT"))
        add("02_magnetics/tilt_angle_with_lineaments.png", lambda: map_png(project, mag["tilt"], "Tilt angle with traced lineaments", "PuOr_r", True, "deg", mask=mag["lineaments"]))
        add("02_magnetics/analytic_signal.png", lambda: map_png(project, mag["asig"], "Analytic signal", "magma", False, "nT/m"))
        add("02_magnetics/total_horizontal_derivative.png", lambda: map_png(project, mag["hg"], "Total horizontal derivative", "magma", False, "nT/m"))
        add("02_magnetics/first_vertical_derivative.png", lambda: map_png(project, mag["vd"], "First vertical derivative", "RdBu_r", True, "nT/m"))
        add("02_magnetics/tilt_depth.png", lambda: map_png(project, mag["depth"], "Tilt-depth to magnetic sources", "viridis_r", False, "m"))
        if len(mag["lineament_table"]):
            add("02_magnetics/lineament_strike_chart.png", lambda: _rose_png(mag["lineament_table"]))
        add("02_magnetics/depth_histogram.png", lambda: _hist_png(mag["depth"], "Distribution of depth estimates", "Depth (m)"))
    if rad is not None:
        add("03_radiometrics/ternary_K_Th_U.png", lambda: map_png(project, rad["ternary"], "Ternary image (K red, eTh green, eU blue)", rgb=True))
        add("03_radiometrics/potassium_enrichment_Kd.png", lambda: map_png(project, rad["kd"], "Potassium enrichment (Kd) with flagged alteration", "RdYlBu_r", True, "", mask=rad["alteration"]))
        add("03_radiometrics/F_parameter.png", lambda: map_png(project, rad["f_param"], "F-parameter (K x eU / eTh)", "YlOrRd"))
        add("03_radiometrics/K_over_eTh.png", lambda: map_png(project, rad["k_eth"], "K / eTh", "YlOrRd"))
        add("03_radiometrics/eU_over_eTh.png", lambda: map_png(project, rad["eu_eth"], "eU / eTh", "PuBu"))
        add("03_radiometrics/K_vs_eTh_crossplot.png", lambda: _crossplot_png(L["k"], L["eth"], rad["kd"], rad["alteration"]))
    if reg is not None:
        add("04_regolith_gravity/regolith_regime.png", lambda: map_png(project, reg["regime"], "Regolith regime (brown relict, cream erosional, blue depositional)", classes=True))
        add("04_regolith_gravity/slope.png", lambda: map_png(project, reg["slope"], "Slope", "Greys", False, "deg"))
    if grav is not None:
        add("04_regolith_gravity/bouguer_gravity.png", lambda: map_png(project, L["grav"] - np.median(L["grav"]), "Bouguer gravity", "Spectral_r", True, "mGal"))
        add("04_regolith_gravity/residual_gravity.png", lambda: map_png(project, grav["residual"], "Residual gravity", "Spectral_r", True, "mGal"))
        add("04_regolith_gravity/vertical_gravity_gradient.png", lambda: map_png(project, grav["vd"] * 1e4, "Vertical gravity gradient (Gzz)", "PuOr_r", True, "Eotvos"))
    add("05_prospectivity/prospectivity_score.png", lambda: map_png(project, score, score_label, "viridis", False, "score"))
    add("05_prospectivity/score_distribution.png", lambda: _hist_png(score, "Score distribution", "Score", 50))
    for name, layer in (evidence or {}).items():
        add("05_prospectivity/evidence/%s.png" % "".join(ch if ch.isalnum() else "_" for ch in name).strip("_"), lambda layer=layer, name=name: map_png(project, layer, name, "viridis"))
    if uncertainty is not None:
        add("05_prospectivity/model_uncertainty.png", lambda: map_png(project, uncertainty, "Model uncertainty (spread between trees)", "Greys"))
    if model_info and model_info.get("importance"):
        add("05_prospectivity/evidence_importance.png", lambda: _bar_png(list(model_info["importance"]), list(model_info["importance"].values()), "What the model relied on"))
    if model_info and model_info.get("fpr"):
        add("05_prospectivity/roc_curve.png", lambda: _roc_png(model_info))
    add("06_targets/score_with_ranked_targets.png", lambda: map_png(project, score, score_label + " with ranked targets", "viridis", False, "score", points=targets))
    if targets is not None and len(targets):
        zf.writestr("06_targets/targets.csv", targets.to_csv(index=False)); made.append("06_targets/targets.csv")
        zf.writestr("06_targets/targets.kml", targets_kml(targets)); made.append("06_targets/targets.kml")
        zf.writestr("06_targets/targets.geojson", targets_geojson(targets)); made.append("06_targets/targets.geojson")
    dem = L.get("dem") if "dem" in L else np.zeros((project["ny"], project["nx"]))
    for i, (az, el) in enumerate(((-60, 32), (35, 28)), 1):
        add("07_3d_view/terrain_score_view_%d.png" % i, lambda az=az, el=el: surface_png(project, dem, score, "viridis", "Terrain coloured by prospectivity, with targets", "score", targets, exag, az, el))
    if mag is not None:
        add("07_3d_view/terrain_magnetic_view.png", lambda: surface_png(project, dem, mag["field"], "RdBu_r", "Terrain coloured by magnetic field", "nT", targets, exag, -60, 32, True))
    if interactive_3d_html:
        zf.writestr("07_3d_view/interactive_3d_view.html", interactive_3d_html); made.append("07_3d_view/interactive_3d_view.html")
    readme = ["RegolithSense figure pack", "Generated %s" % date.today().isoformat(), "Dataset: %s" % source,
              "Study area: %d x %d cells of %.0f m (%.0f km2)" % (project["nx"], project["ny"], project["dx"], project["nx"] * project["ny"] * project["dx"] ** 2 / 1e6),
              "Coordinates: WGS84 UTM zone %dN (kilometres on the axes)" % project["zone"], ""]
    readme += list(notes or [])
    readme += ["", "Scores are rankings of ground for follow-up, not probabilities of a deposit. Not a JORC or NI 43-101 report.", "",
               "Folders: 01 prepared layers, 02 magnetics, 03 radiometrics, 04 regolith and gravity, 05 prospectivity, 06 targets (CSV, KML, GeoJSON),",
               "07 3D view (static images and an interactive HTML file that opens in any browser).", "", "%d files." % len(made)]
    if skipped:
        readme += ["", "Skipped:"] + ["  " + s for s in skipped]
    zf.writestr("README.txt", "\n".join(readme)); zf.close()
    return zf_buf.getvalue()
