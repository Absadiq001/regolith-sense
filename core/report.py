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
