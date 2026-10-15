"""
RegolithSense - Enterprise Exploration Report Exporter (Word DOCX & Excel XLSX)
Compiles all geophysical maps, radiometric alteration indices, machine learning diagnostics,
and JORC/NI 43-101 drill targets into a unified, publication-grade Word document.
"""

import io
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from datetime import datetime

def set_cell_background(cell, hex_color):
    """Sets background color of a Word table cell."""
    tcPr = cell._element.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color.replace('#', ''))
    tcPr.append(shd)

def render_figure_to_bytes(fig):
    """Saves a matplotlib figure to an in-memory PNG bytes buffer."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

def generate_comprehensive_word_report(base_data, prob_map, targets, importances, feat_names, metrics, model_type):
    """
    Generates a complete, single Word document (.docx) containing:
    - Executive Summary & Exploration Context
    - Machine Learning Model Architecture, Diagnostics & ROC-AUC
    - All Geophysical Maps (Aeromagnetics, Bouguer Gravity, Gzz)
    - All Radiometric Maps (TC, K, eTh, eU, Ternary, K/eTh, F-Parameter, Bivariate Crossplot)
    - Full JORC / NI 43-101 Target Table & Coordinates
    """
    doc = docx.Document()
    
    # Page Margins (0.75 in)
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    b = base_data["raw"]["bounds"]
    df_targets = pd.DataFrame(targets)

    # 1. Title Banner
    title_p = doc.add_paragraph()
    title_run = title_p.add_run("REGOLITHSENSE ENTERPRISE EXPLORATION REPORT")
    title_run.font.name = "Arial"
    title_run.font.size = Pt(22)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(15, 23, 42) # Slate 900
    title_p.paragraph_format.space_after = Pt(2)

    sub_p = doc.add_paragraph()
    sub_run = sub_p.add_run("Multi-Physics Inversion & Machine Learning Mineral Prospectivity Mapping for Deeply Concealed Critical Mineral Systems")
    sub_run.font.name = "Arial"
    sub_run.font.size = Pt(10.5)
    sub_run.font.color.rgb = RGBColor(100, 116, 139) # Slate 500
    sub_p.paragraph_format.space_after = Pt(14)

    # 2. Metadata Table
    meta_table = doc.add_table(rows=4, cols=4)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        [("Project / Belt:", True), ("Nasarawa Lithium-Pegmatite Corridor", False), ("Date of Report:", True), (datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"), False)],
        [("Survey Bounds:", True), (f"{b[0]:.2f}°E to {b[1]:.2f}°E, {b[2]:.2f}°N to {b[3]:.2f}°N", False), ("Grid Geometry:", True), (f"{base_data['raw']['tmi'].shape[1]} x {base_data['raw']['tmi'].shape[0]} cells ({base_data['raw']['dx']:.0f}m)", False)],
        [("Target Mineral:", True), ("Lithium (LCT Pegmatites) & REEs", False), ("Mean Regolith Cover:", True), (f"{np.mean(base_data['raw']['regolith_thickness']):.1f} meters", False)],
        [("Active ML Model:", True), (f"{model_type}", False), ("Model ROC-AUC Score:", True), (f"{metrics['auc']:.3f}", False)]
    ]
    for row_idx, row_content in enumerate(meta_data):
        for col_idx, (text, is_bold) in enumerate(row_content):
            cell = meta_table.cell(row_idx, col_idx)
            cell.text = text
            p = cell.paragraphs[0]
            p.runs[0].font.name = "Arial"
            p.runs[0].font.size = Pt(9)
            if is_bold:
                p.runs[0].font.bold = True
                set_cell_background(cell, "#F1F5F9")
            else:
                set_cell_background(cell, "#FAFAFA")
    doc.add_paragraph() # Spacer

    # ----------------- SECTION 1: EXECUTIVE PROSPECTIVITY & TARGETS -----------------
    h1 = doc.add_heading("1. Executive Summary & Ranked Mineral Targets", level=1)
    h1.paragraph_format.space_before = Pt(10)
    
    p_exec = doc.add_paragraph(
        f"This pre-competitive exploration assessment integrates high-resolution potential field geophysics (aeromagnetics and satellite gravity) "
        f"with multi-channel gamma-ray radiometrics to overcome the 15m to 65m tropical regolith blanket in the Central Nigerian Basement Complex. "
        f"A total of {len(df_targets)} high-probability structural targets have been delineated, with {len(df_targets[df_targets['target_tier'].str.contains('Tier-1')])} "
        f"classified as Tier-1 Immediate Reverse Circulation (RC) drill spotting priorities."
    )
    p_exec.style.font.name = "Arial"
    p_exec.style.font.size = Pt(9.5)

    # Render Prospectivity Map Figure
    fig_mpm, ax_mpm = plt.subplots(figsize=(8.5, 5.2))
    im_mpm = ax_mpm.imshow(prob_map * 100, cmap="turbo", extent=[b[0], b[1], b[2], b[3]], vmin=10, vmax=95)
    plt.colorbar(im_mpm, ax=ax_mpm, label="Prospectivity Probability (%)", shrink=0.8)
    for tgt in targets:
        if "Tier-1" in tgt["target_tier"]:
            ax_mpm.plot(tgt["longitude"], tgt["latitude"], 'r*', markersize=14, markeredgecolor='black', label="Tier-1 Target" if tgt == targets[0] else "")
        else:
            ax_mpm.plot(tgt["longitude"], tgt["latitude"], 'yo', markersize=8, markeredgecolor='black', label="Tier-2 Target" if tgt == targets[1] else "")
    for dep in base_data["raw"].get("known_deposits", []):
        ax_mpm.plot(dep["lon"], dep["lat"], 'w^', markersize=9, markeredgecolor='black', label="Known Mine" if dep == base_data["raw"]["known_deposits"][0] else "")
    ax_mpm.legend(loc="upper right", fontsize=8)
    ax_mpm.set_title(f"Figure 1: Machine Learning Mineral Prospectivity Map ({model_type}) with Ranked Drillhole Targets", fontsize=11, fontweight="bold")
    ax_mpm.set_xlabel("Longitude (°E)")
    ax_mpm.set_ylabel("Latitude (°N)")
    buf_mpm = render_figure_to_bytes(fig_mpm)
    doc.add_picture(buf_mpm, width=Inches(6.5))
    doc.add_paragraph()

    # ----------------- SECTION 2: MACHINE LEARNING MODEL DIAGNOSTICS -----------------
    h2 = doc.add_heading("2. Machine Learning Model Architecture & Validation", level=1)
    p_ml = doc.add_paragraph(
        f"The predictive engine operates using a supervised {model_type} architecture trained on verified mineral deposits and multi-criteria evidential layers. "
        f"The model achieved a Receiver Operating Characteristic Area Under Curve (ROC-AUC) score of {metrics['auc']:.3f}, indicating exceptional discrimination "
        f"against barren laterite background."
    )
    p_ml.style.font.name = "Arial"
    p_ml.style.font.size = Pt(9.5)

    # Render ROC Curve & Feature Importance
    fig_diag, (ax_roc, ax_imp) = plt.subplots(1, 2, figsize=(10, 4.2))
    ax_roc.plot(metrics['fpr'], metrics['tpr'], color='#0284C7', lw=2.5, label=f'{model_type} (AUC = {metrics["auc"]:.3f})')
    ax_roc.plot([0, 1], [0, 1], color='#94A3B8', linestyle='--', label='Random Chance')
    ax_roc.set_title("A: Receiver Operating Characteristic (ROC)", fontsize=10, fontweight="bold")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right", fontsize=8)
    ax_roc.grid(True, linestyle=":", alpha=0.6)

    min_len = min(len(feat_names), len(importances))
    ax_imp.barh(feat_names[:min_len], importances[:min_len], color='#0D9488')
    ax_imp.set_title("B: Multi-Physics Feature Importance Weight", fontsize=10, fontweight="bold")
    ax_imp.set_xlabel("Relative Predictive Weight")
    ax_imp.grid(True, linestyle=":", alpha=0.6)
    fig_diag.tight_layout()
    buf_diag = render_figure_to_bytes(fig_diag)
    doc.add_picture(buf_diag, width=Inches(6.5))
    doc.add_paragraph()

    # ----------------- SECTION 3: AEROMAGNETIC WAVEFIELD INVERSION -----------------
    h3 = doc.add_heading("3. Subsurface Aeromagnetic Wavefield Inversion", level=1)
    p_mag = doc.add_paragraph(
        "Aeromagnetic potential fields penetrate non-magnetic regolith without attenuation. "
        "The Tilt Angle Derivative (TDR = arctan(VDR/THDR)) zero-contour precisely delineates regional shear corridors and pegmatite contacts."
    )
    p_mag.style.font.name = "Arial"
    p_mag.style.font.size = Pt(9.5)

    fig_mag, (ax_tmi, ax_tdr) = plt.subplots(1, 2, figsize=(10, 4.2))
    im_tmi = ax_tmi.imshow(base_data["raw"]["tmi"], cmap="jet", extent=[b[0], b[1], b[2], b[3]])
    plt.colorbar(im_tmi, ax=ax_tmi, label="TMI (nT)", shrink=0.75)
    ax_tmi.set_title("A: Total Magnetic Intensity (TMI in nT)", fontsize=10, fontweight="bold")
    ax_tmi.set_xlabel("Longitude (°E)")
    ax_tmi.set_ylabel("Latitude (°N)")

    im_tdr = ax_tdr.imshow(base_data["tdr"], cmap="coolwarm", extent=[b[0], b[1], b[2], b[3]], vmin=-90, vmax=90)
    plt.colorbar(im_tdr, ax=ax_tdr, label="Tilt Angle (°)", shrink=0.75)
    ax_tdr.contour(base_data["tdr"], levels=[0], colors="black", linewidths=1.2, extent=[b[0], b[1], b[2], b[3]])
    ax_tdr.set_title("B: Tilt Angle Derivative (Zero-Contour = Contacts)", fontsize=10, fontweight="bold")
    ax_tdr.set_xlabel("Longitude (°E)")
    ax_tdr.set_ylabel("Latitude (°N)")
    fig_mag.tight_layout()
    buf_mag = render_figure_to_bytes(fig_mag)
    doc.add_picture(buf_mag, width=Inches(6.5))
    doc.add_paragraph()

    # ----------------- SECTION 4: BOUGUER GRAVITY & DENSITY INVERSION -----------------
    h4 = doc.add_heading("4. Subsurface Density Inversion & Bouguer Gravity", level=1)
    p_grav = doc.add_paragraph(
        "Gravity data resolves bulk rock density contrasts (Δρ). Felsic peraluminous leucogranites (the parent magmas of LCT pegmatites) "
        "have lower bulk density (ρ ~ 2.60 g/cm³) compared to dense metamorphic basement gneisses and amphibolites (ρ ~ 2.90 g/cm³), "
        "producing distinct negative Bouguer gravity anomaly lows that pinpoint fertile intrusive roots."
    )
    p_grav.style.font.name = "Arial"
    p_grav.style.font.size = Pt(9.5)

    fig_grav, (ax_bg, ax_gzz) = plt.subplots(1, 2, figsize=(10, 4.2))
    im_bg = ax_bg.imshow(base_data["bouguer"], cmap="Spectral_r", extent=[b[0], b[1], b[2], b[3]])
    plt.colorbar(im_bg, ax=ax_bg, label="Bouguer (mGal)", shrink=0.75)
    ax_bg.contour(base_data["bouguer"], levels=[-38.0, -36.0], colors="blue", linewidths=1.2, extent=[b[0], b[1], b[2], b[3]])
    ax_bg.set_title("A: Complete Bouguer Gravity Anomaly (mGal)", fontsize=10, fontweight="bold")
    ax_bg.set_xlabel("Longitude (°E)")
    ax_bg.set_ylabel("Latitude (°N)")

    im_gzz = ax_gzz.imshow(base_data["gzz"], cmap="PuOr_r", extent=[b[0], b[1], b[2], b[3]])
    plt.colorbar(im_gzz, ax=ax_gzz, label="Gzz (Eotvos)", shrink=0.75)
    ax_gzz.set_title("B: First Vertical Gravity Gradient (Gzz in Eotvos)", fontsize=10, fontweight="bold")
    ax_gzz.set_xlabel("Longitude (°E)")
    ax_gzz.set_ylabel("Latitude (°N)")
    fig_grav.tight_layout()
    buf_grav = render_figure_to_bytes(fig_grav)
    doc.add_picture(buf_grav, width=Inches(6.5))
    doc.add_paragraph()

    # ----------------- SECTION 5: GAMMA-RAY RADIOMETRICS & ALTERATION -----------------
    h5 = doc.add_heading("5. Gamma-Ray Spectrometry & Hydrothermal Alteration Suite", level=1)
    p_rad = doc.add_paragraph(
        "Radiometric processing isolates diagnostic hydrothermal potassic metasomatism (microcline, muscovite, lepidolite) "
        "characteristic of lithium pegmatite swarms. The Efimov / Saunders F-Parameter (F = K * eU / eTh) and K/eTh ratio "
        "successfully discriminate fertile altered zones from barren crustal baseline."
    )
    p_rad.style.font.name = "Arial"
    p_rad.style.font.size = Pt(9.5)

    # 4-panel radiometrics figure
    fig_rad_suite, ((ax_r1, ax_r2), (ax_r3, ax_r4)) = plt.subplots(2, 2, figsize=(10, 8.5))
    
    # 1. Ternary RGB
    ax_r1.imshow(base_data["ternary_rgb"], extent=[b[0], b[1], b[2], b[3]])
    ax_r1.set_title("A: Radiometric Ternary Map (Red:K | Green:eTh | Blue:eU)", fontsize=9.5, fontweight="bold")
    
    # 2. Potassic Alteration Ratio (K / eTh)
    im_kth = ax_r2.imshow(base_data["ratios"]["K_eTh"], cmap="YlOrRd", extent=[b[0], b[1], b[2], b[3]])
    plt.colorbar(im_kth, ax=ax_r2, label="K / eTh Ratio", shrink=0.75)
    ax_r2.set_title("B: Potassic Metasomatism Ratio (K / eTh)", fontsize=9.5, fontweight="bold")

    # 3. Efimov F-Parameter
    im_f = ax_r3.imshow(base_data["ratios"]["F_param"], cmap="hot", extent=[b[0], b[1], b[2], b[3]])
    plt.colorbar(im_f, ax=ax_r3, label="F-Parameter", shrink=0.75)
    ax_r3.set_title("C: Efimov F-Parameter Index [K * eU / eTh]", fontsize=9.5, fontweight="bold")

    # 4. Total Count
    im_tc = ax_r4.imshow(base_data["ratios"]["TC"], cmap="turbo", extent=[b[0], b[1], b[2], b[3]])
    plt.colorbar(im_tc, ax=ax_r4, label="Total Count (cps / Ur)", shrink=0.75)
    ax_r4.set_title("D: Total Radiometric Count (TC)", fontsize=9.5, fontweight="bold")

    fig_rad_suite.tight_layout()
    buf_rad_suite = render_figure_to_bytes(fig_rad_suite)
    doc.add_picture(buf_rad_suite, width=Inches(6.5))
    doc.add_paragraph()

    # Bivariate Scatter Plot
    fig_bivar, ax_biv = plt.subplots(figsize=(8, 4.5))
    step_s = 2
    flat_k = base_data["raw"]["k"][::step_s, ::step_s].ravel()
    flat_th = base_data["raw"]["eth"][::step_s, ::step_s].ravel()
    flat_f = base_data["ratios"]["F_param"][::step_s, ::step_s].ravel()
    
    sc = ax_biv.scatter(flat_th, flat_k, c=flat_f, cmap="turbo", s=12, alpha=0.7)
    plt.colorbar(sc, ax=ax_biv, label="F-Parameter Alteration Index", shrink=0.8)
    th_line = np.linspace(0, float(np.max(flat_th)), 50)
    ax_biv.plot(th_line, th_line / 3.75, 'r--', lw=2, label="Normal Crustal Ratio (eTh/K = 3.75)")
    ax_biv.set_title("Figure 5: Bivariate K (%) vs eTh (ppm) Exploration Alteration Crossplot", fontsize=10.5, fontweight="bold")
    ax_biv.set_xlabel("Equivalent Thorium (eTh ppm)")
    ax_biv.set_ylabel("Potassium Concentration (K %)")
    ax_biv.legend(loc="upper left", fontsize=8)
    ax_biv.grid(True, linestyle=":", alpha=0.5)
    buf_biv = render_figure_to_bytes(fig_bivar)
    doc.add_picture(buf_biv, width=Inches(6.5))
    doc.add_paragraph()

    # ----------------- SECTION 6: JORC / NI 43-101 DRILL TARGET REGISTRY -----------------
    h6 = doc.add_heading("6. Standardized JORC / NI 43-101 Drillhole Spotting Target Registry", level=1)
    doc.add_paragraph(
        "The following table provides the prioritized drilling targets extracted from the multi-physics inversion, "
        "including estimated regolith overburden casing depths and recommended drillhole depths."
    )

    tgt_table = doc.add_table(rows=len(df_targets) + 1, cols=8)
    tgt_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Target ID", "Lat (°N)", "Lon (°E)", "Collar Elev (m)", "Regolith (m)", "Drill Depth (m)", "Score (%)", "Priority"]
    
    for c_idx, h_text in enumerate(headers):
        cell = tgt_table.cell(0, c_idx)
        cell.text = h_text
        set_cell_background(cell, "#0F172A") # Navy dark
        p = cell.paragraphs[0]
        p.runs[0].font.name = "Arial"
        p.runs[0].font.size = Pt(8.5)
        p.runs[0].font.bold = True
        p.runs[0].font.color.rgb = RGBColor(255, 255, 255)

    for r_idx, (_, row) in enumerate(df_targets.iterrows()):
        row_cells = tgt_table.rows[r_idx + 1].cells
        vals = [
            str(row["target_id"]),
            f"{row['latitude']:.4f}",
            f"{row['longitude']:.4f}",
            f"{row['surface_elevation_m']:.1f}",
            f"{row['regolith_cover_m']:.1f}",
            f"{row['target_drill_depth_m']:.1f}",
            f"{row['prospectivity_score']:.1f}",
            str(row["priority"])
        ]
        is_tier1 = "Tier-1" in str(row["target_tier"])
        bg_col = "#FEF2F2" if is_tier1 else "#FFFFFF"
        for c_idx, val in enumerate(vals):
            cell = row_cells[c_idx]
            cell.text = val
            set_cell_background(cell, bg_col)
            p = cell.paragraphs[0]
            p.runs[0].font.name = "Arial"
            p.runs[0].font.size = Pt(8)
            if is_tier1 and c_idx in [0, 6, 7]:
                p.runs[0].font.bold = True
                p.runs[0].font.color.rgb = RGBColor(185, 28, 28)

    doc.add_paragraph()
    doc.add_paragraph(
        "Report Sign-Off: Prepared by RegolithSense Enterprise Multi-Physics Inversion Engine | "
        "Nasarawa State University Keffi Mine-Tech UniPod (UNDP timbuktoo Pre-Incubation Program)."
    )

    doc_buf = io.BytesIO()
    doc.save(doc_buf)
    doc_buf.seek(0)
    return doc_buf.getvalue()

def generate_excel_workbook(df_targets, feat_names, importances, metrics, base_data):
    """
    Generates a structured, multi-sheet Excel workbook (.xlsx) containing:
    - Sheet 1: Ranked_Drill_Targets (JORC Spotting Coordinates & Depths)
    - Sheet 2: Model_Performance_Metrics (AUC, Hyperparameters, Feature Weights)
    - Sheet 3: Survey_Metadata & Statistics
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sheet 1: Targets
        df_targets.to_excel(writer, sheet_name="Ranked_Drill_Targets", index=False)
        
        # Sheet 2: Features & Metrics
        min_len = min(len(feat_names), len(importances))
        df_feat = pd.DataFrame({
            "Evidence_Layer": list(feat_names)[:min_len],
            "Relative_Weight": list(importances)[:min_len],
            "Model_ROC_AUC": [metrics["auc"]] * min_len
        })
        df_feat.to_excel(writer, sheet_name="Model_Performance_Metrics", index=False)
        
        # Sheet 3: Survey Metadata
        b = base_data["raw"]["bounds"]
        meta_rows = [
            {"Parameter": "Survey Name", "Value": "Nasarawa Critical Mineral Corridor"},
            {"Parameter": "Min Longitude", "Value": b[0]},
            {"Parameter": "Max Longitude", "Value": b[1]},
            {"Parameter": "Min Latitude", "Value": b[2]},
            {"Parameter": "Max Latitude", "Value": b[3]},
            {"Parameter": "Grid Cells (Nx x Ny)", "Value": f"{base_data['raw']['tmi'].shape[1]} x {base_data['raw']['tmi'].shape[0]}"},
            {"Parameter": "Grid Resolution", "Value": f"{base_data['raw']['dx']:.1f} meters"},
            {"Parameter": "Mean Regolith Thickness", "Value": f"{np.mean(base_data['raw']['regolith_thickness']):.2f} m"},
            {"Parameter": "Total Prioritized Targets", "Value": len(df_targets)}
        ]
        pd.DataFrame(meta_rows).to_excel(writer, sheet_name="Survey_Metadata", index=False)
        
    output.seek(0)
    return output.getvalue()
