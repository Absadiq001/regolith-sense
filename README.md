# RegolithSense

**Finding the pegmatites that the weathering cover hides.**
A screening tool that turns Nigeria's existing airborne geophysics into a ranked list of places worth visiting.

*Pan-African MineTech Exploration Technology Pre-Incubation Programme — Nigeria Track*
*Nasarawa State University Keffi Mine-Tech UniPod, in partnership with UNDP timbuktoo*

**Live app:** https://absadiq001-regolith-sense-app-xvaabs.streamlit.app
**Team:** GeoSense Dynamics, Federal University of Lafia

---

## The problem

Across the Nigerian Basement Complex, lithium–tantalum pegmatites and the granites that feed them lie under 10 to 60 m of
laterite and saprolite. Optical satellites and surface mapping see the cover, not the rock. Exploration is still done by
walking, pitting and guesswork; artisanal miners find deposits by accident and companies drill blind.

## What RegolithSense does

It takes data that already exists for the whole country and turns it into a target package for a field team:

| Step | Input | What it produces |
|---|---|---|
| 1. Data | NGSA Geosoft `.grd` grids (magnetic, K, eTh, eU, TC), SRTM, optional Bouguer gravity; GeoTIFF, CSV/XYZ | One harmonised UTM grid, gap-filled and despiked, with a preparation report |
| 2. Magnetics | Total magnetic intensity | Reduction to the equator, residual, tilt angle, analytic signal, derivatives, automatic lineaments, tilt-depth to source |
| 3. Radiometrics | K, eTh, eU | Ternary image, K/eTh, eU/eTh, F-parameter, Kd/Ud deviation, potassic-alteration mask, K–eTh crossplot |
| 4. Regolith and gravity | SRTM, Bouguer | Relict / erosional / depositional regimes (where surface evidence can be trusted); residual gravity, Gzz, gradients |
| 5. Prospectivity | Evidence layers, optional known occurrences | Weighted overlay, or Random Forest / Gradient Boosting / logistic regression with spatial block cross-validation, ROC, importances, uncertainty |
| 6. Targets and export | Score | Ranked, spaced targets with regime and depth; CSV, GeoJSON, KML, Excel, XYZ, Word report; per-target evidence table |
| 7. 3D view | SRTM + any layer | Terrain draped with the score, target pins to estimated source depth |
| 8. Decision brief | Everything above | One page for the person deciding where to send the team, with field instructions and an illustrative cost framing |

The app opens with **Wamba sheet 210 (Nasarawa State)** already loaded from real NGSA grids (bundled at 200 m, 1.3 MB),
so the whole workflow can be shown in two minutes. A clearly labelled simulated dataset (with gravity) is also included.

## What it is not

Scores are rankings, not probabilities of a deposit. The regolith classes are rule-based and uncalibrated. Nothing here
replaces field mapping, sampling or drilling, and the outputs are not a JORC or NI 43-101 report. The *Methods and limits*
page in the app lists every method, its reference, and where it can mislead.

## Run it

```bash
git clone https://github.com/Absadiq001/regolith-sense.git
cd regolith-sense
pip install -r requirements.txt
streamlit run app.py
```

Tests: `python -m pytest -q tests`

Deploy: push to GitHub and point [Streamlit Community Cloud](https://share.streamlit.io) at `app.py`.

## Repository layout

```
app.py                  Streamlit interface (ten pages, one map helper, cached processing)
core/geosoft.py         Geosoft .grd reader (plain and compressed)
core/gridio.py          File readers, UTM projection, harmonisation, gap fill, despike, occurrences
core/magnetics.py       FFT filters, RTE, derivatives, tilt-depth, lineament tracing
core/radiometrics.py    Ratios, F-parameter, Kd/Ud, ternary, alteration mask
core/regolith.py        Terrain analysis and regolith regimes
core/prospectivity.py   Evidence stack, weighted overlay, trained models with spatial CV, target picking
core/pipeline.py        Runs each stage; bundled demo and simulated datasets
core/report.py          Word report, Excel workbook, XYZ, GeoJSON, KML
core/synthetic_geology.py  Simulated survey generator
data/wamba_210_demo.npz NGSA Wamba 210 grids resampled to 200 m
tests/test_core.py      Numerical checks
```

## Data

Nigerian Geological Survey Agency airborne magnetic and radiometric grids (2005–2009 national survey, 500 m line spacing,
80 m terrain clearance); SRTM 1-arc-second elevation. Satellite gravity (WGM2012, GGMplus) can be added as a Bouguer grid.

## Team (GeoSense Dynamics)

* **Abubakar Sadiq Musa** — Team Lead & Computational Geophysicist *(Founder)*, Federal University of Lafia
* **Abdulrafiu Sheriff** — GIS, Remote Sensing & AI Lead *(Co-Founder)*
* **Galadima Abdulhamid Abdullahi** — Geologist & Environmental Geochemistry / ESG Lead
* **Sani Muhammad Muktar** — Stakeholder & Community Operations Lead
* **Iposhi Martins David** — Field Exploration & Ground-Truthing Lead

## References

Anand & Paine 2002, *Aust. J. Earth Sci.* 49 · Carranza 2008, *Geochemical Anomaly and Mineral Prospectivity Mapping in GIS* ·
Dickson & Scott 1997, *AGSO J.* 17 · Miller & Singh 1994, *J. Appl. Geophys.* 32 · Pires 1995, *Rev. Bras. Geociências* 25 ·
Roberts et al. 2017, *Ecography* 40 · Roest, Verhoef & Pilkington 1992, *Geophysics* 57 · Salem et al. 2007, *The Leading Edge* 26 ·
Černý & Ercit 2005, *Can. Mineral.* 43.
