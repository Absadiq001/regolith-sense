"""Fast checks of the numerical core. Run with:  python -m pytest -q"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import gridio, magnetics as mg, radiometrics as rd, regolith as rg, prospectivity as pr, pipeline, report  # noqa: E402


def test_utm_round_trip():
    lon, lat = np.array([8.6, 9.1]), np.array([8.9, 9.4])
    e, n = gridio.lonlat_to_utm(lon, lat, 32)
    lon2, lat2 = gridio.utm_to_lonlat(e, n, 32)
    assert np.allclose(lon, lon2, atol=1e-7) and np.allclose(lat, lat2, atol=1e-7)


def test_rte_is_identity_at_zero_inclination():
    rng = np.random.default_rng(0)
    g = rng.normal(size=(64, 64)).cumsum(0).cumsum(1)
    out = mg.reduce_to_equator(g, 100.0, 0.0, 0.0)
    assert np.abs(out - g).max() < 1e-6


def test_tilt_and_depth_on_a_vertical_contact():
    # a step in the field: the tilt angle crosses zero at the contact
    x = np.arange(128) * 100.0
    field = np.tile(np.tanh((x - 6400) / 300.0), (128, 1)) * 100
    tilt = mg.tilt_angle(field, 100.0)
    zero_col = np.argmin(np.abs(tilt[64]))
    assert abs(zero_col - 64) <= 2


def test_radiometric_products_shapes_and_signs():
    rng = np.random.default_rng(1)
    k, th, u = rng.uniform(0.5, 3, (40, 40)), rng.uniform(5, 30, (40, 40)), rng.uniform(1, 6, (40, 40))
    R = rd.products(k, th, u)
    assert set(R) >= {"k_eth", "eu_eth", "f_param", "kd", "ud"}
    assert np.all(R["f_param"] >= 0) and np.isfinite(R["kd"]).all()


def test_simulated_pipeline_end_to_end():
    P = gridio.harmonise(pipeline.simulated_layers(), cell=200.0)
    mag, rad, reg, grav = pipeline.run_magnetics(P), pipeline.run_radiometrics(P), pipeline.run_regolith(P), pipeline.run_gravity(P)
    ev = pipeline.evidence_layers(mag, rad, reg, grav)
    assert len(ev) >= 5
    score = pr.knowledge_score(ev)
    assert score.shape == (P["ny"], P["nx"]) and 0 <= score.min() and score.max() <= 1
    thr = float(np.quantile(score, 0.95))
    T = pr.pick_targets(score, P, thr, 1500.0, regime=reg["regime"], depth=mag["depth"])
    assert len(T) > 0 and (T["Score"].diff().dropna() <= 0).all()  # ranked best first
    assert len(report.targets_kml(T)) > 100 and len(report.targets_geojson(T)) > 100


def test_trained_models_report_cross_validated_auc():
    P = gridio.harmonise(pipeline.simulated_layers(), cell=200.0)
    mag, rad, reg, grav = pipeline.run_magnetics(P), pipeline.run_radiometrics(P), pipeline.run_regolith(P), pipeline.run_gravity(P)
    ev = pipeline.evidence_layers(mag, rad, reg, grav)
    score = pr.knowledge_score(ev)
    rr, cc = np.nonzero(score > np.quantile(score, 0.99))
    pick = np.random.default_rng(0).choice(len(rr), 20, replace=False)
    occ = pd.DataFrame({"row": rr[pick], "col": cc[pick]})
    for kind in pr.MODELS:
        prob, unc, info = pr.train_forest(ev, occ, 200.0, n_trees=50, max_depth=5, model_kind=kind)
        assert prob.shape == score.shape and info["auc"] is None or 0 <= info["auc"] <= 1
        assert (unc is not None) == (kind == "Random Forest")


def test_bundled_demo_loads():
    P = gridio.harmonise(pipeline.demo_layers(), cell=200.0)
    assert {"tmi", "k", "eth", "eu", "dem"} <= set(P["layers"]) and P["zone"] == 32
