"""
Mineral prospectivity mapping for concealed LCT pegmatites.

Two modes:
  * knowledge-driven: a weighted overlay of evidence layers scaled 0-1. Needs
    no training data. The result is a ranking score, not a probability.
  * data-driven: a Random Forest trained on known occurrences, scored by
    spatial block cross-validation so the accuracy figure is not inflated by
    neighbouring cells, with the spread between trees reported as uncertainty.
"""
import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve

DEFAULT_WEIGHTS = {
    "Near magnetic contacts": 1.0,
    "Lineament density": 1.0,
    "Potassium enrichment (Kd)": 1.5,
    "F-parameter": 1.0,
    "K/eTh ratio": 0.5,
    "Gravity low": 1.0,
}


def _scale(a, low=2, high=98):
    lo, hi = np.percentile(a, (low, high))
    return np.clip((a - lo) / (hi - lo + 1e-12), 0, 1)


def build_evidence(dist_contacts=None, lineament_density=None, rad=None, gravity_residual=None, in_situ=None, decay=1000.0):
    """Evidence layers scaled 0-1, higher = more favourable. Layers without data are left out."""
    ev = {}
    if dist_contacts is not None and np.isfinite(dist_contacts).all():
        ev["Near magnetic contacts"] = np.exp(-dist_contacts / decay)
    if lineament_density is not None:
        ev["Lineament density"] = _scale(lineament_density)
    if rad is not None:
        w = in_situ if in_situ is not None else 1.0
        ev["Potassium enrichment (Kd)"] = _scale(rad["kd"]) * w
        ev["F-parameter"] = _scale(rad["f_param"]) * w
        ev["K/eTh ratio"] = _scale(rad["k_eth"]) * w
    if gravity_residual is not None:
        ev["Gravity low"] = _scale(-gravity_residual)
    return ev


def knowledge_score(evidence, weights=None):
    weights = weights or DEFAULT_WEIGHTS
    total = np.zeros(next(iter(evidence.values())).shape); wsum = 0.0
    for name, layer in evidence.items():
        w = float(weights.get(name, 1.0))
        total += w * layer; wsum += w
    if wsum == 0:
        raise ValueError("All evidence weights are zero.")
    return ndimage.gaussian_filter(total / wsum, 1.0)


def train_forest(evidence, occurrences, cell, buffer_m=300.0, n_trees=200, blocks=5, seed=0):
    """
    Train on known occurrences (DataFrame with row, col). Returns
    (probability, uncertainty, info). info holds the cross-validated AUC, the
    ROC curve, feature importances and sample counts.
    """
    names = list(evidence)
    shape = evidence[names[0]].shape
    X = np.column_stack([evidence[n].ravel() for n in names])
    pos = np.zeros(shape, bool)
    pos[occurrences["row"].to_numpy(), occurrences["col"].to_numpy()] = True
    reach = max(int(round(buffer_m / cell)), 1)
    dist = ndimage.distance_transform_edt(~pos)
    pos_mask = dist <= reach
    neg_pool = np.flatnonzero((dist > 6 * reach).ravel())
    pos_idx = np.flatnonzero(pos_mask.ravel())
    rng = np.random.default_rng(seed)
    neg_idx = rng.choice(neg_pool, size=min(len(neg_pool), max(len(pos_idx) * 5, 500)), replace=False)
    idx = np.concatenate([pos_idx, neg_idx]); y = np.concatenate([np.ones(len(pos_idx)), np.zeros(len(neg_idx))])

    rows, cols = np.unravel_index(idx, shape)
    block = (rows * blocks // shape[0]) * blocks + (cols * blocks // shape[1])
    oof = np.full(len(idx), np.nan)
    for b in np.unique(block):
        test = block == b; train = ~test
        if y[test].sum() == 0 or y[train].sum() == 0:
            continue
        m = RandomForestClassifier(n_estimators=n_trees, min_samples_leaf=3, class_weight="balanced", random_state=seed, n_jobs=-1)
        m.fit(X[idx[train]], y[train])
        oof[test] = m.predict_proba(X[idx[test]])[:, 1]
    scored = np.isfinite(oof)
    info = {"n_occurrences": int(len(occurrences)), "n_positive_cells": int(len(pos_idx)), "n_negative_cells": int(len(neg_idx)),
            "auc": None, "fpr": None, "tpr": None, "blocks_scored": int(len(np.unique(block[scored])))}
    if scored.sum() > 0 and 0 < y[scored].sum() < scored.sum():
        info["auc"] = float(roc_auc_score(y[scored], oof[scored]))
        fpr, tpr, _ = roc_curve(y[scored], oof[scored])
        info["fpr"], info["tpr"] = fpr.tolist(), tpr.tolist()

    model = RandomForestClassifier(n_estimators=n_trees, min_samples_leaf=3, class_weight="balanced", random_state=seed, n_jobs=-1)
    model.fit(X[idx], y)
    per_tree = np.stack([t.predict_proba(X)[:, 1] for t in model.estimators_])
    prob = ndimage.gaussian_filter(per_tree.mean(axis=0).reshape(shape), 1.0)
    unc = per_tree.std(axis=0).reshape(shape)
    info["importance"] = dict(zip(names, model.feature_importances_.tolist()))
    return prob, unc, info


def pick_targets(score, project, threshold=0.7, spacing_m=1500.0, regime=None, depth=None, uncertainty=None, max_targets=40):
    """Local maxima of the score above `threshold`, at least `spacing_m` apart, ranked best first."""
    cell = project["dx"]
    size = max(int(spacing_m / cell) | 1, 3)
    peaks = (ndimage.maximum_filter(score, size=size) == score) & (score >= threshold)
    rr, cc = np.nonzero(peaks)
    order = np.argsort(score[rr, cc])[::-1][:max_targets]
    rr, cc = rr[order], cc[order]
    from .gridio import utm_to_lonlat
    x = project["x0"] + cc * cell; y = project["y0"] + rr * cell
    lon, lat = utm_to_lonlat(x, y, project["zone"])
    df = pd.DataFrame({"Target": ["T%02d" % (i + 1) for i in range(len(rr))], "Score": np.round(score[rr, cc], 3),
                       "Easting (m)": np.round(x).astype(int), "Northing (m)": np.round(y).astype(int),
                       "Longitude": np.round(lon, 5), "Latitude": np.round(lat, 5)})
    if uncertainty is not None:
        df["Uncertainty"] = np.round(uncertainty[rr, cc], 3)
    if regime is not None:
        from .regolith import REGIMES
        df["Regolith regime"] = [REGIMES[int(v)] for v in regime[rr, cc]]
    if depth is not None:
        vals = []
        for r, c in zip(rr, cc):
            win = depth[max(r - 7, 0):r + 8, max(c - 7, 0):c + 8]
            vals.append(int(round(float(np.nanmedian(win)))) if np.isfinite(win).any() else None)
        df["Depth to magnetic source (m)"] = vals
    q = score[rr, cc]
    df["Priority"] = np.where(q >= np.quantile(score, 0.995), "High", np.where(q >= np.quantile(score, 0.98), "Medium", "Low"))
    return df


def area_reduction(score, threshold):
    """Share of the survey area (%) excluded when only cells at or above the threshold are kept."""
    return float(100.0 * (1.0 - np.mean(score >= threshold)))
