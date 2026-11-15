"""Runs each processing stage on a harmonised project and returns plain dicts of grids."""
import os

import numpy as np

from . import magnetics as mg
from . import radiometrics as rd
from . import regolith as rg
from . import prospectivity as pr
from .synthetic_geology import generate_nasarawa_geological_grid


def simulated_layers():
    """A simulated survey for trying the app without data. Not real measurements."""
    d = generate_nasarawa_geological_grid(nx=160, ny=160, dx=100.0, dy=100.0)
    base = {"x0": 400000.0, "y0": 950000.0, "dx": 100.0, "dy": 100.0, "units": "m"}
    names = {"tmi": "tmi", "k": "k", "eth": "eth", "eu": "eu", "dem": "elevation", "grav": "bouguer"}
    return {kind: dict(base, grid=np.asarray(d[src], float)) for kind, src in names.items()}


DEMO_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "wamba_210_demo.npz")


def demo_layers():
    """Wamba sheet 210 (Nasarawa State) from NGSA airborne grids, resampled to 200 m for a quick demonstration."""
    d = np.load(DEMO_PATH)
    base = {"x0": float(d["x0"]), "y0": float(d["y0"]), "dx": float(d["dx"]), "dy": float(d["dx"]), "units": "m"}
    return {k: dict(base, grid=np.asarray(d[k], float)) for k in d.files if k not in ("x0", "y0", "dx", "zone")}


def run_magnetics(project, inclination=-7.0, declination=-2.0, reduce=True, sensor_height=80.0, min_length=1500.0):
    L = project["layers"]
    if "tmi" not in L:
        return None
    cell = project["dx"]
    field = mg.reduce_to_equator(L["tmi"], cell, inclination, declination) if reduce else L["tmi"]
    asig = mg.analytic_signal(field, cell)
    tilt = mg.tilt_angle(field, cell)
    lin, table = mg.lineaments(tilt, asig, cell, min_length=min_length)
    return {
        "field": field,
        "residual": mg.residual(field, cell, 2000.0),
        "vd": mg.vertical_derivative(field, cell),
        "hg": mg.horizontal_gradient(field, cell),
        "asig": asig,
        "tilt": tilt,
        "lineaments": lin,
        "lineament_table": table,
        "density": mg.density(lin, cell),
        "distance": mg.distance_to(lin, cell),
        "depth": mg.tilt_depth(tilt, cell, sensor_height=sensor_height),
    }


def run_radiometrics(project):
    L = project["layers"]
    if not all(k in L for k in ("k", "eth", "eu")):
        return None
    out = rd.products(L["k"], L["eth"], L["eu"])
    out["ternary"] = rd.ternary(L["k"], L["eth"], L["eu"])
    out["alteration"] = rd.alteration_zones(out["kd"], out["f_param"])
    return out


def run_regolith(project):
    L = project["layers"]
    if "dem" not in L:
        return None
    regime, slope, tpi = rg.classify(L["dem"], project["dx"], L.get("k"), L.get("eth"))
    return {"regime": regime, "slope": slope, "tpi": tpi, "weight": rg.in_situ_weight(regime)}


def run_gravity(project):
    L = project["layers"]
    if "grav" not in L:
        return None
    cell = project["dx"]
    return {"residual": mg.residual(L["grav"], cell, 5000.0), "hg": mg.horizontal_gradient(L["grav"], cell),
            "vd": mg.vertical_derivative(L["grav"], cell)}


def evidence_layers(mag, rad, reg, grav):
    return pr.build_evidence(
        dist_contacts=mag["distance"] if mag else None,
        lineament_density=mag["density"] if mag else None,
        rad=rad,
        gravity_residual=grav["residual"] if grav else None,
        in_situ=reg["weight"] if reg else None,
    )
