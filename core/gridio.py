"""
Reading survey files and bringing them onto one common grid.

A raw layer is a dict: grid (2D array, row 0 = south), x0, y0 (centre of the
south-west cell), dx, dy, and units ("m" for projected metres, "deg" for
longitude/latitude). `harmonise` turns a set of raw layers into a project:
every layer on the same UTM grid, cleaned and gap-filled.
"""
import io
import re
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.interpolate import griddata

from .geosoft import read_grd

LAYER_NAMES = {
    "tmi": "Magnetic field (nT)",
    "k": "Potassium (%)",
    "eth": "Thorium (ppm)",
    "eu": "Uranium (ppm)",
    "tc": "Total count",
    "dem": "Elevation (m)",
    "grav": "Bouguer gravity (mGal)",
}

_KIND_PATTERNS = [
    ("dem", r"srtm|dem|elev|topo"),
    ("grav", r"grav|boug|ggm|wgm"),
    ("tmi", r"tmi|mag|rtp|rte|aero"),
    ("tc", r"(^|[_\-\s])tc($|[_\-\s.])|total.?count"),
    ("eth", r"(^|[_\-\s])e?th($|[_\-\s.])|thor"),
    ("eu", r"(^|[_\-\s])e?u($|[_\-\s.])|uran"),
    ("k", r"(^|[_\-\s])k($|[_\-\s.])|potas"),
]


def guess_kind(filename):
    """Guess which survey channel a file holds from its name. Returns None if unsure."""
    stem = re.sub(r"\.(grd|tif|tiff|csv|xyz|txt)$", "", filename.lower())
    for kind, pat in _KIND_PATTERNS:
        if re.search(pat, stem):
            return kind
    return None


# ---------------------------------------------------------------- projection
_A = 6378137.0
_F = 1 / 298.257223563
_K0 = 0.9996
_N = _F / (2 - _F)
_AA = _A / (1 + _N) * (1 + _N**2 / 4 + _N**4 / 64)
_ALPHA = (_N / 2 - 2 * _N**2 / 3 + 5 * _N**3 / 16, 13 * _N**2 / 48 - 3 * _N**3 / 5, 61 * _N**3 / 240)
_BETA = (_N / 2 - 2 * _N**2 / 3 + 37 * _N**3 / 96, _N**2 / 48 + _N**3 / 15, 17 * _N**3 / 480)
_DELTA = (2 * _N - 2 * _N**2 / 3 - 2 * _N**3, 7 * _N**2 / 3 - 8 * _N**3 / 5, 56 * _N**3 / 15)


def utm_zone(lon):
    return int(np.floor((np.nanmean(lon) + 180) / 6) + 1)


def lonlat_to_utm(lon, lat, zone):
    """WGS 84 longitude/latitude (degrees) to UTM easting/northing (northern hemisphere)."""
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    lam = np.radians(lon - (zone * 6 - 183)); phi = np.radians(lat)
    e = 2 * np.sqrt(_N) / (1 + _N)
    t = np.sinh(np.arctanh(np.sin(phi)) - e * np.arctanh(e * np.sin(phi)))
    xi = np.arctan2(t, np.cos(lam)); eta = np.arctanh(np.sin(lam) / np.sqrt(1 + t**2))
    x = eta.copy(); y = xi.copy()
    for j, a in enumerate(_ALPHA, 1):
        x = x + a * np.cos(2 * j * xi) * np.sinh(2 * j * eta)
        y = y + a * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
    return 500000.0 + _K0 * _AA * x, _K0 * _AA * y


def utm_to_lonlat(easting, northing, zone):
    """UTM easting/northing (northern hemisphere) to WGS 84 longitude/latitude in degrees."""
    xi = np.asarray(northing, float) / (_K0 * _AA); eta = (np.asarray(easting, float) - 500000.0) / (_K0 * _AA)
    xi_p = xi.copy(); eta_p = eta.copy()
    for j, b in enumerate(_BETA, 1):
        xi_p = xi_p - b * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
        eta_p = eta_p - b * np.cos(2 * j * xi) * np.sinh(2 * j * eta)
    chi = np.arcsin(np.sin(xi_p) / np.cosh(eta_p))
    phi = chi.copy()
    for j, d in enumerate(_DELTA, 1):
        phi = phi + d * np.sin(2 * j * chi)
    lam = np.arctan2(np.sinh(eta_p), np.cos(xi_p))
    return np.degrees(lam) + (zone * 6 - 183), np.degrees(phi)


# ------------------------------------------------------------------- readers
def _units_from_coords(x0, y0, dx):
    return "deg" if (abs(x0) <= 180 and abs(y0) <= 90 and dx < 1) else "m"


def read_geosoft(data):
    g = read_grd(data)
    return {"grid": g["grid"], "x0": g["x0"], "y0": g["y0"], "dx": g["dx"], "dy": g["dy"],
            "units": _units_from_coords(g["x0"], g["y0"], g["dx"])}


def read_geotiff(data):
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover - depends on deployment
        raise ValueError("GeoTIFF support needs the 'rasterio' package on the server.") from exc
    with rasterio.MemoryFile(data) as mem, mem.open() as src:
        arr = src.read(1).astype(float)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        t = src.transform
        dx, dy = abs(t.a), abs(t.e)
        grid = arr[::-1] if t.e < 0 else arr
        x0 = t.c + dx / 2
        y0 = (t.f - dy * arr.shape[0] + dy / 2) if t.e < 0 else t.f + dy / 2
        units = "deg" if (src.crs is not None and src.crs.is_geographic) else _units_from_coords(x0, y0, dx)
    return {"grid": grid, "x0": x0, "y0": y0, "dx": dx, "dy": dy, "units": units}


_COLUMN_ALIASES = {
    "x": ["x", "lon", "long", "longitude", "easting", "east", "utm_x"],
    "y": ["y", "lat", "latitude", "northing", "north", "utm_y"],
    "tmi": ["tmi", "mag", "magnetic", "rtp", "rte", "tmi_nt"],
    "k": ["k", "potassium", "k_pct", "potassium_pct"],
    "eth": ["eth", "th", "thorium", "thorium_ppm", "eth_ppm"],
    "eu": ["eu", "u", "uranium", "uranium_ppm", "eu_ppm"],
    "tc": ["tc", "total_count", "total_count_cps"],
    "dem": ["dem", "elev", "elevation", "elevation_m", "z", "alt"],
    "grav": ["grav", "gravity", "bouguer", "bouguer_mgal"],
}


def read_table(data, filename):
    """Read a CSV / XYZ point file. Returns {kind: raw layer} for every channel found."""
    text = data.decode("utf-8", "replace")
    sep = "," if filename.lower().endswith(".csv") else r"\s+"
    df = pd.read_csv(io.StringIO(text), sep=sep, engine="python", comment="/")
    lower = {c.lower().strip(): c for c in df.columns}
    found = {}
    for key, aliases in _COLUMN_ALIASES.items():
        for a in aliases:
            if a in lower:
                found[key] = lower[a]
                break
    if "x" not in found or "y" not in found:
        raise ValueError("The table needs coordinate columns (x/y, easting/northing or longitude/latitude).")
    channels = [k for k in found if k not in ("x", "y")]
    if not channels:
        raise ValueError("No survey channel recognised. Expected columns such as tmi, k, eth, eu, elevation or bouguer.")
    x = pd.to_numeric(df[found["x"]], errors="coerce").to_numpy(); y = pd.to_numeric(df[found["y"]], errors="coerce").to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 20:
        raise ValueError("Too few valid points to build a grid.")
    x, y = x[ok], y[ok]
    units = "deg" if (np.abs(x).max() <= 180 and np.abs(y).max() <= 90) else "m"
    area = (x.max() - x.min()) * (y.max() - y.min())
    cell = max(np.sqrt(area / len(x)) / 1.5, 1e-9)
    nx = int(min(max((x.max() - x.min()) / cell, 20), 800)); ny = int(min(max((y.max() - y.min()) / cell, 20), 800))
    gx = np.linspace(x.min(), x.max(), nx); gy = np.linspace(y.min(), y.max(), ny)
    GX, GY = np.meshgrid(gx, gy)
    layers = {}
    for kind in channels:
        v = pd.to_numeric(df[found[kind]], errors="coerce").to_numpy()[ok]
        good = np.isfinite(v)
        if good.sum() < 20:
            continue
        grid = griddata((x[good], y[good]), v[good], (GX, GY), method="linear")
        layers[kind] = {"grid": grid, "x0": gx[0], "y0": gy[0], "dx": gx[1] - gx[0], "dy": gy[1] - gy[0], "units": units}
    return layers


def read_file(filename, data):
    """Read one uploaded file. Returns {kind or None: raw layer}; kind None means 'ask the user'."""
    low = filename.lower()
    if low.endswith(".grd"):
        return {guess_kind(filename): read_geosoft(data)}
    if low.endswith((".tif", ".tiff")):
        return {guess_kind(filename): read_geotiff(data)}
    if low.endswith((".csv", ".xyz", ".txt", ".tsv")):
        return read_table(data, filename)
    raise ValueError("Unsupported file type. Use Geosoft .grd, GeoTIFF, CSV or XYZ.")


# ------------------------------------------------------------------ cleaning
def fill_gaps(grid):
    """Fill blank cells with the value of the nearest valid cell."""
    bad = ~np.isfinite(grid)
    if not bad.any():
        return grid
    if bad.all():
        raise ValueError("A layer contains no valid data.")
    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return grid[tuple(idx)]


def despike(grid, size=5, k=8.0):
    """
    Replace spikes with the local median. A cell counts as a spike only when it is
    both far from its neighbours and an extreme value for the whole grid, so sharp
    but real geological boundaries are left alone. Returns (grid, n_fixed).
    """
    med = ndimage.median_filter(grid, size=size, mode="nearest")
    resid = grid - med
    mad = np.median(np.abs(resid - np.median(resid))) * 1.4826 + 1e-12
    lo, hi = np.percentile(grid, (0.05, 99.95))
    spikes = (np.abs(resid) > k * mad) & ((grid > hi) | (grid < lo))
    out = grid.copy(); out[spikes] = med[spikes]
    return out, int(spikes.sum())


# -------------------------------------------------------------- harmonisation
def _extent_utm(layer, zone):
    g = layer["grid"]; ny, nx = g.shape
    xs = layer["x0"] + np.array([0, nx - 1]) * layer["dx"]; ys = layer["y0"] + np.array([0, ny - 1]) * layer["dy"]
    if layer["units"] == "m":
        return xs[0], xs[1], ys[0], ys[1]
    ex, ny_ = lonlat_to_utm([xs[0], xs[1], xs[0], xs[1]], [ys[0], ys[0], ys[1], ys[1]], zone)
    return max(ex[0], ex[2]), min(ex[1], ex[3]), max(ny_[0], ny_[1]), min(ny_[2], ny_[3])


def _resample(layer, X, Y, zone, cell):
    g = layer["grid"]
    if layer["units"] == "deg":
        lon, lat = utm_to_lonlat(X, Y, zone)
        col = (lon - layer["x0"]) / layer["dx"]; row = (lat - layer["y0"]) / layer["dy"]
        native = layer["dx"] * 111320.0 * np.cos(np.radians(np.nanmean(lat)))
    else:
        col = (X - layer["x0"]) / layer["dx"]; row = (Y - layer["y0"]) / layer["dy"]
        native = layer["dx"]
    valid = np.isfinite(g)
    filled = np.where(valid, g, 0.0)
    w = valid.astype(float)
    if cell / native > 1.5:  # coarser target: average first so fine detail does not alias
        size = int(round(cell / native))
        filled = ndimage.uniform_filter(filled, size=size, mode="nearest"); w = ndimage.uniform_filter(w, size=size, mode="nearest")
    num = ndimage.map_coordinates(filled, [row, col], order=1, mode="constant", cval=0.0)
    den = ndimage.map_coordinates(w, [row, col], order=1, mode="constant", cval=0.0)
    out = np.full(X.shape, np.nan)
    ok = den > 0.5
    out[ok] = num[ok] / den[ok]
    return out


def harmonise(raw, cell=100.0, max_cells=1200):
    """
    Put every raw layer on one UTM grid covering the area they all share.

    Returns a project dict: layers, x0, y0, dx, nx, ny, zone, and a log of what
    was done to each layer (shown to the user as the quality report).
    """
    if not raw:
        raise ValueError("No layers to process.")
    zone = None
    for layer in raw.values():
        if layer["units"] == "deg":
            zone = utm_zone(layer["x0"] + np.arange(layer["grid"].shape[1]) * layer["dx"])
            break
    if zone is None:
        zone = 32  # projected input: assumed already in its own UTM zone; the label is informational
    ext = [_extent_utm(l, zone) for l in raw.values()]
    xmin = max(e[0] for e in ext); xmax = min(e[1] for e in ext); ymin = max(e[2] for e in ext); ymax = min(e[3] for e in ext)
    if xmax - xmin < 5 * cell or ymax - ymin < 5 * cell:
        raise ValueError("The uploaded layers do not overlap. Check that they cover the same area.")
    cell = max(cell, (xmax - xmin) / max_cells, (ymax - ymin) / max_cells)
    xmin = np.ceil(xmin / cell) * cell; ymin = np.ceil(ymin / cell) * cell
    nx = int(np.floor((xmax - xmin) / cell)) + 1; ny = int(np.floor((ymax - ymin) / cell)) + 1
    X, Y = np.meshgrid(xmin + np.arange(nx) * cell, ymin + np.arange(ny) * cell)
    layers, log = {}, []
    for kind, layer in raw.items():
        g = _resample(layer, X, Y, zone, cell)
        blank = float(np.mean(~np.isfinite(g)) * 100)
        g = fill_gaps(g)
        n_spikes = 0; n_neg = 0
        if kind in ("k", "eth", "eu", "tc"):
            g, n_spikes = despike(g)
            n_neg = int((g < 0).sum()); g = np.clip(g, 0, None)
        native = layer["dx"] if layer["units"] == "m" else layer["dx"] * 111320.0
        layers[kind] = g
        log.append({"Layer": LAYER_NAMES.get(kind, kind), "Original cell (m)": round(float(native), 1),
                    "Original coordinates": "Longitude/latitude" if layer["units"] == "deg" else "Projected (m)",
                    "Blank cells filled (%)": round(blank, 2), "Spikes removed": n_spikes, "Negative values set to 0": n_neg,
                    "Minimum": round(float(g.min()), 3), "Mean": round(float(g.mean()), 3), "Maximum": round(float(g.max()), 3)})
    return {"layers": layers, "x0": float(xmin), "y0": float(ymin), "dx": float(cell), "nx": nx, "ny": ny, "zone": zone, "log": log}


def cell_to_xy(project, row, col):
    return project["x0"] + np.asarray(col) * project["dx"], project["y0"] + np.asarray(row) * project["dx"]


def xy_to_cell(project, x, y):
    return (np.asarray(y) - project["y0"]) / project["dx"], (np.asarray(x) - project["x0"]) / project["dx"]


def read_occurrences(data, filename, project):
    """Read known mineral occurrences (CSV with easting/northing or longitude/latitude). Returns a DataFrame with row, col."""
    df = pd.read_csv(io.BytesIO(data))
    lower = {c.lower().strip(): c for c in df.columns}

    def pick(names):
        for n in names:
            for c in lower:
                if c == n or c.startswith(n):
                    return lower[c]
        return None

    e = pick(["utm_easting", "easting", "utm_x"]); n = pick(["utm_northing", "northing", "utm_y"])
    lon = pick(["longitude_decimal", "longitude", "lon"]); lat = pick(["latitude_decimal", "latitude", "lat"])
    if e and n:
        x = pd.to_numeric(df[e], errors="coerce"); y = pd.to_numeric(df[n], errors="coerce")
    elif lon and lat:
        x, y = lonlat_to_utm(pd.to_numeric(df[lon], errors="coerce"), pd.to_numeric(df[lat], errors="coerce"), project["zone"])
    else:
        raise ValueError("The occurrence file needs easting/northing or longitude/latitude columns.")
    name_col = pick(["name", "occurrence", "settlement", "site", "deposit"])
    out = pd.DataFrame({"name": df[name_col].astype(str) if name_col else ["Occurrence %d" % (i + 1) for i in range(len(df))],
                        "easting": np.asarray(x, float), "northing": np.asarray(y, float)})
    row, col = xy_to_cell(project, out["easting"], out["northing"])
    out["row"] = np.round(row); out["col"] = np.round(col)
    inside = (out["row"] >= 0) & (out["row"] < project["ny"]) & (out["col"] >= 0) & (out["col"] < project["nx"]) & np.isfinite(out["easting"])
    return out[inside].astype({"row": int, "col": int}).reset_index(drop=True), int((~inside).sum())
