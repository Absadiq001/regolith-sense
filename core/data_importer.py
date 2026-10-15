"""
RegolithSense - Multi-Format Survey Ingestion & Grid Interpolation Engine
Supports CSV, TSV, and XYZ ASCII survey files containing:
- Coordinates (Lon/Lat or Easting/Northing)
- Gravity (Bouguer Anomaly in mGal)
- Radiometrics (K %, eTh ppm, eU ppm, Total Count)
- Aeromagnetics (TMI in nT)
- Elevation (DEM in meters)
"""

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy import ndimage
import io

def detect_survey_columns(df):
    """
    Intelligently maps varying column headers from exploration survey files.
    """
    cols = {col.lower().strip(): col for col in df.columns}
    
    mapping = {
        "x": None,
        "y": None,
        "tmi": None,
        "bouguer": None,
        "k": None,
        "eth": None,
        "eu": None,
        "tc": None,
        "elevation": None
    }
    
    # Matching rules
    patterns = {
        "x": ['lon', 'longitude', 'long', 'x', 'easting', 'utm_x'],
        "y": ['lat', 'latitude', 'y', 'northing', 'utm_y'],
        "tmi": ['tmi', 'mag', 'rtp', 'magnetic', 'total_mag', 'tmi_nt'],
        "bouguer": ['bouguer', 'gravity', 'grav', 'boug', 'faa', 'mgal', 'bouguer_mgal'],
        "k": ['k', 'potassium', 'k_pct', 'k_percent', 'potassium_%', 'k_perc'],
        "eth": ['eth', 'thorium', 'th', 'eth_ppm', 'th_ppm'],
        "eu": ['eu', 'uranium', 'u', 'eu_ppm', 'u_ppm'],
        "tc": ['tc', 'total_count', 'cps', 'dose_rate', 'totalcount'],
        "elevation": ['elev', 'elevation', 'dem', 'alt', 'altitude', 'z', 'elevation_m']
    }
    
    for key, aliases in patterns.items():
        for alias in aliases:
            for lower_col, actual_col in cols.items():
                if lower_col == alias or lower_col.startswith(alias + "_") or lower_col.endswith("_" + alias):
                    mapping[key] = actual_col
                    break
            if mapping[key] is not None:
                break
                
    return mapping

def interpolate_points_to_grid(df, mapping, nx=100, ny=100):
    """
    Interpolates scattered or profile line points onto a regular 2D raster grid.
    """
    x_col = mapping["x"]
    y_col = mapping["y"]
    
    if not x_col or not y_col:
        raise ValueError("Survey file must contain valid X/Longitude and Y/Latitude coordinate columns.")
        
    pts_x = df[x_col].values.astype(float)
    pts_y = df[y_col].values.astype(float)
    
    min_x, max_x = np.nanmin(pts_x), np.nanmax(pts_x)
    min_y, max_y = np.nanmin(pts_y), np.nanmax(pts_y)
    
    grid_x = np.linspace(min_x, max_x, nx)
    grid_y = np.linspace(min_y, max_y, ny)
    XI, YI = np.meshgrid(grid_x, grid_y)
    
    points = np.column_stack([pts_x, pts_y])
    
    grids = {
        "bounds": (min_x, max_x, min_y, max_y),
        "dx": (max_x - min_x) / nx,
        "dy": (max_y - min_y) / ny,
        "nx": nx,
        "ny": ny
    }
    
    # Interpolate each variable
    variables = ["tmi", "bouguer", "k", "eth", "eu", "elevation", "tc"]
    defaults = {
        "tmi": 33200.0,
        "bouguer": -35.0,
        "k": 1.2,
        "eth": 9.0,
        "eu": 2.2,
        "elevation": 380.0,
        "tc": 350.0
    }
    
    for var in variables:
        col = mapping.get(var)
        if col and col in df.columns:
            vals = df[col].values.astype(float)
            # Remove NaNs
            valid = ~np.isnan(vals)
            if np.sum(valid) > 5:
                # Linear interpolation with nearest fallback for convex hull edges
                grid_vals = griddata(points[valid], vals[valid], (XI, YI), method='linear')
                # Fill remaining boundary NaNs with nearest neighbor
                nan_mask = np.isnan(grid_vals)
                if np.any(nan_mask):
                    grid_nearest = griddata(points[valid], vals[valid], (XI, YI), method='nearest')
                    grid_vals[nan_mask] = grid_nearest[nan_mask]
                # Light Gaussian smoothing
                grids[var] = ndimage.gaussian_filter(grid_vals, sigma=0.8)
            else:
                grids[var] = np.full((ny, nx), defaults[var])
        else:
            grids[var] = np.full((ny, nx), defaults[var])
            
    # Calculate synthetic regolith thickness if missing
    if "regolith_thickness" not in grids:
        grids["regolith_thickness"] = np.clip(32.0 + 8.0 * np.sin(XI * 50) - (grids["elevation"] - 380.0) * 0.05, 12.0, 70.0)
        grids["bedrock_elevation"] = grids["elevation"] - grids["regolith_thickness"]

    return grids

def generate_sample_survey_csv():
    """
    Generates a realistic ready-to-test CSV survey file for the user to download.
    Contains 2,500 flight-line points with coordinates, TMI, Bouguer gravity, K, eTh, and eU.
    """
    np.random.seed(42)
    n_points = 2500
    
    lons = np.random.uniform(8.10, 8.45, n_points)
    lats = np.random.uniform(8.65, 8.95, n_points)
    elev = 380.0 + 120.0 * np.sin((lons - 8.10) * 15) + np.random.normal(0, 3, n_points)
    
    # Pegmatite anomaly zone around lon ~ 8.28, lat ~ 8.80
    dist_anomaly = np.sqrt(((lons - 8.28)*80)**2 + ((lats - 8.80)*80)**2)
    anomaly_factor = np.exp(-dist_anomaly**2 / 12.0)
    
    tmi = 33150.0 + 90.0 * np.sin((lons + lats)*25) + anomaly_factor * 110.0 + np.random.normal(0, 2.5, n_points)
    bouguer = -34.5 - 0.03 * (tmi - 33150.0) - anomaly_factor * 7.5 + np.random.normal(0, 0.4, n_points)
    
    k_pct = 1.2 + 0.3 * np.random.rand(n_points) + anomaly_factor * 3.8 # K surges over pegmatites
    eth_ppm = 8.5 + 1.5 * np.random.rand(n_points) - anomaly_factor * 1.5 # Th immobile/depleted
    eu_ppm = 2.1 + 0.6 * np.random.rand(n_points) + anomaly_factor * 3.2 # U mobile surge
    total_count = k_pct * 125.0 + eu_ppm * 38.0 + eth_ppm * 19.5
    
    df_sample = pd.DataFrame({
        "longitude": np.round(lons, 5),
        "latitude": np.round(lats, 5),
        "elevation_m": np.round(elev, 1),
        "tmi_nT": np.round(tmi, 1),
        "bouguer_mgal": np.round(bouguer, 2),
        "potassium_pct": np.round(k_pct, 2),
        "thorium_ppm": np.round(eth_ppm, 2),
        "uranium_ppm": np.round(eu_ppm, 2),
        "total_count_cps": np.round(total_count, 1)
    })
    
    return df_sample.to_csv(index=False).encode('utf-8')
