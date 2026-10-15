"""
RegolithSense - High-Fidelity Geological Baseline Generator
Simulates realistic aeromagnetic and radiometric survey grids covering the
Central Nigerian Lithium-Pegmatite Belt (Nasarawa-Keffi-Kokona corridor).
Accurately models:
- Migmatite-gneiss basement complex
- NE-SW Pan-African shear zones & faults
- Syn- to post-tectonic granite intrusions
- Concealed LCT pegmatite dyke swarms beneath 20-50m of lateritic regolith
"""

import numpy as np
from scipy import ndimage

def generate_nasarawa_geological_grid(nx=120, ny=120, dx=100.0, dy=100.0):
    """
    Generates synthetic high-resolution geophysical grids:
    - bounds: 7.80E to 8.50E, 8.50N to 9.10N (Nasarawa / Keffi / Kokona)
    - Total Magnetic Intensity (TMI in nT)
    - Radiometric Potassium (K in %)
    - Radiometric Thorium (eTh in ppm)
    - Radiometric Uranium (eU in ppm)
    - Regolith thickness (m)
    """
    np.random.seed(101)
    
    # Coordinates grid
    x = np.linspace(0, nx * dx, nx)
    y = np.linspace(0, ny * dy, ny)
    X, Y = np.meshgrid(x, y)
    
    bounds = (7.80, 8.50, 8.50, 9.10) # (min_lon, max_lon, min_lat, max_lat)
    
    # 1. Regional Background: Precambrian Basement Complex (33,000 nT baseline)
    tmi = 33200.0 + 150.0 * np.sin(2 * np.pi * X / (nx * dx * 1.5)) * np.cos(2 * np.pi * Y / (ny * dy * 1.2))
    
    # 2. Granitic Plutons (Older Granite suite - Nasarawa batholiths)
    # Elliptical positive magnetic anomalies
    cx1, cy1 = nx * dx * 0.35, ny * dy * 0.40
    pluton1 = 280.0 * np.exp(-(((X - cx1)/2800)**2 + ((Y - cy1)/1800)**2))
    
    cx2, cy2 = nx * dx * 0.70, ny * dy * 0.65
    pluton2 = 220.0 * np.exp(-(((X - cx2)/2200)**2 + ((Y - cy2)/2600)**2))
    
    tmi += (pluton1 + pluton2)
    
    # 3. Major Pan-African Shear Zones (NE-SW strike ~ 45 degrees)
    # Linear magnetic lows / highs representing mylonite belts and fault splays
    shear1 = 180.0 * np.sin((X + Y * 1.1) / 1200.0) * np.exp(-((X - Y * 0.9 - 1500)**2)/(2 * 1800**2))
    shear2 = -120.0 * np.exp(-((X - Y * 0.85 + 3200)**2)/(2 * 900**2))
    tmi += (shear1 + shear2)
    
    # 4. Pegmatite Dyke Swarms (Sub-parallel en-echelon veins striking NNE)
    pegmatite_mask = np.zeros_like(tmi)
    for offset in [-2500, -800, 1200, 3100, 4800]:
        dyke = np.exp(-((X - Y * 0.45 + offset)**2)/(2 * 220**2)) * (0.8 + 0.4 * np.sin(Y / 600.0))
        # Keep dykes concentrated near shear margins
        near_shear = np.exp(-((X - Y * 0.9 - 1500)**2)/(2 * 4500**2))
        tmi += dyke * 90.0 * near_shear
        pegmatite_mask += dyke * near_shear
        
    pegmatite_mask = np.clip(pegmatite_mask, 0, 1)

    # 5. High-frequency geological texture & slight measurement noise
    noise = np.random.normal(0, 3.5, size=(ny, nx))
    tmi += ndimage.gaussian_filter(noise, sigma=0.8)

    # 6. Radiometrics Simulation
    # Normal background: K ~ 1.2%, eTh ~ 9.0 ppm, eU ~ 2.2 ppm
    k_grid = 1.1 + 0.3 * np.random.rand(ny, nx)
    eth_grid = 8.5 + 1.2 * np.random.rand(ny, nx)
    eu_grid = 2.1 + 0.5 * np.random.rand(ny, nx)
    
    # Plutons have higher Th & U
    eth_grid += (pluton1 + pluton2) * 0.035
    eu_grid += (pluton1 + pluton2) * 0.012
    
    # Pegmatite swarms have intense hydrothermal Potassic metasomatism (K surges to 3.5% - 5.5%)
    # while eTh remains immobile or depleted!
    k_grid += pegmatite_mask * 3.4
    
    # 7. Topographic Elevation (DEM in meters above sea level: 310m to 640m)
    # Hills over granitic plutons, rolling terrain over basement, incised valleys
    elevation = 360.0 + 120.0 * np.exp(-(((X - cx1)/3500)**2 + ((Y - cy1)/2400)**2)) \
                + 160.0 * np.exp(-(((X - cx2)/3000)**2 + ((Y - cy2)/3200)**2)) \
                + 45.0 * np.sin(X / 2200.0) * np.cos(Y / 1800.0)
    elevation += ndimage.gaussian_filter(np.random.normal(0, 3.0, (ny, nx)), sigma=1.0)

    # 8. Regolith Cover Thickness (15m to 65m)
    # Thicker in relict plateaus/valleys, thinner over eroded inselberg ridges
    regolith_thickness = 28.0 + 14.0 * np.sin(X / 2500.0) + 10.0 * np.cos(Y / 3000.0) \
                         - (elevation - 360.0) * 0.08 + np.random.normal(0, 1.5, (ny, nx))
    regolith_thickness = np.clip(regolith_thickness, 12.0, 72.0)
    bedrock_elevation = elevation - regolith_thickness

    # 9. Bouguer Gravity Anomaly Modeling (mGal)
    # Regional baseline: -34.0 mGal
    # Felsic peraluminous leucogranites (parent pegmatite sources) produce negative anomalies (-6 to -12 mGal)
    # Dense amphibolitic/schist belts produce positive anomalies (+4 to +8 mGal)
    bouguer = -34.0 - 0.038 * (pluton1 + pluton2) \
              + 0.032 * (shear1 * (shear1 > 0)) \
              - (regolith_thickness * 0.045) # low density laterite response
    bouguer += ndimage.gaussian_filter(np.random.normal(0, 0.4, (ny, nx)), sigma=1.2)

    # Known deposits database inside this test sector
    known_deposits = [
        {"name": "Nasarawa Eggon Pegmatite Mine", "lon": 8.1820, "lat": 8.7840, "mineral": "Spodumene / Lithium"},
        {"name": "Kokona West Tantalite Outcrop", "lon": 8.0210, "lat": 8.6910, "mineral": "Columbite-Tantalite"},
        {"name": "Udege LCT Pegmatite Swarm", "lon": 8.3150, "lat": 8.9250, "mineral": "Lepidolite / Spodumene"}
    ]

    return {
        "tmi": tmi,
        "bouguer": bouguer,
        "k": k_grid,
        "eth": eth_grid,
        "eu": eu_grid,
        "elevation": elevation,
        "regolith_thickness": regolith_thickness,
        "bedrock_elevation": bedrock_elevation,
        "bounds": bounds,
        "dx": dx,
        "dy": dy,
        "known_deposits": known_deposits
    }

