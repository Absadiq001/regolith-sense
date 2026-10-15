"""
RegolithSense - Advanced Radiometrics & Hydrothermal Alteration Suite
Comprehensive gamma-ray spectrometry processing for critical mineral and pegmatite exploration:
1. Total Count (TC) & Radioelement Concentrations (K %, eTh ppm, eU ppm)
2. Diagnostic Radioelement Ratios: K/eTh, eU/eTh, eU/K
3. Efimov / Saunders F-Parameter Alteration Index (F = K * eU / eTh)
4. Potassium Deviation / Residual Enrichment (Kd = K - eTh / 3.75)
5. Radiometric Ternary RGB Compositing
"""

import numpy as np

def compute_radioelement_ratios(k_grid, eth_grid, eu_grid):
    """
    Computes standardized exploration gamma-ray radioelement metrics:
    - Total Count (TC): Modeled standard dose equivalent in cps / Ur
    - K/eTh: Potassic metasomatism index (hallmark of LCT pegmatites)
    - eTh/K: Crustal weathering ratio
    - eU/eTh: Hydrothermal uranium fractionation ratio
    - eU/K: Late-stage magmatic fraction ratio
    - F-Parameter (Efimov, 1978; Saunders et al., 1987): F = K * (eU / eTh)
    - Potassium Enrichment (Kd): Kd = K - (eTh / 3.75) (relative to average crust)
    """
    safe_k = np.where(k_grid <= 0.01, 0.01, k_grid)
    safe_eth = np.where(eth_grid <= 0.1, 0.1, eth_grid)
    safe_eu = np.where(eu_grid <= 0.01, 0.01, eu_grid)
    
    # 1. Total Count (standard approximation: TC = a*K + b*eU + c*eTh)
    total_count = (safe_k * 125.0) + (safe_eu * 38.0) + (safe_eth * 19.5)
    
    # 2. Ratios
    k_eth = safe_k / safe_eth
    eth_k = safe_eth / safe_k
    eu_eth = safe_eu / safe_eth
    eu_k = safe_eu / safe_k
    
    # 3. Efimov / Saunders F-Parameter Alteration Index
    # High F-parameter denotes simultaneous enrichment in K and mobile U relative to Th
    f_parameter = safe_k * (safe_eu / safe_eth)
    
    # 4. Potassium Enrichment Anomaly (Kd)
    # Average crustal ratio of eTh/K is approximately 3.75.
    # Positive values indicate hydrothermal potassium addition overprinting basement.
    kd = safe_k - (safe_eth / 3.75)

    return {
        "TC": total_count,
        "K_eTh": k_eth,
        "eTh_K": eth_k,
        "eU_eTh": eu_eth,
        "eU_K": eu_k,
        "F_param": f_parameter,
        "Kd": kd
    }

def compute_potassic_alteration_factor(k_grid, eth_grid):
    """
    Computes normalized potassic alteration index:
    F_k = (K * 4.0) / eTh
    Values > 1.8 denote hydrothermal potassic metasomatism (microcline/lepidolite/muscovite)
    overprinting basement gneisses.
    """
    safe_eth = np.where(eth_grid <= 0.1, 0.1, eth_grid)
    normalized_k = (k_grid * 4.0) / safe_eth
    return normalized_k

def generate_ternary_rgb(k_grid, eth_grid, eu_grid):
    """
    Generates a standard geophysical ternary RGB image:
    Red = Potassium (K)
    Green = Thorium (eTh)
    Blue = Uranium (eU)
    Normalized with 2%-98% percentile contrast clipping.
    """
    def normalize_band(b):
        p2, p98 = np.percentile(b, (2, 98))
        clipped = np.clip(b, p2, p98)
        norm = (clipped - p2) / (p98 - p2 + 1e-6)
        return (norm * 255).astype(np.uint8)

    r = normalize_band(k_grid)
    g = normalize_band(eth_grid)
    b = normalize_band(eu_grid)
    
    return np.stack([r, g, b], axis=-1)
