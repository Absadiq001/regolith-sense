"""
First-pass regolith-landform classification from elevation and radiometrics.

Three regimes, after the relict / erosional / depositional scheme used for
exploration through cover (Anand and Butt, 2010):
  1 Relict       preserved lateritic surfaces: high ground, gentle slopes,
                 thorium retained while potassium is leached
  2 Erosional    stripped saprolite and bedrock: steeper or dissected ground,
                 radiometric signal reflects the rock beneath
  3 Depositional transported cover: low, flat valley floors
This is a rule-based screen meant to be checked against field observation.
"""
import numpy as np
from scipy import ndimage

REGIMES = {1: "Relict (lateritic)", 2: "Erosional (in situ)", 3: "Depositional (transported)"}


def terrain(dem, cell, window=1500.0):
    smooth = ndimage.gaussian_filter(dem, 1.0)
    gy, gx = np.gradient(smooth, cell)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    size = max(int(window / cell) | 1, 3)
    tpi = smooth - ndimage.uniform_filter(smooth, size=size, mode="nearest")  # position relative to surroundings
    return slope, tpi


def classify(dem, cell, k=None, eth=None):
    slope, tpi = terrain(dem, cell)
    flat = slope < np.percentile(slope, 45)
    low = tpi < np.percentile(tpi, 30)
    high = tpi > np.percentile(tpi, 60)
    regime = np.full(dem.shape, 2, dtype=int)
    regime[flat & low] = 3
    relict = flat & high
    if k is not None and eth is not None:
        ratio = ndimage.gaussian_filter(eth, 2.0) / np.maximum(ndimage.gaussian_filter(k, 2.0), 0.05)
        relict &= ratio > np.percentile(ratio, 55)  # leached potassium, residual thorium
    regime[relict] = 1
    regime = ndimage.median_filter(regime, size=5)
    return regime, slope, tpi


def in_situ_weight(regime):
    """How far the surface radiometric signal can be trusted to reflect bedrock (0-1)."""
    w = np.ones(regime.shape)
    w[regime == 1] = 0.6
    w[regime == 3] = 0.2
    return w
