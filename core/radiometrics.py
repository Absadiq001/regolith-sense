"""Gamma-ray spectrometry products used to map potassic alteration."""
import numpy as np
from scipy import ndimage


def _safe(a, floor):
    return np.maximum(a, floor)


def products(k, eth, eu):
    """
    Returns a dict of alteration products:
      k_eth    K/eTh ratio; rises where potassium was added without thorium
      eu_eth   eU/eTh ratio
      f_param  F = K * eU / eTh (Efimov), high over K and U enrichment
      kd, ud   deviation of K and U from the value expected for the local
               thorium content (Pires, 1995); positive = enrichment
    """
    k = ndimage.gaussian_filter(k, 1.0); eth = ndimage.gaussian_filter(eth, 1.0); eu = ndimage.gaussian_filter(eu, 1.0)
    th = _safe(eth, 0.5)
    k_ideal = (k.mean() / th.mean()) * th
    u_ideal = (eu.mean() / th.mean()) * th
    return {
        "k_eth": k / th,
        "eu_eth": eu / th,
        "f_param": k * eu / th,
        "kd": (k - k_ideal) / _safe(k_ideal, 1e-3),
        "ud": (eu - u_ideal) / _safe(u_ideal, 1e-3),
    }


def ternary(k, eth, eu, low=2, high=98):
    """RGB image (0-1 floats): red = K, green = eTh, blue = eU, each stretched between percentiles."""
    out = np.zeros(k.shape + (3,))
    for i, band in enumerate((k, eth, eu)):
        lo, hi = np.percentile(band, (low, high))
        out[..., i] = np.clip((band - lo) / (hi - lo + 1e-12), 0, 1)
    return out


def alteration_zones(kd, f_param, k_sigma=1.5):
    """
    Cells flagged as potassic alteration: K enrichment relative to thorium
    more than `k_sigma` standard deviations above the mean, with an F-parameter
    above its median. Small isolated patches are removed.
    """
    z = (kd - kd.mean()) / (kd.std() + 1e-12)
    mask = (z > k_sigma) & (f_param > np.median(f_param))
    return ndimage.binary_opening(mask, structure=np.ones((2, 2)))
