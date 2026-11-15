"""
Potential-field processing for magnetic (and gravity) grids.

All functions take a 2D array with row 0 = south and a square cell size in
metres. Filters are applied in the wavenumber domain on a mirror-padded,
tapered copy of the grid to limit edge effects.
"""
import numpy as np
from scipy import ndimage


def _pad(grid, frac=0.25):
    ny, nx = grid.shape
    py, px = max(int(ny * frac), 8), max(int(nx * frac), 8)
    mean = grid.mean()
    padded = np.pad(grid - mean, ((py, py), (px, px)), mode="reflect")
    wy = np.ones(padded.shape[0]); wx = np.ones(padded.shape[1])
    wy[:py] = 0.5 - 0.5 * np.cos(np.pi * np.arange(py) / py); wy[-py:] = wy[:py][::-1]
    wx[:px] = 0.5 - 0.5 * np.cos(np.pi * np.arange(px) / px); wx[-px:] = wx[:px][::-1]
    return padded * wy[:, None] * wx[None, :], mean, (py, px)


def _wavenumbers(shape, cell):
    ky = 2 * np.pi * np.fft.fftfreq(shape[0], d=cell); kx = 2 * np.pi * np.fft.fftfreq(shape[1], d=cell)
    KX, KY = np.meshgrid(kx, ky)
    return KX, KY, np.hypot(KX, KY)


def _apply(grid, cell, make_filter, keep_mean=False):
    padded, mean, (py, px) = _pad(grid)
    KX, KY, K = _wavenumbers(padded.shape, cell)
    out = np.real(np.fft.ifft2(np.fft.fft2(padded) * make_filter(KX, KY, K)))
    out = out[py:py + grid.shape[0], px:px + grid.shape[1]]
    return out + mean if keep_mean else out


def upward_continue(grid, cell, height):
    """Field as it would be measured `height` metres higher: suppresses shallow, short-wavelength sources."""
    return _apply(grid, cell, lambda KX, KY, K: np.exp(-K * height), keep_mean=True)


def residual(grid, cell, height=2000.0):
    """Residual field: the grid minus its upward continuation (the regional)."""
    return grid - upward_continue(grid, cell, height)


def reduce_to_equator(grid, cell, inclination, declination):
    """
    Reduction to the magnetic equator, the standard low-latitude alternative to
    reduction to the pole. Angles in degrees; declination positive east.
    Assumes induced magnetisation. Anomalies are centred over their sources
    (as lows over magnetic bodies).
    """
    inc, dec = np.radians(inclination), np.radians(declination)

    def filt(KX, KY, K):
        with np.errstate(invalid="ignore", divide="ignore"):
            theta = np.arctan2(KX, KY)  # azimuth of the wavenumber, clockwise from north
            num = (1j * np.cos(theta)) ** 2
            den = (np.sin(inc) + 1j * np.cos(inc) * np.cos(theta - dec)) ** 2
            # the denominator only vanishes when the inclination is almost exactly zero
            floor = 1e-3
            small = np.abs(den) < floor
            f = num / np.where(small, floor * np.exp(1j * np.angle(den)), den)
            f[small & (np.abs(num) < floor)] = 1.0
        f[K == 0] = 1.0
        return f

    return _apply(grid, cell, filt, keep_mean=True)


def vertical_derivative(grid, cell, order=1):
    """Vertical derivative with a gentle high-cut so grid noise is not amplified."""
    def filt(KX, KY, K):
        kc = K.max() * 0.6
        return K ** order / (1.0 + (K / kc) ** 8)
    return _apply(grid, cell, filt)


def horizontal_gradient(grid, cell):
    gy, gx = np.gradient(grid, cell)
    return np.hypot(gx, gy)


def analytic_signal(grid, cell):
    gy, gx = np.gradient(grid, cell)
    vz = vertical_derivative(grid, cell)
    return np.sqrt(gx**2 + gy**2 + vz**2)


def tilt_angle(grid, cell):
    """Tilt derivative in degrees (Miller and Singh, 1994). Its zero contour follows source edges."""
    return np.degrees(np.arctan2(vertical_derivative(grid, cell), horizontal_gradient(grid, cell) + 1e-12))


def tilt_depth(tilt_deg, cell, band=10.0, sensor_height=80.0):
    """
    Depth to the top of magnetic contacts by the tilt-depth method (Salem et al., 2007):
    at a contact, distance to the source = 1 / |horizontal gradient of the tilt angle (radians)|.
    The survey's terrain clearance is subtracted to give depth below ground. Returns
    depths in metres at cells near the zero contour, NaN elsewhere. The grid cell
    size limits how shallow a source can be resolved.
    """
    t = np.radians(ndimage.gaussian_filter(tilt_deg, 1.0))
    gy, gx = np.gradient(t, cell)
    grad = np.hypot(gx, gy)
    depth = np.where(grad > 0, 1.0 / np.maximum(grad, 1e-12), np.nan)
    depth[np.abs(tilt_deg) > band] = np.nan
    depth[depth > 100 * cell] = np.nan
    return np.maximum(depth - sensor_height, 0.0)


def contacts(tilt_deg, strength, percentile=60.0):
    """Cells on the tilt zero contour where the anomaly is strong enough to be a real edge."""
    t = ndimage.gaussian_filter(tilt_deg, 1.0)
    sign = t > 0
    cross = np.zeros_like(sign)
    cross[:-1, :] |= sign[:-1, :] != sign[1:, :]
    cross[:, :-1] |= sign[:, :-1] != sign[:, 1:]
    return cross & (strength >= np.percentile(strength, percentile))


def lineaments(tilt_deg, strength, cell, min_length=1500.0, percentile=60.0):
    """
    Trace magnetic lineaments as connected runs of contact cells. Returns
    (mask, table) where the table lists each lineament's length (m),
    strike (degrees clockwise from north, 0-180) and centroid row and column.
    """
    mask = contacts(tilt_deg, strength, percentile)
    labels, n = ndimage.label(mask, structure=np.ones((3, 3)))
    keep = np.zeros_like(mask); rows = []
    for i, sl in enumerate(ndimage.find_objects(labels), 1):
        rr, cc = np.nonzero(labels[sl] == i)
        if rr.size < 5:
            continue
        length = np.hypot(np.ptp(rr), np.ptp(cc)) * cell
        if length < min_length:
            continue
        coords = np.column_stack([cc - cc.mean(), rr - rr.mean()])
        _, _, vt = np.linalg.svd(coords, full_matrices=False)
        strike = np.degrees(np.arctan2(vt[0, 0], vt[0, 1])) % 180.0
        keep[sl][labels[sl] == i] = True
        rows.append((length, strike, rr.mean() + sl[0].start, cc.mean() + sl[1].start))
    return keep, np.array(rows).reshape(-1, 4)


def density(mask, cell, radius=1500.0):
    """Fraction of nearby cells that are on a lineament, smoothed over `radius` metres."""
    return ndimage.gaussian_filter(mask.astype(float), sigma=radius / cell / 2.0)


def distance_to(mask, cell):
    """Distance in metres from every cell to the nearest True cell."""
    if not mask.any():
        return np.full(mask.shape, np.inf)
    return ndimage.distance_transform_edt(~mask) * cell
