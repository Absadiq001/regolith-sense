"""
RegolithSense - Core Potential Field Geophysics Module
Implements mathematical operators for aeromagnetic wavefield analysis:
1. First Vertical Derivative (1VD / VDR) via 2D FFT
2. Horizontal Derivatives (dx, dy)
3. Total Horizontal Derivative (THDR)
4. Tilt Angle Derivative (TDR) - Zero-contour maps geologic contacts through regolith
5. Analytic Signal Amplitude (AS)
"""

import numpy as np
from scipy import ndimage

def calculate_horizontal_gradients(grid, dx=50.0, dy=50.0):
    """
    Computes spatial horizontal derivatives dT/dx and dT/dy using Sobel / central differences.
    dx, dy: cell grid spacing in meters.
    """
    # Central gradient using numpy gradient
    gy, gx = np.gradient(grid, dy, dx)
    return gx, gy

def calculate_total_horizontal_derivative(gx, gy):
    """
    Total Horizontal Derivative:
    THDR = sqrt((dT/dx)^2 + (dT/dy)^2)
    Highlights structural edges and fault contacts.
    """
    return np.sqrt(gx**2 + gy**2)

def calculate_vertical_derivative_fft(grid, dx=50.0, dy=50.0):
    """
    Calculates the First Vertical Derivative (1VD = dT/dz) in the wavenumber domain using 2D FFT.
    F[dT/dz] = |k| * F[T], where |k| = sqrt(kx^2 + ky^2) is radial wavenumber.
    """
    ny, nx = grid.shape
    
    # 2D Fourier Transform
    fft_grid = np.fft.fft2(grid)
    
    # Wavenumber grids (rad/m)
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    
    # Apply vertical derivative filter in Fourier space
    # High-wavenumber cosine roll-off to suppress high-frequency noise
    k_cutoff = np.max(K) * 0.45
    butterworth = 1.0 / (1.0 + (K / k_cutoff)**6)
    
    fft_vdr = fft_grid * K * butterworth
    
    # Inverse FFT to spatial domain
    vdr = np.real(np.fft.ifft2(fft_vdr))
    return vdr

def calculate_analytic_signal(thdr, vdr):
    """
    3D Analytic Signal Amplitude:
    AS = sqrt((dT/dx)^2 + (dT/dy)^2 + (dT/dz)^2) = sqrt(THDR^2 + VDR^2)
    Independent of geomagnetic inclination; peaks over magnetization contrast boundaries.
    """
    return np.sqrt(thdr**2 + vdr**2)

def calculate_tilt_angle_derivative(vdr, thdr):
    """
    Tilt Angle Derivative (Miller & Singh, 1994):
    TDR = arctan( VDR / THDR ) in degrees [-90 deg to +90 deg]
    The zero-contour (TDR = 0) accurately tracks the contact boundary between differing lithologies
    regardless of whether the magnetic source is shallow or deeply buried beneath laterite.
    """
    denom = np.where(thdr == 0, 1e-6, thdr)
    tdr_rad = np.arctan2(vdr, denom)
    return np.degrees(tdr_rad)

def extract_structural_lineaments(tdr_grid, threshold_deg=5.0):
    """
    Extracts structural lineaments along the zero-contour crossings of the Tilt Angle Derivative.
    Returns binary mask of concealed fault/contact trajectories.
    """
    # Detect zero crossings where absolute TDR is near zero and gradient is steep
    zero_crossing = (np.abs(tdr_grid) < threshold_deg).astype(float)
    # Morphological skeletonization
    smoothed = ndimage.gaussian_filter(zero_crossing, sigma=1.0)
    lineaments = (smoothed > 0.4).astype(int)
    return lineaments

def calculate_gravity_vertical_gradient(bouguer_grid, dx=100.0, dy=100.0):
    """
    Computes First Vertical Gravity Gradient (Gzz = dg/dz in Eotvos / mGal/km)
    via 2D Fourier wavenumber domain filtering:
    F[Gzz] = |k| * F[g_bouguer]
    Enhances shallow density contrasts (granites, faults, pegmatites) over deep regional roots.
    """
    ny, nx = bouguer_grid.shape
    fft_grid = np.fft.fft2(bouguer_grid)
    
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    
    # Filter with smooth high-cut roll-off
    k_cutoff = np.max(K) * 0.40
    butterworth = 1.0 / (1.0 + (K / k_cutoff)**6)
    
    fft_gzz = fft_grid * K * butterworth
    # Convert from 1/m to Eotvos (1 Eotvos = 10^-9 s^-2 = 0.1 mGal/km)
    gzz = np.real(np.fft.ifft2(fft_gzz)) * 1e4
    return gzz

def calculate_horizontal_gravity_gradient(bouguer_grid, dx=100.0, dy=100.0):
    """
    Computes Horizontal Gravity Gradient:
    HGG = sqrt((dg/dx)^2 + (dg/dy)^2)
    Peaks sharply over high-contrast density contacts (e.g. granite intrusive margins).
    """
    gy, gx = np.gradient(bouguer_grid, dy, dx)
    return np.sqrt(gx**2 + gy**2)

def calculate_residual_gravity(bouguer_grid, regional_sigma=9.0):
    """
    Separates regional deep crustal gravity from shallow residual anomalies.
    Residual = Bouguer - Regional_smooth
    Isolates light peraluminous granites (negative residual) and dense schists (positive residual).
    """
    regional = ndimage.gaussian_filter(bouguer_grid, sigma=regional_sigma)
    residual = bouguer_grid - regional
    return residual, regional

