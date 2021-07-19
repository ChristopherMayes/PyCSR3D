import numpy as np

from scipy.signal import savgol_filter

import scipy.constants
mec2 = scipy.constants.value("electron mass energy equivalent in MeV") * 1e6
c_light = scipy.constants.c
e_charge = scipy.constants.e
r_e = scipy.constants.value("classical electron radius")


def csr1d_steady_state_kick_calc(z, weights, nz=100, rho=1, species="electron", normalized_units=False):

    """

    Steady State CSR 1D model kick calc

    
    Parameters
    ----------
    z : np.array
        Bunch z coordinates in [m]    
        
    weights : np.array
        weight array (positive only) in [C]
        This should sum to the total charge in the bunch        
        
    nz : int
        number of z grid points        
        
    rho : float
        bending radius in [m]        
        
    species : str
        Particle species. Currently required to be 'electron'   
        
    normalized_units : bool
        If True, will return in normalized units [1/m^2]
            This multiplied by Qtot / e_charge * r_e * mec2 gives:
        Otherwise, units of [eV/m] are returned (default).
        Default: False
        
    Returns
    -------
    dict with:
    
        denergy_ds : np.array
            energy kick for each particle in [eV/m], or [1/m^2] if normalized_units=True
            
        wake : np.array
            wake array that kicks were interpolated on
            
        zvec : np.array
            z coordinates for wake array
    
    """

    assert species == "electron", f"TODO: support species {species}"

    # Density
    H, edges = np.histogram(z, weights=weights, bins=nz)
    zmin, zmax = edges[0], edges[-1]
    dz = (zmax - zmin) / (nz - 1)

    zvec = np.linspace(zmin, zmax, nz)  # Sloppy with bin centers

    Qtot = np.sum(weights)
    density = H / dz / Qtot

    # Density derivative
    densityp = np.gradient(density) / dz
    densityp_filtered = savgol_filter(densityp, 13, 2)

    # Green function
    zi = np.arange(0, zmax - zmin, dz)
    #factor =
    # factor = -3**(2/3) * Qtot/e_charge * r_e * rho**(-2/3) / gamma  # factor for ddelta/ds [1/m]
    if normalized_units:
        factor =  -3**(2/3) * rho**(-2/3) # factor for normalized uinits [1/m^2]
    else:
        factor =  ( -3**(2/3) * Qtot / e_charge * r_e * mec2 * rho**(-2/3)  )  # factor for denergy/dz [eV/m]
    green = factor * np.diff(zi ** (2 / 3))

    # Convolve to get wake
    wake = np.convolve(densityp_filtered, green, mode="full")[0 : len(zvec)]

    # Interpolate to get the kicks
    delta_kick = np.interp(z, zvec, wake)

    return {"denergy_ds": delta_kick, "zvec": zvec, "wake": wake}
