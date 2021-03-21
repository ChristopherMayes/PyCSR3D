import numpy as np
import scipy.special as ss

from scipy.optimize import root_scalar

from numpy import abs, sin, cos, real, exp, pi, cbrt, sqrt, sign

from csr3d.core import psi_s


def symmetric_vec(n, d):
    """
    Returns a symmetric vector about 0 of length 2*n with spacing d.
    The center = 0 is at [n-1]
    """
    return np.arange(-n+1,n+1,1)*d

def green_meshes(density_shape, deltas, rho=None, beta=None, offset=(0,0,0)):
    """
    Computes Green funcion meshes for psi_s and psi_x simultaneously.
    These meshes are in real space (not scaled space).
    
    Parameters
    ----------
    shape : tuple(int, int, int)
        Shape of the charge mesh
    
    deltas : tuple(float, float, float)
        mesh spacing corresonding to dx, dy, dz
        
    beta : float
        relativistic beta
    
    
    
    Returns:
    tuple of:
        psi_s_grid : np.array
            Double-sized array for the psi_s Green function
        
        psi_x_grid : 
            Double-sized array for the psi_x Green function
        
        zvec2 : array-like
            Coordinate vector in z (real space) [m]

        xvec2 : array-like
            Coordinate vector in x (real space) [m]
    
    """
    

    
    nx, ny, nz = tuple(density_shape)
    dx, dy, dz = tuple(deltas) # Convenience
    
    # Change to internal coordinates
    dx = dx/rho
    dy = dy/rho
    dz = dz/(2*rho)
    
    # Make an offset grid
    vecs = [symmetric_vec(n, delta) for n, delta, o in zip(density_shape, [dx,dy,dz], offset)] 
    meshes = np.meshgrid(*vecs, indexing='ij')

    # Offset center singularity
    center = (nx-1,ny-1,nz-1)
    meshes[0][center] = dx/2
    
    psi_s_grid = psi_s(*meshes, beta=beta)
    psi_s_grid[center] = 0
    
    return psi_s_grid
    
