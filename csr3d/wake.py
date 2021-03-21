import numpy as np
import scipy.special as ss

from scipy.optimize import root_scalar

from numpy import abs, sin, cos, real, exp, pi, cbrt, sqrt, sign

from csr3d.core import psi_calc


def symmetric_vec(n, d):
    """
    Returns a symmetric vector about 0 of length 2*n with spacing d.
    The center = 0 is at [n-1]
    """
    return np.arange(-n+1,n+1,1)*d

def green_meshes(density_shape, deltas, rho=None, beta=None, offset=(0,0,0), components=['s']):
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

    # Offset singularity along z axis (x =0, y = 0)
    # Same as in the CSR 2D code
    center = (nx-1,ny-1,nz-1)    
    meshes[0][nx-1,ny-1,:] = -dx/2
    meshes[0][-1,ny-1,:] = dx/2
    
    # This does all the calcs
    potential = psi_calc(*meshes, beta=beta, components=components)

    # Rename for output
    out = {}
    for k, field in potential.items():
        # Average out this axis
        field[nx-1,ny-1,:] = (field[nx-1,ny-1,:] + field[-1,ny-1,:])/2
        out[k+'_mesh'] = potential[k]

        # Replace origin
#        potential[k][center] = 0
    
    return out
    
