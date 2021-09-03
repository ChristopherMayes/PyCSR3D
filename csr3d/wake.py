import numpy as np
import scipy.special as ss

from scipy.optimize import root_scalar
from scipy import integrate

    
from csr3d.core import psi_x0, psi_xhat0,  psi_y0, psi_s, psi_phi0, Fx_case_B_Chris, Fy_case_B_Chris 


def symmetric_vec(n, d):
    """
    Returns a symmetric vector about 0 of length 2*n with spacing d.
    The center = 0 is at [n-1]
    """
    return np.arange(-n+1,n+1,1)*d

def green_mesh(density_shape, deltas, rho=None, gamma=None, offset=(0,0,0), component='s'):
    """
    Computes Green funcion meshes for a particular component
    These meshes are in real space (not scaled space).
    
    Parameters
    ----------
    shape : tuple(int, int, int)
        Shape of the charge mesh
    
    deltas : tuple(float, float, float)
        mesh spacing corresonding to dx, dy, dz
        
    gamma : float
        relativistic gamma
    
    Returns:
        Double-sized array for the psi_s Green function

    
    """
    
    # handle negative rho
    rho_sign = np.sign(rho)
    rho = abs(rho)
    
    nx, ny, nz = tuple(density_shape)
    dx, dy, dz = tuple(deltas) # Convenience
    
    # Change to internal coordinates
    dx = dx/rho
    dy = dy/rho
    dz = dz/(2*rho)
    
    # Make an offset grid
    vecs = [symmetric_vec(n, delta) for n, delta, o in zip(density_shape, [dx,dy,dz], offset)] 
    vecs[0] = rho_sign*vecs[0] # Flip sign of x
    meshes = np.meshgrid(*vecs, indexing='ij')

    
    
    if component == 'x':
        green = rho_sign*psi_x0(*meshes, gamma, dx, dy, dz)      
    elif component == 'xhat':
        green = rho_sign*psi_xhat0(*meshes, gamma, dx, dy, dz)           
    elif component == 'y':
        green = psi_y0(*meshes, gamma, dx, dy, dz)    
    elif component == 'phi':
        green = rho_sign*psi_phi0(*meshes, gamma, dx, dy, dz)            
    elif component == 's':
        green = psi_s(*meshes, gamma)
        
    elif component in ['Fx_IGF', 'Fy_IGF']:
        
        # Pick the function
        # Takes x, y, z, gamma
        if component == 'Fx_IGF':
            F = Fx_case_B_Chris
        else:
            F = Fy_case_B_Chris 

        
        # Flat meshes
        X = meshes[0].flatten()
        Y = meshes[1].flatten()
        Z = meshes[2].flatten()

        # Select special points for IGF
        ix_for_IGF = np.where(abs(Z)<dz*1.5)
        # Select special points for IGF
       # ix_for_IGF = np.where(np.logical_and( abs(Z)<dz*2, abs(X)<dx*2 ))        
        

        print(f'IGF for {len(ix_for_IGF[0])} points...')
        
        X_special = X[ix_for_IGF]
        Y_special = Y[ix_for_IGF]
        Z_special = Z[ix_for_IGF]

        # evaluate special
        f3 = lambda x, y, z: IGF_z(F, x, y, z, dx, dy, dz, gamma)/dz
        
        res = map(f3, X_special, Y_special, Z_special)
        G_short = np.array(list(res))
        
        print(f'Done. Starting midpoint method...')
        
        # Simple midpoint evaluation
        G = F(X, Y, Z, gamma)
        # Fill with IGF
        G[ix_for_IGF] = G_short
        
        # reshape
        green = G.reshape(meshes[0].shape)
        
    else:
        raise ValueError(f'Unknown component: {component}')
    
    return green
    

    
def IGF_z(func, x, y, z, dx, dy, dz, gamma):
    """
    Special Integrated Green Function (IGF) in the z direction only
    
    """
    
    func_x = lambda x: func(x, y, z, gamma)
    func_z = lambda z: func(x, y, z, gamma)

    if abs(z) < 1e-14:
        if (abs(x) < 1e-14) and (abs(y)< 1e-14):
            return 0

    return integrate.quad(func_z, z-dz/2, z+dz/2, 
                          points = [z], 
                          epsrel=1e-4, # Coarse
                          limit=50)[0]        
