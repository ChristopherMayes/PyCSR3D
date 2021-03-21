import numpy as np
import scipy.fft as sp_fft


def fftconvolve3(rho, *greens):
    """
    Efficiently perform a 3D convolution of a charge density rho and multiple Green functions. 
    
    Parameters
    ----------
    
    rho : np.array (3D)
        Charge mesh
        
    *greens : np.arrays (3D)
        Charge meshes for the Green functions, which should be twice the size of rho    
        
        
    Returns
    -------
    
    fields : tuple of np.arrays with the same shape as rho. 
    
    """

    # FFT Configuration
    fft  = lambda x: sp_fft.fftn(x,  overwrite_x=True)
    ifft = lambda x: sp_fft.ifftn(x, overwrite_x=True)    
    
    # Place rho in double-sized array. Should match the shape of green
    nx, ny, nz = rho.shape
    crho = np.zeros( (2*nx, 2*ny, 2*nz))
    crho[0:nx,0:ny,0:nz] = rho[0:nx,0:ny,0:nz]
    
    # FFT
    crho = fft(crho)    
   
    results = []
    for green in greens:
        assert crho.shape == green.shape, f'Green array shape {green.shape} should be twice rho shape {rho.shape}'
        result = ifft(crho*fft(green))
        # Extract the result
        result = np.real(result[nx-1:2*nx-1,ny-1:2*ny-1,nz-1:2*nz-1])
        results.append(result)
    
    return tuple(results)