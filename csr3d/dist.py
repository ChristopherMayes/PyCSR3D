import numpy as np

def gauss(x, sigma=1, mu=0): 
    return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2) )

                                                   
def lambda_gauss3(x, y, z, sigma_x=10e-6, sigma_y=10e-6, sigma_z=10e-6): 
    """
    3D Gaussian G(z,x)
    """
    return gauss(x, sigma_x)* gauss(y, sigma_y)* gauss(z, sigma_z)

def lambda_gauss3_prime(x, y, z, sigma_x=10e-6, sigma_y=10e-6, sigma_z=10e-6): 
    """
    The z derivative of a 2D Gaussian G(z,x)
    """
    return lambda_gauss3(x, y, z, sigma_x=sigma_x, sigma_y=sigma_y, sigma_z=sigma_z)*(-z/sigma_z**2)