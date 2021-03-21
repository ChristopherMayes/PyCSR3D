import numpy as np
import scipy.special as ss

from scipy.optimize import root_scalar

from numpy import abs, sin, cos, real, exp, pi, cbrt, sqrt, sign





#----------------------------------------------
# psi_s



def psi_s(x, y, z, beta):
    """
    
    Eq. 23 from Ref[X] without the prefactor e beta^2 / (2 rho^2)
    
    """

    beta2 = beta**2
    
    
    alp = alpha(x, y, z, beta2)
    
    kap = 2*(alp - z)/beta # Simpler form of kappa
    
    sin2a = sin(2*alp)
    cos2a = cos(2*alp)    
    
    out = (cos2a - 1/(1+x)) / (kap - beta*(1+x)*sin2a)
    
    return out
    










#----------------------------------------------
# Alpha calc

def alpha_where_z_not_zero(x, y, z, beta2):
    """
    
    Eq. (6) from Ref[X] using the solution in 
    Eq. (A4) from Ref[1] 
    
    
    """

    # Terms of the depressed quartic equation
    eta = -6 * z / (beta2 * (1+x))
    nu = 3 * (1/beta2 - 1 - x) / (1+x)
    zeta = (3/4) * (4* z**2 /beta2 - x**2 - y**2) / (1+x)
    
    # Omega calc and cube root
    temp = (eta**2/16 - zeta * nu/6 + nu**3/216)  
    Omega =  temp + sqrt(temp**2 - (zeta/3 + nu**2/36)**3)  
    omega3 = cbrt(Omega) 
    
    # Eq. (A2) from Ref[1]
    m = -nu/3 + (zeta/3 + nu**2/36) /omega3 + omega3
     
    arg1 = np.sqrt(2 * abs(m))
    arg2 = -2 * (m + nu)
    arg3 = 2 * eta / arg1
    
    zsign= sign(z)
    
    return (zsign*arg1 + sqrt(abs(arg2 -zsign*arg3)))/2


def alpha_where_z_equals_zero(x, y, beta2):
    """
    Evaluate alpha(x, y, z) when z is zero.
    Eq. (6) from Ref[1] simplifies to a quadratic equation for alpha^2.
    """
    #b = 3 * (1/beta2 - 1 - x) / (1+x)
    b = 3 * (1 - beta2 - beta2*x) / beta2 / (1+x)    
    c = -3*(x**2 + y**2)/(4*(1+x))
    
    root1 = (-b + sqrt(b**2 - 4*c))/2
    # root2 = (-b - np.sqrt(b**2 - 4*c))/2   
    # since b>0, root2 is always negative and discarded
    
    return sqrt(root1)


def alpha(x, y, z, beta2):
    """
    Retarded angle alpha
    
    Eq. (6) from Ref[X] using the solution in 
    Eq. (A4) from Ref[1] 

    
    """
    
    
    
    on_x_axis = z == 0
    # Check for scalar, then return the normal functions
    if not isinstance(z, np.ndarray):
        if on_x_axis:
            return alpha_where_z_equals_zero(x, y, beta2)
        else:
            return alpha_where_z_not_zero(x, y, z, beta2)
    # Array z
    out = np.empty(z.shape)
    ix1 = np.where(on_x_axis)
    ix2 = np.where(~on_x_axis)
    
    if len(ix1)==0:
        print('ix1:', ix1)
        print(z)
    # Check for arrays
    if isinstance(x, np.ndarray):
        x1 = x[ix1]
        x2 = x[ix2]
    else:
        x1 = x
        x2 = x
    if isinstance(y, np.ndarray):
        y1 = y[ix1]
        y2 = y[ix2]
    else:
        y1 = y
        y2 = y        
        
    out[ix1] = alpha_where_z_equals_zero(x1, y1, beta2)
    out[ix2] = alpha_where_z_not_zero(x2, y2, z[ix2], beta2)
    return out


@np.vectorize
def alpha_exact(x, y, z, beta2):
    """
    Exact alpha calculation using numerical root finding.

    
    Eq. (5) from Ref[X]
    """
    beta = sqrt(beta2)

    f = lambda a: a - beta/2*np.sqrt(x**2 + y**2 + 4*(1+x)*np.sin(a)**2 ) - z
    
    res = root_scalar(f, bracket=(-1,1))
    
    return res.root