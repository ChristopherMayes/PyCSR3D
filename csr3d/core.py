
from numba import vectorize, float64

import numpy as np
import scipy.special as ss

import csr3d.special as my_ss



from scipy.optimize import root_scalar

from numpy import abs, sin, cos, real, exp, pi, cbrt, sqrt, sign
import scipy.special as ss




#----------------------------------------------
# psi_s



def psi_calc(x, y, z, gamma, components=['x', 'y', 's']):
    """
    Eq. 24 from Ref[X] without the prefactor e beta^2 / (2 rho^2)
    """
    
    potential = {}
    
    beta2 = 1-1/gamma**2
    beta = sqrt(beta2)
    
    alp = old_alpha(x, y, z, gamma)
    kap = 2*(alp - z)/beta # Simpler form of kappa
    #kap = sqrt(x**2 + y**2 + 4*(1+x) * sin(alp)**2) 

    # Common patterns
    sin2a = sin(2*alp)
    cos2a = cos(2*alp)

    if 's' in components:        
        potential['psi_s'] = (cos2a - 1/(1+x)) / (kap - beta*(1+x)*sin2a)
        if len(components) == 1:
            return potential

    kap2 = kap**2
    sin2a2 = sin2a**2
    
    x2 = x**2 
    y2 = y**2
    y4 = y2**2
    xp = x + 1
    xp2 = xp**2
    xy2 = x2 + y2
    xy = np.sqrt(xy2)
    
    # More complicated pattens
    f1 = 2 + 2*x +x2
    f2 = (2+x)**2
    arg2 = -4 * xp / xy2 
    
    F = ss.ellipkinc(alp, arg2) # Incomplete elliptic integral of the first kind K(phi, m), also called F(phi, m)
    E = ss.ellipeinc(alp, arg2)# Incomplete elliptic integral of the second kind E(phi, m)
    
    #print((2/beta2)* F/xy)
    
    # psi_x (actually psi_x_hat that includes the psi_phi term)
    # There is an extra ] in the numerator of the second term. All terms should multiply E. 
    if 'x' in components:
        psi_x_out = f1*F / (xp*xy) - (x2*f2 + y2*f1)*E / (xp*(y2+f2)*xy)  \
            + ( kap2 - 2*beta2*xp2 + beta2*xp*f1*cos2a  ) / (beta *xp*(kap2 - beta2*xp2*sin2a2)) \
            + kap*( y4 - x2*f2 - 2*beta2*y2*xp2 )*sin2a / ( xy2*(y2 + f2)*(kap2-beta2*xp2*sin2a2)  ) \
            + kap*beta2*xp*( x2*f2 + y2*f1 )*sin2a*cos2a / ( xy2*(y2+f2)*(kap2-beta2*xp2*sin2a2)  )
          
        potential['psi_x'] = psi_x_out - (2/beta2)* F/xy # Include the phi term
        
    
    # psi_phi
    if 'phi' in components:
        potential['psi_phi'] = (2/beta2)* F/xy  # Prefactor to be consistent with other returns
        
    
    # psi_y
    if 'y' in components:    
        psi_y_out = y * ( \
                    F/xy - (x*(2+x)+y2)*E / ((y2+f2)*xy) \
                    - beta*(1-xp*cos2a) / (kap2-beta2*xp2*sin2a2) \
                    + kap*xp*( -(2+beta2)*y2 + (-2+beta2)*x*(2+x) ) * sin2a / ( (y4 + x2*f2 + 2*y2*f1)*( kap2-beta2*xp2*sin2a2 ) ) \
                    + kap*beta2*xp2*(y2 + x*(2+x))*sin2a*cos2a / ( ( y4 + x2*f2 + 2*y2*f1)*(kap2 -beta2*xp2*sin2a2)  ) \
                    )
        potential['psi_y'] = psi_y_out
    
    
    return potential
    
# Conveniences
def old_psi_x(x, y, z, gamma):
    return psi_calc(x, y, z, gamma, components=['x'])['psi_x']
    
def old_psi_y(x, y, z, gamma):
    return psi_calc(x, y, z, gamma, components=['y'])['psi_y']

def old_psi_s(x, y, z, gamma):
    return psi_calc(x, y, z, gamma, components=['s'])['psi_s']

def old_psi_phi(x, y, z, gamma):
    return psi_calc(x, y, z, gamma, components=['phi'])['psi_phi']







#----------------------------------------------
# Alpha calc

def alpha_where_z_not_zero(x, y, z, gamma):
    """
    
    Eq. (6) from Ref[X] using the solution in 
    Eq. (A4) from Ref[1] 
    
    
    """
    
    beta2 = 1-1/gamma**2

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


def alpha_where_z_equals_zero(x, y, gamma):
    """
    Evaluate alpha(x, y, z) when z is zero.
    Eq. (6) from Ref[1] simplifies to a quadratic equation for alpha^2.
    """
    beta2 = 1-1/gamma**2
    
    #b = 3 * (1/beta2 - 1 - x) / (1+x)
    b = 3 * (1 - beta2 - beta2*x) / beta2 / (1+x)    
    c = -3*(x**2 + y**2)/(4*(1+x))
    
    root1 = (-b + sqrt(b**2 - 4*c))/2
    # root2 = (-b - np.sqrt(b**2 - 4*c))/2   
    # since b>0, root2 is always negative and discarded
    
    return sqrt(root1)


def old_alpha(x, y, z, gamma):
    """
    Retarded angle alpha
    
    Eq. (6) from Ref[X] using the solution in 
    Eq. (A4) from Ref[1] 

    
    """
    
    beta2 = 1-1/gamma**2
    
    on_x_axis = z == 0
    # Check for scalar, then return the normal functions
    if not isinstance(z, np.ndarray):
        if on_x_axis:
            return alpha_where_z_equals_zero(x, y, gamma)
        else:
            return alpha_where_z_not_zero(x, y, z, gamma)
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
        
    out[ix1] = alpha_where_z_equals_zero(x1, y1, gamma)
    out[ix2] = alpha_where_z_not_zero(x2, y2, z[ix2], gamma)
    return out


@np.vectorize
def alpha_exact(x, y, z, gamma):
    """
    Exact alpha calculation using numerical root finding.

    
    Eq. (5) from Ref[X]
    """
    beta = sqrt(1-1/gamma**2)
    

    f = lambda a: a - beta/2*np.sqrt(x**2 + y**2 + 4*(1+x)*np.sin(a)**2 ) - z
    
    res = root_scalar(f, bracket=(-1,1))
    
    return res.root




#----------------------------------------------
# Numba


@vectorize([float64(float64, float64, float64, float64)])
def alpha(x, y, z, gamma):
    """
    Numba vectorized form of alpha.
    See: https://numba.pydata.org/numba-doc/dev/user/vectorize.html
    
    
    Eq. (6) from Ref[X] using the solution in 
    Eq. (A4) from Ref[1] 
    
    
    """
    
    beta2 = 1-1/gamma**2
    
    if z == 0:
        # Quadratic solution
        
        b = 3 * (1 - beta2 - beta2*x) / beta2 / (1+x)    
        c = -3*(x**2 + y**2)/(4*(1+x))
    
        root1 = (-b + np.sqrt(b**2 - 4*c))/2
        
        return np.sqrt(root1)
        
    # Quartic solution 
        
    # Terms of the depressed quartic equation
    eta = -6 * z / (beta2 * (1+x))
    nu = 3 * (1/beta2 - 1 - x) / (1+x)
    zeta = (3/4) * (4* z**2 /beta2 - x**2 - y**2) / (1+x)
    
    # Omega calc and cube root
    temp = (eta**2/16 - zeta * nu/6 + nu**3/216)  
    Omega =  temp + np.sqrt(temp**2 - (zeta/3 + nu**2/36)**3)  
    #omega3 = np.cbrt(Omega) # Not supported in Numba! See: https://github.com/numba/numba/issues/5385
    omega3= Omega**(1/3)
    
    # Eq. (A2) from Ref[1]
    m = -nu/3 + (zeta/3 + nu**2/36) /omega3 + omega3
     
    arg1 = np.sqrt(2 * abs(m))
    arg2 = -2 * (m + nu)
    arg3 = 2 * eta / arg1
    
    zsign= np.sign(z)
    
    return (zsign*arg1 + np.sqrt(abs(arg2 -zsign*arg3)))/2


@vectorize([float64(float64, float64, float64, float64)], target='parallel')
def psi_s(x, y, z, gamma):
    """
    
    Numba, parallel
    
    Eq. 24 from Ref[X] without the prefactor e beta^2 / (2 rho^2)
    """
    
    # Check for the origin
    if x == 0 and y == 0 and z == 0:
        return 0
    
    beta2 = 1-1/gamma**2
    beta = sqrt(beta2)
    
    alp = alpha(x, y, z, gamma)
    kap = 2*(alp - z)/beta # Simpler form of kappa
    #kap = sqrt(x**2 + y**2 + 4*(1+x) * sin(alp)**2) 

    out = (cos(2*alp) - 1/(1+x)) / (kap - beta*(1+x)*sin(2*alp) )

    return out


@vectorize([float64(float64, float64, float64, float64)])
def psi_xhat(x, y, z, gamma):
    """
    Psi_x - Psi_phi from Eq. 24 from Ref[X] without the prefactor e beta^2 / (2 rho^2)
    """
        
    beta2 = 1-1/gamma**2
    beta = sqrt(beta2)
    
    alp = alpha(x, y, z, gamma)
    kap = 2*(alp - z)/beta # Simpler form of kappa
    #kap = sqrt(x**2 + y**2 + 4*(1+x) * sin(alp)**2) 

    # Common patterns
    sin2a = sin(2*alp)
    cos2a = cos(2*alp)

    kap2 = kap**2
    sin2a2 = sin2a**2
    
    x2 = x**2 
    y2 = y**2
    y4 = y2**2
    xp = x + 1
    xp2 = xp**2
    xy2 = x2 + y2
    xy = np.sqrt(xy2)
    
    # More complicated pattens
    f1 = 2 + 2*x +x2
    f2 = (2+x)**2
    arg2 = -4 * xp / xy2 
    
    # Use my numba wrapped routines
    F = my_ss.ellipkinc(alp, arg2) # Incomplete elliptic integral of the first kind K(phi, m), also called F(phi, m)
    E = my_ss.ellipeinc(alp, arg2)# Incomplete elliptic integral of the second kind E(phi, m)
        
    # psi_x (actually psi_x_hat that includes the psi_phi term)
    # There is an extra ] in the numerator of the second term. All terms should multiply E. 

    psi_x_out = f1*F / (xp*xy) - (x2*f2 + y2*f1)*E / (xp*(y2+f2)*xy)  \
            + ( kap2 - 2*beta2*xp2 + beta2*xp*f1*cos2a  ) / (beta *xp*(kap2 - beta2*xp2*sin2a2)) \
            + kap*( y4 - x2*f2 - 2*beta2*y2*xp2 )*sin2a / ( xy2*(y2 + f2)*(kap2-beta2*xp2*sin2a2)  ) \
            + kap*beta2*xp*( x2*f2 + y2*f1 )*sin2a*cos2a / ( xy2*(y2+f2)*(kap2-beta2*xp2*sin2a2)  )
          
    psi_phi_out = (2/beta2)* F/xy # Include the phi term        
        
    psi_xhat_out = psi_x_out  - psi_phi_out
    
    return psi_xhat_out

@vectorize([float64(float64, float64, float64, float64)])
def psi_x(x, y, z, gamma):
    """
    Psi_x from Eq. 24 from Ref[X] without the prefactor e beta^2 / (2 rho^2)
    """
        
    beta2 = 1-1/gamma**2
    beta = sqrt(beta2)
    
    alp = alpha(x, y, z, gamma)
    kap = 2*(alp - z)/beta # Simpler form of kappa
    #kap = sqrt(x**2 + y**2 + 4*(1+x) * sin(alp)**2) 

    # Common patterns
    sin2a = sin(2*alp)
    cos2a = cos(2*alp)

    kap2 = kap**2
    sin2a2 = sin2a**2
    
    x2 = x**2 
    y2 = y**2
    y4 = y2**2
    xp = x + 1
    xp2 = xp**2
    xy2 = x2 + y2
    xy = np.sqrt(xy2)
    
    # More complicated pattens
    f1 = 2 + 2*x +x2
    f2 = (2+x)**2
    arg2 = -4 * xp / xy2 
    
    # Use my numba wrapped routines
    F = my_ss.ellipkinc(alp, arg2) # Incomplete elliptic integral of the first kind K(phi, m), also called F(phi, m)
    E = my_ss.ellipeinc(alp, arg2)# Incomplete elliptic integral of the second kind E(phi, m)
        
    # psi_x (see psi_xhat that includes the psi_phi term)
    # There is an extra ] in the numerator of the second term. All terms should multiply E. 

    psi_x_out = f1*F / (xp*xy) - (x2*f2 + y2*f1)*E / (xp*(y2+f2)*xy)  \
            + ( kap2 - 2*beta2*xp2 + beta2*xp*f1*cos2a  ) / (beta *xp*(kap2 - beta2*xp2*sin2a2)) \
            + kap*( y4 - x2*f2 - 2*beta2*y2*xp2 )*sin2a / ( xy2*(y2 + f2)*(kap2-beta2*xp2*sin2a2)  ) \
            + kap*beta2*xp*( x2*f2 + y2*f1 )*sin2a*cos2a / ( xy2*(y2+f2)*(kap2-beta2*xp2*sin2a2)  )
          
    
    # Space Charge term
    #psi_x_sc = F/(xp*xy) \
    #        + (x*(2+x)-y2)*E/(xp*(y2+f2)*xy) \
    #        + beta*(cos2a-xp)/(kap2 - beta2*xp2*sin2a2) \
    #        - kap*sin2a * ( (x*(2+x)*(beta2*xp2-2) + y2*(2+beta2*xp2) ) + beta2*xp*(x*(2+x)-y2)*cos2a ) \
    #            / ( (y4 + x2*f2 + 2*y2*f1) * (kap2 - beta2*xp2*sin2a2 ) )
    #
    #psi_x_sc /= (gamma**2-1) # prefactor 1/(gamma*beta)^2 = 1/(gamma^2-1)to agree with the prefactor of psi_x_out
    #psi_x_out += psi_x_sc
        
    return psi_x_out



@vectorize([float64(float64, float64, float64, float64)])
def psi_phi(x, y, z, gamma):
    """

    Psi_phi from Eq. 24 from Ref[X] without the prefactor e beta^2 / (2 rho^2)
    """
        
    beta2 = 1-1/gamma**2
    beta = sqrt(beta2)
    
    alp = alpha(x, y, z, gamma)
    xy2 = x**2 + y**2 
    xy = np.sqrt(xy2)
    
    arg2 = -4 * (x + 1) / xy2
    F = my_ss.ellipkinc(alp, arg2) 

    psi_phi_out = (2/beta2)* F/xy 

    return psi_phi_out

@vectorize([float64(float64, float64, float64, float64, float64, float64, float64)], target='parallel')
def psi_phi0(x, y, z, gamma, dx, dy, dz):
    """
    Same as psi_phi, but checks for evaluation on the z and x axes.
    
    Numba, parallel
    
    There are singularities along the z and x axes.
    This attempts to average them out. 
    
    """
    

    # There are singularities along these axes. 
    # 
    if x == 0 and y == 0:
        # Along z axis
        #print('Along z axis')
        res = (psi_phi(-dx/2, y, z, gamma) +  psi_phi(dx/2, y, z, gamma))/2 # Average over x (same as CSR2D)
        
    elif y == 0 and z == 0:
        # Along x axis
        #print('Along x axis')
        res = (psi_phi(x, -dy/2, z, gamma) +  psi_phi(x, dy/2, z, gamma))/2 # Average over y
  
    else:
        res =  psi_phi(x, y, z, gamma)

    return res



@vectorize([float64(float64, float64, float64, float64, float64, float64, float64)], target='parallel')
def psi_x0(x, y, z, gamma, dx, dy, dz):
    """
    Same as psi_x, but checks for evaluation on the z and x axes.
    
    Numba, parallel
    
    There are singularities along the z and x axes.
    This attempts to average them out. 
    
    """
    

    # There are singularities along these axes. 
    # 
    if x == 0 and y == 0:
        # Along z axis
        #print('Along z axis')
        res = (psi_x(-dx/2, y, z, gamma) +  psi_x(dx/2, y, z, gamma))/2 # Average over x (same as CSR2D)
        
    elif y == 0 and z == 0:
        # Along x axis
        #print('Along x axis')
        res = (psi_x(x, -dy/2, z, gamma) +  psi_x(x, dy/2, z, gamma))/2 # Average over y
  
    else:
        res =  psi_x(x, y, z, gamma)

    return res

@vectorize([float64(float64, float64, float64, float64, float64, float64, float64)], target='parallel')
def psi_xhat0(x, y, z, gamma, dx, dy, dz):
    """
    Same as psi_xhat, but checks for evaluation on the z and x axes.
    
    Numba, parallel
    
    There are singularities along the z and x axes.
    This attempts to average them out. 
    
    """
    

    # There are singularities along these axes. 
    # 
    if x == 0 and y == 0:
        # Along z axis
        #print('Along z axis')
        res = (psi_xhat(-dx/2, y, z, gamma) +  psi_xhat(dx/2, y, z, gamma))/2 # Average over x (same as CSR2D)
        
    elif y == 0 and z == 0:
        # Along x axis
        #print('Along x axis')
        res = (psi_xhat(x, -dy/2, z, gamma) +  psi_xhat(x, dy/2, z, gamma))/2 # Average over y
  
    else:
        res =  psi_xhat(x, y, z, gamma)

    return res




@vectorize([float64(float64, float64, float64, float64)])
def psi_y(x, y, z, gamma):
    """
    Eq. 25 from Ref[X] without the prefactor e beta^2 / (2 rho^2)
    """
        
    beta2 = 1-1/gamma**2
    beta = sqrt(beta2)
    
    alp = alpha(x, y, z, gamma)
    kap = 2*(alp - z)/beta # Simpler form of kappa
    #kap = sqrt(x**2 + y**2 + 4*(1+x) * sin(alp)**2) 

    # Common patterns
    sin2a = sin(2*alp)
    cos2a = cos(2*alp)

    kap2 = kap**2
    sin2a2 = sin2a**2
    
    x2 = x**2 
    y2 = y**2
    y4 = y2**2
    xp = x + 1
    xp2 = xp**2
    xy2 = x2 + y2
    xy = np.sqrt(xy2)
    
    # More complicated pattens
    f1 = 2 + 2*x +x2
    f2 = (2+x)**2
    arg2 = -4 * xp / xy2 
    
    # Use my numba wrapped routines
    F = my_ss.ellipkinc(alp, arg2) # Incomplete elliptic integral of the first kind K(phi, m), also called F(phi, m)
    E = my_ss.ellipeinc(alp, arg2)# Incomplete elliptic integral of the second kind E(phi, m)
        
    psi_y_out = y * ( \
                    F/xy - (x*(2+x)+y2)*E / ((y2+f2)*xy) \
                    - beta*(1-xp*cos2a) / (kap2-beta2*xp2*sin2a2) \
                    + kap*xp*( -(2+beta2)*y2 + (-2+beta2)*x*(2+x) ) * sin2a / ( (y4 + x2*f2 + 2*y2*f1)*( kap2-beta2*xp2*sin2a2 ) ) \
                    + kap*beta2*xp2*(y2 + x*(2+x))*sin2a*cos2a / ( ( y4 + x2*f2 + 2*y2*f1)*(kap2 -beta2*xp2*sin2a2)  ) \
                    )
    
    return psi_y_out


@vectorize([float64(float64, float64, float64, float64, float64, float64, float64)], target='parallel')
def psi_y0(x, y, z, gamma, dx, dy, dz):
    """
    Same as psi_y, but checks for evaluation on the z and x axes.
    
    Numba, parallel
    
    There are singularities along the z and x axes.
    This attempts to average them out. 
    
    """
    

    # There are singularities along these axes. 
    # 
    if x == 0 and y == 0:
        # Along z axis
        #print('Along z axis')
        res = (psi_y(-dx/2, y, z, gamma) +  psi_y(dx/2, y, z, gamma))/2 # Average over x (same as CSR2D)
        #res = (psi_y(x, -dy/2, z, gamma) +  psi_y(x, dy/2, z, gamma))/2 # Average over y
        # Average over 4 points:
        #res = (psi_y(-dx/2, -dy/2, z, gamma) +  psi_y(dx/2, -dy/2, z, gamma) \
        #      +psi_y(-dx/2,  dy/2, z, gamma) +  psi_y(dx/2,  dy/2, z, gamma))/4        
        
        #res = 0
        
    elif y == 0 and z == 0:
        # Along x axis
        #print('Along x axis')
        res = (psi_y(x, -dy/2, z, gamma) +  psi_y(x, dy/2, z, gamma))/2 # Average over y
        #res = (psi_y(x, y, -dz/2, gamma) +  psi_y(x, y, dz/2, gamma))/2 # Average over z    
  
    else:
        res =  psi_y(x, y, z, gamma)

    return res

