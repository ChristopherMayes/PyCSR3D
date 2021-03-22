from numba import vectorize, float64, njit
# For special functions
from numba.extending import get_cython_function_address
import ctypes

# Include special functions for Numba
#
# Tip from: https://github.com/numba/numba/issues/3086
# and http://numba.pydata.org/numba-doc/latest/extending/high-level.html
#
addr1 = get_cython_function_address('scipy.special.cython_special', 'ellipkinc')
addr2 = get_cython_function_address('scipy.special.cython_special', 'ellipeinc')
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)

ellipkinc = functype(addr1)
ellipeinc = functype(addr2)

