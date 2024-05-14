import numpy as np
from numba import njit
from .linalg import matrix_matrix



@njit
def pade_expansion_ls(npoles, z, y, tol=1e-8):
    assert z.ndim==1 and y.ndim==1
    assert z.size == y.size

    pade_coefs = np.ones((2*npoles,), dtype=np.complex128)

    Ndata = z.size
    Nparam = pade_coefs.size
    
    conv = 1e5
    while conv>tol: # Condition must be take into account convergence
        jacobian = np.zeros((Ndata,Nparam), dtype=np.complex128)
        P = np.zeros((Ndata), dtype=np.complex128)
        Q = np.ones((Ndata), dtype=np.complex128)
        for i in range(Ndata):
            for l in range(npoles):
                Q[i] += pade_coefs[npoles+l] * z[i]**(l+1)
                P[i] += pade_coefs[l] * z[i]**l
            for j in range(npoles):
                jacobian[i,j] = z[i]**j / Q[i]
                jacobian[i,j+npoles] = -z[i]**(j+1) * P[i] / Q[i]**2
        
        delta_coefs = np.zeros_like(pade_coefs)
        meta_jacobian = matrix_matrix(np.linalg.inv(matrix_matrix(jacobian.T, jacobian)), jacobian.T)
        for s in range(Nparam):
            for r in range(Ndata):
                delta_coefs[s] += meta_jacobian[s,r] * (y[r] - P[r]/Q[r])
        
        pade_coefs += delta_coefs

        # Check for convergence
        convsq = 0
        for k in range(Ndata):
            convsq += delta_coefs[k].real * delta_coefs[k].real + delta_coefs[k].imag * delta_coefs[k].imag
        conv = np.sqrt(convsq)
    
    return pade_coefs[:npoles], np.append(np.array([1.+0.j]),pade_coefs[npoles:])


@njit
def pade_continuation(x, a, b):
    Ndata = x.size
    P = np.zeros((Ndata,Ndata), dtype=np.complex128)
    Q = np.zeros((Ndata,Ndata), dtype=np.complex128)
    
    for r in range(a.size):
        P += a[r] * x**r
    for s in range(b.size):
        Q += b[s] * x**s
    
    return P/Q
