import numpy as np
from numba import njit
from .linalg import matrix_matrix

# z and y are 1-dim arrays of the same length 
# z represents the "x" variable (independent variable). In this case, the matsubara frequency.
# y represents the dependent variable. In this case, the Matsubara green function for an specific frequency.
# npoles is the number of poles of what?


@njit
def pade_expansion_ls(npoles, z, y, tol=1e-8):
    assert z.ndim==1 and y.ndim==1
    assert z.size == y.size # numpy.size() function count the number of elements along a given axis.

    pade_coefs = np.ones((2*npoles,), dtype=np.complex128)

    Ndata = z.size
    Nparam = pade_coefs.size # N param= 2*npoles
    
    conv = 1e5
    while conv>tol: # Condition must be take into account convergence
        jacobian = np.zeros((Ndata,Nparam), dtype=np.complex128) # Jacobian is Ndata x Nparam matrix.
        P = np.zeros((Ndata), dtype=np.complex128)
        Q = np.ones((Ndata), dtype=np.complex128)
        for i in range(Ndata):
            for l in range(npoles):
                Q[i] += pade_coefs[npoles+l] * z[i]**(l+1) #  stores the value of Q(x) for each data value. 
                P[i] += pade_coefs[l] * z[i]**l  #  stores the value of P(x) for each data value. 
                # It can be seen that the degree of Q(x) is one unit greater than the degree of P(x).
            for j in range(npoles): # equations (154) 
                jacobian[i,j] = z[i]**j / Q[i]
                jacobian[i,j+npoles] = -z[i]**(j+1) * P[i] / Q[i]**2
        
        delta_coefs = np.zeros_like(pade_coefs) #delta beta
        #matrix_matrix(a, b) returns a matrix c= a.b ( product of matrices)
        #jacobian.T is the transpose of jacobian.
        meta_jacobian = matrix_matrix(np.linalg.inv(matrix_matrix(jacobian.T, jacobian)), jacobian.T) # eq. (152)
        for s in range(Nparam):
            for r in range(Ndata):
                delta_coefs[s] += meta_jacobian[s,r] * (y[r] - P[r]/Q[r]) # eq.(152)
        
        pade_coefs += delta_coefs # beta (k+1)=beta (k) + delta beta

        # Check for convergence
        convsq = 0
        #for k in range(Ndata):
        for k in range(Nparam):
            convsq += delta_coefs[k].real * delta_coefs[k].real + delta_coefs[k].imag * delta_coefs[k].imag
        conv = np.sqrt(convsq)
    # pade_coefs[:npoles] returns coefficients until the position npoles-1 (does not consider the one in npoles position)-->a
    #pade_coefs[npoles:] returns coeff from npoles position to the final element--> b
    return pade_coefs[:npoles], np.append(np.array([1.+0.j]),pade_coefs[npoles:])
    
    #a=[a0,a1,a2,..,a_(npoles-1)] --> a= pade_expansion_ls(npoles, z, y, tol=1e-8)[0]
    #b=[1,b1,b2,b3,...,b_npoles] --> b= pade_expansion_ls(npoles, z, y, tol=1e-8)[1]
    # a.size = b.size + 1


@njit
#a is the list of coefficients for P(x)
#b is the list of coefficientes for Q(x)
def pade_continuation(x, a, b):
    Ndata = x.size
    P = np.zeros((Ndata,), dtype=np.complex128)
    Q = np.zeros((Ndata,), dtype=np.complex128)
    
    for r in range(a.size):
        P += a[r] * x**r
    for s in range(b.size):
        Q += b[s] * x**s
    
    return P/Q
