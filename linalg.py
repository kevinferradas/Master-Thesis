import numpy as np
from numba import njit
from .printing import exp_string



@njit
def matrix_vector(M, v):
    assert M.shape[1] == v.shape[0], "Shape does not coincide"
    
    c = np.zeros_like(v)
    
    for ii in range(M.shape[0]):
        for jj in range(v.shape[0]):
            c[jj] += M[ii,jj] * v[jj]
    
    return c


@njit
def matrix_matrix(a, b):
    assert a.shape[1] == b.shape[0], "Shape does not coincide"
    
    c = np.zeros((a.shape[0],b.shape[1]), dtype=np.complex128)
    
    for ii in range(a.shape[0]):
        for jj in range(a.shape[1]):
            for kk in range(b.shape[1]):
                c[ii,kk] += a[ii,jj] * b[jj,kk]
    
    return c


@njit
def matrix_matrix2(a, b):
    c = np.zeros((a.shape[0],b.shape[0],b.shape[1],a.shape[1]), dtype=np.complex128)
    
    for mm in range(a.shape[0]):
        for nn in range(b.shape[0]):
            for kk in range(b.shape[1]):
                for ll in range(a.shape[1]):
                    c[mm,nn,kk,ll] += a[mm,ll] * b[nn,kk]
    
    return c


@njit
def matrix_tensor(a, b):
    assert a.shape[0] == b.shape[2], "Shape does not coincide"
    assert a.shape[1] == b.shape[1], "Shape does not coincide"
    
    c = np.zeros((b.shape[0], b.shape[3]), dtype=np.complex128)
    
    for mm in range(b.shape[0]):
        for nn in range(b.shape[3]):
            for ll in range(a.shape[0]):
                for kk in range(a.shape[1]):
                    c[mm,nn] += a[ll,kk] * b[mm,kk,ll,nn]
    
    return c


@njit
def tensor_tensor(a, b):
    assert a.shape[1] == b.shape[2], "Shape does not coincide"
    assert a.shape[3] == b.shape[0], "Shape does not coincide"
    
    c = np.zeros((a.shape[0],b.shape[1],a.shape[2],b.shape[3]), dtype=np.complex128)
    
    for mm in range(a.shape[0]):
        for aa in range(a.shape[1]):
            for ll in range(a.shape[2]):
                for ee in range(a.shape[3]):
                    for xx in range(b.shape[1]):
                        for bb in range(b.shape[3]):
                            c[mm,xx,ll,bb] += a[mm,aa,ll,ee] * b[ee,xx,aa,bb]
    
    return c


@njit
def norm2_matrix(arr):
    assert arr.ndim==2, "Norm 2 for matrices"
    
    arrsq = np.zeros((arr.shape[0],arr.shape[0]), dtype=np.complex128)
    for ii in range(arr.shape[0]):
        for jj in range(arr.shape[0]):
            for kk in range(arr.shape[1]):
                arrsq[ii,jj] += arr[ii,kk] * arr[jj,kk].conjugate()
    
    if np.all(arrsq==np.zeros_like(arrsq)):
        return 0.0
    else:
        w,_ = np.linalg.eig(arrsq)
        return np.sqrt(w.real.max())
