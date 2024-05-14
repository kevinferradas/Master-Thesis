import numpy as np
from numba import njit, int64, float64
from .linalg import norm2_matrix



@njit
def xor_particle(a, b):
    assert a==1 or a==0
    assert b==1 or b==0
    
    if a!=b:
        return 1
    else:
        return 0


@njit
def gdist_mat(A, B):
    ntau = A.shape[0]
    assert B.shape[0]==ntau
    
    d = 0
    for tt in range(ntau):
        d += norm2_matrix(A[tt] - B[tt])
    
    return d


@njit
def gdist_real_time(t, A, B):
    ntau = A.get_mat().shape[0]
    d = 0
    for jj in range(t+1):
        d += norm2_matrix(A.get_ret()[t,jj] - B.get_ret()[t,jj])
        d += norm2_matrix(A.get_les()[t,jj] - B.get_les()[t,jj])
    for kk in range(ntau):
        d += norm2_matrix(A.get_lmx()[t,kk] - B.get_lmx()[t,kk])
    return d


@njit
def factorial(n: int64) -> int64:
    assert n > -1, "Factorial is not defined for negative numbers"
    
    if n == 0 or n == 1:
        return 1
    
    else:
        f = 1
        for k in range(2,n+1):
            f *= k
        return f


@njit
def binomial(m: int64, n: int64) -> float64:
    assert m >= n, "Binomial coefficient is not defined for m < n"
    
    return factorial(m) / factorial(m-n) / factorial(n)
