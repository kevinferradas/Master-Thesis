import numpy as np
from numba import njit
from .utils import xor_particle



@njit
def conv_mat1(tt, a, b, interpol, h, mat_prod): # Integrates CM from 0 to tt
    assert a.get_mat().shape[0] == b.get_mat().shape[0], "Arrays must be same imaginary time steps"
    assert a.particle_type == b.particle_type
    
    particle = a.particle_type
    N = a.get_mat().shape[0]-1
    w = interpol.gregory_weights(N)

    c = np.zeros_like(a.get_mat()[0])

    if tt <= interpol.k:
        for jj in range(interpol.k+1):
            for ll in range(interpol.k+1):
                c += interpol.R[tt,jj,ll] * mat_prod(a.get_mat()[jj], b.get_mat()[ll])
    else:
        for ll in range(tt+1):
            c += w[tt,ll] * mat_prod(a.get_mat()[tt-ll], b.get_mat()[ll])
    
    return h * c


@njit
def conv_mat2(tt, a, b, interpol, h, mat_prod): # Integrates CM from tt to beta
    assert a.get_mat().shape[0] == b.get_mat().shape[0], "Arrays must be same imaginary time steps"
    assert a.particle_type == b.particle_type
    
    particle = a.particle_type
    
    N = a.get_mat().shape[0]-1
    w = interpol.gregory_weights(N)
    
    c = np.zeros_like(a.get_mat()[0])
    if tt >= N-interpol.k:
        for jj in range(interpol.k+1):
            for kk in range(interpol.k+1):
                c += (-1)**particle * interpol.R[N-tt,jj,kk] * mat_prod(a.get_mat()[N-jj], b.get_mat()[N-kk])
    else:
        for kk in range(N-tt+1):
            c += (-1)**particle * w[N-tt,kk] * mat_prod(a.get_mat()[tt+kk], b.get_mat()[N-kk])
    
    return h * c


@njit
def conv_lmx1(tt, ss, a, b, interpol, h, mat_prod):
    assert a.get_ret().shape[1] == b.get_lmx().shape[0], "Arrays must be same time steps"
    assert a.particle_type == b.particle_type

    c = np.zeros_like(a.get_mat()[0])
    if tt > interpol.k:
        w = interpol.gregory_weights(tt)
        for jj in range(tt+1):
            c += w[tt,jj] * mat_prod(a.get_ret()[tt,jj], b.get_lmx()[jj,ss])
    else:
        w = interpol.gregory_weights(interpol.k)
        acomp = a.get_ret_comp()[:interpol.k+1,:interpol.k+1]
        for jj in range(interpol.k+1):
            c += w[tt,jj] * mat_prod(acomp[tt,jj], b.get_lmx()[jj,ss])
    
    return h * c


@njit
def conv_lmx2(tt, ss, a, b, interpol, h, mat_prod):
    assert a.get_lmx().shape[1] == b.get_mat().shape[0], "Arrays must be same imaginary time steps"
    assert a.particle_type == b.particle_type
    
    particle = a.particle_type
    
    N = b.get_mat().shape[0]-1
    w = interpol.gregory_weights(N)
    
    c = np.zeros_like(b.get_mat()[0])
    if ss <= interpol.k:
        for jj in range(interpol.k+1):
            for ll in range(interpol.k+1):
                c += (-1)**particle * interpol.R[ss,jj,ll] * mat_prod(a.get_lmx()[tt,ll], b.get_mat()[N-jj])
    else:
        for ll in range(ss+1):
            c += (-1)**particle * w[ss,ll] * mat_prod(a.get_lmx()[tt,ss-ll], b.get_mat()[N-ll])

    return h * c


@njit
def conv_lmx3(tt, ss, a, b, interpol, h, mat_prod):
    assert a.get_lmx().shape[1] == b.get_mat().shape[0], "Arrays must be same imaginary time steps"
    assert a.particle_type == b.particle_type
    
    particle = a.particle_type
    
    N = b.get_mat().shape[0]-1
    w = interpol.gregory_weights(N)
    
    c = np.zeros_like(b.get_mat()[0])
    
    if ss >= N-interpol.k:
        for jj in range(interpol.k+1):
            for ll in range(interpol.k+1):
                c += interpol.R[N-ss,jj,ll] * mat_prod(a.get_lmx()[tt,N-ll], b.get_mat()[jj])
    else:
        for ll in range(N-ss+1):
            c += w[N-ss,ll] * mat_prod(a.get_lmx()[tt,ss+ll], b.get_mat()[ll])

    return h * c


@njit
def conv_les1(tt, ss, a, b, interpol, h, mat_prod):
    assert a.get_ret().shape[1] == b.get_les().shape[0], "Arrays must be same time steps"
    assert a.particle_type == b.particle_type

    c = np.zeros_like(a.get_mat()[0])
    if tt > interpol.k:
        w = interpol.gregory_weights(tt)
        for jj in range(tt+1):
            c += w[tt,jj] * mat_prod(a.get_ret()[tt,jj], b.get_les()[jj,ss])
    else:
        w = interpol.gregory_weights(interpol.k)
        acomp = a.get_ret_comp()[:interpol.k+1,:interpol.k+1]
        for jj in range(interpol.k+1):
            c += w[tt,jj] * mat_prod(acomp[tt,jj], b.get_les()[jj,ss])
    
    return h * c


@njit
def conv_les2(tt, ss, a, b, interpol, h, mat_prod):
    assert a.get_les().shape[1] == b.get_adv().shape[0], "Arrays must be same time steps"
    assert a.particle_type == b.particle_type

    c = np.zeros_like(a.get_mat()[0])
    if tt > interpol.k:
        w = interpol.gregory_weights(ss)
        for jj in range(ss+1):
            c += w[tt,jj] * mat_prod(a.get_les()[tt,jj], b.get_adv()[jj,ss])
    else:
        w = interpol.gregory_weights(interpol.k)
        bcomp = b.get_adv_comp()[:interpol.k+1,:interpol.k+1]
        for jj in range(interpol.k+1):
            c += w[tt,jj] * mat_prod(a.get_les()[tt,jj], bcomp[jj,ss])
    
    return h * c


@njit
def conv_les3(tt, ss, a, b, interpol, h, mat_prod):
    assert a.get_lmx().shape[1] == b.get_rmx().shape[0], "Arrays must be same imaginary time steps"
    assert a.particle_type == b.particle_type
    
    N = a.get_mat().shape[0] - 1
    w = interpol.gregory_weights(N-1)
    
    c = np.zeros_like(a.get_mat()[0])
    for kk in range(N+1):
        c += w[N,kk] * mat_prod(a.get_lmx()[tt,kk], b.get_rmx()[kk,ss])
    
    return -1j * h * c


@njit
def conv_ret(tt, ss, a, b, interpol, h, mat_prod):
    assert a.get_ret().shape[1] == b.get_ret().shape[0], "Arrays must be same imaginary time steps"
    assert a.particle_type == b.particle_type

    c = np.zeros_like(a.get_mat()[0])
    if tt > interpol.k and tt-ss > interpol.k:
        w = interpol.gregory_weights(tt-ss)
        for jj in range(ss,tt+1):
            c += w[tt-ss,jj-ss] * mat_prod(a.get_ret()[tt,jj], b.get_ret()[jj,ss])
    elif tt > interpol.k and tt-ss <= interpol.k:
        w = interpol.gregory_weights(interpol.k)
        bcomp = b.get_ret_comp()
        for jj in range(interpol.k+1):
            c += w[tt-ss,jj] * mat_prod(a.get_ret()[tt,tt-jj], bcomp[tt-jj,ss])
    else:
        acomp = a.get_ret_comp()
        bcomp = b.get_ret_comp()
        for jj in range(interpol.k+1):
            c += interpol.I[ss,tt,jj] * mat_prod(acomp[tt,jj], bcomp[jj,ss])
    
    return h * c


@njit
def conv_mat_extern(a, b, interpol, h, mat_prod, particle):
    assert a.shape[0] == b.shape[0], "Arrays must be same imaginary time steps"

    N = a.shape[0]-1
    c1 = np.zeros_like(a)
    c2 = np.zeros_like(a)
    w = interpol.gregory_weights(N)
    
    for mm in range(N+1):
        if mm <= interpol.k:
            for jj in range(interpol.k+1):
                for ll in range(interpol.k+1):
                    c1[mm] += interpol.R[mm,jj,ll] * mat_prod(a[jj], b[ll])
        else:
            for ll in range(mm+1):
                c1[mm] += w[mm,ll] * mat_prod(a[mm-ll], b[ll])
        
        if mm >= N-interpol.k:
            for jj in range(interpol.k+1):
                for kk in range(interpol.k+1):
                    c2[mm] += (-1)**particle * interpol.R[N-mm,jj,kk] * mat_prod(a[N-jj], b[N-kk])
        else:
            for kk in range(N-mm+1):
                c2[mm] += (-1)**particle * w[N-mm,kk] * mat_prod(a[mm+kk], b[N-kk])
    
    return h * (c1 + c2)


@njit
def conv_mat(a, b, c, interpol, h, mat_prod, particle=0):
    assert a.get_mat().shape[0] == b.get_mat().shape[0], "Arrays must be same imaginary time steps"
    assert a.particle_type == b.particle_type
    
    particle = a.particle_type
    assert particle == c.particle_type
    
    N = a.get_mat().shape[0]-1
    w = interpol.gregory_weights(N)
    
    for mm in range(N+1):
        c1 = np.zeros_like(a.get_mat()[0])
        c2 = np.zeros_like(c1)
        if mm <= interpol.k:
            for jj in range(interpol.k+1):
                for ll in range(interpol.k+1):
                    c1 += interpol.R[mm,jj,ll] * mat_prod(a.get_mat()[jj], b.get_mat()[ll])
        else:
            for ll in range(mm+1):
                c1 += w[mm,ll] * mat_prod(a.get_mat()[mm-ll], b.get_mat()[ll])
        
        if mm >= N-interpol.k:
            for jj in range(interpol.k+1):
                for kk in range(interpol.k+1):
                    c2 += (-1)**particle * interpol.R[N-mm,jj,kk] * mat_prod(a.get_mat()[N-jj], b.get_mat()[N-kk])
        else:
            for kk in range(N-mm+1):
                c2 += (-1)**particle * w[N-mm,kk] * mat_prod(a.get_mat()[mm+kk], b.get_mat()[N-kk])
                
        c.set_mat_loc(mm, h * (c1 + c2))


@njit
def conv_lmx_res(t, a, b, c, interpol, h, mat_prod):
    assert a.get_lmx().shape[1] == b.get_mat().shape[0], "Arrays must be same imaginary time steps"
    assert a.particle_type == b.particle_type
    
    particle = a.particle_type
    assert particle == c.particle_type
    
    N = b.get_mat().shape[0]-1
    w = interpol.gregory_weights(N)
    
    for mm in range(a.shape[1]):
        c2 = np.zeros_like(b.get_mat()[0])
        c3 = np.zeros_like(c2)
        if mm <= interpol.k:
            for jj in range(interpol.k+1):
                for ll in range(interpol.k+1):
                    c2 += (-1)**particle * interpol.R[mm,jj,ll] * mat_prod(a.get_lmx()[t,ll], b.get_mat()[N-jj])
        else:
            for ll in range(mm+1):
                c2 += (-1)**particle * w[mm,ll] * mat_prod(a.get_lmx()[t,mm-ll], b.get_mat()[N-ll])
        
        if mm >= N-interpol.k:
            for jj in range(interpol.k+1):
                for ll in range(interpol.k+1):
                    c3 += interpol.R[N-mm,jj,ll] * mat_prod(a.get_lmx()[t,N-ll], b.get_mat()[jj])
        else:
            for ll in range(N-mm+1):
                c3 += w[N-mm,ll] * mat_prod(a.get_lmx()[t,mm+ll], b.get_mat()[ll])

        c.set_lmx_loc(t, mm, h * (c2 + c3))


@njit
def conv_les_res(t, tp_max, a, b, c, interpol, h, mat_prod):
    assert a.get_les().shape[1] == b.get_adv().shape[0], "Arrays must be same time steps"
    assert a.get_lmx().shape[1] == b.get_rmx().shape[0], "Arrays must be same imaginary time steps"
    assert a.particle_type == b.particle_type
    
    particle = a.particle_type
    assert particle == c.particle_type
    
    w = interpol.gregory_weights(a.shape[1]-1)
    N = a.get_mat().shape[0] - 1
    
    for mm in range(tp_max+1):
        cla = np.zeros_like(a.get_mat()[0])
        cij = np.zeros_like(cla)
        for jj in range(max(interpol.k,mm)+1):
            cla += w[mm,jj] * mat_prod(a.get_les()[t,jj], b.get_adv()[jj,mm])
        for kk in range(N+1):
            cij += w[N,kk] * mat_prod(a.get_lmx()[t,kk], b.get_rmx()[jj,tp_max])
    
    c.set_les_loc(t, tp_max, h * (cla - 1j*cij))
