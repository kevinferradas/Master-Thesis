import numpy as np
from numba import njit, prange, int64, float64, config
from .interpolator import Interpolator
from .linalg import matrix_vector
# config.NUMBA_FASTMATH = False



@njit
def vide_startR(interpol: Interpolator, h: float64, p: np.ndarray, q: np.ndarray, K: np.ndarray, y0: np.ndarray, conjugate=False) -> np.ndarray:
    """
    Computes the initial bootstraping of a function described by a Volterra integro-differential equation for retarded component.
    
    Parameters
    ----------
    interpol : Interpolator
        Class with interpolation weights.
    p : ndarray
        Weight (one-time-label array).
    q : ndarray
        Source (one-time-label array).
    K : ndarray
        Kernell (two-time-label array).
    y0 : ndarray
        Initial condition (one-time-label array).
    conjugate : boolean, optional
        If True conjugate VIDE is computed. Default id False.

    Returns
    -------
    ndarray
        Bootstrapped function (one-time-label array).

    """
    p = p[:interpol.k+1]
    q = q[:interpol.k+1]
    K = K[:interpol.k+1,:interpol.k+1]
    
    assert p.ndim==3 and q.ndim==3 and K.ndim==4 and y0.ndim==3, "Non-compatible dimensions"
    assert y0.shape[0] <= interpol.k, "y0 cannot be considered as initial condition"
    init_ts = y0.shape[0]-1
    M = np.zeros((q.size,q.shape[0]*y0[0].size), dtype=np.complex128)
    if conjugate:
        assert y0.shape[-1]==p.shape[1], "Non-compatible weight-init shape"
        assert y0.shape[-1]==K.shape[2], "Non-compatible kernell-init shape"
        assert y0.shape[-2]==q.shape[1], "Non-compatible source-init shape"
        assert p.shape[2]==q.shape[2], "Non-compatibel weight-source shape"
        assert K.shape[3]==q.shape[2], "Non-compatible kernell-source shape"
        
        for mm in range(q.shape[0]):
            for ll in range(q.shape[0]):
                for ii in range(q.shape[1]):
                    for jj in range(q.shape[2]):
                        for nn in range(y0.shape[-1]):
                            idxMq = jj + q.shape[2]*ii + q.shape[2]*q.shape[1]*mm
                            idxMy = nn + y0.shape[-1]*ii + y0.shape[-1]*y0.shape[-2]*ll
                            M[idxMq,idxMy] += interpol.D[mm,ll] / h * (nn==jj)
                            M[idxMq,idxMy] += (mm==ll) * p[mm,nn,jj]
                            M[idxMq,idxMy] += h * interpol.I[init_ts,mm,ll] * K[ll,mm,nn,jj]
        
    else:
        assert y0.shape[-2]==p.shape[2], "Non-compatible weight-init shape"
        assert y0.shape[-2]==K.shape[3], "Non-compatible kernell-init shape"
        assert y0.shape[-1]==q.shape[2], "Non-compatible source-init shape"
        assert p.shape[1]==q.shape[1], "Non-compatibel weight-source shape"
        assert K.shape[2]==q.shape[1], "Non-compatible kernell-source shape"
        
        for mm in range(q.shape[0]):
            for ll in range(q.shape[0]):
                for ii in range(q.shape[1]):
                    for jj in range(q.shape[2]):
                        for kk in range(y0.shape[-2]):
                            idxMq = jj + q.shape[2]*ii + q.shape[2]*q.shape[1]*mm
                            idxMy = jj + y0.shape[-1]*kk + y0.shape[-1]*y0.shape[-2]*ll
                            M[idxMq,idxMy] += interpol.D[mm,ll] / h * (ii==kk)
                            M[idxMq,idxMy] += (mm==ll) * p[mm,ii,kk]
                            M[idxMq,idxMy] += h * interpol.I[init_ts,mm,ll] * K[mm,ll,ii,kk]
    
    """
    for mm in range(q.shape[0]):
        for ll in range(q.shape[0]):
            for ii in range(q.shape[1]):
                for jj in range(q.shape[2]):
                    for kk in range(y0.shape[0]):
                        for nn in range(y0.shape[1]):
                            idxMq = jj + q.shape[2]*ii + q.shape[2]*q.shape[1]*mm
                            idxMy = nn + y0.shape[1]*kk + y0.shape[1]*y0.shape[0]*ll
                            M[idxMq,idxMy] += interpol.D[mm,ll] / interpol.h * (ii==kk) * (nn==jj)
                            if conjugate:
                                M[idxMq,idxMy] += (mm==ll) * p[mm,nn,jj] * (ii==kk)
                                M[idxMq,idxMy] += interpol.h * interpol.s[mm,ll] * K[ll,mm,nn,jj] * (ii==kk)
                            else:
                                M[idxMq,idxMy] += (mm==ll) * p[mm,ii,kk] * (nn==jj)
                                M[idxMq,idxMy] += interpol.h * interpol.s[mm,ll] * K[mm,ll,ii,kk] * (nn==jj)
    """
    sz = y0[0].size
    Mcut = M[sz*(init_ts+1):,sz*(init_ts+1):]
    Minit = np.zeros_like(q[init_ts+1:].flatten())
    for ii in range(init_ts+1):
        Minit += matrix_vector(M[sz*(init_ts+1):,sz*ii:sz*(ii+1)], y0[ii].flatten())
    yboot = matrix_vector(np.linalg.inv(Mcut), q[init_ts+1:].flatten()-Minit)
    return yboot.reshape((q.shape[0]-init_ts-1,)+y0[0].shape)


@njit
def vide_start(interpol: Interpolator, h: float64, p: np.ndarray, q: np.ndarray, K: np.ndarray, y0: np.ndarray, conjugate=False) -> np.ndarray:
    """
    Computes the initial bootstraping of a function described by a Volterra integro-differential equation.
    
    Parameters
    ----------
    interpol : Interpolator
        Class with interpolation weights.
    p : ndarray
        Weight (one-time-label array).
    q : ndarray
        Source (one-time-label array).
    K : ndarray
        Kernell (two-time-label array).
    y0 : ndarray
        Initial condition (zero-time-label array).
    conjugate : boolean, optional
        If True conjugate VIDE is computed. Default id False.

    Returns
    -------
    ndarray
        Bootstrapped function (one-time-label array).

    """
    p = p[:interpol.k+1]
    q = q[:interpol.k+1]
    K = K[:interpol.k+1,:interpol.k+1]
    
    assert p.ndim==3 and q.ndim==3 and K.ndim==4 and y0.ndim==2, "Non-compatible dimensions"
    M = np.zeros((q.size,q.shape[0]*y0.size), dtype=np.complex128)
    if conjugate:
        assert y0.shape[1]==p.shape[1], "Non-compatible weight-init shape"
        assert y0.shape[1]==K.shape[2], "Non-compatible kernell-init shape"
        assert y0.shape[0]==q.shape[1], "Non-compatible source-init shape"
        assert p.shape[2]==q.shape[2], "Non-compatibel weight-source shape"
        assert K.shape[3]==q.shape[2], "Non-compatible kernell-source shape"
        
        for mm in range(q.shape[0]):
            for ll in range(q.shape[0]):
                for ii in range(q.shape[1]):
                    for jj in range(q.shape[2]):
                        for nn in range(y0.shape[1]):
                            idxMq = jj + q.shape[2]*ii + q.shape[2]*q.shape[1]*mm
                            idxMy = nn + y0.shape[1]*ii + y0.shape[1]*y0.shape[0]*ll
                            M[idxMq,idxMy] += interpol.D[mm,ll] / h * (nn==jj)
                            M[idxMq,idxMy] += (mm==ll) * p[mm,nn,jj]
                            M[idxMq,idxMy] += h * interpol.s[mm,ll] * K[ll,mm,nn,jj]
        
    else:
        assert y0.shape[0]==p.shape[2], "Non-compatible weight-init shape"
        assert y0.shape[0]==K.shape[3], "Non-compatible kernell-init shape"
        assert y0.shape[1]==q.shape[2], "Non-compatible source-init shape"
        assert p.shape[1]==q.shape[1], "Non-compatibel weight-source shape"
        assert K.shape[2]==q.shape[1], "Non-compatible kernell-source shape"
        
        for mm in range(q.shape[0]):
            for ll in range(q.shape[0]):
                for ii in range(q.shape[1]):
                    for jj in range(q.shape[2]):
                        for kk in range(y0.shape[0]):
                            idxMq = jj + q.shape[2]*ii + q.shape[2]*q.shape[1]*mm
                            idxMy = jj + y0.shape[1]*kk + y0.shape[1]*y0.shape[0]*ll
                            M[idxMq,idxMy] += interpol.D[mm,ll] / h * (ii==kk)
                            M[idxMq,idxMy] += (mm==ll) * p[mm,ii,kk]
                            M[idxMq,idxMy] += h * interpol.s[mm,ll] * K[mm,ll,ii,kk]
    
    """
    for mm in range(q.shape[0]):
        for ll in range(q.shape[0]):
            for ii in range(q.shape[1]):
                for jj in range(q.shape[2]):
                    for kk in range(y0.shape[0]):
                        for nn in range(y0.shape[1]):
                            idxMq = jj + q.shape[2]*ii + q.shape[2]*q.shape[1]*mm
                            idxMy = nn + y0.shape[1]*kk + y0.shape[1]*y0.shape[0]*ll
                            M[idxMq,idxMy] += interpol.D[mm,ll] / interpol.h * (ii==kk) * (nn==jj)
                            if conjugate:
                                M[idxMq,idxMy] += (mm==ll) * p[mm,nn,jj] * (ii==kk)
                                M[idxMq,idxMy] += interpol.h * interpol.s[mm,ll] * K[ll,mm,nn,jj] * (ii==kk)
                            else:
                                M[idxMq,idxMy] += (mm==ll) * p[mm,ii,kk] * (nn==jj)
                                M[idxMq,idxMy] += interpol.h * interpol.s[mm,ll] * K[mm,ll,ii,kk] * (nn==jj)
    """
    
    Mcut = M[y0.size:,y0.size:]
    Minit = matrix_vector(M[y0.size:,:y0.size], y0.flatten())
    yboot = matrix_vector(np.linalg.inv(Mcut), q[1:].flatten()-Minit)
    return yboot.reshape((q.shape[0]-1,)+y0.shape)


@njit
def vie_startR(interpol, q, K, y0, conjugate=False):
    """
    Computes the initial bootstraping of a function described by a Volterra integral equation for retarded component.
    
    Parameters
    ----------
    interpol : Interpolator
        Class with interpolation weights.
    q : ndarray
        Source (one-time-label array).
    K : ndarray
        Kernell (two-time-label array).
    y0 : ndarray
        Initial condition (one-time-label array).
    conjugate : boolean, optional
        If True conjugate VIDE is computed. Default id False.

    Returns
    -------
    ndarray
        Bootstrapped function (one-time-label array).

    """
    q = q[:interpol.k+1]
    K = K[:interpol.k+1,:interpol.k+1]
    
    assert q.ndim==5 and K.ndim==6 and y0.ndim==5, "Non-compatible dimensions"
    assert y0.shape[0] <= interpol.k, "y0 cannot be considered as initial condition"
    init_ts = y0.shape[0]-1
    M = np.zeros((q.size,q.shape[0]*y0[0].size), dtype=np.complex128)
    if conjugate:
        assert y0.shape[1]==q.shape[1], "Non-compatible source-init shape"
        assert y0.shape[3]==q.shape[3], "Non-compatible source-init shape"
        assert y0.shape[4]==K.shape[2], "Non-compatible kernell-init shape"
        assert y0.shape[2]==K.shape[4], "Non-compatible kernell-init shape"
        assert K.shape[3]==q.shape[2], "Non-compatible kernell-source shape"
        assert K.shape[5]==q.shape[4], "Non-compatible kernell-source shape"
        
        for ss in range(q.shape[0]):
            for tt in range(q.shape[0]):
                for ii in range(q.shape[1]):
                    for jj in range(q.shape[2]):
                        for kk in range(q.shape[3]):
                            for ll in range(q.shape[4]):
                                for nn in range(y0.shape[2]):
                                    for qq in range(y0.shape[4]):
                                        idxMq = ll + q.shape[-1]*(kk + q.shape[-2]*(jj + q.shape[-3]*(ii + q.shape[-4]*ss)))
                                        idxMy = qq + q.shape[-1]*(kk + q.shape[-2]*(nn + q.shape[-3]*(ii + q.shape[-4]*tt)))
                                        M[idxMq,idxMy] += 1 if (ss==tt) and (jj==nn) and (ll==qq) else 0
                                        M[idxMq,idxMy] += interpol.h * interpol.s[ss,tt] * K[tt,ss,qq,jj,nn,ll]
        
    else:
        assert y0.shape[2]==q.shape[2], "Non-compatible source-init shape"
        assert y0.shape[4]==q.shape[4], "Non-compatible source-init shape"
        assert y0.shape[1]==K.shape[5], "Non-compatible kernell-init shape"
        assert y0.shape[3]==K.shape[3], "Non-compatible kernell-init shape"
        assert K.shape[2]==q.shape[1], "Non-compatible kernell-source shape"
        assert K.shape[4]==q.shape[3], "Non-compatible kernell-source shape"
        
        for ss in range(q.shape[0]):
            for tt in range(q.shape[0]):
                for ii in range(q.shape[1]):
                    for jj in range(q.shape[2]):
                        for kk in range(q.shape[3]):
                            for ll in range(q.shape[4]):
                                for mm in range(y0.shape[1]):
                                    for pp in range(y0.shape[3]):
                                        idxMq = ll + q.shape[-1]*(kk + q.shape[-2]*(jj + q.shape[-3]*(ii + q.shape[-4]*ss)))
                                        idxMy = ll + q.shape[-1]*(pp + q.shape[-2]*(jj + q.shape[-3]*(mm + q.shape[-4]*tt)))
                                        M[idxMq,idxMy] += 1 if (ss==tt) and (ii==mm) and (kk==pp) else 0
                                        M[idxMq,idxMy] += interpol.h * interpol.s[ss,tt] * K[ss,tt,ii,pp,kk,mm]
    
    """
    for ss in range(q.shape[0]):
        for tt in range(q.shape[0]):
            for ii in range(q.shape[1]):
                for jj in range(q.shape[2]):
                    for kk in range(q.shape[3]):
                        for ll in range(q.shape[4]):
                            for mm in range(y0.shape[0]):
                                for nn in range(y0.shape[1]):
                                    for pp in range(y0.shape[2]):
                                        for qq in range(y0.shape[3]):
                                            idxMq = ll + q.shape[-1]*(kk + q.shape[-2]*(jj + q.shape[-3]*(ii + q.shape[-4]*ss)))
                                            idxMy = qq + q.shape[-1]*(pp + q.shape[-2]*(nn + q.shape[-3]*(mm + q.shape[-4]*tt)))
                                            M[idxMq,idxMy] += 1 * (ss==tt) * (ii==mm) * (jj==nn) * (kk==pp) * (ll==qq)
                                            if conjugate:
                                                M[idxMq,idxMy] += interpol.h * interpol.s[ss,tt] * K[tt,ss,qq,jj,nn,ll] * (ii==mm) * (kk==pp)
                                            else:
                                                M[idxMq,idxMy] += interpol.h * interpol.s[ss,tt] * K[ss,tt,ii,pp,kk,mm] * (jj==nn) * (ll==qq)
    """
    sz = y0[0].size
    Mcut = M[sz*(init_ts+1):,sz*(init_ts+1):]
    Minit = np.zeros_like(q[init_ts+1:].flatten())
    for ii in range(init_ts+1):
        Minit += matrix_vector(M[sz*(init_ts+1):,sz*ii:sz*(ii+1)], y0[ii].flatten())
    yboot = matrix_vector(np.linalg.inv(Mcut), q[init_ts+1:].flatten()-Minit)
    return yboot.reshape((q.shape[0]-init_ts-1,)+y0[0].shape)


@njit
def vide_step(interpol: Interpolator, h: float64, p: np.ndarray, q: np.ndarray, K: np.ndarray, y: np.ndarray, conjugate=False) -> np.ndarray:
    """
    Computes the following time-step of a function described by a Volterra integro-differential equation.
    
    Parameters
    ----------
    interpol : Interpolator
        Class with interpolation weights.
    p : ndarray
        Weight (one-time-label array).
        Its length indicates the current time-steps computed.
    q : ndarray
        Source (one-time-label array).
    K : ndarray
        Kernell (two-time-label array).
    y : ndarray
        Function to solve (one-time-label array).
    conjugate : boolean, optional
        If True conjugate VIDE is computed. Default is False.

    Returns
    -------
    ndarray
        Next time-step (zero-time-label array).

    """
    n = y.shape[0] #Current time-step
    w = interpol.gregory_weights(n)[n,:] #Only current time-step, one-time-label
    
    if p.ndim==3:
        p = p[n] #Only curent time-step, zero-time-label
    K = K[:n+1,n] if conjugate else K[n,:n+1] #Only current time-step, one-time-label
    if q.ndim==3:
        q = q[n] #Only current time-step, zero-time label
    
    assert p.ndim==2 and q.ndim==2 and K.ndim==3 and y.ndim==3, "Non-compatible dimensions"
    M = np.zeros((q.size,y.size+y.shape[1]*y.shape[2]), dtype=np.complex128)
    if conjugate:
        assert y.shape[2]==p.shape[0], "Non-compatible weight-func shape"
        assert y.shape[2]==K.shape[1], "Non-compatible kernell-func shape"
        assert y.shape[1]==q.shape[0], "Non-compatible source-func shape"
        assert p.shape[1]==q.shape[1], "Non-compatibel weight-source shape"
        assert K.shape[2]==q.shape[1], "Non-compatible kernell-source shape"
        
        for ll in range(n+1):
            for ii in range(q.shape[0]):
                for jj in range(q.shape[1]):
                    for nn in range(y.shape[2]):
                        idxMq = jj + q.shape[1]*ii
                        idxMy = nn + y.shape[2]*ii + y.shape[2]*y.shape[1]*ll
                        if ll > n - (interpol.k+1):
                            M[idxMq,idxMy] += interpol.a[n-ll] / h * (jj==nn)
                        M[idxMq,idxMy] += (n==ll) * p[nn,jj]
                        M[idxMq,idxMy] += h * w[ll] * K[ll,nn,jj]
        
    else:
        assert y.shape[1]==p.shape[1], "Non-compatible weight-func shape"
        assert y.shape[1]==K.shape[2], "Non-compatible kernell-func shape"
        assert y.shape[2]==q.shape[1], "Non-compatible source-func shape"
        assert p.shape[0]==q.shape[0], "Non-compatibel weight-source shape"
        assert K.shape[1]==q.shape[0], "Non-compatible kernell-source shape"
        
        for ll in range(n+1):
            for ii in range(q.shape[0]):
                for jj in range(q.shape[1]):
                    for kk in range(y.shape[1]):
                        idxMq = jj + q.shape[1]*ii
                        idxMy = jj + y.shape[2]*kk + y.shape[2]*y.shape[1]*ll
                        if ll > n - (interpol.k+1):
                            M[idxMq,idxMy] += interpol.a[n-ll] / h * (ii==kk)
                        M[idxMq,idxMy] += (n==ll) * p[ii,kk]
                        M[idxMq,idxMy] += h * w[ll] * K[ll,ii,kk]
    
    """
    for ll in range(n+1):
        for ii in range(q.shape[0]):
            for jj in range(q.shape[1]):
                for kk in range(y.shape[1]):
                    for nn in range(y.shape[2]):
                        idxMq = jj + q.shape[1]*ii
                        idxMy = nn + y.shape[2]*kk + y.shape[2]*y.shape[1]*ll
                        if ll > n - (interpol.k+1):
                            M[idxMq,idxMy] += interpol.a[n-ll] / interpol.h * (ii==kk) * (jj==nn)
                        if conjugate:
                            M[idxMq,idxMy] += (n==ll) * p[nn,jj] * (ii==kk)
                            M[idxMq,idxMy] += interpol.h * w[ll] * K[ll,nn,jj] * (ii==kk)
                        else:
                            M[idxMq,idxMy] += (n==ll) * p[ii,kk] * (nn==jj)
                            M[idxMq,idxMy] += interpol.h * w[ll] * K[ll,ii,kk] * (nn==jj)
    """
    
    Mcut = matrix_vector(M[:,:-y.shape[1]*y.shape[2]], y[:n].flatten())
    Mfinal = M[:,-y.shape[1]*y.shape[2]:]
    return matrix_vector(np.linalg.inv(Mfinal), (q.flatten()-Mcut).reshape(q.shape))


@njit
def vie_start(interpol, q, K, y0, conjugate=False):
    """
    Computes the initial bootstraping of a function described by a Volterra integral equation.
    
    Parameters
    ----------
    interpol : Interpolator
        Class with interpolation weights.
    q : ndarray
        Source (one-time-label array).
    K : ndarray
        Kernell (two-time-label array).
    y0 : ndarray
        Initial condition (zero-time-label array).
    conjugate : boolean, optional
        If True conjugate VIDE is computed. Default id False.

    Returns
    -------
    ndarray
        Bootstrapped function (one-time-label array).

    """
    q = q[:interpol.k+1]
    K = K[:interpol.k+1,:interpol.k+1]
    
    assert q.ndim==5 and K.ndim==6 and y0.ndim==4, "Non-compatible dimensions"
    M = np.zeros((q.size,q.shape[0]*y0.size), dtype=np.complex128)
    if conjugate:
        assert y0.shape[0]==q.shape[1], "Non-compatible source-init shape"
        assert y0.shape[2]==q.shape[3], "Non-compatible source-init shape"
        assert y0.shape[3]==K.shape[2], "Non-compatible kernell-init shape"
        assert y0.shape[1]==K.shape[4], "Non-compatible kernell-init shape"
        assert K.shape[3]==q.shape[2], "Non-compatible kernell-source shape"
        assert K.shape[5]==q.shape[4], "Non-compatible kernell-source shape"
        
        for ss in range(q.shape[0]):
            for tt in range(q.shape[0]):
                for ii in range(q.shape[1]):
                    for jj in range(q.shape[2]):
                        for kk in range(q.shape[3]):
                            for ll in range(q.shape[4]):
                                for nn in range(y0.shape[1]):
                                    for qq in range(y0.shape[3]):
                                        idxMq = ll + q.shape[-1]*(kk + q.shape[-2]*(jj + q.shape[-3]*(ii + q.shape[-4]*ss)))
                                        idxMy = qq + q.shape[-1]*(kk + q.shape[-2]*(nn + q.shape[-3]*(ii + q.shape[-4]*tt)))
                                        M[idxMq,idxMy] += 1 if (ss==tt) and (jj==nn) and (ll==qq) else 0
                                        M[idxMq,idxMy] += interpol.h * interpol.s[ss,tt] * K[tt,ss,qq,jj,nn,ll]
        
    else:
        assert y0.shape[1]==q.shape[2], "Non-compatible source-init shape"
        assert y0.shape[3]==q.shape[4], "Non-compatible source-init shape"
        assert y0.shape[0]==K.shape[5], "Non-compatible kernell-init shape"
        assert y0.shape[2]==K.shape[3], "Non-compatible kernell-init shape"
        assert K.shape[2]==q.shape[1], "Non-compatible kernell-source shape"
        assert K.shape[4]==q.shape[3], "Non-compatible kernell-source shape"
        
        for ss in range(q.shape[0]):
            for tt in range(q.shape[0]):
                for ii in range(q.shape[1]):
                    for jj in range(q.shape[2]):
                        for kk in range(q.shape[3]):
                            for ll in range(q.shape[4]):
                                for mm in range(y0.shape[0]):
                                    for pp in range(y0.shape[2]):
                                        idxMq = ll + q.shape[-1]*(kk + q.shape[-2]*(jj + q.shape[-3]*(ii + q.shape[-4]*ss)))
                                        idxMy = ll + q.shape[-1]*(pp + q.shape[-2]*(jj + q.shape[-3]*(mm + q.shape[-4]*tt)))
                                        M[idxMq,idxMy] += 1 if (ss==tt) and (ii==mm) and (kk==pp) else 0
                                        M[idxMq,idxMy] += interpol.h * interpol.s[ss,tt] * K[ss,tt,ii,pp,kk,mm]
    
    """
    for ss in range(q.shape[0]):
        for tt in range(q.shape[0]):
            for ii in range(q.shape[1]):
                for jj in range(q.shape[2]):
                    for kk in range(q.shape[3]):
                        for ll in range(q.shape[4]):
                            for mm in range(y0.shape[0]):
                                for nn in range(y0.shape[1]):
                                    for pp in range(y0.shape[2]):
                                        for qq in range(y0.shape[3]):
                                            idxMq = ll + q.shape[-1]*(kk + q.shape[-2]*(jj + q.shape[-3]*(ii + q.shape[-4]*ss)))
                                            idxMy = qq + q.shape[-1]*(pp + q.shape[-2]*(nn + q.shape[-3]*(mm + q.shape[-4]*tt)))
                                            M[idxMq,idxMy] += 1 * (ss==tt) * (ii==mm) * (jj==nn) * (kk==pp) * (ll==qq)
                                            if conjugate:
                                                M[idxMq,idxMy] += interpol.h * interpol.s[ss,tt] * K[tt,ss,qq,jj,nn,ll] * (ii==mm) * (kk==pp)
                                            else:
                                                M[idxMq,idxMy] += interpol.h * interpol.s[ss,tt] * K[ss,tt,ii,pp,kk,mm] * (jj==nn) * (ll==qq)
    """
    
    Mcut = M[y0.size:,y0.size:]
    Minit = matrix_vector(M[y0.size:,:y0.size], y0.flatten())
    yboot = matrix_vector(np.linalg.inv(Mcut), q[1:].flatten()-Minit)
    return yboot.reshape((q.shape[0]-1,)+y0.shape)


@njit
def vie_step(interpol, q, K, y, conjugate=False):
    """
    Computes the following of a function described by a Volterra integral equation.
    
    Parameters
    ----------
    interpol : Interpolator
        Class with interpolation weights.
    q : ndarray
        Source (one-time-label array).
    K : ndarray
        Kernell (two-time-label array).
    y : ndarray
        Function to solve (one-time-label array).
        Its length indicates the current time-steps computed.
    conjugate : boolean, optional
        If True conjugate VIDE is computed. Default if False.

    Returns
    -------
    ndarray
        Next time-step (zero-time-label array).

    """
    n = y.shape[0] #Current time-step
    w = interpol.gregory_weights(n)[n,:] #Only current time-step, one-time-label
    
    K = K[:n+1,n] if conjugate else K[n,:n+1] #Only current time-step, one-time-label
    if q.ndim==5:
        q = q[n] #Only current time-step, zero-time label
    
    assert q.ndim==4 and K.ndim==5 and y.ndim==5, "Non-compatible dimensions"
    M = np.zeros((q.size,y.size+y.shape[1]*y.shape[2]*y.shape[3]*y.shape[4]), dtype=np.complex128)
    if conjugate:
        assert y.shape[1]==q.shape[0], "Non-compatible source-init shape"
        assert y.shape[3]==q.shape[2], "Non-compatible source-init shape"
        assert y.shape[4]==K.shape[1], "Non-compatible kernell-init shape"
        assert y.shape[2]==K.shape[3], "Non-compatible kernell-init shape"
        assert K.shape[2]==q.shape[1], "Non-compatible kernell-source shape"
        assert K.shape[4]==q.shape[3], "Non-compatible kernell-source shape"
        
        for tt in range(n+1):
            for ii in range(q.shape[0]):
                for jj in range(q.shape[1]):
                    for kk in range(q.shape[2]):
                        for ll in range(q.shape[3]):
                            for nn in range(y.shape[2]):
                                for qq in range(y.shape[4]):
                                    idxMq = ll + q.shape[-1]*(kk + q.shape[-2]*(jj + q.shape[-3]*ii))
                                    idxMy = qq + q.shape[-1]*(kk + q.shape[-2]*(nn + q.shape[-3]*(ii + q.shape[-4]*tt)))
                                    M[idxMq,idxMy] += 1 if (n==tt) and (jj==nn) and (ll==qq) else 0
                                    M[idxMq,idxMy] += interpol.h * w[tt] * K[tt,qq,jj,nn,ll]
    
    else:
        assert y.shape[2]==q.shape[1], "Non-compatible source-init shape"
        assert y.shape[4]==q.shape[3], "Non-compatible source-init shape"
        assert y.shape[1]==K.shape[4], "Non-compatible kernell-init shape"
        assert y.shape[3]==K.shape[2], "Non-compatible kernell-init shape"
        assert K.shape[1]==q.shape[0], "Non-compatible kernell-source shape"
        assert K.shape[3]==q.shape[2], "Non-compatible kernell-source shape"
        
        for tt in range(n+1):
            for ii in range(q.shape[0]):
                for jj in range(q.shape[1]):
                    for kk in range(q.shape[2]):
                        for ll in range(q.shape[3]):
                            for mm in range(y.shape[1]):
                                for pp in range(y.shape[3]):
                                    idxMq = ll + q.shape[-1]*(kk + q.shape[-2]*(jj + q.shape[-3]*ii))
                                    idxMy = ll + q.shape[-1]*(pp + q.shape[-2]*(jj + q.shape[-3]*(mm + q.shape[-4]*tt)))
                                    M[idxMq,idxMy] += 1 if (n==tt) and (ii==mm) and (kk==pp) else 0
                                    M[idxMq,idxMy] += interpol.h * w[tt] * K[tt,ii,pp,kk,mm]
    
    """
    for tt in range(n+1):
        for ii in range(q.shape[0]):
            for jj in range(q.shape[1]):
                for kk in range(q.shape[2]):
                    for ll in range(q.shape[3]):
                        for mm in range(y.shape[1]):
                            for nn in range(y.shape[2]):
                                for pp in range(y.shape[3]):
                                    for qq in range(y.shape[4]):
                                        idxMq = ll + q.shape[-1]*(kk + q.shape[-2]*(jj + q.shape[-3]*ii))
                                        idxMy = qq + q.shape[-1]*(pp + q.shape[-2]*(nn + q.shape[-3]*(mm + q.shape[-4]*tt)))
                                        M[idxMq,idxMy] += 1 * (n==tt) * (ii==mm) * (jj==nn) * (kk==pp) * (ll==qq)
                                        if conjugate:
                                            M[idxMq,idxMy] += interpol.h * w[tt] * K[tt,qq,jj,nn,ll] * (ii==mm) * (kk==pp)
                                        else:
                                            M[idxMq,idxMy] += interpol.h * w[tt] * K[tt,ii,pp,kk,mm] * (jj==nn) * (ll==qq)
    """
    
    Mcut = matrix_vector(M[:,:-y.shape[1]*y.shape[2]*y.shape[3]*y.shape[4]], y[:n].flatten())
    Mfinal = M[:,-y.shape[1]*y.shape[2]*y.shape[3]*y.shape[4]:]
    return matrix_vector(np.linalg.inv(Mfinal), (q.flatten()-Mcut).reshape(q.shape))
