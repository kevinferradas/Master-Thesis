import numpy as np
from numba import njit
from .linalg import matrix_matrix



@njit
def time_evolution_boot(H, interp, h):
    a1 = (3-2*np.sqrt(3))/12
    a2 = (3+2*np.sqrt(3))/12
    c1 = (1-1/np.sqrt(3))/2
    c2 = (1+1/np.sqrt(3))/2

    U = np.zeros((interp.k,)+H[0].shape, dtype=np.complex128)

    for tt in range(interp.k):
        J1 = np.zeros_like(H[0])
        J2 = np.zeros_like(H[0])
        for rr in range(interp.k+1):
            for ll in range(interp.k+1):
                J1 += (tt+c1)**rr * interp.P[rr,ll] * H[ll]
                J2 += (tt+c2)**rr * interp.P[rr,ll] * H[ll]
        
        O1 = a1*J1 + a2*J2
        O2 = a2*J1 + a1*J2
        assert not np.any(np.isnan(O1))
        assert not np.any(np.isnan(O2))
        
        w1, P1 = np.linalg.eig(O1)
        w2, P2 = np.linalg.eig(O2)
        
        V1 = np.zeros_like(O1)
        V2 = np.zeros_like(O2)
        for ii in range(V1.shape[0]):
            for jj in range(V1.shape[1]):
                for kk in range(V2.shape[1]):
                    V1[ii,kk] += P1[ii,jj] * np.exp(-1j*w1[jj] * h) * P1[kk,jj].conjugate()
                    V2[ii,kk] += P2[ii,jj] * np.exp(-1j*w2[jj] * h) * P2[kk,jj].conjugate()
                    # print("i,j,k", ii, jj, kk)
                    # if np.isnan(V1[ii,kk]):
                    #     print("V1 nan at i,j,k", ii, jj, kk)
                    # print("P1[i,j]", P1[ii,jj])
                    # print("P1[k,j]", P1[kk,jj])
                    # print("w1[j]", w1[jj])
                    # if np.isnan(V2[ii,kk]):
                    #     print("V2 nan at i,j,k", ii, jj, kk)
                    # print("P2[i,j]", P2[ii,jj])
                    # print("P2[k,j]", P2[kk,jj])
                    # print("w2[j]", w2[jj])
        assert not np.any(np.isnan(V1))
        assert not np.any(np.isnan(V2))

        U[tt] = matrix_matrix(V1, V2)
        assert not np.any(np.isnan(U[tt]))
    
    return U


@njit
def time_evolution_step(H, interp, h):
    a1 = (3-2*np.sqrt(3))/12
    a2 = (3+2*np.sqrt(3))/12
    c1 = (1-1/np.sqrt(3))/2
    c2 = (1+1/np.sqrt(3))/2

    J1 = np.zeros_like(H[0])
    J2 = np.zeros_like(H[0])
    for rr in range(interp.k+1):
        for ll in range(interp.k+1):
            J1 += (interp.k-1+c1)**rr * interp.P[rr,ll] * H[ll]
            J2 += (interp.k-1+c2)**rr * interp.P[rr,ll] * H[ll]
            if np.any(np.isnan(J1)):
                print("J1 nan at r,l", rr, ll)
                # print((interp.k-1+c1)**rr)
                # print(H[tt-interp.k+ll])
            if np.any(np.isnan(J2)):
                print("J2 nan at r,l", rr, ll)
                # print((interp.k-1+c2)**rr)
                # print(H[tt-interp.k+ll])
    
    assert not np.any(np.isnan(J1))
    assert not np.any(np.isnan(J2))
    O1 = a1*J1 + a2*J2
    O2 = a2*J1 + a1*J2
    assert not np.any(np.isnan(O1))
    assert not np.any(np.isnan(O2))

    w1, P1 = np.linalg.eig(O1)
    w2, P2 = np.linalg.eig(O2)

    V1 = np.zeros_like(O1)
    V2 = np.zeros_like(O2)
    for ii in range(V1.shape[0]):
        for jj in range(V1.shape[1]):
            for kk in range(V2.shape[1]):
                V1[ii,kk] += P1[ii,jj] * np.exp(-1j*w1[jj] * h) * P1[kk,jj].conjugate()
                V2[ii,kk] += P2[ii,jj] * np.exp(-1j*w2[jj] * h) * P2[kk,jj].conjugate()
                if np.isnan(V1[ii,kk]):
                    print("V1 nan at i,j,k", ii, jj, kk)
                    # print("P1[i,j]", P1[ii,jj])
                    # print("P1[k,j]", P1[kk,jj])
                    print(O1)
                    print("w1[j]", w1[jj])
                if np.isnan(V2[ii,kk]):
                    print("V2 nan at i,j,k", ii, jj, kk)
                    # print("P2[i,j]", P2[ii,jj])
                    # print("P2[k,j]", P2[kk,jj])
                    print(O2)
                    print("w2[j]", w2[jj])

    return matrix_matrix(V1, V2)
