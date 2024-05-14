import numpy as np
from numba import njit
from .linalg import matrix_matrix
from .convolution import conv_les1, conv_les2, conv_les3


@njit
def particle_density(tt, Gloc):
    return 1j * Gloc.particle_sign * np.trace(Gloc.get_les()[tt,tt])


@njit
def kinetic_energy(tt, H_kin, Gk, lattice):
    nkvec = len(Gk)
    p_sign = Gk[0].particle_sign
    norb = H_kin.shape[-1]
    assert Gk[0].get_mat().shape[-1] == norb

    Ekin = 0.0j
    for kk in range(nkvec):
        k_vec = lattice.get_vec(kk)
        Hk = 2*H_kin[0,tt]*np.cos(k_vec[0]) + 2*H_kin[1,tt]*np.cos(k_vec[1]) + 2*H_kin[2,tt]*np.cos(k_vec[2])
        for ii in range(norb):
            for jj in range(norb):
                Ekin += 1j*p_sign * Hk[ii,jj] * Gk[kk].get_les()[tt,tt,jj,ii] / nkvec
    
    return Ekin


@njit
def local_energy(tt, mu, Hloc, Gloc):
    p_sign = Gloc.particle_sign
    norb = Hloc.shape[-1]
    assert Gloc.get_mat().shape[-1] == norb

    Eloc = -mu + 0.0j
    for ii in range(norb):
        for jj in range(norb):
            Eloc += 1j * p_sign * Hloc[ii,jj] * Gloc.get_les()[tt,tt,jj,ii]
    return  Eloc


@njit
def interaction_energy(tt, Gloc, S, interp, h):
    assert Gloc.get_mat().shape[-1] == S.get_mat().shape[-1]
    
    Eint = 0.0j
    Eint -= 1j * np.trace(conv_les1(tt, tt, Gloc, S, interp, h, matrix_matrix))
    Eint -= 1j * np.trace(conv_les2(tt, tt, Gloc, S, interp, h, matrix_matrix))
    Eint -= 1j * np.trace(conv_les3(tt, tt, Gloc, S, interp, h, matrix_matrix))
    return Eint


@njit
def current_vector(tt, H_kin, Gk, lattice):
    nkvec = len(Gk)
    p_sign = Gk[0].particle_sign
    norb = H_kin.shape[-1]
    assert Gk[0].get_mat().shape[-1] == norb

    jvec = np.zeros((3,), dtype=np.complex128)
    for kk in range(nkvec):
        k_vec = lattice.get_vec(kk)
        for ii in range(3):
            dHk = -2*H_kin[ii,tt]*np.sin(k_vec[ii])
            for aa in range(norb):
                for bb in range(norb):
                    jvec[ii] += 1j*p_sign * dHk[aa,bb] * Gk[kk].get_les()[tt,tt,bb,aa]

    return jvec
