import numpy as np
from numba import njit
from .linalg import matrix_matrix, matrix_matrix2, matrix_tensor, tensor_tensor
from .contour_funcs import Gmatrix, Vmatrix
from .convolution import conv_mat_extern
from .utils import gdist_mat
from .printing import float_string, exp_string


@njit
def sgn(x):
    if x < 0:
        return -1
    elif x > 0:
        return +1
    else:
        return 0


@njit
def g_nonint_init(ntau, N, mu, H0, beta=1, particle=0, mu_jump=0.5, tolN=1e-6):
    assert H0.ndim==2, "Only constant hamiltonian"
    assert H0.shape[0]==H0.shape[1], "Hamiltonian is a squared matrix"
    n_orb = H0.shape[0]
    particle_sign = (-1)**particle
    tau = np.linspace(0, beta, ntau)
    
    assert not np.any(np.isnan(H0))
    
    w, P = np.linalg.eig(H0)
    Pinv = np.linalg.inv(P)
    
    last_sign = 2
    if N>0:
        print("Checking number of particle for non-interacting case")
        while True:
            N0 = 0.0
            for jj in range(n_orb):
                e = w[jj].real - mu
                N0 += 1/(particle_sign - np.exp(e * beta))
            
            DN = N - N0
            DNsign = sgn(DN)
            if abs(DN) < tolN:
                break
            if DNsign!=last_sign and last_sign!=2:
                mu_jump /= 2
            mu += DNsign * mu_jump
            last_sign = DNsign
    
    g = np.zeros((ntau, H0.shape[0], H0.shape[1]), dtype=np.complex128)
    for ii in range(n_orb):
        for jj in range(n_orb):
            for kk in range(n_orb):
                e = w[jj] - mu
                g[...,ii,kk] += P[ii,jj] * np.exp(-e*tau)/(particle_sign * np.exp(-e*beta) - 1) * Pinv[jj,kk] # P[kk,jj].conjugate()
    
    if np.any(np.abs(g) > 1e14):
        print("Warning: possible overload on non-interactive green's function")
    return g, mu


@njit
def matsubara_branch_init(N, mu, H0, G, S, vP, W, E, v, interpol, beta=1, particle=0, mu_jump=0.5, max_iter=100000, tol=1e-6):
    print("Initilizing Matsubara branch")
    ntau = G.get_mat().shape[0]
    particle_sign = (-1)**particle
    
    ntau = G.get_mat().shape[0]
    htau = beta / (ntau-1)
    
    last_sign = 2
    while True:
        conv = 1e5
        loop_iterations = 0
        print("Starting Matsubara loop for mu="+float_string(mu, 5))
        while conv>=tol:
            for tt in range(ntau):
                vP.set_mat_loc(tt, -tensor_tensor(v, matrix_matrix2(G.get_mat()[tt], G.neg_imag_time_mat()[tt])))
                E.set_mat_loc(tt, tensor_tensor(vP[tt], v))
            W.set_mat(E.get_mat() + conv_mat_extern(vP.get_mat(), W.get_mat(), interpol, htau, tensor_tensor, 0))
            
            S.set_hf_loc(0, -particle_sign * matrix_tensor(G.get_mat()[-1], v-np.swapaxes(v, -1, -2)))
            for tt in range(ntau):
                S.set_mat_loc(tt, matrix_tensor(G.get_mat()[tt], W.get_mat()[tt]))
            
            g,_ = g_nonint_init(ntau, -1, mu, H0 - mu*np.eye(H0.shape[0]) + S.get_hf()[0], beta, particle)
            F = conv_mat_extern(g, S.get_mat(), interpol, htau, matrix_matrix, particle)
            newGM = g + conv_mat_extern(F, G.get_mat(), interpol, htau, matrix_matrix, particle)
            
            conv = gdist_mat(newGM, G.get_mat())
            G.set_mat(newGM)
            
            loop_iterations += 1
            assert loop_iterations <= max_iter
        
        print("Convergence for Matsubara branch and mu="+float_string(mu, 5))
        print("Norm "+exp_string(conv, 5)+"\n\n")
        
        N0 = -np.trace(G.get_mat()[-1])
        DN = N - N0.real
        DNsign = sgn(DN)
        if abs(DN) < tol:
            print("Reached convergence for number of particles "+float_string(N0.real, 5))
            print("----------------------------------------")
            break
        if DNsign!=last_sign and last_sign!=2:
            mu_jump /= 2
        mu += DNsign * mu_jump
        last_sign = DNsign
        
        print("Computed number of particles "+float_string(N0.real, 5))
        print("New mu="+float_string(mu, 5))
    
    return mu


@njit
def matsubara_branch_init_gw0(N, mu, H0, G, S, v, interpol, beta=1, particle=0, mu_jump=0.5, max_iter=100000, tol=1e-6):
    print("Initilizing Matsubara branch")
    ntau = G.get_mat().shape[0]
    particle_sign = (-1)**particle
    
    ntau = G.get_mat().shape[0]
    htau = beta / (ntau-1)
    
    last_sign = 2
    while True:
        conv = 1e5
        loop_iterations = 0
        print("Starting Matsubara loop for mu="+float_string(mu, 5))
        while conv>=tol:
            
            S.set_hf_loc(0, -particle_sign * matrix_tensor(G.get_mat()[-1], v-np.swapaxes(v, -1, -2)))
            print("Shf(t=0) computed")
            print(S.get_hf()[0])
            print("\n")
            assert not np.any(np.isnan(v))
            assert not np.any(np.isnan(S.get_hf()[0]))
            for tt in range(ntau):
                S.set_mat_loc(tt, -matrix_tensor(G.get_mat()[tt], tensor_tensor(tensor_tensor(v, matrix_matrix2(G.get_mat()[tt], G.neg_imag_time_mat()[tt])), v)))
            print("SM computed")
            
            g,no_use = g_nonint_init(ntau, -1, mu, H0 - mu*np.eye(H0.shape[0]) + S.get_hf()[0], beta, particle)
            print("Non-interactive g computed")
            F = conv_mat_extern(g, S.get_mat(), interpol, htau, matrix_matrix, particle)
            print("G and self-energy convoluted")
            newGM = g + conv_mat_extern(F, G.get_mat(), interpol, htau, matrix_matrix, particle)
            print("New GM convoluted")
            
            conv = gdist_mat(newGM, G.get_mat())
            print("Distance computed")
            G.set_mat(newGM)
            print("New GM set")
            
            loop_iterations += 1
            assert loop_iterations <= max_iter
        
        print("Convergence for Matsubara branch and mu="+float_string(mu, 5))
        print("Norm "+exp_string(conv, 5)+"\n\n")
        
        N0 = -np.trace(G.get_mat()[-1])
        DN = N - N0.real
        DNsign = sgn(DN)
        if abs(DN) < tol:
            print("Reached convergence for number of particles "+float_string(N0.real, 5))
            print("----------------------------------------")
            break
        if DNsign!=last_sign and last_sign!=2:
            mu_jump /= 2
        mu += DNsign * mu_jump
        last_sign = DNsign
        
        print("Computed number of particles "+float_string(N0.real, 5))
        print("New mu="+float_string(mu, 5))
    
    return mu


@njit
def non_interactive_matsubara_kspace(N, mu, lattice, H0, H0_kin, Gk, Gloc, beta=1, particle=0, mu_jump=0.5, tol=1e-6):
    print("Estimating Matsubara branch for non-interactive case")
    ntau = Gloc.get_mat().shape[0]
    nkvec = len(Gk)

    last_sign = 2
    while True:
        newGlocM = np.zeros_like(Gloc.get_mat())
        for kk in range(nkvec):
            k_vec = lattice.get_vec(kk)
            Hk0 = H0 + 2*H0_kin[0]*np.cos(k_vec[0]) + 2*H0_kin[1]*np.cos(k_vec[1]) + 2*H0_kin[2]*np.cos(k_vec[2]) - mu*np.eye(H0.shape[0])
            assert not np.any(np.isnan(Hk0))
            assert np.all(np.abs(Hk0) < 1e14)
            Gk[kk].set_mat(g_nonint_init(ntau, -1, mu, Hk0, beta, particle)[0])
            newGlocM += Gk[kk].get_mat() / nkvec
        Gloc.set_mat(newGlocM)
        
        print("Non-interactive gloc approximated for mu="+float_string(mu, 5))
        
        N0 = -np.trace(Gloc.get_mat()[-1])
        DN = N - N0.real
        DNsign = sgn(DN)
        if abs(DN) < tol:
            break
        if DNsign!=last_sign and last_sign!=2:
            mu_jump /= 2
        mu += DNsign * mu_jump
        last_sign = DNsign

        print("Particle density of "+float_string(N0.real, 5))
    
    print("Computed non-interactive case with mu="+float_string(mu, 5))

    return mu


@njit
def matsubara_branch_init_gw0_kspace(N, mu, lattice, H0, H0_kin, Gk, Gloc, Pk, S, v, interpol, beta=1, particle=0, mu_jump=0.5, max_iter=100000, tol=1e-6):
    print("Initilizing Matsubara branch")
    ntau = Gloc.get_mat().shape[0]
    norb = Gloc.get_mat().shape[1]
    nkvec = len(Gk)
    particle_sign = (-1)**particle
    
    htau = beta / (ntau-1)

    mu = non_interactive_matsubara_kspace(N, mu, lattice, H0, H0_kin, Gk, Gloc, beta, particle, mu_jump)
    
    last_sign = 2
    while True:
        conv = 1e5
        loop_iterations = 0
        print("Starting Matsubara loop for mu="+float_string(mu, 5))
        while conv>=tol:
            
            S.set_hf_loc(0, -particle_sign * matrix_tensor(Gloc.get_mat()[-1], v-np.swapaxes(v, -1, -2)))
            # if not np.all(np.abs(S.get_hf()[0] - S.get_hf()[0].T.conjugate()) < 1e-15*np.ones(S.get_hf()[0].shape, dtype=np.float64)):
            #     print("Warning: equilibrium HF self-energy is not hermitian")
            assert not np.any(np.isnan(S.get_hf()[0]))
            for tt in range(ntau):
                # tempPk = []
                for qq in range(nkvec):
                    # tempPk.append(np.zeros((norb,norb,norb,norb), dtype=np.complex128))
                    for kk in range(nkvec):
                        k_plus_q = lattice.sum_indices(qq, kk)
                        Pk[qq].set_mat_loc(tt, -matrix_matrix2( Gk[k_plus_q].get_mat()[tt], Gk[kk].neg_imag_time_mat()[tt] ) )
                tempSloc = np.zeros((norb,norb), dtype=np.complex128)
                for kk in range(nkvec):
                    for qq in range(nkvec):
                        k_minus_q = lattice.diff_indices(kk, qq)
                        tempSloc += matrix_tensor( Gk[k_minus_q].get_mat()[tt], tensor_tensor( tensor_tensor ( v, Pk[qq].get_mat()[tt] ), v ) ) / nkvec**3
                S.set_mat_loc(tt, tempSloc)
                # if not np.all(np.abs(S.get_mat()[tt] - S.get_mat()[tt].T.conjugate()) < 1e-15*np.ones(S.get_mat()[tt].shape, dtype=np.float64)):
                #     print("Warning: Matsubara self-energy is not hermitian at time step", tt)
            print("Matsubara self-energy set")
            
            newGlocM = np.zeros_like(Gloc.get_mat())
            for kk in range(nkvec):
                # print("Initializing Dyson for k index ", kk)
                k_vec = lattice.get_vec(kk)
                HkMF = H0 + 2*H0_kin[0]*np.cos(k_vec[0]) + 2*H0_kin[1]*np.cos(k_vec[1]) + 2*H0_kin[2]*np.cos(k_vec[2]) - mu*np.eye(H0.shape[0]) + S.get_hf()[0]
                g,no_use = g_nonint_init(ntau, -1, mu, HkMF, beta, particle)
                # print("Mean field Green's function computed")
                F = conv_mat_extern(g, S.get_mat(), interpol, htau, matrix_matrix, particle)
                Gk[kk].set_mat(g + conv_mat_extern(F, Gk[kk].get_mat(), interpol, htau, matrix_matrix, particle))
                if np.any(np.abs(Gk[kk].get_mat()) > 1e14):
                    print("Warning: possible overload at k vector", kk)
                # print("Dyson equation computed")
                newGlocM += Gk[kk].get_mat() / nkvec
            # for tt in range(ntau):
            #     if not np.all(np.abs(Gloc.get_mat()[tt] - Gloc.get_mat()[tt].T.conjugate()) < 1e-15*np.ones(Gloc.get_mat()[tt].shape, dtype=np.float64)):
            #         print("Warning: Matsubara green's function is not hermitian at time step", tt)
            print("Gloc set")
            
            conv = gdist_mat(newGlocM, Gloc.get_mat())
            print("Convergence at "+float_string(conv,5)+" at iteration",loop_iterations+1)
            Gloc.set_mat(newGlocM)
            
            loop_iterations += 1
            assert loop_iterations <= max_iter
        
        print("Convergence for Matsubara branch and mu="+float_string(mu, 5))
        print("Norm "+exp_string(conv, 5)+"\n\n")
        
        N0 = -np.trace(Gloc.get_mat()[-1])
        DN = N - N0.real
        DNsign = sgn(DN)
        if abs(DN) < tol:
            print("Reached convergence for number of particles "+float_string(N0.real, 5))
            print("----------------------------------------")
            break
        if DNsign!=last_sign and last_sign!=2:
            mu_jump /= 2
        mu += DNsign * mu_jump
        last_sign = DNsign
        
        print("Computed number of particles "+float_string(N0.real, 5))
        print("New mu="+float_string(mu, 5))
    
    return mu



@njit
def matsubara_branch_init_hf_kspace(N, mu, lattice, H0, H0_kin, Gk, Gloc, S, v, interpol, beta=1, particle=0, mu_jump=0.5, max_iter=100000, tol=1e-6):
    print("Initilizing Matsubara branch")
    ntau = Gloc.get_mat().shape[0]
    # norb = Gloc.get_mat().shape[1]
    nkvec = len(Gk)
    particle_sign = (-1)**particle
    
    # htau = beta / (ntau-1)

    mu = non_interactive_matsubara_kspace(N, mu, lattice, H0, H0_kin, Gk, Gloc, beta, particle, mu_jump)
    
    last_sign = 2
    while True:
        conv = 1e5
        loop_iterations = 0
        print("Starting Matsubara loop for mu="+float_string(mu, 5))
        while conv>=tol:
            
            S.set_hf_loc(0, -particle_sign * matrix_tensor(Gloc.get_mat()[-1], v-np.swapaxes(v, -1, -2)))
            # print("Hartree-Fock self-energy set")
            
            newGlocM = np.zeros_like(Gloc.get_mat())
            for kk in range(nkvec):
                k_vec = lattice.get_vec(kk)
                HkMF = H0 + 2*H0_kin[0]*np.cos(k_vec[0]) + 2*H0_kin[1]*np.cos(k_vec[1]) + 2*H0_kin[2]*np.cos(k_vec[2]) - mu*np.eye(H0.shape[0]) + S.get_hf()[0]
                g,no_use = g_nonint_init(ntau, -1, mu, HkMF, beta, particle)
                Gk[kk].set_mat(g)
                newGlocM += Gk[kk].get_mat() / nkvec
            # print("Gloc set")
            
            conv = gdist_mat(newGlocM, Gloc.get_mat())
            print("Convergence at "+exp_string(conv,5)+" at iteration",loop_iterations+1)
            Gloc.set_mat(newGlocM)
            
            loop_iterations += 1
            assert loop_iterations <= max_iter
        
        print("Convergence for Matsubara branch and mu="+float_string(mu, 5))
        print("Norm "+exp_string(conv, 5)+"\n\n")
        
        N0 = -np.trace(Gloc.get_mat()[-1])
        DN = N - N0.real
        DNsign = sgn(DN)
        if abs(DN) < tol:
            print("Reached convergence for number of particles "+float_string(N0.real, 5))
            print("----------------------------------------")
            break
        if DNsign!=last_sign and last_sign!=2:
            mu_jump /= 2
        mu += DNsign * mu_jump
        last_sign = DNsign
        
        print("Computed number of particles "+float_string(N0.real, 5))
        print("New mu="+float_string(mu, 5))
    
    return mu
