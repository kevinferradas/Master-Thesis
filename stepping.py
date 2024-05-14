import numpy as np
from numba import njit
from .vide import vide_step, vie_step
from .contour_funcs import matrix_matrix, matrix_matrix2, matrix_tensor, tensor_tensor, copy_gmatrix
from .convolution import conv_lmx_res, conv_les_res
from .convolution import conv_lmx2, conv_lmx3, conv_les2, conv_les3
from .time_evolution import time_evolution_step
from .utils import gdist_real_time
from .printing import float_string, exp_string
from .observables import kinetic_energy, particle_density, local_energy, interaction_energy, current_vector



@njit
def step_gret(t, G, S, H, mu, h, interpol):
    norb = G.get_mat()[-1]
    Gaux = G.get_ret_comp()[:t+1,:t+1]
    Hhf = H + S.get_hf()[t] - mu * np.eye(norb, dtype=np.complex128)
    for jj in range(t):
        G.set_ret_loc(t, jj, vide_step(interpol, h, 1j*Hhf, -1j*np.zeros_like(G.get_mat()[0]), 1j*S.get_ret(), Gaux[:t,jj]))
    G.set_ret_loc(t, t, -1j*np.eye(norb))


@njit
def step_glmx(t, G, S, H, mu, h, interpol):
    ntau, norb = G.get_mat()[:-1]
    Hhf = H + S.get_hf()[t] - mu * np.eye(norb, dtype=np.complex128)
    for jj in range(ntau):
        Q = conv_lmx2(t, jj, S, G, interpol, h, matrix_matrix) + conv_lmx3(t, jj, S, G, interpol, h, matrix_matrix)
        G.set_lmx_loc(t, jj, vide_step(interpol, h, 1j*Hhf, -1j*Q, 1j*S.get_ret(), G.get_lmx()[:t,jj]))


@njit
def step_gles(t, G, S, Q, H, mu, h, interpol):
    norb = G.get_mat()[-1]
    Hhf = H + S.get_hf()[t] - mu * np.eye(norb, dtype=np.complex128)
    for jj in range(t+1):
        Q = conv_les2(t, jj, S, G, interpol, h, matrix_matrix) + conv_les3(t, jj, S, G, interpol, h, matrix_matrix)
        G.set_les_loc(t, jj, vide_step(interpol, h, 1j*Hhf, -1j*Q.get_les()[:,jj], 1j*S.get_ret(), G.get_les()[:t,jj]))
        if jj < t:
            G.set_les_loc(jj, t, -G.get_les()[t, jj].T.conjugate())


@njit
def step_wret(t, W, vP, E, interpol):
    Waux = W.get_ret_comp()[:t+1,:t+1]
    for jj in range(t):
        W.set_ret_loc(t, jj, vie_step(interpol, E.get_ret()[:,jj], -vP.get_ret(), Waux[:t,jj]))
    W.set_ret_loc(t, t, E.get_ret()[t,t])


@njit
def step_wlmx(t, W, vP, E, interpol):
    ntau = W.get_mat().shape[0]
    for jj in range(ntau):
        W.set_lmx_loc(t, jj, vide_step(interpol, E.get_lmx[:,jj], -vP.get_ret(), W.get_lmx()[:t,jj]))


@njit
def step_wles(t, W, vP, E, interpol):
    for jj in range(t+1):
        W.set_les_loc(t, jj, vie_step(interpol, E.get_les()[:,jj], -vP.get_ret(), W.get_les()[:t,jj]))
        if jj < t:
            W.set_les_loc(jj, t, -np.swapaxes(np.swapaxes(W.get_les()[t,jj].conjugate(), 0, 3), 1, 2))



@njit
def step_loop_gw(t, G, S, vP, W, Q, E, H, v, mu, interpol, h, tol=1e-6, max_iter=100000):
    print("Starting convergence loop for time step "+str(t))
    ntau, norb = G.get_mat().shape[:-1]
    
    #G initial conditions
    G.set_ret_loc(t, t, -1j*np.eye(norb))
    #Init for GI already set
    
    conv = 1e5
    loop_iterations = 0
    while conv > tol:
        #Update bubbles
        for ss in range(t+1):
            vP.set_ret_loc(t, ss, -1j*tensor_tensor(v, matrix_matrix2(G.get_ret()[t,ss], G.get_les()[ss,t]) +
                                                       matrix_matrix2(G.get_les()[t,ss], G.get_adv()[ss,t])))
            vP.set_les_loc(t, ss, -1j*tensor_tensor(v, matrix_matrix2(G.get_les()[t,ss], G.get_gtr()[ss,t])))
            if ss < t:
                vP.set_les_loc(ss, t, -np.swapaxes(np.swapaxes(W.get_les()[t,ss].conjugate(), 0, 3), 1, 2))
        for kk in range(ntau):
            vP.set_lmx(t, kk, -1j*tensor_tensor(v, matrix_matrix2(G.get_lmx()[t,kk], G.get_rmx()[kk,t])))
        
        #Step W
        conv_lmx_res(t, vP, W, E, interpol, h, tensor_tensor)
        for kk in range(ntau):
            E.add_to_lmx_loc(t, kk, tensor_tensor(vP.get_lmx()[t,kk], v))
            #Init for WI already set
        for ss in range(t+1):
            E.set_ret_loc(t, ss, tensor_tensor(vP.get_ret()[t, ss], v))
            if ss==t:
                W.set_ret_loc(t, t, E.get_ret()[t,t])
            conv_les_res(t, ss, vP, W, E, interpol, h, tensor_tensor)
            E.add_to_les_loc(t, ss, tensor_tensor(vP.get_les()[t,ss], v))
            #Init for WL not necessary because of c.h.
            if ss < t:
                E.set_les_loc(ss, t, -np.swapaxes(np.swapaxes(E.get_les()[t,ss].conjugate(), 0, 3), 1, 2))
        step_wret(t, W, vP, E, interpol)
        step_wlmx(t, W, vP, E, interpol)
        step_wles(t, W, vP, E, interpol)
        
        #Update S
        S.set_hf_loc(t, matrix_tensor(G.get_les()[t,t], v - np.swapaxes(v, -1, -2)))
        for ss in range(interpol.k+1):
            S.set_ret_loc(t, ss, 1j*matrix_tensor(G.get_ret()[t,ss], W.get_gtr()[t,ss]) +
                                 1j*matrix_tensor(G.get_les()[t,ss], W.get_ret()[t,ss]))
            S.set_les_loc(t, ss, 1j*matrix_tensor(G.get_les()[t,ss], W.get_les()[t,ss]))
            if ss < t:
                S.set_les_loc(ss, t, -S.get_les()[t,ss].T.conjugate())
        for kk in range(ntau):
            S.set_lmx_loc(t, kk, 1j*matrix_tensor(G.get_lmx()[t,kk], W.get_lmx()[t,kk]))
        
        #Step G
        lastG = copy_gmatrix(G)
        conv_lmx_res(t, S, G, Q, interpol, h, matrix_matrix)
        for ss in range(interpol.k+1):
            conv_les_res(t, ss, S, G, Q, interpol, h, matrix_matrix)
        step_gret(G, S, Q, H, mu, interpol)
        step_glmx(G, S, Q, H, mu, interpol)
        #Init for GL not necessary because of c.h.
        step_gles(G, S, Q, H, mu, interpol)
        
        # Convergence
        conv = gdist_real_time(t, lastG, G)
        loop_iterations += 1
        
        # Reched max iterations
        assert loop_iterations < max_iter
    
    print("Convergence for time step "+str(t)+" reached")
    print("Norm "+exp_string(conv, 5)+"\n\n")


@njit
def step_loop_gw0(t, G, S, Q, H, v, mu, interpol, h, tol=1e-6, max_iter=100000):
    print("Starting convergence loop for time step "+str(t))
    ntau, norb = G.get_mat().shape[:-1]
    
    #G initial conditions
    G.set_ret_loc(t, t, -1j*np.eye(norb))
    #Init for GI already set
    
    conv = 1e5
    loop_iterations = 0
    while conv > tol:
        
        #Update S
        S.set_hf_loc(t, matrix_tensor(G.get_les()[t,t], v - np.swapaxes(v, -1, -2)))
        for ss in range(t+1):
            S.set_ret_loc(t, ss, matrix_tensor(G.get_ret()[t,ss], tensor_tensor(v, tensor_tensor(matrix_matrix2(G.get_gtr()[t,ss], G.get_les()[ss,t]), v))) +
                                  matrix_tensor(G.get_les()[t,ss], tensor_tensor(v, tensor_tensor(matrix_matrix2(G.get_ret()[t,ss], G.get_les()[ss,t]), v))) +
                                  matrix_tensor(G.get_les()[t,ss], tensor_tensor(v, tensor_tensor(matrix_matrix2(G.get_les()[t,ss], G.get_adv()[ss,t]), v))) )
            S.set_les_loc(t, ss, matrix_tensor(G.get_les()[t,ss], tensor_tensor(v, tensor_tensor(matrix_matrix2(G.get_les()[t,ss], G.get_gtr()[ss,t]), v))) )
            if ss < t:
                S.set_les_loc(ss, t, -S.get_les()[t,ss].T.conjugate())
        for kk in range(ntau):
            S.set_lmx_loc(t, kk, matrix_tensor(G.get_lmx()[t,kk], tensor_tensor(v, tensor_tensor(matrix_matrix2(G.get_lmx()[t,kk], G.get_rmx()[kk,t]), v))))
        
        #Step G
        lastG = copy_gmatrix(G)
        conv_lmx_res(t, S, G, Q, interpol, h, matrix_matrix)
        for ss in range(interpol.k+1):
            conv_les_res(t, ss, S, G, Q, interpol, h, matrix_matrix)
        step_gret(G, S, Q, H, mu, interpol)
        step_glmx(G, S, Q, H, mu, interpol)
        #Init for GL not necessary because of c.h.
        step_gles(G, S, Q, H, mu, interpol)
        
        # Convergence
        conv = gdist_real_time(t, lastG, G)
        loop_iterations += 1
        
        # Reched max iterations
        assert loop_iterations < max_iter
    
    print("Convergence for time step "+str(t)+" reached")
    print("Norm "+exp_string(conv, 5)+"\n\n")


@njit
def step_loop_gw0_kspace(t, lattice, Gk, Gloc, Pk, S, Q, H, H_kin, v, mu, interpol, h, tol=1e-6, max_iter=100000):
    print("Starting convergence loop for time step "+str(t))
    ntau, norb = Gloc.get_mat().shape[:-1]
    nkvec = len(Gk)
    
    #G initial conditions
    for kk in range(nkvec):
        Gk[kk].set_ret_loc(t, t, -1j*np.eye(norb))
    #Init for GI already set
    
    conv = 1e5
    loop_iterations = 0
    while conv > tol:
        
        # #Update S
        # S.set_hf_loc(t, matrix_tensor(Gloc.get_les()[t,t], v - np.swapaxes(v, -1, -2)))
        # for k1 in range(nkvec):
        #     for k2 in range(nkvec):
        #         k4 = lattice.diff_indices(k1, k2)
        #         for k3 in range(nkvec):
        #             k5 = lattice.sum_indices(k3, k2)
        #             for ss in range(t+1):
        #                 S.add_to_ret_loc(t, ss, matrix_tensor(Gk[k4].get_ret()[t,ss], tensor_tensor(v, tensor_tensor(matrix_matrix2(Gk[k5].get_gtr()[t,ss], Gk[k3].get_les()[ss,t]), v)))/nkvec +
        #                                         matrix_tensor(Gk[k4].get_les()[t,ss], tensor_tensor(v, tensor_tensor(matrix_matrix2(Gk[k5].get_ret()[t,ss], Gk[k3].get_les()[ss,t]), v)))/nkvec +
        #                                         matrix_tensor(Gk[k4].get_les()[t,ss], tensor_tensor(v, tensor_tensor(matrix_matrix2(Gk[k5].get_les()[t,ss], Gk[k3].get_adv()[ss,t]), v)))/nkvec )
        #                 S.add_to_les_loc(t, ss, matrix_tensor(Gk[k4].get_les()[t,ss], tensor_tensor(v, tensor_tensor(matrix_matrix2(Gk[k5].get_les()[t,ss], Gk[k3].get_gtr()[ss,t]), v)))/nkvec )
        #                 if ss < t:
        #                     S.set_les_loc(ss, t, -S.get_les()[t,ss].T.conjugate())
        #             for kk in range(ntau):
        #                 S.add_to_lmx_loc(t, kk, matrix_tensor(Gk[k4].get_lmx()[t,kk], tensor_tensor(v, tensor_tensor(matrix_matrix2(Gk[k5].get_lmx()[t,kk], Gk[k3].get_rmx()[kk,t]), v)))/nkvec )
        
        # Update S
        S.set_hf_loc(t, matrix_tensor(Gloc.get_les()[t,t], v - np.swapaxes(v, -1, -2)))
        for qq in range(nkvec):
            # tempPk.append(np.zeros((norb,norb,norb,norb), dtype=np.complex128))
            for kk in range(nkvec):
                k_plus_q = lattice.sum_indices(qq, kk)
                for ss in range(t+1):
                    Pk[qq].set_ret_loc(t, ss, -1j*matrix_matrix2( Gk[k_plus_q].get_ret()[t,ss], Gk[kk].get_les()[ss,t] )
                                               -1j*matrix_matrix2( Gk[k_plus_q].get_les()[t,ss], Gk[kk].get_adv()[ss,t] ) )
                    Pk[qq].set_les_loc(t, ss, -1j*matrix_matrix2( Gk[k_plus_q].get_les()[t,ss], Gk[kk].get_gtr()[ss,t] ) )
                    if ss < t:
                        Pk[qq].set_les_loc(ss, t, -np.swapaxes(np.swapaxes(Pk[qq].get_les()[ss,t], 1, 2), 0, 3))
                for rr in range(ntau):
                    Pk[qq].set_lmx_loc(t, rr, -1j*matrix_matrix2( Gk[k_plus_q].get_lmx()[t,rr], Gk[kk].get_rmx()[rr,t] ) )
        tempSlocR = np.zeros((t+1,norb,norb), dtype=np.complex128)
        tempSlocL = np.zeros((t+1,norb,norb), dtype=np.complex128)
        tempSlocI = np.zeros((t+1,ntau,norb,norb), dtype=np.complex128)
        for kk in range(nkvec):
            for qq in range(nkvec):
                k_minus_q = lattice.diff_indices(kk, qq)
                for ss in range(t+1):
                    tempSlocR[ss] += 1j*(matrix_tensor( Gk[k_minus_q].get_les()[t,ss], tensor_tensor( tensor_tensor ( v, Pk[qq].get_ret()[t,ss] ), v ) ) +
                                           matrix_tensor( Gk[k_minus_q].get_ret()[t,ss], tensor_tensor( tensor_tensor ( v, Pk[qq].get_gtr()[t,ss] ), v ) ) ) / nkvec**3
                    tempSlocL[ss] += 1j* matrix_tensor( Gk[k_minus_q].get_les()[t,ss], tensor_tensor( tensor_tensor ( v, Pk[qq].get_les()[t,ss] ), v ) ) / nkvec**3
                for rr in range(ntau):
                    tempSlocI[rr] += 1j* matrix_tensor( Gk[k_minus_q].get_lmx()[t,rr], tensor_tensor( tensor_tensor ( v, Pk[qq].get_lmx()[t,rr] ), v ) ) / nkvec**3
        for ss in range(t+1):
            S.set_ret_loc(t, ss, tempSlocR[ss])
            S.set_les_loc(t, ss, tempSlocL[ss])
            if ss < t:
                S.set_les_loc(ss, t, -S.get_les()[t,ss].T.conjugate())
        for rr in range(ntau):
            S.set_lmx_loc(t, rr, tempSlocI[rr])
        
        #Step G
        lastG = copy_gmatrix(Gloc)
        for kk in range(nkvec):
            k_vec = lattice.get_vec(kk)
            Hk = H[t] + 2*H_kin[0,t]*np.cos(k_vec[0]) + 2*H_kin[1,t]*np.cos(k_vec[1]) + 2*H_kin[2,t]*np.cos(k_vec[2])
            step_gret(Gk[kk], S, Q, Hk, mu, h, interpol)
            step_glmx(Gk[kk], S, Q, Hk, mu, h, interpol)
            #Init for GL not necessary because of c.h.
            step_gles(Gk[kk], S, Q, Hk, mu, h, interpol)
        
        # Convergence
        conv = gdist_real_time(t, lastG, Gloc)
        loop_iterations += 1
        
        # Reched max iterations
        assert loop_iterations < max_iter
    
    print("Convergence for time step "+str(t)+" reached")
    print("Norm "+exp_string(conv, 5)+"\n\n")


@njit
def step_loop_hf_kspace(lattice, Gk, Gloc, S, H, H_kin, v, mu, interpol, h, obsv_map, tol=1e-6, max_iter=100000):
    n = Gloc.get_ret().shape[0]
    # obsv = np.zeros((n,16), dtype=np.float64)
    for t in range(interpol.k+1,n):
        print("Starting convergence loop for time step "+str(t))
        ntau, norb = Gloc.get_mat().shape[:-1]
        nkvec = len(Gk)
        
        #G initial conditions
        for kk in range(nkvec):
            Gk[kk].set_ret_loc(t, t, -1j*np.eye(norb))
        #Init for GI already set
        
        # Numba bug avoids definition of new arrays out of first execution
        # Defiition of loop auxiliary empty arrays
        # k_vec = np.empty((3,), dtype=np.float64)
        # Hk = np.empty((interpol.k+1,norb,norb), dtype=np.complex128)
        # time_evo_k = np.empty((norb,norb), dtype=np.complex128)
        
        conv = 1e5
        loop_iterations = 0
        while conv > tol:
            
            
            # Update S
            S.set_hf_loc(t, 1j*matrix_tensor(Gloc.get_les()[t,t], v - np.swapaxes(v, -1, -2)))
            assert not np.any(np.isnan(S.get_hf()[t]))
            assert not np.any(np.isinf(S.get_hf()[t]))
            # print(S.get_hf()[t])
            
            #Step G
            lastG = copy_gmatrix(Gloc)
            for kk in range(nkvec):
                # print("k vec", kk)
                k_vec = lattice.get_vec(kk)
                # print("kvec set")
                Hk = 2*H_kin[0,t-interpol.k:t+1]*np.cos(k_vec[0]) + 2*H_kin[1,t-interpol.k:t+1]*np.cos(k_vec[1]) + 2*H_kin[2,t-interpol.k:t+1]*np.cos(k_vec[2]) + S.get_hf()[t-interpol.k:t+1]
                # print("Hamiltonian set")
                assert not np.any(np.isnan(Hk))
                assert not np.any(np.isinf(Hk))
                for uu in range(interpol.k+1):
                    Hk[uu] += H - mu * np.eye(norb)
                # print("Hamiltonian corrected")
                # print(Hk)
                assert not np.any(np.isnan(Hk))
                assert not np.any(np.isinf(Hk))
                time_evo_k = time_evolution_step(Hk, interpol, h)
                # print("Time evo set")
                assert not np.any(np.isnan(time_evo_k))
                assert not np.any(np.isinf(time_evo_k))
                for rr in range(ntau):
                    Gk[kk].set_lmx_loc(t, rr, matrix_matrix(time_evo_k, Gk[kk].get_lmx()[t-1,rr]))
                    if kk==0:
                        Gloc.set_lmx_loc(t, rr, Gk[kk].get_lmx()[t,rr] / nkvec)
                    else:
                        Gloc.add_to_lmx_loc(t, rr, Gk[kk].get_lmx()[t,rr] / nkvec)
                for ss in range(t+1):
                    if ss < t:
                        Gk[kk].set_ret_loc(t, ss, matrix_matrix(time_evo_k, Gk[kk].get_ret()[t-1,ss]))
                    Gk[kk].set_les_loc(t, ss, matrix_matrix(time_evo_k, Gk[kk].get_les()[t-1,ss]))
                    if ss < t:
                        Gk[kk].set_les_loc(ss, t, -Gk[kk].get_les()[t, ss].T.conjugate())
                    if kk==0:
                        Gloc.set_ret_loc(t, ss, Gk[kk].get_ret()[t,ss] / nkvec)
                        Gloc.set_les_loc(t, ss, Gk[kk].get_les()[t,ss] / nkvec)
                    else:
                        Gloc.add_to_ret_loc(t, ss, Gk[kk].get_ret()[t,ss] / nkvec)
                        Gloc.add_to_les_loc(t, ss, Gk[kk].get_les()[t,ss] / nkvec)
            for ss in range(t):
                Gloc.set_les_loc(ss, t, -Gloc.get_les()[t,ss].T.conjugate())
            
            # print(Gloc.get_ret()[t,:t+1,0,0])
            # print(Gloc.get_les()[t,:t+1,0,0])
            # print(Gloc.get_lmx()[t,:,0,0])
            # Convergence
            conv = gdist_real_time(t, lastG, Gloc)
            loop_iterations += 1
            print("Convergence at "+exp_string(conv,5)+" at iteration",loop_iterations)
            
            # Reched max iterations
            assert loop_iterations < max_iter
        
        # Printing out observables
        Ndens = particle_density(t, Gloc)
        Eloc = local_energy(t, mu, H + S.get_hf()[t], Gloc)
        Ekin = kinetic_energy(t, H_kin, Gk, lattice)
        Etot = Eloc + Ekin
        jvec = current_vector(t, H_kin, Gk, lattice)
        obsv_map[t,0] = t
        obsv_map[t,1] = h*t
        obsv_map[t,2] = Ndens.real
        obsv_map[t,3] = Ndens.imag
        obsv_map[t,4] = Eloc.real
        obsv_map[t,5] = Eloc.imag
        obsv_map[t,6] = Ekin.real
        obsv_map[t,7] = Ekin.imag
        obsv_map[t,8] = Etot.real
        obsv_map[t,9] = Etot.imag
        obsv_map[t,10] = jvec[0].real
        obsv_map[t,11] = jvec[0].imag
        obsv_map[t,12] = jvec[1].real
        obsv_map[t,13] = jvec[1].imag
        obsv_map[t,14] = jvec[2].real
        obsv_map[t,15] = jvec[2].imag
        # np.savetxt("observables", obsv)
        
        print("Convergence for time step "+str(t)+" reached")
        print("Norm "+exp_string(conv, 5)+"\n\n")
        # print(S.get_hf()[t])
        # print(Gloc.get_ret()[t,:t+1,0,0])
        # print(Gloc.get_les()[t,:t+1,0,0])
        # print(Gloc.get_les()[t,:t+1,1,1])
