import numpy as np
from numba import njit
from .vide import vide_startR, vide_start, vie_startR, vie_start
from .contour_funcs import matrix_matrix, matrix_matrix2, matrix_tensor, tensor_tensor, copy_gmatrix
from .convolution import conv_lmx_res, conv_les_res
from .convolution import conv_lmx2, conv_lmx3, conv_les2, conv_les3
from .time_evolution import time_evolution_boot
from .utils import gdist_real_time
from .printing import float_string, exp_string
from .observables import kinetic_energy, particle_density, local_energy, interaction_energy, current_vector



@njit
def boot_gret(G, S, H, mu, h, interpol):
    norb = G.get_ret().shape[-1]
    Gaux = G.get_ret_comp()[:interpol.k+1,:interpol.k+1]
    Hhf = H[:interpol.k+1] + S.get_hf()[:interpol.k+1]
    for tt in range(Hhf.shape[0]):
        Hhf -= mu * np.eye(norb)
    for tt in range(interpol.k+1):
        for oo in range(norb):
            Gaux[tt,tt,oo,oo] = -1j
    
    for tt in range(interpol.k+1):
        Gaux[tt+1:,tt] = vide_startR(interpol, h, 1j*Hhf, np.zeros_like(Hhf), 1j*S.get_ret(), Gaux[:tt+1,tt])
        Gaux[tt,tt+1:] = -np.swapaxes(Gaux[tt+1:,tt].conjugate(), -1, -2)
        for ss in range(tt, interpol.k+1):
            G.set_ret_loc(ss, tt, Gaux[ss,tt])


@njit
def boot_glmx(G, S, H, mu, h, interpol):
    ntau = G.get_mat().shape[0]
    norb = G.get_mat().shape[1]
    Hhf = H[:interpol.k+1] + S.get_hf()[:interpol.k+1]
    for tt in range(Hhf.shape[0]):
        Hhf -= mu * np.eye(norb)
    for tt in range(ntau):
        Q = np.empty((interpol.k+1,norb,norb), dtype=np.complex128)
        for rr in range(interpol.k+1):
            Q[rr] = conv_lmx2(rr, tt, S, G, interpol, h, matrix_matrix) + conv_lmx3(rr, tt, S, G, interpol, h, matrix_matrix)
        bootGI = vide_start(interpol, h, 1j*Hhf, -1j*Q, 1j*S.get_ret(), G.get_lmx()[0,tt])
        for ss in range(1,interpol.k+1):
            G.set_lmx_loc(ss, tt, bootGI[ss-1])


@njit
def boot_gles(G, S, H, mu, h, interpol):
    norb = G.get_mat().shape[1]
    Hhf = H[:interpol.k+1] + S.get_hf()[:interpol.k+1]
    for tt in range(Hhf.shape[0]):
        Hhf -= mu * np.eye(norb)
    for tt in range(interpol.k+1):
        Q = np.empty((interpol.k+1,norb,norb), dtype=np.complex128)
        for rr in range(interpol.k+1):
            Q[rr] = conv_les2(rr, tt, S, G, interpol, h, matrix_matrix) + conv_les3(rr, tt, S, G, interpol, h, matrix_matrix)
        bootGL = vide_start(interpol, h, 1j*Hhf, -1j*Q, 1j*S.get_ret(), G.get_les()[0,tt])
        for ss in range(1, interpol.k+1):
            G.set_les_loc(ss, tt, bootGL[ss-1])



@njit
def boot_wret(W, vP, E, interpol):
    norb = W.get_ret().shape[-1]
    Waux = np.zeros((interpol.k+1,interpol.k+1,norb,norb,norb,norb))
    for tt in range(interpol.k+1):
        Waux[tt,tt] = E.get_ret()[tt,tt]
        Waux[tt+1:,tt] = vie_startR(interpol, E.get_ret()[:,tt], -vP.get_ret(), Waux[:tt+1,tt])
        Waux[tt,tt+1:] = -np.swapaxes(np.swapaxes(Waux[tt+1:,tt].conjugate(), -2, -3), -1, -4)
        for ss in range(tt, interpol.k+1):
            W.set_ret_loc(ss, tt, Waux[ss,tt])


@njit
def boot_wlmx(W, vP, E, interpol):
    ntau = W.get_mat().shape[0]
    for tt in range(ntau):
        bootWI = vie_start(interpol, E.get_lmx()[:,tt], -vP.get_ret(), W.get_lmx[0,tt])
        for ss in range(1,interpol.k+1):
            W.set_lmx_loc(ss, tt, bootWI[ss-1,tt])


@njit
def boot_wles(W, vP, E, interpol):
    for tt in range(interpol.k+1):
        bootWL = vie_start(interpol, E.get_les()[:,tt], -vP.get_ret(), W.get_les[0,tt])
        for ss in range(1,interpol.k+1):
            W.set_les_loc(ss, tt, bootWL[ss-1,tt])




@njit
def boot_loop_gw(G, S, vP, W, Q, E, H, v, mu, interpol, h, tol=1e-6, max_iter=100000):
    print("Starting bootstrapping loop")
    ntau, norb = G.get_mat().shape[:-1]
    particle_sign = (-1)**G.particle_type
    
    #G initial conditions
    for tt in range(interpol.k+1):
        G.set_ret_loc(tt, tt, -1j*np.eye(norb))
    for rr in range(ntau):
        G.set_lmx_loc(0, rr, 1j * particle_sign * G.get_mat()[-1-rr])
    
    conv = 1e5
    loop_iterations = 0
    while conv>tol:
        # Update bubbles
        for tt in range(interpol.k+1):
            for ss in range(interpol.k+1):
                vP.set_ret_loc(tt, ss, -1j*tensor_tensor(v, matrix_matrix2(G.get_ret()[tt,ss], G.get_les()[ss,tt]) +
                                                            matrix_matrix2(G.get_les()[tt,ss], G.get_adv()[ss,tt])))
                vP.set_les_loc(tt, ss, -1j*tensor_tensor(v, matrix_matrix2(G.get_les()[tt,ss], G.get_gtr()[ss,tt])))
            for rr in range(ntau):
                vP.set_lmx_loc(tt, rr, -1j*tensor_tensor(v, matrix_matrix2(G.get_lmx()[tt,rr], G.get_rmx()[rr,tt])))
        
        # Boot W
        for tt in range(interpol.k+1):
            conv_lmx_res(tt, vP, W, E, interpol, h, tensor_tensor)
            for rr in range(ntau):
                E.add_to_lmx_loc(tensor_tensor(v, vP.get_lmx()[tt,rr]))
                if tt==0:
                    W.set_lmx_loc(0, rr, E.get_lmx()[0,rr])
            for ss in range(interpol.k+1):
                E.set_ret_loc(tt, ss, tensor_tensor(vP.get_ret()[tt,ss], v))
                if ss==tt:
                    W.set_ret_loc(ss, ss, E.get_ret()[ss,ss])
                conv_les_res(tt, ss, vP, W, E, interpol, h, tensor_tensor)
                E.add_to_les_loc(tensor_tensor(v, vP.get_les()[tt,ss]))
                if tt==0:
                    W.set_les_loc(0, ss, E.get_les()[0, ss])
        boot_wret(W, vP, E, interpol)
        boot_wlmx(W, vP, E, interpol)
        boot_wles(W, vP, E, interpol)
        
        # Update S
        for tt in range(interpol.k+1):
            S.set_hf_loc(tt, matrix_tensor(G.get_les()[tt,tt], v - np.swapaxes(v, -1, -2)))
            for ss in range(interpol.k+1):
                S.set_ret_loc(tt, ss, 1j*matrix_tensor(G.get_ret()[tt,ss], W.get_gtr()[tt,ss]) +
                                      1j*matrix_tensor(G.get_les()[tt,ss], W.get_ret()[tt,ss]))
                S.set_les_loc(tt, ss, 1j*matrix_tensor(G.get_les()[tt,ss], W.get_les()[tt,ss]))
            for rr in range(ntau):
                S.set_lmx_loc(tt, rr, 1j*matrix_tensor(G.get_lmx()[tt,rr], W.get_lmx()[tt,rr]))
        
        # Boot G
        lastG = copy_gmatrix(G)
        for tt in range(interpol.k+1):
            conv_lmx_res(tt, S, G, Q, interpol, h, matrix_matrix)
            for ss in range(interpol.k+1):
                conv_les_res(tt, ss, S, G, Q, interpol, h, matrix_matrix)
        boot_gret(G, S, H, mu, interpol)
        boot_glmx(G, S, Q, H, mu, interpol)
        for tt in range(interpol.k+1):
            G.set_les_loc(0, tt, G.get_rmx()[0,tt])
        boot_gles(G, S, Q, H, mu, interpol)
        
        # Convergence
        conv = gdist_real_time(interpol.k, lastG, G)
        loop_iterations += 1
        
        # Reched max iterations
        assert loop_iterations < max_iter
    
    print("Convergence for boostrap region reached")
    print("Norm "+exp_string(conv, 5)+"\n\n")


@njit
def boot_loop_gw0(G, S, Q, H, v, mu, interpol, h, tol=1e-6, max_iter=100000):
    print("Starting bootstrapping loop")
    ntau, norb = G.get_mat().shape[:-1]
    particle_sign = (-1)**G.particle_type
    
    #G initial conditions
    for tt in range(interpol.k+1):
        G.set_ret_loc(tt, tt, -1j*np.eye(norb))
    for rr in range(ntau):
        G.set_lmx_loc(0, rr, 1j * particle_sign * G.get_mat()[-1-rr])
    
    conv = 1e5
    loop_iterations = 0
    while conv>tol:
        
        # Update S
        for tt in range(interpol.k+1):
            S.set_hf_loc(tt, matrix_tensor(G.get_les()[tt,tt], v - np.swapaxes(v, -1, -2)))
            for ss in range(interpol.k+1):
                S.set_ret_loc(tt, ss, matrix_tensor(G.get_ret()[tt,ss], tensor_tensor(v, tensor_tensor(matrix_matrix2(G.get_gtr()[tt,ss], G.get_les()[ss,tt]), v))) +
                                      matrix_tensor(G.get_les()[tt,ss], tensor_tensor(v, tensor_tensor(matrix_matrix2(G.get_ret()[tt,ss], G.get_les()[ss,tt]), v))) +
                                      matrix_tensor(G.get_les()[tt,ss], tensor_tensor(v, tensor_tensor(matrix_matrix2(G.get_les()[tt,ss], G.get_adv()[ss,tt]), v))) )
                S.set_les_loc(tt, ss, matrix_tensor(G.get_les()[tt,ss], tensor_tensor(v, tensor_tensor(matrix_matrix2(G.get_les()[tt,ss], G.get_gtr()[ss,tt]), v))) )
            for rr in range(ntau):
                S.set_lmx_loc(tt, rr, matrix_tensor(G.get_lmx()[tt,rr], tensor_tensor(v, tensor_tensor(matrix_matrix2(G.get_lmx()[tt,rr], G.get_rmx()[rr,tt]), v))))
        
        # Boot G
        lastG = copy_gmatrix(G)
        for tt in range(interpol.k+1):
            conv_lmx_res(tt, S, G, Q, interpol, h, matrix_matrix)
            for ss in range(interpol.k+1):
                conv_les_res(tt, ss, S, G, Q, interpol, h, matrix_matrix)
        boot_gret(G, S, H, mu, interpol)
        boot_glmx(G, S, Q, H, mu, interpol)
        for tt in range(interpol.k+1):
            G.set_les_loc(0, tt, G.get_rmx()[0,tt])
        boot_gles(G, S, Q, H, mu, interpol)
        
        # Convergence
        conv = gdist_real_time(interpol.k, lastG, G)
        loop_iterations += 1
        
        # Reched max iterations
        assert loop_iterations < max_iter
    
    print("Convergence for boostrap region reached")
    print("Norm "+exp_string(conv, 5)+"\n\n")


@njit
def boot_loop_gw0_kspace(lattice, Gk, Gloc, Pk, S, H, H_kin, v, mu, interpol, h, tol=1e-6, max_iter=100000):
    print("Starting bootstrapping loop")
    ntau, norb = Gloc.get_mat().shape[:-1]
    nkvec = len(Gk)
    particle_sign = (-1)**Gloc.particle_type
    
    #G initial conditions
    for kk in range(nkvec):
        for tt in range(interpol.k+1):
            Gk[kk].set_ret_loc(tt, tt, -1j*np.eye(norb))
        for rr in range(ntau):
            Gk[kk].set_lmx_loc(0, rr, 1j * particle_sign * Gk[kk].get_mat()[-1-rr])
    
    conv = 1e5
    loop_iterations = 0
    while conv>tol:
        
        # # Update S
        # S.init_ret()
        # S.init_lmx()
        # S.init_les()
        # for tt in range(interpol.k+1):
        #     print("Updating self-energy for time step", tt)
        #     S.set_hf_loc(tt, matrix_tensor(Gloc.get_les()[tt,tt], v - np.swapaxes(v, -1, -2)))
        #     for k1 in range(nkvec):
        #         print("Updating self-energy for k vector", k1)
        #         for k2 in range(nkvec):
        #             print("k2", k2)
        #             k4 = lattice.diff_indices(k1, k2)
        #             for k3 in range(nkvec):
        #                 k5 = lattice.sum_indices(k3, k2)
        #                 for ss in range(interpol.k+1):
        #                     S.add_to_ret_loc(tt, ss, matrix_tensor(Gk[k4].get_ret()[tt,ss], tensor_tensor(v, tensor_tensor(matrix_matrix2(Gk[k5].get_gtr()[tt,ss], Gk[k3].get_les()[ss,tt]), v)))/nkvec +
        #                                              matrix_tensor(Gk[k4].get_les()[tt,ss], tensor_tensor(v, tensor_tensor(matrix_matrix2(Gk[k5].get_ret()[tt,ss], Gk[k3].get_les()[ss,tt]), v)))/nkvec +
        #                                              matrix_tensor(Gk[k4].get_les()[tt,ss], tensor_tensor(v, tensor_tensor(matrix_matrix2(Gk[k5].get_les()[tt,ss], Gk[k3].get_adv()[ss,tt]), v)))/nkvec )
        #                     S.add_to_les_loc(tt, ss, matrix_tensor(Gk[k4].get_les()[tt,ss], tensor_tensor(v, tensor_tensor(matrix_matrix2(Gk[k5].get_les()[tt,ss], Gk[k3].get_gtr()[ss,tt]), v)))/nkvec )
        #                 for rr in range(ntau):
        #                     S.add_to_lmx_loc(tt, rr, matrix_tensor(Gk[k4].get_lmx()[tt,rr], tensor_tensor(v, tensor_tensor(matrix_matrix2(Gk[k5].get_lmx()[tt,rr], Gk[k3].get_rmx()[rr,tt]), v))))
        # print("Self-energy booted")
        
        # Update S
        for tt in range(interpol.k+1):
            print("Self-energy for time step", tt)
            S.set_hf_loc(tt, matrix_tensor(Gloc.get_les()[tt,tt], v - np.swapaxes(v, -1, -2)))
            for qq in range(nkvec):
                print("Computing polarization bubble for k vector", qq)
                # tempPk.append(np.zeros((norb,norb,norb,norb), dtype=np.complex128))
                for kk in range(nkvec):
                    k_plus_q = lattice.sum_indices(qq, kk)
                    for ss in range(interpol.k+1):
                        Pk[qq].set_ret_loc(tt, ss, -1j*matrix_matrix2( Gk[k_plus_q].get_ret()[tt,ss], Gk[kk].get_les()[ss,tt] )
                                                   -1j*matrix_matrix2( Gk[k_plus_q].get_les()[tt,ss], Gk[kk].get_adv()[ss,tt] ) )
                        Pk[qq].set_les_loc(tt, ss, -1j*matrix_matrix2( Gk[k_plus_q].get_les()[tt,ss], Gk[kk].get_gtr()[ss,tt] ) )
                    for rr in range(ntau):
                        Pk[qq].set_lmx_loc(tt, rr, -1j*matrix_matrix2( Gk[k_plus_q].get_lmx()[tt,rr], Gk[kk].get_rmx()[rr,tt] ) )
            tempSlocR = np.zeros((interpol.k+1,interpol.k+1,norb,norb), dtype=np.complex128)
            tempSlocL = np.zeros((interpol.k+1,interpol.k+1,norb,norb), dtype=np.complex128)
            tempSlocI = np.zeros((interpol.k+1,ntau,norb,norb), dtype=np.complex128)
            for kk in range(nkvec):
                print("Computing self-energy for k vector", kk)
                for qq in range(nkvec):
                    k_minus_q = lattice.diff_indices(kk, qq)
                    for ss in range(interpol.k+1):
                        tempSlocR[tt,ss] += 1j*(matrix_tensor( Gk[k_minus_q].get_les()[tt,ss], tensor_tensor( tensor_tensor ( v, Pk[qq].get_ret()[tt,ss] ), v ) ) +
                                                matrix_tensor( Gk[k_minus_q].get_ret()[tt,ss], tensor_tensor( tensor_tensor ( v, Pk[qq].get_gtr()[tt,ss] ), v ) ) ) / nkvec**3
                        tempSlocL[tt,ss] += 1j* matrix_tensor( Gk[k_minus_q].get_les()[tt,ss], tensor_tensor( tensor_tensor ( v, Pk[qq].get_les()[tt,ss] ), v ) ) / nkvec**3
                    for rr in range(ntau):
                        tempSlocI[tt,rr] += 1j* matrix_tensor( Gk[k_minus_q].get_lmx()[tt,rr], tensor_tensor( tensor_tensor ( v, Pk[qq].get_lmx()[tt,rr] ), v ) ) / nkvec**3
            for ss in range(interpol.k+1):
                S.set_ret_loc(tt, ss, tempSlocR[tt,ss])
                S.set_les_loc(tt, ss, tempSlocL[tt,ss])
            for rr in range(ntau):
                S.set_lmx_loc(tt, rr, tempSlocI[tt,rr])
        print("Self-energy booted")

        # Boot G
        lastG = copy_gmatrix(Gloc)
        for kk in range(nkvec):
            # print("Booting Green's function k", kk)
            k_vec = lattice.get_vec(kk)
            Hk = H[:interpol.k+1] + 2*H_kin[0,:interpol.k+1]*np.cos(k_vec[0]) + 2*H_kin[1,:interpol.k+1]*np.cos(k_vec[1]) + 2*H_kin[2,:interpol.k+1]*np.cos(k_vec[2])
            boot_gret(Gk[kk], S, Hk, mu, h, interpol)
            boot_glmx(Gk[kk], S, Hk, mu, h, interpol)
            for tt in range(interpol.k+1):
                Gk[kk].set_les_loc(0, tt, Gk[kk].get_rmx()[0,tt])
            boot_gles(Gk[kk], S, Hk, mu, h, interpol)

            for tt in range(interpol.k+1):
                for ss in range(interpol.k+1):
                    if kk==0:
                        Gloc.set_ret_loc(tt, ss, Gk[kk].get_ret()[tt,ss] / nkvec)
                        Gloc.set_les_loc(tt, ss, Gk[kk].get_les()[tt,ss] / nkvec)
                    else:
                        Gloc.add_to_ret_loc(tt, ss, Gk[kk].get_ret()[tt,ss] / nkvec)
                        Gloc.add_to_les_loc(tt, ss, Gk[kk].get_les()[tt,ss] / nkvec)
                for rr in range(ntau):
                    if kk==0:
                        Gloc.set_lmx_loc(tt, rr, Gk[kk].get_lmx()[tt,rr] / nkvec)
                    else:
                        Gloc.add_to_lmx_loc(tt, rr, Gk[kk].get_lmx()[tt,rr] / nkvec)
        print("Green's function booted")

        # Convergence
        conv = gdist_real_time(interpol.k, lastG, Gloc)
        loop_iterations += 1
        print("At convergence iteration",loop_iterations,"the convergence error is "+exp_string(conv, 5))
        
        # Reched max iterations
        assert loop_iterations < max_iter
    
    print("Convergence for boostrap region reached")
    print("Norm "+exp_string(conv, 5)+"\n\n")


@njit
def boot_loop_hf_kspace(lattice, Gk, Gloc, S, H, H_kin, v, mu, interpol, h, obsv_map, tol=1e-6, max_iter=100000):
    print("Starting bootstrapping loop")
    ntau, norb = Gloc.get_mat().shape[:-1]
    nkvec = len(Gk)
    particle_sign = (-1)**Gloc.particle_type
    
    #G initial conditions
    for kk in range(nkvec):
        for tt in range(interpol.k+1):
            Gk[kk].set_ret_loc(tt, tt, -1j*np.eye(norb))
        for rr in range(ntau):
            Gk[kk].set_lmx_loc(0, rr, 1j * particle_sign * Gk[kk].get_mat()[-1-rr])
    
    conv = 1e5
    loop_iterations = 0
    while conv>tol:

        # Update S
        for tt in range(1,interpol.k+1):
            S.set_hf_loc(tt, 1j*matrix_tensor(Gloc.get_les()[tt,tt], v - np.swapaxes(v, -1, -2)))
        # print("S updated")
        
        # Boot G
        lastG = copy_gmatrix(Gloc)
        for kk in range(nkvec):
            # print("Booting Green's function k", kk)
            k_vec = lattice.get_vec(kk)
            Hk = 2*H_kin[0,:interpol.k+1]*np.cos(k_vec[0]) + 2*H_kin[1,:interpol.k+1]*np.cos(k_vec[1]) + 2*H_kin[2,:interpol.k+1]*np.cos(k_vec[2]) + S.get_hf()[:interpol.k+1]
            for tt in range(interpol.k+1):
                Hk[tt] += H - mu * np.eye(norb)
            time_evo_k = time_evolution_boot(Hk, interpol, h)

            for tt in range(interpol.k+1):
                # print("Time step", tt)
                for rr in range(ntau):
                    if tt > 0:
                        Gk[kk].set_lmx_loc(tt, rr, matrix_matrix(time_evo_k[tt-1], Gk[kk].get_lmx()[tt-1,rr]))
                    if kk==0:
                        Gloc.set_lmx_loc(tt, rr, Gk[kk].get_lmx()[tt,rr] / nkvec)
                    else:
                        Gloc.add_to_lmx_loc(tt, rr, Gk[kk].get_lmx()[tt,rr] / nkvec)
                Gk[kk].set_les_loc(0, tt, Gk[kk].get_rmx()[0,tt])
                for ss in range(interpol.k+1):
                    if tt > 0 and ss < tt:
                        Gk[kk].set_ret_loc(tt, ss, matrix_matrix(time_evo_k[tt-1], Gk[kk].get_ret()[tt-1,ss]))
                    if kk==0:
                        Gloc.set_ret_loc(tt, ss, Gk[kk].get_ret()[tt,ss] / nkvec)
                    else:
                        Gloc.add_to_ret_loc(tt, ss, Gk[kk].get_ret()[tt,ss] / nkvec)
            
            for tt in range(interpol.k+1):
                for ss in range(interpol.k+1):
                    if tt > 0:
                        Gk[kk].set_les_loc(tt, ss, matrix_matrix(time_evo_k[tt-1], Gk[kk].get_les()[tt-1,ss]))
                    if kk==0:
                        Gloc.set_les_loc(tt, ss, Gk[kk].get_les()[tt,ss] / nkvec)
                    else:
                        Gloc.add_to_les_loc(tt, ss, Gk[kk].get_les()[tt,ss] / nkvec)
        # print("Retarded")
        # print(Gloc.get_ret()[interpol.k,:interpol.k+1])
        # print("Last retarded")
        # print(lastG.get_ret()[interpol.k,:interpol.k+1])
        # print("Local lesser")
        # print(Gloc.get_les()[:interpol.k+1,:interpol.k+1,0,0])
        # assert False
        # print("Lasr lesser")
        # print(lastG.get_les()[interpol.k,:interpol.k+1])
        # for rr in range(ntau):
        #     print(rr)
        #     print("Mixed")
        #     print(Gloc.get_lmx()[interpol.k,rr])
        #     print("Last mixed")
        #     print(Gloc.get_lmx()[interpol.k,rr])
        
        # Convergence
        conv = gdist_real_time(interpol.k, lastG, Gloc)
        loop_iterations += 1
        print("At convergence iteration",loop_iterations,"the convergence error is "+exp_string(conv, 5))
        
        # Reched max iterations
        assert loop_iterations < max_iter
    
    # Printing out observables
    # obsv = np.zeros((interpol.k+1,16), dtype=np.float64)
    for tt in range(interpol.k+1):
        Ndens = particle_density(tt, Gloc)
        Eloc = local_energy(tt, mu, H + S.get_hf()[tt], Gloc)
        Ekin = kinetic_energy(tt, H_kin, Gk, lattice)
        Etot = Eloc + Ekin
        jvec = current_vector(tt, H_kin, Gk, lattice)
        obsv_map[tt,0] = tt
        obsv_map[tt,1] = h*tt
        obsv_map[tt,2] = Ndens.real
        obsv_map[tt,3] = Ndens.imag
        obsv_map[tt,4] = Eloc.real
        obsv_map[tt,5] = Eloc.imag
        obsv_map[tt,6] = Ekin.real
        obsv_map[tt,7] = Ekin.imag
        obsv_map[tt,8] = Etot.real
        obsv_map[tt,9] = Etot.imag
        obsv_map[tt,10] = jvec[0].real
        obsv_map[tt,11] = jvec[0].imag
        obsv_map[tt,12] = jvec[1].real
        obsv_map[tt,13] = jvec[1].imag
        obsv_map[tt,14] = jvec[2].real
        obsv_map[tt,15] = jvec[2].imag
    # np.savetxt("observables", obsv)
    
    print("Convergence for boostrap region reached")
    print("Norm "+exp_string(conv, 5)+"\n\n")
