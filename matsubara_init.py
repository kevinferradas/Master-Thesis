import numpy as np
from numba import njit
from .linalg import matrix_matrix, matrix_matrix2, matrix_tensor, tensor_tensor
from .contour_funcs import Gmatrix, Vmatrix
from .convolution import conv_mat_extern
from .utils import gdist_mat
from .printing import float_string, exp_string

#función signo
@njit
def sgn(x):
    if x < 0:
        return -1
    elif x > 0:
        return +1
    else:
        return 0

# ntau is the number of elements of the list tau. Then, ntau determines the space between each element of the list.
# N is the (desired) number of particles
#mu is the chemical potential (initial value)
#H0 is the non interacting hamiltonian
# beta is the thermal energy (kb.T)^-1
#  particle : 0--> boson, 1--> fermion
# mu_jump is the variation in the chemical potential
# tolN is a tolerance

#ndim and shape comes from numpy library.
@njit
def g_nonint_init(ntau, N, mu, H0, beta=1, particle=0, mu_jump=0.5, tolN=1e-6):
   ## The assert keyword lets you test if a condition in your code returns True, if not, the program will raise an AssertionError.
    # in numpy .ndim is the same as axis/axes
    assert H0.ndim==2, "Only constant hamiltonian" # dim 2 means we are working with matrices
     #shape describes how many data (or the range) along each available axis.
    assert H0.shape[0]==H0.shape[1], "Hamiltonian is a squared matrix" # 0--> rows , 1--> columns
    n_orb = H0.shape[0] #number of rows
    particle_sign = (-1)**particle
    tau = np.linspace(0, beta, ntau) # Return evenly spaced numbers over a specified interval.
    
    
    assert not np.any(np.isnan(H0)) ## np.any test whether any array element along a given axis evaluates to True.
                                    ##np.isnan tests element-wise whether it is NaN (not a Number) or not and returns the result as a boolean array
    
    ## The linalg library (from numpy) specializes in linear algebra with matrices and vectors provided by numpy.
    # .eig --> Compute the eigenvalues and right eigenvectors of a square array. w (eigenvalues) , P (eigenvectors matrix; column "i" corresponds to eigenvalue"i")
    # .inv --> Compute the (multiplicative) inverse of a matrix. 
    w, P = np.linalg.eig(H0)
    Pinv = np.linalg.inv(P)
    
    last_sign = 2 # is a parameter used to adjust the value of mu
  
    if N>0:
        print("Checking number of particle for non-interacting case")
        while True:
            N0 = 0.0 # N0 is the expected number of particles
            for jj in range(n_orb): ## range(start, stop, step) start--> 0 ( by default) ; step-->1 (by default); stop ( not included in the sequence).
                e = w[jj].real - mu 
                #.real is an attribute for the complex math library in Python to obtain the real part of a complex number.
                # e is the difference between the energies and the chemical potential
                
                N0 -= 1/(particle_sign - np.exp(e * beta)) # N0=-tr (G^M(beta)) . The expresion comes from replacing t=beta in eq. 202.
            
            DN = N - N0 # differential of the number of particles 
            DNsign = sgn(DN)
            if abs(DN) < tolN:
                break
            if DNsign!=last_sign and last_sign!=2: # != --> does not equal 
                mu_jump /= 2 # x /= 2 equivalent to x = x / 2
            mu += DNsign * mu_jump
            last_sign = DNsign
            
    # numpy.zeros(shape, dtype=float, order='C', *, like=None)
    g = np.zeros((ntau, H0.shape[0], H0.shape[1]), dtype=np.complex128) #np.zeros--> Return a new array of given shape and type, filled with zeros. # 128-bit complex floating-point number
    # dim(g)=3
    # Nested loops to complete the elements of an array.
    for ii in range(n_orb): # n_orb = H0.shape[0]
        for jj in range(n_orb):
            for kk in range(n_orb):
                e = w[jj] - mu
                g[...,ii,kk] += P[ii,jj] * np.exp(-e*tau)/(particle_sign * np.exp(-e*beta) - 1) * Pinv[jj,kk] # P[kk,jj].conjugate()
                #Ellipsis--> ... In this case, we want the matrix elements gik for all values of tau ( imaginary time)
                # We obtain the matrix element gik for the green function in the basis of the hamiltonian
                # g captures the Green's function values in the Hamiltonian's basis for all tau values.
                # g(tau)= np.exp(-e*tau)/(particle_sign * np.exp(-e*beta) - 1) is eqn. 202 . Is diagonal!
    
    if np.any(np.abs(g) > 1e14):
        print("Warning: possible overload on non-interactive green's function")
    return g, mu

###########################################################################

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

############################################################################################################
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

#################################################################################################################

# N is the (desired) number of particles
#mu is the chemical potential (initial value)
#lattice is an instance (object) of the class Lattice.
#H0 is the local part of the non interacting hamiltonian
#H0_kin is the hopping kinetic term of the non interacting hamiltonian. Is represented as a list of three diagonal matrices, one for each dimension.
# Gloc--> Is an instance (object) of the class Gmatrix
#Gk--> List of Gmatrix objects.
# beta is the thermal energy (kb.T)^-1
#  particle : 0--> boson, 1--> fermion
# mu_jump is the variation in the chemical potential
# tolN is a tolerance
# ntau is the number of partitions between zero and beta


@njit
def non_interactive_matsubara_kspace(N, mu, lattice, H0, H0_kin, Gk, Gloc, beta=1, particle=0, mu_jump=0.5, tol=1e-6):
    print("Estimating Matsubara branch for non-interactive case")
    ntau = Gloc.get_mat().shape[0] # .get_mat() returns self.GM = np.zeros((ntau, orb, orb), dtype=np.complex128) ( dim=3 array) 
    nkvec = len(Gk)

    last_sign = 2
    while True:
        newGlocM = np.zeros_like(Gloc.get_mat())
        for kk in range(nkvec): 
            k_vec = lattice.get_vec(kk) #wave vector 
            Hk0 = H0 + 2*H0_kin[0]*np.cos(k_vec[0]) + 2*H0_kin[1]*np.cos(k_vec[1]) + 2*H0_kin[2]*np.cos(k_vec[2]) - mu*np.eye(H0.shape[0]) # nearest neighbors approx.
                                                                         #np.eye(N)--> Return a 2-D array of N X N with ones on the diagonal and zeros elsewhere.
                                                                         # The term - mu*np.eye(H0.shape[0]) should be erased!!
            assert not np.any(np.isnan(Hk0))
            assert np.all(np.abs(Hk0) < 1e14)
            Gk[kk].set_mat(g_nonint_init(ntau, -1, mu, Hk0, beta, particle)[0]) 
            # g_nonint_init()[0] returns g; # g.shape= (ntau, Hk0.shape[0], Hk0.shape[1]) 
            # set_mat(self, g): 1. assert g.shape == self.GM.shape --> 2. self.GM = np.copy(g)
            # since N < 0, mu does not change.                                                                     
                
            newGlocM += Gk[kk].get_mat() / nkvec 
            # For each imaginary time, there is a sum, over all available k's, of Matsubara's Green function Matrices 
            #  returns the new self.GM, that is self.GM=np.copy(g) #eqn. 299, with i=j. 
            # Is nkvec equal to the number of lattice sites?
        Gloc.set_mat(newGlocM) # self.GM = np.copy(newGlocM) # After this step, self.GM is our G local (eq.299) 
        
        print("Non-interactive gloc approximated for mu="+float_string(mu, 5)) # mu with 5 digits of precision.
        
        N0 = -np.trace(Gloc.get_mat()[-1]) ## N0=-tr (G^M(beta)) 
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

###############################################################################
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
                HkMF = H0 + 2*H0_kin[0]*np.cos(k_vec[0]) + 2*H0_kin[1]*np.cos(k_vec[1]) + 2*H0_kin[2]*np.cos(k_vec[2]) + S.get_hf()[0]
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

######################################################

# N is the (desired) number of particles
#mu is the chemical potential (initial value)
#lattice is an instance (object) of the class Lattice (lattice.py)
#H0 is the local part of the non interacting hamiltonian
#H0_kin is the hopping kinetic term of the non interacting hamiltonian. Is represented as a list of three diagonal matrices, one for each spatial dimensions (x,y,z) 
# Gloc--> Is an instance (object) of the class Gmatrix
#Gk--> List of Gmatrix objects.
#S -->   Is an instance (object) of the class Gmatrix. It representes the Hartree-Fock Self energy
#v--> Parameter related with self-energy.
#interpol
# beta is the thermal energy (kb.T)^-1
#  particle : 0--> boson, 1--> fermion
# mu_jump is the variation in the chemical potential
# tolN is a tolerance
# ntau is the number of time steps between  zero and beta.

@njit
def matsubara_branch_init_hf_kspace(N, mu, lattice, H0, H0_kin, Gk, Gloc, S, v, interpol, beta=1, particle=0, mu_jump=0.5, max_iter=100000, tol=1e-6):
    print("Initilizing Matsubara branch")
    ntau = Gloc.get_mat().shape[0] # .get_mat() returns self.GM = np.zeros((ntau, orb, orb), dtype=np.complex128) ( dim=3 array) 
    #  norb = Gloc.get_mat().shape[1]
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
            
            S.set_hf_loc(0, -particle_sign * matrix_tensor(Gloc.get_mat()[-1], v-np.swapaxes(v, -1, -2))) #Eq.172  
            # numpy.swapaxes(a, axis1, axis2)[source] --> Interchange two axes of an input array a-
            #This defines de Hartree-Fock self-energy at (imaginary?)time 0.
              # def set_hf_loc(self, t, arr):
            # assert arr.shape == self.Ghf[0].shape
             #self.Ghf[t] = arr
            # self.Ghf = np.zeros((n, orb, orb), dtype=np.complex128)
            # print("Hartree-Fock self-energy set")
            
            newGlocM = np.zeros_like(Gloc.get_mat())
            for kk in range(nkvec):
                k_vec = lattice.get_vec(kk) #wave vector k 
                HkMF = H0 + 2*H0_kin[0]*np.cos(k_vec[0]) + 2*H0_kin[1]*np.cos(k_vec[1]) + 2*H0_kin[2]*np.cos(k_vec[2]) + S.get_hf()[0]
                #HkMF --> Mean field Hamiltonian in the momentum space
                # nearest neighbors approx.
                #S.get_hf()[0]--> return self.Ghf --> self.Ghf = np.zeros((n, orb, orb), dtype=np.complex128) # 0 means the initial time S is taken.
            assert not np.any(np.isnan(Hk0))
                g,no_use = g_nonint_init(ntau, -1, mu, HkMF, beta, particle)
                #g--> green functions matrix , no_use--> mu (chemical potential)
                Gk[kk].set_mat(g) # set_mat(self, g): 1. assert g.shape == self.GM.shape --> 2. self.GM = np.copy(g)
                newGlocM += Gk[kk].get_mat() / nkvec
            # print("Gloc set")
            
            conv = gdist_mat(newGlocM, Gloc.get_mat())
            print("Convergence at "+exp_string(conv,5)+" at iteration",loop_iterations+1)
            Gloc.set_mat(newGlocM)
            
            loop_iterations += 1
            assert loop_iterations <= max_iter
        
        print("Convergence for Matsubara branch and mu="+float_string(mu, 5))
        print("Norm "+exp_string(conv, 5)+"\n\n")
        
        N0 = -np.trace(Gloc.get_mat()[-1]) ## N0=-tr (G^M(beta))
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
