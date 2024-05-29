import numpy as np
from numba import njit, typed
import pyNEGFv5 as negf

print("Imports done")




fl = open("parameters") #The open() function opens a file, and returns it as a file object.
lines = fl.readlines() # The readlines() method returns a list containing each line in the file as a list item.
fl.close() # The close() method closes an open file.

for line in lines:
    if line[:2] == "--":
        splitted = line[2:].split(' ')
        while True:
            try:
                splitted.remove('')
            except:
                break
        locals()[splitted[0]] = float(splitted[2])
    
    if line[:2] == "++":
        splitted = line[2:].split(' ')
        while True:
            try:
                splitted.remove('')
            except:
                break
        locals()[splitted[0]] = int(splitted[2])

print("Parameters set")



###################################################################
# Previous definitions
# Interpolator
###################################################################
interp = negf.Interpolator(k)

print("Interpolator set")



###################################################################
# On-site e-e interactions
# V = sum_{ijkl} v_ijkl cci ccj cal cak
# v_ijkl = int int dr1 dr2 psi_i^dagger(r1) psi_j^dagger(r2) (1/|r1-r2|) psi_k(r1) psi_l(r2)
###################################################################

Vloc = np.zeros((2,2,2,2), dtype=np.complex128)
Vloc[1,0,1,0] = U/2
Vloc[0,1,0,1] = U/2
Vloc[1,0,0,1] = -U/2
Vloc[0,1,1,0] = -U/2



###################################################################
# Intersite hoppings
###################################################################

Hkin = np.array([-t*np.eye(2),-t*np.eye(2),-t*np.eye(2)], dtype=np.complex128) # Hkin.shape=(3,2,2)
#np.eye(N) --> Return a 2-D, N X N  array (matrix) with ones on the diagonal and zeros elsewhere.



@njit
def generateGk(nk_points, n, ntau, states, particle_type=0):
    Gk = []
    for i in range(nk_points):
        Gk.append(negf.Gmatrix(n, ntau, states, particle_type)) # The append() method appends an element to the end of the list.
    return Gk




k_lattice = negf.Lattice(nkvec,nkvec,nkvec)
Gk = generateGk(k_lattice.length, n, ntau, 2, 1) # There are just 2 states
Gloc = negf.Gmatrix(n, ntau, 2, 1)
S = negf.Gmatrix(n, ntau, 2, 1) # Purely local


Hloc = np.zeros((2,2), dtype=np.complex128)


mu = negf.matsubara_branch_init_hf_kspace(N, 0, k_lattice, Hloc, Hkin, Gk, Gloc, S, Vloc, interp, beta, 1)
np.savetxt("Gloc_mat_up", Gloc.get_mat()[:,0,0]) # savetxt --> Save an array to a text file.
np.savetxt("Gloc_mat_dn", Gloc.get_mat()[:,1,1])
# Gloc.get_mat()[:,0,0]


print("Work finished succesfully")
