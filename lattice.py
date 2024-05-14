import numpy as np
from numba.experimental import jitclass
from numba import int32, float64, complex128



spec = [("nk1", int32),
        ("nk2", int32),
        ("nk3", int32),
        ("length", int32),
        ("basis1", float64[:]),
        ("basis2", float64[:]),
        ("basis3", float64[:])]
@jitclass(spec)
class Lattice:
    def __init__(self, nk1=1, nk2=1, nk3=1, basis1=np.array([2*np.pi,0.,0.]), basis2=np.array([0.,2*np.pi,0.]), basis3=np.array([0.,0.,2*np.pi])):
        self.nk1 = nk1
        self.nk2 = nk2
        self.nk3 = nk3
        
        self.basis1 = basis1
        self.basis2 = basis2
        self.basis3 = basis3
        
        self.length = nk1 * nk2 * nk3
    
    
    def __getitem__(self, idx):
        # idx = i3 + i2*n3 + i1*n2*n3
        # idx%n3 = i3
        # idx//n3 = i2 + i1*n2
        # (idx//n3)%n2 = i2
        # (idx//n3)//n2 = i1
        return np.array([(idx//self.nk3)//self.nk2, (idx//self.nk3)%self.nk2, idx%self.nk3], dtype=np.int32)
    
    
    def get_vec(self, idx):
        # print("I'm inside get vec")
        mesh_vec = self[idx]
        # print("Mesh vector got")
        return mesh_vec[0]*self.basis1/self.nk1 + mesh_vec[1]*self.basis2/self.nk2 + mesh_vec[2]*self.basis3/self.nk3
        # print("I will return", ret)
        # return ret
    
    
    def sum_indices(self, a, b):
        mesh_vec_a = self[a]
        mesh_vec_b = self[b]
        c1 = (mesh_vec_a[0] + mesh_vec_b[0]) % self.nk1
        c2 = (mesh_vec_a[1] + mesh_vec_b[1]) % self.nk2
        c3 = (mesh_vec_a[2] + mesh_vec_b[2]) % self.nk3
        return c3 + self.nk3 * (c2 + self.nk1 * c1)
    
    
    def diff_indices(self, a, b):
        mesh_vec_a = self[a]
        mesh_vec_b = self[b]
        c1 = (mesh_vec_a[0] - mesh_vec_b[0]) % self.nk1
        c2 = (mesh_vec_a[1] - mesh_vec_b[1]) % self.nk2
        c3 = (mesh_vec_a[2] - mesh_vec_b[2]) % self.nk3
        return c3 + self.nk3 * (c2 + self.nk1 * c1)
