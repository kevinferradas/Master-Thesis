import numpy as np
from numba import njit
from numba.experimental import jitclass
from numba import prange, float64, int32, complex128



spec = [('n', int32),
        ('ntau', int32),
        ('orb', int32),
        ('particle_sign', int32),
        ('particle_type', int32),
        ('GM', complex128[:,:,:]),
        ('GI', complex128[:,:,:,:]),
        ('GR', complex128[:,:,:,:]),
        ('GL', complex128[:,:,:,:]),
        ('Ghf', complex128[:,:,:])]
@jitclass(spec)
class Gmatrix:
    """
    """
    
    def __init__(self, n, ntau, orb, particle_type):
        
        if particle_type==0:
            self.particle_sign = +1
        elif particle_type==1:
            self.particle_sign = -1
        self.particle_type = particle_type
        
        self.GM = np.zeros((ntau, orb, orb), dtype=np.complex128)
        self.GI = np.zeros((n, ntau, orb, orb), dtype=np.complex128)
        self.GR = np.zeros((n, n, orb, orb), dtype=np.complex128)
        self.GL = np.zeros((n, n, orb, orb), dtype=np.complex128)
        
        self.Ghf = np.zeros((n, orb, orb), dtype=np.complex128)
    
    
    def get_particle(self):
        return self.particle_type
    def init_mat(self):
        self.GM = np.zeros_like(self.GM)
    def init_lmx(self):
        self.GI = np.zeros_like(self.GI)
    def init_ret(self):
        self.GR = np.zeros_like(self.GR)
    def init_les(self):
        self.GL = np.zeros_like(self.GL)
    def init_hf(self):
        self.Ghf = np.zeros_like(self.Ghf)
    
    
    def get_mat(self):
        return self.GM
    def get_lmx(self):
        return self.GI
    def get_rmx(self):
        return self.particle_sign * np.swapaxes(np.swapaxes(self.GI.conjugate(), 0, 1), 2, 3)
    def get_ret(self):
        return self.GR
    def get_hf(self):
        return self.Ghf
    def get_adv(self):
        return np.swapaxes(np.swapaxes(self.GR.conjugate(), 0, 1), 2, 3)
    def get_les(self):
        return self.GL
    def get_gtr(self):
        return self.GR - self.get_adv() + self.GL
    def get_ret_comp(self):
        Gcomp = np.zeros_like(self.GR)
        for n in range(self.GR.shape[0]):
            Gcomp[n:] = self.GR[n:]
            if n < self.GR.shape[0]-1:
                Gcomp[:,n+1] = -np.swapaxes(self.GR[:,n+1:].conjugate(), -1, -2)
        return Gcomp
    def get_adv_comp(self):
        return -self.get_ret_comp()
    
    def add_to_mat(self, arr):
        assert self.GM.shape == arr.shape
        self.GM += arr
    def add_to_ret(self, arr):
        assert self.GR.shape == arr.shape
        self.GR += arr
    def add_to_lmx(self, arr):
        assert self.GI.shape == arr.shape
        self.GI += arr
    def add_to_les(self, arr):
        assert self.GL.shape == arr.shape
        self.GL += arr
    def add_to_hf(self, arr):
        assert self.Ghf.shape == arr.shape
        self.Ghf += arr
        
    def add_to_mat_loc(self, tau, arr):
        assert arr.shape == self.GM[0].shape
        self.GM[tau] += arr
    def add_to_lmx_loc(self, t, tau, arr):
        assert arr.shape == self.GI[0,0].shape
        self.GI[t, tau] += arr
    def add_to_ret_loc(self, t1, t2, arr):
        assert arr.shape == self.GR[0,0].shape
        self.GR[t1, t2] += arr
    def add_to_les_loc(self, t1, t2, arr):
        assert arr.shape == self.GL[0,0].shape
        self.GL[t1, t2] += arr
    def add_to_hf_loc(self, t, arr):
        assert self.Ghf.shape[0] == arr.shape
        self.Ghf[t] += arr
    
    def set_mat(self, GM):
        assert GM.shape == self.GM.shape
        self.GM = np.copy(GM)
    def set_lmx(self, GI):
        assert GI.shape == self.GI.shape
        self.GI = np.copy(GI)
    def set_ret(self, GR):
        assert GR.shape == self.GR.shape
        self.GR = np.copy(GR)
    def set_les(self, GL):
        assert GL.shape == self.GL.shape
        self.GL = np.copy(GL)
    def set_hf(self, Ghf):
        assert Ghf.shape == self.Ghf.shape
        self.Ghf = np.copy(Ghf)
    
    
    def set_mat_loc(self, tau, arr):
        assert arr.shape == self.GM[0].shape
        self.GM[tau] = arr
    def set_lmx_loc(self, t, tau, arr):
        assert arr.shape == self.GI[0,0].shape
        self.GI[t, tau] = arr
    def set_ret_loc(self, t1, t2, arr):
        assert arr.shape == self.GR[0,0].shape
        self.GR[t1, t2] = arr
    def set_les_loc(self, t1, t2, arr):
        assert arr.shape == self.GL[0,0].shape
        self.GL[t1, t2] = arr
    def set_hf_loc(self, t, arr):
        assert arr.shape == self.Ghf[0].shape
        self.Ghf[t] = arr
    
    
    def set_lmx_row(self, tau, arr):
        assert arr.shape == self.GI[:,0].shape
        self.GI[:, tau] = arr
    def set_ret_row(self, t2, arr):
        assert arr.shape == self.GR[:,0].shape
        self.GR[:, t2] = arr
    def set_les_row(self, t2, arr):
        assert arr.shape == self.GL[:,0].shape
        self.GL[:, t2] = arr
    
    
    def set_lmx_col(self, t, arr):
        assert arr.shape == self.GI[0,:].shape
        self.GI[t, :] = arr
    def set_ret_col(self, t1, arr):
        assert arr.shape == self.GR[0,:].shape
        self.GR[t1, :] = arr
    def set_les_col(self, t1, arr):
        assert arr.shape == self.GL[0,:].shape
        self.GL[t1, :] = arr
    
    
    def neg_imag_time_mat(self):
        return self.particle_sign * self.GM[::-1]



spec = [('n', int32),
        ('ntau', int32),
        ('orb', int32),
        ('particle_sign', int32),
        ('particle_type', int32),
        ('VM', complex128[:,:,:,:,:]),
        ('VI', complex128[:,:,:,:,:,:]),
        ('VR', complex128[:,:,:,:,:,:]),
        ('VL', complex128[:,:,:,:,:,:]),]
@jitclass(spec)
class Vmatrix:
    """
    """
    
    def __init__(self, n, ntau, orb, particle_type):
        
        if particle_type==0:
            self.particle_sign = +1
        elif particle_type==1:
            self.particle_sign = -1
        self.particle_type = particle_type
        
        self.VM = np.zeros((ntau, orb, orb, orb, orb), dtype=np.complex128)
        self.VI = np.zeros((n, ntau, orb, orb, orb, orb), dtype=np.complex128)
        self.VR = np.zeros((n, n, orb, orb, orb, orb), dtype=np.complex128)
        self.VL = np.zeros((n, n, orb, orb, orb, orb), dtype=np.complex128)
    
    
    def init_mat(self):
        self.VM = np.zeros_like(self.VM)
    def init_lmx(self):
        self.VI = np.zeros_like(self.VI)
    def init_ret(self):
        self.VR = np.zeros_like(self.VR)
    def init_les(self):
        self.VL = np.zeros_like(self.VL)
    
    
    def get_particle(self):
        return self.particle_type
    def get_mat(self):
        return self.VM
    def get_lmx(self):
        return self.VI
    def get_rmx(self):
        return self.particle_sign * np.swapaxes(np.swapaxes(np.swapaxes(self.VI.conjugate(), 0, 1), 2, 5), 4, 3)
    def get_ret(self):
        return self.VR
    def get_adv(self):
        return np.swapaxes(np.swapaxes(np.swapaxes(self.VR.conjugate(), 0, 1), 2, 5), 4, 3)
    def get_les(self):
        return self.VL
    def get_gtr(self):
        return self.VR - self.get_adv() + self.VL
    def get_ret_comp(self):
        Vcomp = np.zeros_like(self.VR)
        for n in range(self.VR.shape[0]):
            Vcomp[n:] = self.VR[n:]
            if n < self.VR.shape[0]-1:
                Vcomp[:,n+1] = -np.swapaxes(np.swapaxes(self.VR[:,n+1:].conjugate(), -4, -1), -3, -2)
        return Vcomp
    def get_adv_comp(self):
        return -self.get_ret_comp()
        
    
    def add_to_mat(self, arr):
        assert self.VM.shape == arr.shape
        self.VM += arr
    def add_to_ret(self, arr):
        assert self.VR.shape == arr.shape
        self.VR += arr
    def add_to_lmx(self, arr):
        assert self.VI.shape == arr.shape
        self.VI += arr
    def add_to_les(self, arr):
        assert self.VL.shape == arr.shape
        self.VL += arr
    
    def add_to_mat_loc(self, tau, arr):
        assert arr.shape == self.VM[0].shape
        self.VM[tau] += arr
    def add_to_lmx_loc(self, t, tau, arr):
        assert arr.shape == self.VI[0,0].shape
        self.VI[t, tau] += arr
    def add_to_ret_loc(self, t1, t2, arr):
        assert arr.shape == self.VR[0,0].shape
        self.VR[t1, t2] += arr
    def add_to_les_loc(self, t1, t2, arr):
        assert arr.shape == self.VL[0,0].shape
        self.VL[t1, t2] += arr
    def add_to_hf_loc(self, t, arr):
        assert self.Ghf.shape[0] == arr.shape
        self.Ghf[t] += arr
    
    def set_mat(self, VM):
        assert VM.shape == self.VM.shape
        self.VM = np.copy(VM)
    def set_lmx(self, VI):
        assert VI.shape == self.VI.shape
        self.VI = np.copy(VI)
    def set_ret(self, VR):
        assert VR.shape == self.VR.shape
        self.VR = np.copy(VR)
    def set_les(self, VL):
        assert VL.shape == self.VL.shape
        self.VL = np.copy(VL)
    
    
    def set_mat_loc(self, tau, arr):
        assert arr.shape == self.VM[0].shape
        self.VM[tau] = arr
    def set_lmx_loc(self, t, tau, arr):
        assert arr.shape == self.VI[0,0].shape
        self.VI[t, tau] = arr
    def set_ret_loc(self, t1, t2, arr):
        assert arr.shape == self.VR[0,0].shape
        self.VR[t1, t2] = arr
    def set_les_loc(self, t1, t2, arr):
        assert arr.shape == self.VL[0,0].shape
        self.VL[t1, t2] = arr
    
    
    def set_lmx_row(self, tau, arr):
        assert arr.shape == self.VI[:,0].shape
        self.VI[:, tau] = arr
    def set_ret_row(self, t2, arr):
        assert arr.shape == self.VR[:,0].shape
        self.VR[:, t2] = arr
    def set_les_row(self, t2, arr):
        assert arr.shape == self.VL[:,0].shape
        self.VL[:, t2] = arr
    
    
    def set_lmx_col(self, t, arr):
        assert arr.shape == self.VI[0,:].shape
        self.VI[t, :] = arr
    def set_ret_col(self, t1, arr):
        assert arr.shape == self.VR[0,:].shape
        self.VR[t1, :] = arr
    def set_les_col(self, t1, arr):
        assert arr.shape == self.VL[0,:].shape
        self.VL[t1, :] = arr
    
    
    def neg_imag_time_mat(self):
        return self.particle_sign * self.VM[::-1]




@njit
def copy_gmatrix(G):
    nt, ntau, orb = G.get_lmx().shape[:-1]
    newG = Gmatrix(nt, ntau, orb, G.particle_type)
    newG.set_mat(np.copy(G.get_mat()))
    newG.set_lmx(np.copy(G.get_lmx()))
    newG.set_ret(np.copy(G.get_ret()))
    newG.set_les(np.copy(G.get_les()))
    newG.set_hf( np.copy(G.get_hf() ))
    return newG


@njit
def copy_vmatrix(V):
    nt, ntau, orb = V.get_lmx().shape[:-1]
    newV = Vmatrix(nt, ntau, orb, V.particle_type)
    newV.set_mat(np.copy(V.get_mat()))
    newV.set_lmx(np.copy(V.get_lmx()))
    newV.set_ret(np.copy(V.get_ret()))
    newV.set_les(np.copy(V.get_les()))
    return newV



@njit
def matrix_matrix(a, b):
    assert a.shape[1] == b.shape[0], "Shape does not coincide"
    
    c = np.zeros((a.shape[0],b.shape[1]), dtype=np.complex128)
    
    for ii in range(a.shape[0]):
        for jj in range(a.shape[1]):
            for kk in range(b.shape[1]):
                c[ii,kk] += a[ii,jj] * b[jj,kk]
    
    return c


@njit
def matrix_matrix2(a, b):
    c = np.zeros((a.shape[0],b.shape[0],b.shape[1],a.shape[1]), dtype=np.complex128)
    
    for mm in range(a.shape[0]):
        for nn in range(b.shape[0]):
            for kk in range(b.shape[1]):
                for ll in range(a.shape[1]):
                    c[mm,nn,kk,ll] += a[mm,ll] * b[nn,kk]
    
    return c


@njit
def matrix_tensor(a, b):
    assert a.shape[0] == b.shape[2], "Shape does not coincide"
    assert a.shape[1] == b.shape[1], "Shape does not coincide"
    
    c = np.zeros((b.shape[0], b.shape[3]), dtype=np.complex128)
    
    for mm in range(b.shape[0]):
        for nn in range(b.shape[3]):
            for ll in range(a.shape[0]):
                for kk in range(a.shape[1]):
                    c[mm,nn] += a[ll,kk] * b[mm,kk,ll,nn]
    
    return c


@njit
def tensor_tensor(a, b):
    assert a.shape[1] == b.shape[2], "Shape does not coincide"
    assert a.shape[3] == b.shape[0], "Shape does not coincide"
    
    c = np.zeros((a.shape[0],b.shape[1],a.shape[2],b.shape[3]), dtype=np.complex128)
    
    for mm in range(a.shape[0]):
        for aa in range(a.shape[1]):
            for ll in range(a.shape[2]):
                for ee in range(a.shape[3]):
                    for xx in range(b.shape[1]):
                        for bb in range(b.shape[3]):
                            c[mm,xx,ll,bb] += a[mm,aa,ll,ee] * b[ee,xx,aa,bb]
    
    return c
