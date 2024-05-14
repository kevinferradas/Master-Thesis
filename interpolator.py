import numpy as np
from numba.experimental import jitclass
from numba import prange, float64, int32, int8
from .utils import factorial, binomial
from .linalg import matrix_matrix




spec = [
        ('k', int8),
        ('P', float64[:,:]),
        ('D', float64[:,:]),
        ('a', float64[:]),
        ('s', float64[:,:]),
        ('w', float64[:,:]),
        ('R', float64[:,:,:]),
        ('I', float64[:,:,:])
        ]
@jitclass(spec)
class Interpolator:
    """
    Interpolator class computes and storage the interpolation weights and the corresponding for derivation, backward differentiation, gregory integration and convolution.
    
    Parameters
    ----------
    k : int
        Order of interpolation.
    h : float, optional
        Time step of samples to interpolate. Default is 1.
    
    """
    def __init__(self, k):
        assert k < 10, "Avoid overload on interpolation coeffitients"
        row = np.empty((k+1,k+1), dtype=np.int32); col = np.empty((k+1,k+1), dtype=np.int32)
        for i in range(k+1):
            row[i,:] = i
            col[:,i] = i
        self.k = k
        self.P = np.linalg.inv((row**col).astype(np.float64))
        self.a = -self.P[1,:]
        self.D = np.zeros((k+1,k+1), dtype=np.float64)
        self.s = np.zeros((k+1,k+1))
        for i in range(k+1):
            for j in range(k+1):
                for l in range(k+1):
                    self.s[i,j] += (row[i,l]*1.0)**(col[i,l]+1)/((col[i,l]+1)*1.0) * self.P[l,j]
                    if l > 0:
                        self.D[i,j] += col[i,l] * row[i,l]**col[i,l] * self.P[l,j]
        #self.D = matrix_matrix(col[:,1:].astype(np.float64) * row[:,1:].astype(np.float64)**(col[:,1:].astype(np.float64)-1), self.P[1:,:])
        #self.s = matrix_matrix(row.astype(np.float64)**(col.astype(np.float64)+1)/(col.astype(np.float64)+1), self.P)
        self.w = np.copy(self.s)
        
        self.R = np.zeros((self.k+1, self.k+1, self.k+1), dtype=np.float64)
        for mm in range(self.k+1):
            for ll in range(self.k+1):
                for bb in range(self.k+1):
                    for rr in range(self.k+1):
                        for aa in range(self.k+1):
                            for qq in range(rr+1):
                                self.R[mm,ll,bb] += binomial(rr,qq) * (-1)**qq * self.P[rr,ll] * self.P[aa,bb] * mm**(aa+rr+1) / (aa+qq+1)
        
        self.I = np.zeros_like(self.R)
        for mm in range(self.k+1):
            for nn in range(self.k+1):
                for ll in range(self.k+1):
                    for rr in range(self.k+1):
                        self.I[nn,mm,ll] += (mm**(rr+1) - nn**(rr+1)) * self.P[rr,ll] / (rr+1)
    
    # def gregory_weights_matrix(self, n):
    #     if n<=self.k:
    #         return self.s[:n+1,:n+1]
        
    #     S = np.zeros((n+1,n+1))
    #     A = np.zeros((n+1,n+1))
    #     S[np.arange(self.k+1,n+1),np.arange(self.k+1,n+1)] += 1
    #     A[np.arange(self.k+1),np.arange(self.k+1)] += self.h
    #     S[:self.k+1,:self.k+1] += self.h*self.s
    #     for m in range(self.k+1,n+1):
    #         A[m,m:m-self.k-1:-1] += self.a
    #     return np.linalg.inv(A) @ S
    
    def gregory_weights(self, n):
        """
        Computes the gregory integration wiegths by an iterative method.

        Parameters
        ----------
        n : int
            Integration order.

        Returns
        -------
        ndarray
            Gregory weights since order n.

        """
        if n+1 <= self.w.shape[0]:
            return self.w[:n+1,:n+1]
        
        for k in range(n+1 - self.w.shape[0]):
            self.__gregory_add()
        return self.w
    
    def __gregory_add(self):
        """
        Adds a new row to gregory weigths matrix.

        Returns
        -------
        None.

        """
        m = self.w.shape[0]
        new_row = np.zeros(m+1)
        new_row[-1] = 1/self.a[0]
        aux_a = np.empty((self.a.size-1,1), dtype=self.a.dtype)
        aux_a[:,0] = self.a[1:]
        add_terms = -new_row[-1] * aux_a * self.w[m-1:m-self.k-1:-1,:]
        for i in range(add_terms.shape[0]):
            new_row[:-1] += add_terms[i]
        new_w = np.zeros((self.w.shape[0]+1,self.w.shape[1]+1), dtype=self.w.dtype)
        new_w[:-1,:-1] = self.w
        new_w[-1,:] = new_row
        self.w = np.copy(new_w)
    
    def conv_weights(self, m):
        """
        Computes the convolution integration weights from t=0 to jh with j=0,...,m.

        Parameters
        ----------
        m : int
            Time steps to convolve.

        Returns
        -------
        ndarray
            Convolution weights.

        """
        # a = np.empty((self.k+1,self.k+1), dtype=np.int32); b = np.empty((self.k+1,self.k+1), dtype=np.int32)
        # for i in prange(self.k+1):
        #     for j in prange(self.k+1):
        #         a[i,j] = i
        #         b[j,i] = i
        # factor = (-1)**b * factorial(a) * factorial(b) * m**(a+b+1) / factorial(a+b+1)
        # return self.P.T @ factor @ self.P
        
        R = np.zeros_like(self.P)
        
        for ll in range(self.k+1):
            for bb in range(self.k+1):
                for rr in range(self.k+1):
                    for aa in range(self.k+1):
                        for qq in range(rr+1):
                            R[ll,bb] += binomial(rr, qq) * (-1)**qq * self.P[rr,ll] * self.P[aa,bb] * m**(aa+rr+1) / (aa+qq+1)
        
        return R


# spec = [
#         ('k', int32),
#         ('h', float64),
#         ('P', float64[:,:]),
#         ('D', float64[:,:]),
#         ('a', float64[:]),
#         ('s', float64[:,:]),
#         ('w', float64[:,:]),
#         ('sigma', float64[:,:]),
#         ('omega', float64[:])
#         ]
# @jitclass(spec)
# class Interpolator:
#     """
#     Interpolator class computes and storage the interpolation weights and the corresponding for derivation, backward differentiation, gregory integration and convolution.
    
#     Parameters
#     ----------
#     k : int
#         Order of interpolation.
#     h : float, optional
#         Time step of samples to interpolate. Default is 1.
    
#     """
#     def __init__(self, k, h):
#         row = np.empty((k+1,k+1), dtype=np.int32); col = np.empty((k+1,k+1), dtype=np.int32)
#         for i in range(k+1):
#             row[i,:] = i
#             col[:,i] = i
#         self.k = k
#         self.h = h
#         self.P = np.linalg.inv((row**col).astype(np.float64))
#         self.D = np.ascontiguousarray((col[:,1:] * row[:,1:]**(col[:,1:]-1)).astype(np.float64)) @ np.ascontiguousarray(self.P[1:,:])
#         self.a = -self.P[1,:]
#         self.s = np.ascontiguousarray((row**(col+1)/(col+1))) @ np.ascontiguousarray(self.P)
#         self.w = np.copy(self.s)
        
#         sigma = np.empty(((self.k+1)*(self.k+1),), dtype=np.float64)
#         self.omega = np.empty((self.k+1,), dtype=np.float64)
        
#         if self.k==1:
#             sigma[0]=  4.16666666666666666670e-01;
#             sigma[1]=  1.16666666666666666670e+00;
#             sigma[2]=  4.16666666666666666670e-01;
#             sigma[3]=  1.08333333333333333330e+00;
#             self.omega[0]=  4.16666666666666666670e-01;
#             self.omega[1]=  1.08333333333333333330e+00;
#         elif self.k==2:
#             sigma[0]=  3.75000000000000000000e-01;
#             sigma[1]=  1.12500000000000000000e+00;
#             sigma[2]=  1.12500000000000000000e+00;
#             sigma[3]=  3.75000000000000000000e-01;
#             sigma[4]=  1.16666666666666666670e+00;
#             sigma[5]=  9.16666666666666666670e-01;
#             sigma[6]=  3.75000000000000000000e-01;
#             sigma[7]=  1.16666666666666666670e+00;
#             sigma[8]=  9.58333333333333333330e-01;
#             self.omega[0]=  3.75000000000000000000e-01;
#             self.omega[1]=  1.16666666666666666670e+00;
#             self.omega[2]=  9.58333333333333333330e-01;
#         elif self.k==3:
#             sigma[0]=  3.48611111111111111110e-01;
#             sigma[1]=  1.27222222222222222220e+00;
#             sigma[2]=  7.58333333333333333330e-01;
#             sigma[3]=  1.27222222222222222220e+00;
#             sigma[4]=  3.48611111111111111110e-01;
#             sigma[5]=  1.24583333333333333330e+00;
#             sigma[6]=  9.05555555555555555560e-01;
#             sigma[7]=  9.05555555555555555560e-01;
#             sigma[8]=  3.48611111111111111110e-01;
#             sigma[9]=  1.24583333333333333330e+00;
#             sigma[10]=  8.79166666666666666670e-01;
#             sigma[11]=  1.05277777777777777780e+00;
#             sigma[12]=  3.48611111111111111110e-01;
#             sigma[13]=  1.24583333333333333330e+00;
#             sigma[14]=  8.79166666666666666670e-01;
#             sigma[15]=  1.02638888888888888890e+00;
#             self.omega[0]=  3.48611111111111111110e-01;
#             self.omega[1]=  1.24583333333333333330e+00;
#             self.omega[2]=  8.79166666666666666670e-01;
#             self.omega[3]=  1.02638888888888888890e+00;
#         elif self.k==4:
#             sigma[0]=  3.29861111111111111110e-01;
#             sigma[1]=  1.30208333333333333330e+00;
#             sigma[2]=  8.68055555555555555560e-01;
#             sigma[3]=  8.68055555555555555560e-01;
#             sigma[4]=  1.30208333333333333330e+00;
#             sigma[5]=  3.29861111111111111110e-01;
#             sigma[6]=  1.32083333333333333330e+00;
#             sigma[7]=  7.47916666666666666670e-01;
#             sigma[8]=  1.20277777777777777780e+00;
#             sigma[9]=  7.47916666666666666670e-01;
#             sigma[10]=  3.29861111111111111110e-01;
#             sigma[11]=  1.32083333333333333330e+00;
#             sigma[12]=  7.66666666666666666670e-01;
#             sigma[13]=  1.08263888888888888890e+00;
#             sigma[14]=  1.08263888888888888890e+00;
#             sigma[15]=  3.29861111111111111110e-01;
#             sigma[16]=  1.32083333333333333330e+00;
#             sigma[17]=  7.66666666666666666670e-01;
#             sigma[18]=  1.10138888888888888890e+00;
#             sigma[19]=  9.62500000000000000000e-01;
#             sigma[20]=  3.29861111111111111110e-01;
#             sigma[21]=  1.32083333333333333330e+00;
#             sigma[22]=  7.66666666666666666670e-01;
#             sigma[23]=  1.10138888888888888890e+00;
#             sigma[24]=  9.81250000000000000000e-01;
#             self.omega[0]=  3.29861111111111111110e-01;
#             self.omega[1]=  1.32083333333333333330e+00;
#             self.omega[2]=  7.66666666666666666670e-01;
#             self.omega[3]=  1.10138888888888888890e+00;
#             self.omega[4]=  9.81250000000000000000e-01;
#         elif self.k==5:
#             sigma[0]=  3.15591931216931216930e-01;
#             sigma[1]=  1.40644841269841269840e+00;
#             sigma[2]=  5.33878968253968253970e-01;
#             sigma[3]=  1.48816137566137566140e+00;
#             sigma[4]=  5.33878968253968253970e-01;
#             sigma[5]=  1.40644841269841269840e+00;
#             sigma[6]=  3.15591931216931216930e-01;
#             sigma[7]=  1.39217923280423280420e+00;
#             sigma[8]=  6.38244047619047619050e-01;
#             sigma[9]=  1.15398478835978835980e+00;
#             sigma[10]=  1.15398478835978835980e+00;
#             sigma[11]=  6.38244047619047619050e-01;
#             sigma[12]=  3.15591931216931216930e-01;
#             sigma[13]=  1.39217923280423280420e+00;
#             sigma[14]=  6.23974867724867724870e-01;
#             sigma[15]=  1.25834986772486772490e+00;
#             sigma[16]=  8.19808201058201058200e-01;
#             sigma[17]=  1.25834986772486772490e+00;
#             sigma[18]=  3.15591931216931216930e-01;
#             sigma[19]=  1.39217923280423280420e+00;
#             sigma[20]=  6.23974867724867724870e-01;
#             sigma[21]=  1.24408068783068783070e+00;
#             sigma[22]=  9.24173280423280423280e-01;
#             sigma[23]=  9.24173280423280423280e-01;
#             sigma[24]=  3.15591931216931216930e-01;
#             sigma[25]=  1.39217923280423280420e+00;
#             sigma[26]=  6.23974867724867724870e-01;
#             sigma[27]=  1.24408068783068783070e+00;
#             sigma[28]=  9.09904100529100529100e-01;
#             sigma[29]=  1.02853835978835978840e+00;
#             sigma[30]=  3.15591931216931216930e-01;
#             sigma[31]=  1.39217923280423280420e+00;
#             sigma[32]=  6.23974867724867724870e-01;
#             sigma[33]=  1.24408068783068783070e+00;
#             sigma[34]=  9.09904100529100529100e-01;
#             sigma[35]=  1.01426917989417989420e+00;
#             self.omega[0]=  3.15591931216931216930e-01;
#             self.omega[1]=  1.39217923280423280420e+00;
#             self.omega[2]=  6.23974867724867724870e-01;
#             self.omega[3]=  1.24408068783068783070e+00;
#             self.omega[4]=  9.09904100529100529100e-01;
#             self.omega[5]=  1.01426917989417989420e+00;
#         elif self.k==6:
#             sigma[0]=  3.04224537037037037040e-01;
#             sigma[1]=  1.44901620370370370370e+00;
#             sigma[2]=  5.35937500000000000000e-01;
#             sigma[3]=  1.21082175925925925930e+00;
#             sigma[4]=  1.21082175925925925930e+00;
#             sigma[5]=  5.35937500000000000000e-01;
#             sigma[6]=  1.44901620370370370370e+00;
#             sigma[7]=  3.04224537037037037040e-01;
#             sigma[8]=  1.46038359788359788360e+00;
#             sigma[9]=  4.42096560846560846560e-01;
#             sigma[10]=  1.55390211640211640210e+00;
#             sigma[11]=  4.78786375661375661380e-01;
#             sigma[12]=  1.55390211640211640210e+00;
#             sigma[13]=  4.42096560846560846560e-01;
#             sigma[14]=  3.04224537037037037040e-01;
#             sigma[15]=  1.46038359788359788360e+00;
#             sigma[16]=  4.53463955026455026460e-01;
#             sigma[17]=  1.46006117724867724870e+00;
#             sigma[18]=  8.21866732804232804230e-01;
#             sigma[19]=  8.21866732804232804230e-01;
#             sigma[20]=  1.46006117724867724870e+00;
#             sigma[21]=  3.04224537037037037040e-01;
#             sigma[22]=  1.46038359788359788360e+00;
#             sigma[23]=  4.53463955026455026460e-01;
#             sigma[24]=  1.47142857142857142860e+00;
#             sigma[25]=  7.28025793650793650790e-01;
#             sigma[26]=  1.16494708994708994710e+00;
#             sigma[27]=  7.28025793650793650790e-01;
#             sigma[28]=  3.04224537037037037040e-01;
#             sigma[29]=  1.46038359788359788360e+00;
#             sigma[30]=  4.53463955026455026460e-01;
#             sigma[31]=  1.47142857142857142860e+00;
#             sigma[32]=  7.39393187830687830690e-01;
#             sigma[33]=  1.07110615079365079370e+00;
#             sigma[34]=  1.07110615079365079370e+00;
#             sigma[35]=  3.04224537037037037040e-01;
#             sigma[36]=  1.46038359788359788360e+00;
#             sigma[37]=  4.53463955026455026460e-01;
#             sigma[38]=  1.47142857142857142860e+00;
#             sigma[39]=  7.39393187830687830690e-01;
#             sigma[40]=  1.08247354497354497350e+00;
#             sigma[41]=  9.77265211640211640210e-01;
#             sigma[42]=  3.04224537037037037040e-01;
#             sigma[43]=  1.46038359788359788360e+00;
#             sigma[44]=  4.53463955026455026460e-01;
#             sigma[45]=  1.47142857142857142860e+00;
#             sigma[46]=  7.39393187830687830690e-01;
#             sigma[47]=  1.08247354497354497350e+00;
#             sigma[48]=  9.88632605820105820110e-01;
#             self.omega[0]=  3.04224537037037037040e-01;
#             self.omega[1]=  1.46038359788359788360e+00;
#             self.omega[2]=  4.53463955026455026460e-01;
#             self.omega[3]=  1.47142857142857142860e+00;
#             self.omega[4]=  7.39393187830687830690e-01;
#             self.omega[5]=  1.08247354497354497350e+00;
#             self.omega[6]=  9.88632605820105820110e-01;
#         elif self.k==7:
#             sigma[0]=  2.94868000440917107580e-01;
#             sigma[1]=  1.53523589065255731920e+00;
#             sigma[2]=  1.80113536155202821870e-01;
#             sigma[3]=  2.07786816578483245150e+00;
#             sigma[4]= -1.76171186067019400350e-01;
#             sigma[5]=  2.07786816578483245150e+00;
#             sigma[6]=  1.80113536155202821870e-01;
#             sigma[7]=  1.53523589065255731920e+00;
#             sigma[8]=  2.94868000440917107580e-01;
#             sigma[9]=  1.52587935405643738980e+00;
#             sigma[10]=  2.66333223104056437390e-01;
#             sigma[11]=  1.72204420194003527340e+00;
#             sigma[12]=  6.90875220458553791890e-01;
#             sigma[13]=  6.90875220458553791890e-01;
#             sigma[14]=  1.72204420194003527340e+00;
#             sigma[15]=  2.66333223104056437390e-01;
#             sigma[16]=  2.94868000440917107580e-01;
#             sigma[17]=  1.52587935405643738980e+00;
#             sigma[18]=  2.56976686507936507940e-01;
#             sigma[19]=  1.80826388888888888890e+00;
#             sigma[20]=  3.35051256613756613760e-01;
#             sigma[21]=  1.55792162698412698410e+00;
#             sigma[22]=  3.35051256613756613760e-01;
#             sigma[23]=  1.80826388888888888890e+00;
#             sigma[24]=  2.94868000440917107580e-01;
#             sigma[25]=  1.52587935405643738980e+00;
#             sigma[26]=  2.56976686507936507940e-01;
#             sigma[27]=  1.79890735229276895940e+00;
#             sigma[28]=  4.21270943562610229280e-01;
#             sigma[29]=  1.20209766313932980600e+00;
#             sigma[30]=  1.20209766313932980600e+00;
#             sigma[31]=  4.21270943562610229280e-01;
#             sigma[32]=  2.94868000440917107580e-01;
#             sigma[33]=  1.52587935405643738980e+00;
#             sigma[34]=  2.56976686507936507940e-01;
#             sigma[35]=  1.79890735229276895940e+00;
#             sigma[36]=  4.11914406966490299820e-01;
#             sigma[37]=  1.28831735008818342150e+00;
#             sigma[38]=  8.46273699294532627870e-01;
#             sigma[39]=  1.28831735008818342150e+00;
#             sigma[40]=  2.94868000440917107580e-01;
#             sigma[41]=  1.52587935405643738980e+00;
#             sigma[42]=  2.56976686507936507940e-01;
#             sigma[43]=  1.79890735229276895940e+00;
#             sigma[44]=  4.11914406966490299820e-01;
#             sigma[45]=  1.27896081349206349210e+00;
#             sigma[46]=  9.32493386243386243390e-01;
#             sigma[47]=  9.32493386243386243390e-01;
#             sigma[48]=  2.94868000440917107580e-01;
#             sigma[49]=  1.52587935405643738980e+00;
#             sigma[50]=  2.56976686507936507940e-01;
#             sigma[51]=  1.79890735229276895940e+00;
#             sigma[52]=  4.11914406966490299820e-01;
#             sigma[53]=  1.27896081349206349210e+00;
#             sigma[54]=  9.23136849647266313930e-01;
#             sigma[55]=  1.01871307319223985890e+00;
#             sigma[56]=  2.94868000440917107580e-01;
#             sigma[57]=  1.52587935405643738980e+00;
#             sigma[58]=  2.56976686507936507940e-01;
#             sigma[59]=  1.79890735229276895940e+00;
#             sigma[60]=  4.11914406966490299820e-01;
#             sigma[61]=  1.27896081349206349210e+00;
#             sigma[62]=  9.23136849647266313930e-01;
#             sigma[63]=  1.00935653659611992950e+00;
#             self.omega[0]=  2.94868000440917107580e-01;
#             self.omega[1]=  1.52587935405643738980e+00;
#             self.omega[2]=  2.56976686507936507940e-01;
#             self.omega[3]=  1.79890735229276895940e+00;
#             self.omega[4]=  4.11914406966490299820e-01;
#             self.omega[5]=  1.27896081349206349210e+00;
#             self.omega[6]=  9.23136849647266313930e-01;
#             self.omega[7]=  1.00935653659611992950e+00;
#         elif self.k==8:
#             sigma[0]=  2.86975446428571428570e-01;
#             sigma[1]=  1.58112723214285714290e+00;
#             sigma[2]=  1.08482142857142857140e-01;
#             sigma[3]=  1.94303571428571428570e+00;
#             sigma[4]=  5.80379464285714285710e-01;
#             sigma[5]=  5.80379464285714285710e-01;
#             sigma[6]=  1.94303571428571428570e+00;
#             sigma[7]=  1.08482142857142857140e-01;
#             sigma[8]=  1.58112723214285714290e+00;
#             sigma[9]=  2.86975446428571428570e-01;
#             sigma[10]=  1.58901978615520282190e+00;
#             sigma[11]=  2.80926201499118165780e-02;
#             sigma[12]=  2.31338734567901234570e+00;
#             sigma[13]= -4.38419036596119929450e-01;
#             sigma[14]=  2.44188767636684303350e+00;
#             sigma[15]= -4.38419036596119929450e-01;
#             sigma[16]=  2.31338734567901234570e+00;
#             sigma[17]=  2.80926201499118165780e-02;
#             sigma[18]=  2.86975446428571428570e-01;
#             sigma[19]=  1.58901978615520282190e+00;
#             sigma[20]=  3.59851741622574955910e-02;
#             sigma[21]=  2.23299782297178130510e+00;
#             sigma[22]= -6.80674052028218694890e-02;
#             sigma[23]=  1.42308917548500881830e+00;
#             sigma[24]=  1.42308917548500881830e+00;
#             sigma[25]= -6.80674052028218694890e-02;
#             sigma[26]=  2.23299782297178130510e+00;
#             sigma[27]=  2.86975446428571428570e-01;
#             sigma[28]=  1.58901978615520282190e+00;
#             sigma[29]=  3.59851741622574955910e-02;
#             sigma[30]=  2.24089037698412698410e+00;
#             sigma[31]= -1.48456927910052910050e-01;
#             sigma[32]=  1.79344080687830687830e+00;
#             sigma[33]=  4.04290674603174603170e-01;
#             sigma[34]=  1.79344080687830687830e+00;
#             sigma[35]= -1.48456927910052910050e-01;
#             sigma[36]=  2.86975446428571428570e-01;
#             sigma[37]=  1.58901978615520282190e+00;
#             sigma[38]=  3.59851741622574955910e-02;
#             sigma[39]=  2.24089037698412698410e+00;
#             sigma[40]= -1.40564373897707231040e-01;
#             sigma[41]=  1.71305128417107583770e+00;
#             sigma[42]=  7.74642305996472663140e-01;
#             sigma[43]=  7.74642305996472663140e-01;
#             sigma[44]=  1.71305128417107583770e+00;
#             sigma[45]=  2.86975446428571428570e-01;
#             sigma[46]=  1.58901978615520282190e+00;
#             sigma[47]=  3.59851741622574955910e-02;
#             sigma[48]=  2.24089037698412698410e+00;
#             sigma[49]= -1.40564373897707231040e-01;
#             sigma[50]=  1.72094383818342151680e+00;
#             sigma[51]=  6.94252783289241622570e-01;
#             sigma[52]=  1.14499393738977072310e+00;
#             sigma[53]=  6.94252783289241622570e-01;
#             sigma[54]=  2.86975446428571428570e-01;
#             sigma[55]=  1.58901978615520282190e+00;
#             sigma[56]=  3.59851741622574955910e-02;
#             sigma[57]=  2.24089037698412698410e+00;
#             sigma[58]= -1.40564373897707231040e-01;
#             sigma[59]=  1.72094383818342151680e+00;
#             sigma[60]=  7.02145337301587301590e-01;
#             sigma[61]=  1.06460441468253968250e+00;
#             sigma[62]=  1.06460441468253968250e+00;
#             sigma[63]=  2.86975446428571428570e-01;
#             sigma[64]=  1.58901978615520282190e+00;
#             sigma[65]=  3.59851741622574955910e-02;
#             sigma[66]=  2.24089037698412698410e+00;
#             sigma[67]= -1.40564373897707231040e-01;
#             sigma[68]=  1.72094383818342151680e+00;
#             sigma[69]=  7.02145337301587301590e-01;
#             sigma[70]=  1.07249696869488536160e+00;
#             sigma[71]=  9.84214891975308641980e-01;
#             sigma[72]=  2.86975446428571428570e-01;
#             sigma[73]=  1.58901978615520282190e+00;
#             sigma[74]=  3.59851741622574955910e-02;
#             sigma[75]=  2.24089037698412698410e+00;
#             sigma[76]= -1.40564373897707231040e-01;
#             sigma[77]=  1.72094383818342151680e+00;
#             sigma[78]=  7.02145337301587301590e-01;
#             sigma[79]=  1.07249696869488536160e+00;
#             sigma[80]=  9.92107445987654320990e-01;
#             self.omega[0]=  2.86975446428571428570e-01;
#             self.omega[1]=  1.58901978615520282190e+00;
#             self.omega[2]=  3.59851741622574955910e-02;
#             self.omega[3]=  2.24089037698412698410e+00;
#             self.omega[4]= -1.40564373897707231040e-01;
#             self.omega[5]=  1.72094383818342151680e+00;
#             self.omega[6]=  7.02145337301587301590e-01;
#             self.omega[7]=  1.07249696869488536160e+00;
#             self.omega[8]=  9.92107445987654320990e-01;
#         else:
#             pass
        
#         self.sigma = np.ascontiguousarray(sigma).reshape((self.k+1,self.k+1))
    
#     # def gregory_weights_matrix(self, n):
#     #     if n<=self.k:
#     #         return self.s[:n+1,:n+1]
        
#     #     S = np.zeros((n+1,n+1))
#     #     A = np.zeros((n+1,n+1))
#     #     S[np.arange(self.k+1,n+1),np.arange(self.k+1,n+1)] += 1
#     #     A[np.arange(self.k+1),np.arange(self.k+1)] += self.h
#     #     S[:self.k+1,:self.k+1] += self.h*self.s
#     #     for m in range(self.k+1,n+1):
#     #         A[m,m:m-self.k-1:-1] += self.a
#     #     return np.linalg.inv(A) @ S
    
#     def gregory_weights(self, n):
#         """
#         Computes the gregory integration wiegths by an iterative method.

#         Parameters
#         ----------
#         n : int
#             Integration order.

#         Returns
#         -------
#         ndarray
#             Gregory weights since order n.

#         """
#         if n+1 <= self.w.shape[0]:
#             return self.w[:n+1,:n+1]
        
#         for k in range(n+1 - self.w.shape[0]):
#             self.__gregory_add()
#         return self.w
    
#     def __gregory_add(self):
#         """
#         Adds a new row to gregory weigths matrix.

#         Returns
#         -------
#         None.

#         """
#         m = self.w.shape[0]
#         new_row = np.zeros(m+1)
#         if m < 2*(self.k+1):
#             new_row[:self.k+1] = self.sigma[m-self.k-1,:]
#             new_row[self.k+1:m+1] = self.omega[m-self.k-1::-1]
#         else:
#             new_row[:self.k+1] = self.omega[:]
#             new_row[self.k+1:m-self.k] += 1
#             new_row[m-self.k:m+1] = self.omega[::-1]
#         new_w = np.zeros((self.w.shape[0]+1,self.w.shape[1]+1), dtype=self.w.dtype)
#         new_w[:-1,:-1] = self.w
#         new_w[-1,:] = new_row
#         self.w = np.copy(new_w)
    
#     def conv_weights(self, m):
#         """
#         Computes the convolution integration weights from t=0 to jh with j=0,...,m.

#         Parameters
#         ----------
#         m : int
#             Time steps to convolve.

#         Returns
#         -------
#         ndarray
#             Convolution weights.

#         """
#         a = np.empty((self.k+1,self.k+1), dtype=np.int32); b = np.empty((self.k+1,self.k+1), dtype=np.int32)
#         for i in prange(self.k+1):
#             for j in prange(self.k+1):
#                 a[i,j] = i
#                 b[j,i] = i
#         factor = (-1)**b * factorial(a) * factorial(b) * m**(a+b+1) / factorial(a+b+1)
#         return self.P.T @ factor @ self.P
