""" semidefinite optimization of localizable entanglement, given the reduced
density matrix of a segment and translational invariance """

import numpy as np
from cvxopt import matrix, solvers
from scipy.stats import threshold
import math
import itertools

# Constants
ir2 = 1 / np.sqrt(2)
z0 = np.array([1., 0.])
z1 = np.array([0., 1.])
P = ir2 * np.array([1., 1.])
M = ir2 * np.array([1., -1.])
I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
Pauli = [I, X, Y, Z]
table = {"0": z0, "1": z1, "P": P, "M": M, "I": I, "X": X, "Y": Y, "Z": Z}
"partial transpose acting on vectorized 2qb density matrix in computational basis"
PT=np.array([[1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],\
             [0.,0.,0.,0.,2.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],\
             [0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],\
             [0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],\
             [0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],\
             [0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],\
             [0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],\
             [0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.],\
             [0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],\
             [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.],\
             [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,1.,0.],\
             [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],\
             [0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.],\
             [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.],\
             [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.],\
             [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.]])
PhiPlus=0.5*np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])


def int2base(x, base):
    """ Represent a number in some base """
    return np.base_repr(x, base).lower()


def array2base(x, base, ndig):
    """ Represent an array in some base """
    return [int2base(i, base).zfill(ndig) for i in x]


def TrOp(l):
    """
    TrOp gives the operator that traces over qubits at positions with a 1
    in the list l, acting in the computational basis
    """
    nzero = len(np.where(np.array(l) == 0)[0])
    none = len(np.where(np.array(l) == 1)[0])
    O = [[0] * 4 ** len(l)] * 4 ** nzero
    for k in xrange(2 ** none):
        L = np.array(1)
        counter = 0
        index = int2base(k, 2).zfill(none)
        for i in xrange(len(l)):
            if l[i] == 0:
                L = np.kron(L, np.identity(2))
            else:
                if int(index[counter]) == 0:
                    L = np.kron(L, z0)
                else:
                    L = np.kron(L, z1)
                counter = counter + 1
        L = np.kron(L, L)
        O = O + L
    return O


def vectorize(LinOp):
    """ vectorize vectorizes the linear operator (matrix) LinOp """
    return np.reshape(LinOp, len(LinOp) * len(LinOp[0]))


def devectorize(HectorTheVector):
    """ devectorize makes gives the density matrix represented by the input vector of len d**2 """
    nd = np.sqrt(len(HectorTheVector))
    return np.reshape(HectorTheVector, (nd, nd))


def toBloch(n):
    """ transformation from number basis vectorization to n qubit Bloch vector"""
    U = np.array([[0. + 0.j] * 4 ** n] * 4 ** n)
    for i in xrange(4 ** n):
        L = np.array(1.)
        for k in xrange(n):
            indexx = int(int2base(i, 4).zfill(n)[k])
            L = np.kron(L, Pauli[indexx])
        U[i] = np.conj(vectorize(L))
    return U / (2 ** (n / 2.))


def TMany(x):
    """ tensor product of matrices in list x """
    return reduce(np.kron, x)

def DMany(x):
    """ dot product of matrices in list x """
    return reduce(np.dot,x)


def k(s):
    """ Shortcut to tensor many single-qubit operators """
    return TMany(table[i] for i in s)


def ClusterState(n):
    """ makes density matrix of n-qubit cluster state in computational basis
    n >= 3"""
    generator=[0]*n
    generator[0]=k("XZ"+"I"*(n-2))
    for i in range(1,n-1):
        generator[i]=k("I"*(i-1)+"ZXZ"+"I"*(n-2-i))
    generator[n-1]=k("I"*(n-2)+"ZX")
    stabilizer=[]
    for i in range(2,n+1):
        combos=list(itertools.combinations(range(n),i))
        for v in xrange(len(combos)):
            stabilizer.append(DMany(generator[m] for m in combos[v]))
    stabilizer=stabilizer + generator
    dmat=1/(2.**n)*(np.identity(2**n)+sum(stabilizer[a] for a in range(len(stabilizer))))
    return dmat


if __name__ == '__main__':
    " making the objective function "
    al=TMany([z0,z0,P,z0,z0])
    al=np.kron(al,al.T)
    be=TMany([z0,z0,P,z1,z0])
    be=np.kron(be,be.T)
    ga=TMany([z0,z1,P,z0,z0])
    ga=np.kron(ga,ga.T)
    de=TMany([z0,z1,P,z1,z0])
    de=np.kron(de,de.T)
    abcd=np.dot(toBloch(5),al+be+ga+de)
    PP=8.*np.kron(TMany([z0,I,P,I,z0]),TMany([z0,I,P,I,z0]))
    abcd=np.dot([1.]+[0.]*15,PP)
    c=np.real(np.hstack(([0.]*1024,[-1.]+[0.]*1023))) + [0.]*2048
    c=matrix(c)
    
    " making the ideal reduced state of three qubits "
    C3 = 1 / np.sqrt(8.) * np.array([1., 1., 1., -1., 1., 1., -1., 1.])
    C3 = np.kron(C3, C3)
    O1 = np.kron(k("ZII"), k("ZII").T)
    O2 = np.kron(k("IIZ"), k("IIZ").T)
    O3 = np.kron(k("ZIZ"), k("ZIZ").T)
    red = 0.25 * (C3 + np.dot(O1, C3) + np.dot(O2, C3) + np.dot(O3, C3))
    red = np.real(np.dot(toBloch(3), red))

    " making G "
    GTr1 = np.real(
        np.dot(toBloch(3), np.dot(TrOp([0, 0, 0, 1, 1]), np.conj(toBloch(5)).T)))
    GTr1=np.dot(GTr1,np.kron([1.,1.],np.identity(1024)))
    
    GTr2 = np.real(
        np.dot(toBloch(3), np.dot(TrOp([1, 0, 0, 0, 1]), np.conj(toBloch(5)).T)))
    GTr2=np.dot(GTr2,np.kron([1.,1.],np.identity(1024)))
    
    GTr3 = np.real(
        np.dot(toBloch(3), np.dot(TrOp([1, 1, 0, 0, 0]), np.conj(toBloch(5)).T)))
    GTr3=np.dot(GTr3,np.kron([1.,1.],np.identity(1024)))
    
    Gid = 2**(5./2.)*np.array([1.]+[0.]*1023+[1.]+[0.]*1023)
        
    Gnorm5 = np.vstack(([0.] * 2048,-np.kron([1.,1.],np.identity(1024))))

    Gnorm51 = np.vstack(([0.] * 2048,-np.kron([1.,0.],np.identity(1024))))

    Gnorm52 = np.vstack(([0.] * 2048,-np.kron([0.,-1.],np.identity(1024))))
    
    Gpos = -np.real(np.conj(toBloch(5)).T)
    Gpos=np.dot(Gpos,np.kron([1.,1.],np.identity(1024)))
    
    GposPT1 = - np.real(DMany([PT,PP,np.conj(toBloch(5)).T,np.kron([1.,0.],np.identity(1024))]))

    GposPT2 = np.real(DMany([PT,PP,np.conj(toBloch(5)).T,np.kron([0.,1.],np.identity(1024))]))
    
    G = np.vstack((GTr1, GTr2, GTr3, -GTr1, -GTr2, -GTr3,Gnorm5,Gnorm51,Gnorm52,Gpos,GposPT1,GposPT2))
    G = matrix(G)

    hnorm5 = np.hstack(([1.], [0.] * 1024))
    
    hpos5 = [0.] * 1024

    hposPT1=[0.]*16

    hposPT2=[0.]*16
    
    h = np.hstack(
        (red, red, red, -red, -red, -red,hnorm5,hnorm5,hnorm5,hpos5,hposPT1,hposPT2))
    h = matrix(h)
    
    dims = {'l': 384, 'q': [1025,1025,1025], 's': [32,4,4]}
    
    sol = solvers.conelp(c, G, h, dims)
    print(
        array2base(np.nonzero(threshold(sol['x'], 1e-7))[0], 4, int(math.log(len(sol['x']), 4))))
