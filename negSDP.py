""" semidefinite optimization of localizable entanglement, given the reduced
density matrix of a segment and translational invariance """

import numpy as np
from cvxopt import matrix, solvers
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
             [0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],\
             [0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],\
             [0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],\
             [0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],\
             [0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],\
             [0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],\
             [0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.],\
             [0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],\
             [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.],\
             [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.],\
             [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.],\
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
    fBloch=np.conj(toBloch(5)).T
    pro=8*TMany([z0,I,P,I,z0,z0,I,P,I,z0])

    c=-DMany([np.kron([0.,1.],fBloch).T,pro.T,PT.T,vectorize(np.identity(4))])
    c=matrix(np.real(c) + [0.]*2048)

    G1=np.kron([1.,1.],DMany([toBloch(3),TrOp([1,1,0,0,0]),fBloch]))
    h1=DMany([toBloch(3),TrOp([1,1,0,0,0]),vectorize(ClusterState(5))])

    G2=np.kron([1.,1.],DMany([toBloch(3),TrOp([1,0,0,0,1]),fBloch]))
    h2=DMany([toBloch(3),TrOp([1,0,0,0,1]),vectorize(ClusterState(5))])

    G3=np.kron([1.,1.],DMany([toBloch(3),TrOp([0,0,0,1,1]),fBloch]))
    h3=DMany([toBloch(3),TrOp([0,0,0,1,1]),vectorize(ClusterState(5))])

    Gnorm1=np.vstack(([0.]*2048,np.kron([0.,-1.],np.identity(1024))))
    Gnorm2=np.vstack(([0.]*2048,np.kron([-1.,0.],np.identity(1024))))
    hnorm=[100.]+[0.]*1024

    G4=-np.kron([1.,0.],DMany([PT,pro,fBloch]))
    G5=np.kron([0.,1.],DMany([PT,pro,fBloch]))
    hpos1=[0.]*16
    G6=-np.kron([1.,1.],fBloch)
    hpos2=[0.]*1024

    G=matrix(np.real(np.vstack((G1,-G1,G2,-G2,G3,-G3,Gnorm1,Gnorm2,G4,G5,G6))) + [[0.]*2048]*3490)
    h=matrix(np.real(np.hstack((h1,-h1,h2,-h2,h3,-h3,hnorm,hnorm,hpos1,hpos1,hpos2)))+[0.]*3490)

    dims={'l':384 ,'q':[1025,1025], 's':[4,4,32]}
    sol = solvers.conelp(c, G, h, dims)
