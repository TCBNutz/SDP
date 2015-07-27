""" semidefinite optimization of localizable entanglement, given the reduced
density matrix of a segment and translational invariance """

import numpy as np
from cvxopt import matrix, solvers
from scipy.stats import threshold
import math

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
    """ tensor product of many matrices """
    return reduce(np.kron, x)


def k(s):
    """ Shortcut to tensor many single-qubit operators """
    return TMany(table[i] for i in s)


if __name__ == '__main__':
    " making the objective function "

    " 0 + 0 "
    x1 = ir2 * (k("00P00") + k("01P10"))
    ZPZ = np.kron(x1, x1)

    " 0 + 1 "
    x1 = ir2 * (k("00P01") - k("01P11"))
    ZPO = np.kron(x1, x1)

    " 0 - 0 "
    x1 = ir2 * (k("00M10") + k("01M00"))
    ZMZ = np.kron(x1, x1)

    " 0 - 1 "
    x1 = ir2 * (k("00M11") - k("01M01"))
    ZMO = np.kron(x1, x1)

    " 1 + 0 "
    x1 = ir2 * (k("10P00") - k("11P10"))
    OPZ = np.kron(x1, x1)

    " 1 + 1 "
    x1 = ir2 * (k("10P01") + k("11P11"))
    OPO = np.kron(x1, x1)

    " 1 - 0 "
    x1 = ir2 * (k("10M10") - k("11M00"))
    OMZ = np.kron(x1, x1)

    " 1 - 1 "
    x1 = ir2 * (k("10M11") + k("11M01"))
    OMO = np.kron(x1, x1)

    c = np.real(np.dot(toBloch(5), ZPZ + ZPO + ZMZ + ZMO + OPZ + OPO + OMZ + OMO)) + \
        [0.] * 1024
    c = matrix(c)

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
    GTr2 = np.real(
        np.dot(toBloch(3), np.dot(TrOp([1, 0, 0, 0, 1]), np.conj(toBloch(5)).T)))
    GTr3 = np.real(
        np.dot(toBloch(3), np.dot(TrOp([1, 1, 0, 0, 0]), np.conj(toBloch(5)).T)))
    Gid = np.real(np.dot(toBloch(5), vectorize(np.identity(32))))
    Gnorm5 = np.vstack(([0.] * 1024, -np.identity(1024)))
    Gpos = -np.real(np.conj(toBloch(5)).T)

    G = np.vstack((GTr1, GTr2, GTr3, -GTr1, -GTr2, -GTr3, Gid, Gnorm5, Gpos))
    G = matrix(G)

    hnorm5 = np.hstack(([1.], [0.] * 1024))
    hpos5 = [0.] * 1024
    h = np.hstack(
        (red, red, red, -red, -red, -red, np.array(1.), hnorm5, hpos5))
    h = matrix(h)

    dims = {'l': 385, 'q': [1025], 's': [32]}
    sol = solvers.conelp(c, G, h, dims)
    print(
        array2base(np.nonzero(threshold(sol['x'], 1e-7))[0], 4, int(math.log(len(sol['x']), 4))))
