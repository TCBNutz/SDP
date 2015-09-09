import numpy as np
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
dici={'0':'I','1':'X','2':'Y','3':'Z'}

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

def state2dm(state):
    """ makes density matrix for pure state """
    return np.kron([[state[i]] for i in xrange(len(state))],state)
    
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

def CNOT(t,ph):
    """
    CNOT between emitter and target photon t in a state of emitter + ph photons. t>=1
    """
    P1=np.kron(state2dm(z0),k('I'*ph))
    P2=np.kron(state2dm(z1),k('I'*(t-1)+'X'+'I'*(ph-t)))
    return P1+P2

def CPmap(rho,OpEl):
    """
    CPmap with [Operation Elements] on density matrix rho
    """
    return reduce(lambda X,Y: X + DMany([Y,rho,np.conj(Y).T]),OpEl,0)
    
def faultyCluster(n,M):
    """
    density matrix of Cluster state approximation with n photons, M is list of single
    qubit operation elements (as np.arrays) 
    """
    rh=state2dm(k('0'*(n+1)))
    Mbig=[0]*len(M)
    for i in xrange(len(M)):
        Mbig[i]=np.kron(M[i],k('I'*n))
    for i in xrange(n):
        rh=CPmap(rh,Mbig)
        rh=DMany([CNOT(n-i,n),rh,CNOT(n-i,n)])
    return CPmap(rh,Mbig)

def YfaultyCluster(n,p):
    """
    case of probability p of Y error before every emission
    """
    M=[np.sqrt((1-p)/2.)*np.array([[1,1],[1,-1]]),1j*np.sqrt(p/2.)*np.array([[-1,1],[1,1]])]
    return faultyCluster(n,M)

def negativity(rho):
    """
    computes negativity of two-qubit density matrix rho
    """
    pt=devectorize(np.dot(PT,vectorize(rho)))
    return 0.5*(sum(np.abs(np.linalg.eig(pt)[0]))-1)

def Num2Op(string):
    """
    replaces a string of integers with a string of {I,X,Y,Z}, as in
    '313' to 'ZXZ'
    """
    string=list(string)
    stringnew=[dici[string[k]] for k in xrange(len(string))]
    return "".join(stringnew)

def skim(BlochVector):
    """
    finds the operators that have expectation values > 1e-5.
    """
    n=int(math.log(len(BlochVector),4))
    BV=np.around(BlochVector,5)
    pos=np.nonzero(BV)[0]
    ski=array2base(pos,4,n)
    ski1=[Num2Op(ski[i])+'='+str(BV[pos[i]]) for i in xrange(len(ski))]
    return np.matrix(ski1)
    
