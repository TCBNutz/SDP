""" making the density matrix of a Cluster state approximation assuming Markovian noise
on the emitter and perfect CNOT gates"""

from sdpete import *
from negSDP import *

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
