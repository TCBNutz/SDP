""" semidefinite optimization of localizable entanglement, given the reduced
density matrix of a segment and translational invariance """

import numpy as np
from cvxopt import matrix, solvers
import string
from scipy.stats import threshold

digs = string.digits + string.letters

def int2base(x, base):
  if x < 0: sign = -1
  elif x == 0: return digs[0]
  else: sign = 1
  x *= sign
  digits = []
  while x:
    digits.append(digs[x % base])
    x /= base
  if sign < 0:
    digits.append('-')
  digits.reverse()
  return ''.join(digits)

z0=np.array([1,0])
z1=np.array([0,1])

I=np.array([[1,0],[0,1]])
X=np.array([[0,1],[1,0]])
Y=np.array([[0,-1j],[1j,0]])
Z=np.array([[1,0],[0,-1]])
Pauli=[I,X,Y,Z]


""" TrOp gives the operator that traces over qubits at positions with a 1
in the list l, acting in the computational basis"""
def TrOp(l):
    nzero=len(np.where(np.array(l)==0)[0])
    none=len(np.where(np.array(l)==1)[0])
    O=[[0]*4**len(l)]*4**nzero
    for k in xrange(2**none):
        L=np.array(1)
        counter=0
        index=int2base(k,2).zfill(none)
        for i in xrange(len(l)):
            if l[i]==0:
                L=np.kron(L,np.identity(2))
            else:
                if int(index[counter])==0:
                    L=np.kron(L,z0)
                else:
                    L=np.kron(L,z1)
                counter=counter+1
        L=np.kron(L,L)
        O=O+L
    return(O)
                   

""" vectorize vectorizes the linear operator (matrix) LinOp """    
def vectorize(LinOp):
    return(np.reshape(LinOp,len(LinOp)*len(LinOp[0])))

""" devectorize makes gives the density matrix represented by the input vector of len d**2 """
def devectorize(HectorTheVector):
    nd=np.sqrt(len(HectorTheVector))
    return(np.reshape(HectorTheVector,(nd,nd)))

"transformation from number basis vectorization to n qubit Bloch vector"
def toBloch(n):
    U=np.array([[0.+0.j]*4**n]*4**n)
    for i in xrange(4**n):
        L=np.array(1.)
        for k in xrange(n):  
            indexx=int(int2base(i,4).zfill(n)[k])
            L=np.kron(L,Pauli[indexx])
        U[i]=np.conj(vectorize(L))
    return(U/(2**(n/2.)))
"""
c=np.real(np.dot(toBloch(2),vectorize(np.kron([[1.,0.],[0.,0.]],[[1.,0.],[0.,0.]]))))+np.array([0.0]*16)
c=matrix(c)

" positive orthant constraints "
T1=np.real(np.dot(toBloch(1),np.dot(TrOp([1,0]),np.conj(toBloch(2)).T)))
h1=np.real(np.dot(toBloch(1),np.array([1.,0.,0.,0.])))
T2=np.real(np.dot(toBloch(1),np.dot(TrOp([0,1]),np.conj(toBloch(2)).T)))
h2=h1
vecId=np.real(np.dot(toBloch(2),vectorize(np.identity(4))))
h3=np.array(1.)

" second order constraints "
Gnorm=np.vstack(([0.]*16,-np.identity(16)))
hnorm=np.hstack(([1.],[0.]*16))

" positive-semidefinite cone "
Gpos=-np.real(np.conj(toBloch(2)).T)
hpos=-np.hstack(([0.],[0.]*15))

" putting junk together "
G=np.vstack((T1,T2,-T1,-T2,vecId,Gnorm,Gpos))
G=matrix(G)

h=np.hstack((h1,h2,-h1,-h2,h3,hnorm,hpos))
h=matrix(h)

dims = {'l': 17, 'q': [17], 's': [4]}
"sol = solvers.conelp(c, G, h, dims)"
"""

" three-qubit cluster state stuff "
PhiPlusP=0.25*(np.kron([[1,0],[0,0]],np.kron([[1,1],[1,1]],[[1,0],[0,0]]))+\
               np.kron([[0,1],[0,0]],np.kron([[1,1],[1,1]],[[0,1],[0,0]]))+\
               np.kron([[0,0],[1,0]],np.kron([[1,1],[1,1]],[[0,0],[1,0]]))+\
               np.kron([[0,0],[0,1]],np.kron([[1,1],[1,1]],[[0,0],[0,1]])))

PsiPlusM=0.25*(np.kron([[1,0],[0,0]],np.kron([[1,-1],[-1,1]],[[0,0],[0,1]]))+\
               np.kron([[0,0],[1,0]],np.kron([[1,-1],[-1,1]],[[0,1],[0,0]]))+\
               np.kron([[0,1],[0,0]],np.kron([[1,-1],[-1,1]],[[0,0],[1,0]]))+\
               np.kron([[0,0],[0,1]],np.kron([[1,-1],[-1,1]],[[1,0],[0,0]])))

c=np.real(np.dot(toBloch(3),vectorize(PhiPlusP+PsiPlusM)))+[0.]*64
c=matrix(c)

"reduced density matrix of two qbs after tracing out a boundary qb in a 3qb cluster state"
Had=np.kron(np.identity(2),1/np.sqrt(2)*np.array([[1,1],[1,-1]]))
halfhalf=0.25*np.array([[1,0,0,1],[0,1,1,0],[0,1,1,0],[1,0,0,1]])
dmat=np.dot(Had,np.dot(halfhalf,Had))
red3=np.real(np.dot(toBloch(2),vectorize(dmat)))

halfhalf=0.5*np.array([[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]])
dmat=np.dot(Had,np.dot(halfhalf,Had))
red1=np.real(np.dot(toBloch(2),vectorize(dmat)))

hnorm3=np.hstack(([1.],[0.]*64))
hpos3=np.hstack(([0.],[0.]*63))
h=np.hstack((red1,red3,-red1,-red3,np.array(1.),hnorm3,hpos3))
h=matrix(h)

GTr1=np.real(np.dot(toBloch(2),np.dot(TrOp([1,0,0]),np.conj(toBloch(3)).T)))
GTr3=np.real(np.dot(toBloch(2),np.dot(TrOp([0,0,1]),np.conj(toBloch(3)).T)))
G3=np.real(np.dot(toBloch(3),vectorize(np.identity(8))))
Gnorm3=np.vstack(([0.]*64,-np.identity(64)))
Gpos=-np.real(np.conj(toBloch(3)).T)

G=np.vstack((GTr1,GTr3,-GTr1,-GTr3,G3,Gnorm3,Gpos))
G=matrix(G)

dims = {'l': 65, 'q': [65], 's': [8]}
sol = solvers.conelp(c, G, h, dims)
Bloch=threshold(sol['x'], 0.0001)
print(1/np.sqrt(8.)*Bloch)
"3qb cluster state as Bloch vector"
C3=np.real(np.dot(toBloch(3),np.kron(1/np.sqrt(8.)*np.array([1.,1.,1.,-1.,1.,1.,-1.,1.]),1/np.sqrt(8.)*np.array([1.,1.,1.,-1.,1.,1.,-1.,1.]))))
S=np.dot(GTr3,C3)
