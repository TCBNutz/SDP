""" semidefinite optimization of localizable entanglement, given the reduced
density matrix of a segment and translational invariance """

import numpy as np
from cvxopt import matrix, solvers
import string
from scipy.stats import threshold
import math

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

def array2base(x,base,ndig):
    output=[0]*len(x)
    for i in xrange(len(x)):
        output[i]=int2base(x[i],base).zfill(ndig)
    return(output)

z0=np.array([1.,0.])
z1=np.array([0.,1.])
P=1/np.sqrt(2)*np.array([1.,1.])
M=1/np.sqrt(2)*np.array([1.,-1.])

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

" tensor product of many matrices "
def TMany(x):
    outcome=np.array(1.)
    for i in xrange(len(x)):
        outcome=np.kron(outcome,x[i])
    return(outcome)

" 5 qb SDP "

" making the objective function "

" 0 + 0 "
x1=1/np.sqrt(2.)*TMany(np.array([z0,z0,P,z0,z0])) +\
    1/np.sqrt(2.)*TMany(np.array([z0,z1,P,z1,z0]))
ZPZ=np.kron(x1,x1)

" 0 + 1 "
x1=1/np.sqrt(2.)*TMany(np.array([z0,z0,P,z0,z1])) -\
    1/np.sqrt(2.)*TMany(np.array([z0,z1,P,z1,z1]))
ZPO=np.kron(x1,x1)

" 0 - 0 "
x1=1/np.sqrt(2.)*TMany(np.array([z0,z0,M,z1,z0])) +\
    1/np.sqrt(2.)*TMany(np.array([z0,z1,M,z0,z0]))
ZMZ=np.kron(x1,x1)

" 0 - 1 "
x1=1/np.sqrt(2.)*TMany(np.array([z0,z0,M,z1,z1])) -\
    1/np.sqrt(2.)*TMany(np.array([z0,z1,M,z0,z1]))
ZMO=np.kron(x1,x1)

" 1 + 0 "
x1=1/np.sqrt(2.)*TMany(np.array([z1,z0,P,z0,z0])) -\
    1/np.sqrt(2.)*TMany(np.array([z1,z1,P,z1,z0]))
OPZ=np.kron(x1,x1)

" 1 + 1 "
x1=1/np.sqrt(2.)*TMany(np.array([z1,z0,P,z0,z1])) +\
    1/np.sqrt(2.)*TMany(np.array([z1,z1,P,z1,z1]))
OPO=np.kron(x1,x1)

" 1 - 0 "
x1=1/np.sqrt(2.)*TMany(np.array([z1,z0,M,z1,z0])) -\
    1/np.sqrt(2.)*TMany(np.array([z1,z1,M,z0,z0]))
OMZ=np.kron(x1,x1)

" 1 - 1 "
x1=1/np.sqrt(2.)*TMany(np.array([z1,z0,M,z1,z1])) +\
    1/np.sqrt(2.)*TMany(np.array([z1,z1,M,z0,z1]))
OMO=np.kron(x1,x1)

c=np.real(np.dot(toBloch(5),ZPZ+ZPO+ZMZ+ZMO+OPZ+OPO+OMZ+OMO))+[0.]*1024
c=matrix(c)


" making the reduced state of three qubits "
C3=1/np.sqrt(8.)*np.array([1.,1.,1.,-1.,1.,1.,-1.,1.])
C3=np.kron(C3,C3)
O1=np.kron(TMany(np.array([Z,I,I])),TMany(np.array([Z,I,I])).T)
O2=np.kron(TMany(np.array([I,I,Z])),TMany(np.array([I,I,Z])).T)
O3=np.kron(TMany(np.array([Z,I,Z])),TMany(np.array([Z,I,Z])).T)
red=0.25*(C3+np.dot(O1,C3)+np.dot(O2,C3)+np.dot(O3,C3))
red=np.real(np.dot(toBloch(3),red))

" making G "

GTr1=np.real(np.dot(toBloch(3),np.dot(TrOp([0,0,0,1,1]),np.conj(toBloch(5)).T)))
GTr2=np.real(np.dot(toBloch(3),np.dot(TrOp([1,0,0,0,1]),np.conj(toBloch(5)).T)))
GTr3=np.real(np.dot(toBloch(3),np.dot(TrOp([1,1,0,0,0]),np.conj(toBloch(5)).T)))
Gid=np.real(np.dot(toBloch(5),vectorize(np.identity(32))))
Gnorm5=np.vstack(([0.]*1024,-np.identity(1024)))
Gpos=-np.real(np.conj(toBloch(5)).T)

G=np.vstack((GTr1,GTr2,GTr3,-GTr1,-GTr2,-GTr3,Gid,Gnorm5,Gpos))
G=matrix(G)


hnorm5=np.hstack(([1.],[0.]*1024))
hpos5=[0.]*1024
h=np.hstack((red,red,red,-red,-red,-red,np.array(1.),hnorm5,hpos5))
h=matrix(h)

dims = {'l': 385, 'q': [1025], 's': [32]}
sol = solvers.conelp(c, G, h, dims)
print(array2base(np.nonzero(threshold(sol['x'], 1e-5))[0],4,int(math.log(len(sol['x']),4))))
