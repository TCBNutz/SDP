import numpy as np
from cvxopt import matrix,solvers
from negSDP import devectorize,vectorize,toBloch,PT,DMany,TMany,PhiPlus,I,P,ClusterState, TrOp

fBloch=np.conj(toBloch(3)).T
pro=2*TMany([I,P,I,I,P,I])

c=-DMany([np.kron([0,1],fBloch).T,pro.T,PT.T,vectorize(np.identity(4))])
c=matrix(np.real(c) + [0.]*128)

G1=np.kron([1.,1.],DMany([toBloch(2),TrOp([1,0,0]),fBloch]))
h1=DMany([toBloch(2),TrOp([1,0,0]),vectorize(ClusterState(3))])

G2=np.kron([1.,1.],DMany([toBloch(2),TrOp([0,0,1]),fBloch]))
h2=DMany([toBloch(2),TrOp([0,0,1]),vectorize(ClusterState(3))])

Gnorm1=np.vstack(([0.]*128,np.kron([0.,-1.],np.identity(64))))
Gnorm2=np.vstack(([0.]*128,np.kron([-1.,0.],np.identity(64))))
hnorm=[100.]+[0.]*64

G3=-np.kron([1.,0.],DMany([PT,pro,fBloch]))
G4=np.kron([0.,1.],DMany([PT,pro,fBloch]))
hpos1=[0.]*16
G5=-np.kron([1.,1.],fBloch)
hpos2=[0.]*64

G=matrix(np.real(np.vstack((G1,-G1,G2,-G2,Gnorm1,Gnorm2,G3,G4,G5))) + [[0.]*128]*290)
h=matrix(np.real(np.hstack((h1,-h1,h2,-h2,hnorm,hnorm,hpos1,hpos1,hpos2))) + [0.]*290)

dims={'l':64 ,'q':[65,65], 's':[4,4,8]}
sol = solvers.conelp(c, G, h, dims)

p=sol['x']
print np.linalg.eig(devectorize(np.dot(fBloch,p[:64]+p[64:])))[0]
