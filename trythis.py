import numpy as np
from cvxopt import matrix,solvers
from negSDP import devectorize,vectorize,toBloch,PT,DMany
from scipy.stats import threshold
c=[0.]*16+[-1.]+[0.]*15
c=matrix(c)

G1=np.kron([1.,1.],np.identity(16))
h1=np.dot(np.real(toBloch(2)),vectorize(0.5*np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])))

G2=np.array([1.]+[0.]*15)
G2=np.dot(G2,G1)
h2=np.array(1./2)

G3=np.vstack(([0.]*16,-np.identity(16)))
G3=np.dot(G3,G1)
h3=[1.]+[0.]*16

G31=np.vstack(([0.]*16,-np.identity(16)))
G31=np.dot(G31,np.kron([1.,0.],np.identity(16)))
h31=[1.]+[0.]*16

G32=np.vstack(([0.]*16,-np.identity(16)))
G32=np.dot(G32,np.kron([0.,-1.],np.identity(16)))
h32=[1.]+[0.]*16

G4=-np.dot(np.real(np.conj(toBloch(2)).T),G1)
h4=[0.]*16
G5=-DMany([PT,np.real(np.conj(toBloch(2)).T),np.kron([1.,0.],np.identity(16))])
h5=h4
G6=DMany([PT,np.real(np.conj(toBloch(2)).T),np.kron([0.,1.],np.identity(16))])
h6=h4

G=matrix(np.vstack((G1,-G1,G3,G31,G32,G4,G5,G6)))
h=np.hstack((h1,-h1,h3,h31,h32,h4,h5,h6))
h=matrix(h)
dims={'l':32, 'q':[17,17,17], 's':[4,4,4]}
sol = solvers.conelp(c, G, h, dims)
print np.linalg.eig(devectorize(DMany([PT,np.real(np.conj(toBloch(2)).T),sol['x'][:16]])))[0]
