import numpy as np
from cvxopt import matrix,solvers
from negSDP import devectorize,vectorize,toBloch,PT,DMany,PhiPlus
from scipy.stats import threshold
c=[0.]*16+[-1.]+[0.]*15
c=matrix(c)

G1=np.kron([1.,1.],np.identity(16))
h1=np.real(DMany([toBloch(2),PT,vectorize(PhiPlus)]))

Gnorm1=np.vstack(([0.]*32,np.kron([0.,-1.],np.identity(16))))
Gnorm2=np.vstack(([0.]*32,np.kron([-1.,0.],np.identity(16))))
hnorm=[10.]+[0.]*16

G2=np.real(np.dot(np.conj(toBloch(2)).T,np.kron([0.,1.],np.identity(16))))
h2=[0.]*16

G3=-np.real(np.dot(np.conj(toBloch(2)).T,np.kron([1.,0.],np.identity(16))))
h3=[0.]*16

G=np.vstack((G1,-G1,Gnorm1,Gnorm2,G2,G3))
G=matrix(G)

h=matrix(np.hstack((h1,-h1,hnorm,hnorm,h2,h3)).T)

dims={'l':32, 'q':[17,17], 's':[4,4]}
sol = solvers.conelp(c, G, h, dims)

AB=np.array([[0.5,0.,0.,0.],[0.,0.,0.5,0.],[0.,0.5,0.,0.],[0.,0.,0.,0.5]])
ab=np.real(np.dot(toBloch(2),vectorize(AB)))
A=np.array([[0.5,0.,0.,0.],[0.,0.25,0.25,0.],[0.,0.25,0.25,0.],[0.,0.,0.,0.5]])
B=np.array([[0.,0.,0.,0.],[0.,-0.25,0.25,0.],[0.,0.25,-0.25,0.],[0.,0.,0.,0.]])
a=np.real(np.dot(toBloch(2),vectorize(A)))
b=np.real(np.dot(toBloch(2),vectorize(B)))
p=np.hstack((a,b))

print np.max(sol['x'] - matrix(p))
