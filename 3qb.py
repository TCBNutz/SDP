""" semidefinite optimization of localizable entanglement, given the reduced
density matrix of a segment and translational invariance  """

import numpy as np
from cvxopt import matrix, solvers, sparse
from stuff import *

if __name__ == '__main__':
        fBloch=np.conj(toBloch(3)).T#3.125% sparse
        pro=2*TMany([I,P,I,I,P,I])#0.4% sparse

        c=-DMany([np.kron([0.,1.],fBloch).T,pro.T,PT.T,vectorize(np.identity(4))])
        c=matrix(np.real(c) + [0.]*128)

        G1=np.kron([1.,1.],DMany([toBloch(2),TrOp([1,0,0]),fBloch])) #0.8% sparse


        G2=np.kron([1.,1.],DMany([toBloch(2),TrOp([0,0,1]),fBloch])) #0.8% sparse

        Gnorm1=np.vstack(([0.]*128,np.kron([0.,-1.],np.identity(64))))
        Gnorm2=np.vstack(([0.]*128,np.kron([-1.,0.],np.identity(64))))
        hnorm=[100.]+[0.]*64

        G5=-np.kron([1.,0.],DMany([PT,pro,fBloch]))
        G6=np.kron([0.,1.],DMany([PT,pro,fBloch]))
        hpos1=[0.]*16
        G7=-np.kron([1.,1.],fBloch)
        hpos2=[0.]*64

        G=sparse(matrix(np.real(np.vstack((G1,-G1,G2,-G2,Gnorm1,Gnorm2,G5,G6,G7))) + [[0.]*128]*290))

        pro2=TMany([I,np.array([[2.,0.],[0.,0.]]),I,I,I])
        state=DMany([TrOp([1,1,0,0,0]),np.kron(pro2,pro2),vectorize(YfaultyCluster(4,0.000))])
        h1=DMany([toBloch(2),TrOp([1,0,0]),state])
        h2=DMany([toBloch(2),TrOp([0,0,1]),state])
        h=matrix(np.real(np.hstack((h1,-h1,h2,-h2,hnorm,hnorm,hpos1,hpos1,hpos2)))+[0.]*290)

        dims={'l':64 ,'q':[65,65], 's':[4,4,8]}

        sol=solvers.conelp(c, G, h, dims)
