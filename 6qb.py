""" semidefinite optimization of localizable entanglement, given the reduced
density matrix of a segment and translational invariance  """

import numpy as np
from cvxopt import matrix, solvers, sparse
from multiprocessing import Pool
import timeit
from stuff import *

if __name__ == '__main__':
        fBloch=np.conj(toBloch(6)).T#3.125% sparse
        pro=8*TMany([z0,I,P,P,I,z0,z0,I,P,P,I,z0])#0.4% sparse

        c=-DMany([np.kron([0.,1.],fBloch).T,pro.T,PT.T,vectorize(np.identity(4))])
        c=matrix(np.real(c) + [0.]*8192)

        G1=np.kron([1.,1.],DMany([toBloch(3),TrOp([1,1,1,0,0,0]),fBloch])) #0.8% sparse


        G2=np.kron([1.,1.],DMany([toBloch(3),TrOp([1,1,0,0,0,1]),fBloch])) #0.8% sparse

        G3=np.kron([1.,1.],DMany([toBloch(3),TrOp([1,0,0,0,1,1]),fBloch])) #0.8% sparse

	G4=np.kron([1.,1.],DMany([toBloch(3),TrOp([0,0,0,1,1,1]),fBloch]))

        Gnorm1=np.vstack(([0.]*8192,np.kron([0.,-1.],np.identity(4096))))
        Gnorm2=np.vstack(([0.]*8192,np.kron([-1.,0.],np.identity(4096))))
        hnorm=[100.]+[0.]*4096

        G5=-np.kron([1.,0.],DMany([PT,pro,fBloch]))
        G6=np.kron([0.,1.],DMany([PT,pro,fBloch]))
        hpos1=[0.]*16
        G7=-np.kron([1.,1.],fBloch)
        hpos2=[0.]*4096

        G=sparse(matrix(np.real(np.vstack((G1,-G1,G2,-G2,G3,-G3,G4,-G4,Gnorm1,Gnorm2,G5,G6,G7))) + [[0.]*8192]*12834))

        state=YfaultyCluster(5,0.01)
        h1=DMany([toBloch(3),TrOp([1,1,0,0,0,1]),vectorize(state)])
        h=matrix(np.real(np.hstack((h1,-h1,h1,-h1,h1,-h1,h1,-h1,hnorm,hnorm,hpos1,hpos1,hpos2)))+[0.]*12834)

        dims={'l':512 ,'q':[4097,4097], 's':[4,4,64]}

        def SDP():
                return solvers.conelp(c, G, h, dims)

        t=timeit.Timer("SDP()","from __main__ import SDP")
        time=t.timeit(1)
        print time

        """
	def SDPp():
		p=Pool(4)
		a=p.map(SDP,[1])
		return a
        """
