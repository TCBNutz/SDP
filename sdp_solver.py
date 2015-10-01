""" semidefinite optimization of localizable entanglement, given the reduced
density matrix of a segment and translational invariance  """

import numpy as np
from cvxopt import matrix, solvers, sparse
from stuff import *

if __name__ == '__main__':
        fBloch=np.conj(toBloch(5)).T#3.125% sparse
        pro=8*TMany([z0,I,P,I,z0,z0,I,P,I,z0])#0.4% sparse

        c=-DMany([np.kron([0.,1.],fBloch).T,pro.T,PT.T,vectorize(np.identity(4))])
        c=matrix(np.real(c) + [0.]*2048)

        G1=np.kron([1.,1.],DMany([toBloch(3),TrOp([1,1,0,0,0]),fBloch])) #0.8% sparse


        G2=np.kron([1.,1.],DMany([toBloch(3),TrOp([1,0,0,0,1]),fBloch])) #0.8% sparse

        G3=np.kron([1.,1.],DMany([toBloch(3),TrOp([0,0,0,1,1]),fBloch])) #0.8% sparse

        Gnorm1=np.vstack(([0.]*2048,np.kron([0.,-1.],np.identity(1024))))
        Gnorm2=np.vstack(([0.]*2048,np.kron([-1.,0.],np.identity(1024))))
        hnorm=[100.]+[0.]*1024

        G5=-np.kron([1.,0.],DMany([PT,pro,fBloch]))
        G6=np.kron([0.,1.],DMany([PT,pro,fBloch]))
        hpos1=[0.]*16
        G7=-np.kron([1.,1.],fBloch)
        hpos2=[0.]*1024

        G=sparse(matrix(np.real(np.vstack((G1,-G1,G2,-G2,G3,-G3,Gnorm1,Gnorm2,G5,G6,G7))) + [[0.]*2048]*3490))

        state=YfaultyCluster(5,0.01)
        h1=DMany([toBloch(3),TrOp([1,1,0,0,0,1]),vectorize(state)])
        h=matrix(np.real(np.hstack((h1,-h1,h1,-h1,h1,-h1,hnorm,hnorm,hpos1,hpos1,hpos2)))+[0.]*3490)

        dims={'l':384 ,'q':[1025,1025], 's':[4,4,32]}

        #sol=solvers.conelp(c, G, h, dims)

        GfourR=np.kron([1.,1.],DMany([toBloch(4),TrOp([1,0,0,0,0]),fBloch]))
        GfourL=np.kron([1.,1.],DMany([toBloch(4),TrOp([0,0,0,0,1]),fBloch]))
        Gfour=GfourR-GfourL
        hfour=np.array([0.]*256)

        Gone=sparse(matrix(np.real(np.vstack((G1,-G1,Gfour,-Gfour))) + [[0.]*2048]*640))
        Gtwo=[sparse(matrix(np.real(np.vstack((G5))) + [[0.]*2048]*16))]
        Gtwo +=[sparse(matrix(np.real(np.vstack((G6))) + [[0.]*2048]*16))]
        Gtwo +=[sparse(matrix(np.real(np.vstack((G7))) + [[0.]*2048]*1024))]
        hone=matrix(np.real(np.hstack((h1,-h1,hfour,-hfour)))+[0.]*640)
        htwo=[matrix([[0.]*4]*4)]
        htwo +=[matrix([[0.]*4]*4)]
        htwo +=[matrix([[0.]*32]*32)]
        solvers.options['reltol']=1e-11
        sol=solvers.sdp(c,Gl=Gone,hl=hone,Gs=Gtwo,hs=htwo,solver='dsdp')
        print np.dot(c.T,sol['x'])
