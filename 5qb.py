""" semidefinite optimization of localizable entanglement, given the reduced
density matrix of a segment and translational invariance  """

import numpy as np
from cvxopt import matrix, solvers, sparse
from stuff import *

if __name__ == '__main__':
        M1=np.array([[0]*1888]*1024)
        M2=np.array([[0]*1888]*1024)
        l=range(64)+range(64,253,4)+range(256,1009,16)
        counter=0
        for k in xrange(1024):
                if k in l:
                        M1[k][k+864]=-1
                else:
                        M1[k][counter]=1
                        M2[k][counter]=1
                        M2[k][k+864]=1
                        counter=counter+1
                        
        state=YfaultyCluster(5,0.01)
        redState=DMany([toBloch(3),TrOp([1,1,0,0,0,1]),vectorize(state)])
        kb=[0.]*1024 #known bit of the five-qubit state
        kb[0:64]=0.5*redState
        for m in xrange(48):
                kb[64+m*4]=0.5*redState[m+16]
                kb[256+m*16]=0.5*redState[m+16]

        fBloch=np.conj(toBloch(5)).T#3.125% sparse
        pro=8*TMany([z0,I,P,I,z0,z0,I,P,I,z0])#0.4% sparse

        c=-DMany([fBloch.T,pro.T,PT.T,vectorize(np.identity(4))])
        c=np.concatenate((np.array([0.]*864),c))
        c=matrix(np.real(c) + [0.]*1888)

        Gnorm1=np.vstack(([0.]*1888,np.hstack((-1.*np.identity(864),[[0.]*1024]*864 )) ))
        Gnorm2=np.vstack(([0.]*1888,np.hstack(([[0.]*864]*1024,-1.*np.identity(1024) )) ))
        
        hnorm1=[100.]+[0.]*864
        hnorm2=[100.]+[0.]*1024

        G5=-DMany([PT,pro,fBloch,M1])
        hpos1=DMany([PT,pro,fBloch,kb])
        G6=np.hstack(([[0.]*864]*16, DMany([PT,pro,fBloch]) ))
        hpos2=[0.]*16
        G7=-np.dot(fBloch,M2)
        hpos3=np.dot(fBloch,kb)
        
        G=sparse(matrix(np.real(np.vstack((Gnorm1,Gnorm2,G5,G6,G7))) + [[0.]*1888]*2946))

        h=matrix(np.real(np.hstack((hnorm1,hnorm2,hpos1,hpos2,hpos3)))+[0.]*2946)
        
        dims={'l':0 ,'q':[865,1025], 's':[4,4,32]}

        sol=solvers.conelp(c, G, h, dims)
