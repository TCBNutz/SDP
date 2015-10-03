import mosek.fusion 
from   mosek.fusion import * 
from stuff import *
import numpy as np
import sys

# Werner state with lambda=0.5
"""
rho=0.125*np.identity(4)+0.5*state2dm(ir2*np.array([1,0,0,1]))

def ParTrans(r):
        return(devectorize(np.dot(PT,vectorize(r))))

rhoTB=ParTrans(rho)
"""
rhoTB=[[ 0.375,  0.   ,  0.   ,  0.   ],
       [ 0.   ,  0.125,  0.25 ,  0.   ],
       [ 0.   ,  0.25 ,  0.125,  0.   ],
       [ 0.   ,  0.   ,  0.   ,  0.375]]


def main():

    with Model("Negativity") as M:
        Xx=M.variable("Xx",Domain.inPSDCone(4))
        Yy=M.variable("Yy",Domain.inPSDCone(4))
        C=M.constraint("this",Expr.sub(Xx,Yy),Domain.equalsTo(DenseMatrix(rhoTB)))
        M.objective(ObjectiveSense.Minimize,Expr.sum(Yy.diag()))
        M.solve()
        return np.reshape(Yy.level(),(4,4))

 
if __name__ == '__main__': 
    print main()
    sys.exit(0)     

