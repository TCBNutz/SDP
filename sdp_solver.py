import mosek.fusion 
from   mosek.fusion import * 
from stuff import *
import numpy as np
import math
import sys
from scipy import sparse

def makeSparse(mat):
    titi=sparse.coo_matrix(np.around(mat,6))
    a=[int(list(titi.row)[i]) for i in xrange(len(titi.row))]
    b=[int(list(titi.col)[i]) for i in xrange(len(titi.col))]
    c=list(titi.data)
    return Matrix.sparse(len(mat),len(mat[0]),a,b,c)

n=5
m=3
state=YfaultyCluster(5,0.01)
h1=devectorize(np.dot(TrOp([1,1,0,0,0,1]),vectorize(state)))
red=h1
pro=8*TMany([z0,I,P,I,z0,z0,I,P,I,z0])
fBloch=np.conj(toBloch(n)).T
m=int(math.log(len(red),2))
redB=np.real(np.dot(toBloch(m),vectorize(red)))
PTproR=makeSparse(np.real(DMany([PT,8*TMany([z0,I,P,I,z0,z0,I,P,I,z0]),np.conj(toBloch(5)).T])))
PTproI=makeSparse(np.imag(DMany([PT,8*TMany([z0,I,P,I,z0,z0,I,P,I,z0]),np.conj(toBloch(5)).T])))
T12=makeSparse(np.real(DMany([toBloch(m),TrOp([1,1,0,0,0]),fBloch])))
T15=makeSparse(np.real(DMany([toBloch(m),TrOp([1,0,0,0,1]),fBloch])))
T45=makeSparse(np.real(DMany([toBloch(m),TrOp([0,0,0,1,1]),fBloch])))
T1215=makeSparse(np.real(DMany([toBloch(m),TrOp([1,1,0,0,0])-TrOp([1,0,0,0,1]),fBloch])))
T1245=makeSparse(np.real(DMany([toBloch(m),TrOp([1,1,0,0,0])-TrOp([0,0,0,1,1]),fBloch])))
c=-np.real(DMany([np.kron([0.,1.],fBloch).T,pro.T,PT.T,vectorize(np.identity(4))]))

  
def main(red,n):
        with Model("Negativity") as M:
                t=M.variable("slack",1,Domain.unbounded())
                Ra=M.variable("Ra",4**n,Domain.inRange(-1.,1.))
                Rb=M.variable("Rb",4**n,Domain.inRange(-1.,1.))
                RaE=Expr.reshape(Ra.asExpr(),NDSet(2**n,2**n))
                RbE=Expr.reshape(Rb.asExpr(),NDSet(2**n,2**n))
                
                cTr12=M.constraint("C1",Expr.mul(T12,Expr.add(Ra,Rb)),Domain.equalsTo(redB))
                cTr15=M.constraint("C2",Expr.mul(T1215,Expr.add(Ra,Rb)),Domain.equalsTo([0.]*64))
                cTr45=M.constraint("C3",Expr.mul(T1245,Expr.add(Ra,Rb)),Domain.equalsTo([0.]*64))

                ImPall=Expr.reshape(Expr.mul(DenseMatrix(np.imag(fBloch)),Expr.add(Ra,Rb)),NDSet(2**n,2**n))
                RePall=Expr.reshape(Expr.mul(DenseMatrix(np.real(fBloch)),Expr.add(Ra,Rb)),NDSet(2**n,2**n))
                #PSDall=Expr.vstack(Expr.hstack(RePall,Expr.mul(-1.,ImPall)),Expr.hstack(ImPall,RePall))
                
                ImPa=Expr.reshape(Expr.mul(PTproI,Ra),NDSet(4,4))
                RePa=Expr.reshape(Expr.mul(PTproR,Ra),NDSet(4,4))
                #PSD1=Expr.vstack(Expr.hstack(RePa,Expr.mul(-1.,ImPa)),Expr.hstack(ImPa,RePa))
                
                ImPb=Expr.reshape(Expr.mul(PTproI,Rb),NDSet(4,4))
                RePb=Expr.reshape(Expr.mul(PTproR,Rb),NDSet(4,4))
                #PSD2=Expr.mul(-1.,Expr.vstack(Expr.hstack(RePb,Expr.mul(-1.,ImPb)),Expr.hstack(ImPb,RePb)))
                
                cPSDall=M.constraint(RePall,Domain.inPSDCone())
                cPSDa=M.constraint(RePa,Domain.inPSDCone())
                cPSDb=M.constraint(Expr.mul(-1.,RePb),Domain.inPSDCone())

                M.constraint(Expr.sub(t.asExpr(),Expr.dot(vectorize(-np.identity(4)),Expr.mul(PTproR,Rb))),Domain.greaterThan(0.))

                M.objective(ObjectiveSense.Minimize,t)
                M.solve()
                return t.level()


if __name__ == '__main__':
        print main(h1,5)
        sys.exit(0)
