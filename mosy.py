import sys
import mosek
import mosek.fusion
from   mosek.fusion import *


def main(args):
    rho=

    with Model("Negativity") as M:

      # Setting up the variables
      X = M.variable("X", Domain.inPSDCone(N))
      t = M.variable("t", 1, Domain.unbounded())

      # (t, vec (A-X)) \in Q
      M.constraint("C1", Expr.vstack(t, vec(Expr.sub(DenseMatrix(A),X))), Domain.inQCone() );

      # diag(X) = e
      M.constraint("C2",X.diag(), Domain.equalsTo(1.0))

      # Objective: Minimize t
      M.objective(ObjectiveSense.Minimize, t)
                        
      # Solve the problem
      M.solve()

#      M.writeTask('nearestcorr.task')

      # Get the solution values
      print "X = ", X.level()
      
      print t.level()

if __name__ == '__main__':
    main(sys.argv[1:])
