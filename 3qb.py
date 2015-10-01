""" semidefinite optimization of localizable entanglement, given the reduced
density matrix of a segment and translational invariance  """

import numpy as np
from cvxopt import matrix, solvers, sparse
from stuff import *

if __name__ == '__main__':
        fBloch=np.conj(toBloch(3)).T#3.125% sparse

