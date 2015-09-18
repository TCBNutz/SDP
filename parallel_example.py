#!/apps/python/2.7.3/bin/python

#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=20:mem=5gb
#PBS -o /work/pshadbol/logs/
#PBS -e /work/pshadbol/logs/

from numpy import *
import multiprocessing
from signal import signal, SIGINT, SIG_IGN
from cvxopt import matrix, solvers, sparse
import os
import json
import time

CPUs = multiprocessing.cpu_count()
IS_CLUSTER = CPUs > 4
DATA_DIR = "/work/pshadbol/fc2/" if IS_CLUSTER else "."


def run(i):
    """ A slow process """
    time.sleep(10)
    print i


if __name__ == '__main__':
    attempts = 10 if IS_CLUSTER else 4
    pool = multiprocessing.Pool(CPUs, lambda: signal(SIGINT, SIG_IGN))
    wins = pool.map(run, range(attempts))

