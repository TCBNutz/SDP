import json
from stuff import skim
import numpy as np
f=open('5qbSolB','r')
x=2**(5./2.)*np.array(json.load(f))
print skim(x)
