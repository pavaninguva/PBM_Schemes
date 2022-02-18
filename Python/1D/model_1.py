"""
Script to solve PBMs with constant growth rate of the form:

df/dt + u*df/dx = 0, 

with the following boundary conditions:
1. Left end: f=0 at the ghost node
2. Right end: df/dx = 0
"""

import numpy as np


def model_1(Ncells, f0fun, u,tvals,xvals,tvals,scheme)