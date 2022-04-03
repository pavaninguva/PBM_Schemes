from model4 import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator,RectBivariateSpline,interp2d,griddata

#formatting
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')

"""
Functions
"""

def f0_fun(x,y):
    f0 = 50.0*np.exp(-((x-0.4)**2)/0.005 -((y-0.4)**2)/0.005)
    return f0

def hfun(t,x,y):
    h = 1.0 + x*y
    return h


"""
Generate High Resolution Simulation
"""

f_exact, X1,Y1 = model4_split(2001,[2.0,2.0],[1.0,1.0],[0.0,1.0],hfun,f0_fun)

"""
Export as CSV
"""
pd.DataFrame(f_exact).to_csv("f_ana.csv",header=None,index=None)
pd.DataFrame(X1).to_csv("Xvals.csv",header=None,index=None)
pd.DataFrame(Y1).to_csv("Yvals.csv",header=None,index=None)


"""
Load data for plotting as test
"""

# fvals = pd.read_csv("f_ana.csv").to_numpy()
# xvals = pd.read_csv("Xvals.csv").to_numpy()
# yvals = pd.read_csv("Yvals.csv").to_numpy()

# #Perform interpolation to coarser mesh
# f_test, Xtest, Ytest = model4_split(51,[2.0,2.0],[1.0,1.0],[0.0,1.0],hfun,f0_fun)

# f_interp = griddata((xvals.ravel(),yvals.ravel()), fvals.ravel(), (Xtest,Ytest),method="nearest")

# print (f_interp)


# #Plot Analytical Solution
# fig1 = plt.figure(num=1)
# plt.pcolormesh(xvals,yvals,fvals, cmap="jet",shading='gouraud')
# plt.colorbar(label=r"$f$")
# plt.xlabel(r"$a_{1}$")
# plt.ylabel(r"$a_{2}$")
# plt.clim(vmin=0.0)
# plt.tight_layout()

# #Plot Coarse Solution
# fig2 = plt.figure(num=2)
# plt.pcolormesh(Xtest,Ytest,f_test, cmap="jet",shading='gouraud')
# plt.colorbar(label=r"$f$")
# plt.xlabel(r"$a_{1}$")
# plt.ylabel(r"$a_{2}$")
# plt.clim(vmin=0.0)
# plt.tight_layout()

# #Plot interpolated solution
# fig3 = plt.figure(num=3)
# plt.pcolormesh(Xtest,Ytest,f_interp, cmap="jet",shading='gouraud')
# plt.colorbar(label=r"$f$")
# plt.xlabel(r"$a_{1}$")
# plt.ylabel(r"$a_{2}$")
# plt.clim(vmin=0.0)
# plt.tight_layout()





# plt.show()
