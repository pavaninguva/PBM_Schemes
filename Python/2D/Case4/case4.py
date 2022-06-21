from model4 import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#formatting
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')

"""
Functions
"""

def f0_fun(x,y):
    f0 = 10.0*np.exp(-((x-0.4)**2)/0.005 -((y-0.4)**2)/0.005)
    return f0

def hfun(t,x,y):
    h = 1.0 + x*y
    return h

"""
Simulations
"""

f_upwind, X, Y = model4_upwind([101,101],[2.0,2.0],[1.0,1.0],[0.0,1.0],1.0,hfun,f0_fun)

f_exact, X1,Y1 = model4_split(101,[2.0,2.0],[1.0,1.0],[0.0,1.0],hfun,f0_fun)
# pd.DataFrame(f_exact).to_csv("f_anaFD.csv",header=None,index=None)
# pd.DataFrame(X1).to_csv("XvalsFD.csv",header=None,index=None)
# pd.DataFrame(Y1).to_csv("YvalsFD.csv",header=None,index=None)




"""
Plotting
"""
# #Plot Upwind
fig1 = plt.figure(num=1)
plt.pcolormesh(X,Y,f_upwind, cmap="jet",shading='gouraud')
plt.colorbar(label=r"$f$")
plt.xlabel(r"$a_{1}$")
plt.ylabel(r"$a_{2}$")
plt.clim(vmin=0.0,vmax=12)
plt.tight_layout()
plt.savefig("case4_upwind.png",dpi=300)

# #Plot Split Exact
fig2 = plt.figure(num=2)
plt.pcolormesh(X1,Y1,f_exact, cmap="jet",shading='gouraud')
plt.colorbar(label=r"$f$")
plt.xlabel(r"$a_{1}$")
plt.ylabel(r"$a_{2}$")
plt.clim(vmin=0.0,vmax=12)
plt.tight_layout()
plt.savefig("case4_exact.png",dpi=300)


# f_ana1 = pd.read_csv("f_ana.csv").to_numpy()
# x_ana = pd.read_csv("Xvals.csv").to_numpy()
# y_ana = pd.read_csv("Yvals.csv").to_numpy()
# #Plot Reference
# fig3 = plt.figure(num=3)
# plt.pcolormesh(x_ana,y_ana,f_ana1, cmap="jet",shading='gouraud')
# plt.colorbar(label=r"$f$")
# plt.xlabel(r"$a_{1}$")
# plt.ylabel(r"$a_{2}$")
# plt.clim(vmin=0.0,vmax=12)
# plt.tight_layout()
# plt.savefig("case4_ref.png",dpi=300)


plt.show()