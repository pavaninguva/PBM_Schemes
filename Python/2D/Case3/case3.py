import matplotlib.pyplot as plt
import numpy as np
from model3 import *
import awkward as ak

#formatting
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')

"""
Define Functions
"""

def f0_fun(x,y):
    f0 = 50.0*np.exp(-((x-0.4)**2)/0.005 -((y-0.4)**2)/0.005)
    return f0

def g1fun(x,y):
    g1 = 0.25 + 0.5*(x+y)
    return g1

def g2fun(x,y):
    g2 = 0.5 + 0.25*(x+y)
    return g2


"""
Perform Simulations
"""

val_upwind ,x,y= model3_conservative_upwind([51,51],[2.0,2.0],g1fun,g2fun,[0.0,1.0],f0_fun)

# X,Y, val_exact, foo = model3_split_transform([201,201],[2.0,2.0],g1fun,g2fun,[0.0,0.01],0.01,f0_fun)

val_split, X, Y = model3_split_conservative([51,51],[2.0,2.0],g1fun,g2fun,[0.0,1.0],f0_fun)

val_trans_split, X2,Y2 = model3_split_transform([51,51],[2.0,2.0],g1fun,g2fun,[0.0,1.0],f0_fun)

X_1, Y_1, X_2, Y_2 = mesh_constructor([2.0,2.0],0.05,g1fun,g2fun)

f0_vals = f0_fun(X_1,Y_1)

# print(X_1,Y_1)


"""
Plotting
"""

#Upwind
fig1 = plt.figure(num=1)
plt.pcolormesh(x,y,val_upwind[:,:,-1], cmap="jet",shading='gouraud')
plt.colorbar(label=r"$f$")
plt.xlabel(r"$a_{1}$")
plt.ylabel(r"$a_{2}$")
plt.clim(0,50)
plt.tight_layout()
# plt.savefig("case3_upwind.png",dpi=300)

# #Plot mesh
# fig2 = plt.figure(num=2)
# plt.plot(X,Y,".k")
# plt.xlabel(r"$a_{1}$")
# plt.ylabel(r"$a_{2}$")

#Exact
# fig3 = plt.figure(num=3)
# plt.pcolormesh(X,Y,val_exact, cmap="jet",shading='gouraud')
# plt.colorbar(label=r"$f$")
# plt.xlabel(r"$a_{1}$")
# plt.ylabel(r"$a_{2}$")
# plt.clim(0,50)
# plt.tight_layout()
# plt.savefig("case3_exact.png",dpi=300)

#Conservative Split
fig4 = plt.figure(num=4)
plt.pcolormesh(X,Y,val_split[:,:,-1], cmap="jet",shading='gouraud')
plt.colorbar(label=r"$f$")
plt.xlabel(r"$a_{1}$")
plt.ylabel(r"$a_{2}$")
plt.clim(0,50)
plt.tight_layout()
# plt.savefig("case3_split_naive.png",dpi=300)

#Transformed Split
fig5 = plt.figure(num=5)
plt.pcolormesh(X2,Y2,val_trans_split[:,:,-1], cmap="jet",shading='gouraud')
plt.colorbar(label=r"$f$")
plt.xlabel(r"$a_{1}$")
plt.ylabel(r"$a_{2}$")
plt.clim(0,50)
plt.tight_layout()

#Plot CFL=1 Mesh
fig6 = plt.figure(num=6)
plt.plot(ak.flatten(X_1,axis=None),ak.flatten(Y_1,axis=None),".k")
# plt.axhline(y=0)
plt.xlabel(r"$a_{1}$")
plt.ylabel(r"$a_{2}$")

#Plot IC on CFL=1 Mesh
fig7 = plt.figure(num=7)
plt.tripcolor(ak.flatten(X_1,axis=None),ak.flatten(Y_1,axis=None),ak.flatten(f0_vals,axis=None),cmap="jet", shading="gouraud")
plt.colorbar(label=r"$f$")
plt.xlabel(r"$a_{1}$")
plt.ylabel(r"$a_{2}$")
plt.xlim((0,2))
plt.ylim((0,2))
plt.clim(0,50)
plt.tight_layout()



plt.show()


