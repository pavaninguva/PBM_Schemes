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

def analytical(x,y,t):
    #Compute shifted var
    c1 = 3*x + 0.75*t + 2 - 2*np.exp(0.75*t)
    c2 = 3*y - 0.75*t + 1 - np.exp(0.75*t)

    A = 1 + 2*np.exp(0.75*t)
    B = 2*np.exp(0.75*t) - 2
    C = np.exp(0.75*t) -1 
    D = np.exp(0.75*t) + 2

    x_trans = (B*c2 -D*c1)/(B*C - D*A)
    y_trans = (C*c1 -A*c2)/(B*C - D*A)

    f = f0_fun(x_trans,y_trans)*np.exp(-0.75*t)
    return f

#Plotting
x_vals = np.linspace(0.0,2.0,200)
y_vals = np.linspace(0.0,2.0,200)

xx,yy = np.meshgrid(x_vals,y_vals)




"""
Perform Simulations
"""

# val_upwind ,x,y= model3_conservative_upwind([201,201],[2.0,2.0],g1fun,g2fun,[0.0,1.0],f0_fun)

# val_split, X, Y = model3_split_conservative([201,201],[2.0,2.0],g1fun,g2fun,[0.0,1.0],f0_fun)

# val_trans_split, X2,Y2 = model3_split_transform([201,201],[2.0,2.0],g1fun,g2fun,[0.0,1.0],f0_fun)

# val_cfl, X_CFL, Y_CFL = model3_split_transform_cfl([2.0,2.0],[0.0,1.0],0.01,g1fun,g2fun,f0_fun)

# print (ak.count(val_cfl,axis=None))

val_exact, X1, Y1= model3_scratch([201,201],[2.0,2.0],g1fun,g2fun,[0.0,1.0],0.05,f0_fun)

# val_exact, X1, Y1= model3_split_exact([201,201],[2.0,2.0],g1fun,g2fun,[0.0,1.0],0.05,f0_fun)


"""
Plotting
"""


#Upwind
# fig1 = plt.figure(num=1)
# plt.pcolormesh(x,y,val_upwind[:,:,-1], cmap="jet",shading='gouraud')
# plt.colorbar(label=r"$f$")
# plt.xlabel(r"$a_{1}$")
# plt.ylabel(r"$a_{2}$")
# plt.clim(0,50)
# plt.tight_layout()
# plt.savefig("case3_upwind.png",dpi=300)

#Plot nonuniform mesh
# fig2 = plt.figure(num=2)
# # plt.plot(X,Y,".k")
# plt.plot(ak.flatten(X_CFL,axis=None),ak.flatten(Y_CFL,axis=None),".k")
# plt.xlabel(r"$a_{1}$")
# plt.ylabel(r"$a_{2}$")
# plt.savefig("case3_split_mesh.png",dpi=300)

#Exact
# fig3 = plt.figure(num=3)
# plt.pcolormesh(X1,Y1,val_exact, cmap="jet",shading='gouraud')
# plt.colorbar(label=r"$f$")
# plt.xlabel(r"$a_{1}$")
# plt.ylabel(r"$a_{2}$")
# plt.clim(0,50)
# plt.tight_layout()
# plt.savefig("case3_exact.png",dpi=300)

#Conservative Split
# fig4 = plt.figure(num=4)
# plt.pcolormesh(X,Y,val_split[:,:,-1], cmap="jet",shading='gouraud')
# plt.colorbar(label=r"$f$")
# plt.xlabel(r"$a_{1}$")
# plt.ylabel(r"$a_{2}$")
# plt.clim(0,50)
# plt.tight_layout()
# plt.savefig("case3_split_naive.png",dpi=300)

#Transformed Split
# fig5 = plt.figure(num=5)
# plt.pcolormesh(X2,Y2,val_trans_split[:,:,-1], cmap="jet",shading='gouraud')
# plt.colorbar(label=r"$f$")
# plt.xlabel(r"$a_{1}$")
# plt.ylabel(r"$a_{2}$")
# plt.clim(0,50)
# plt.tight_layout()
# plt.savefig("case3_split_trans.png",dpi=300)


#Plot CFL=1 Mesh
# fig6 = plt.figure(num=6)
# plt.tripcolor(ak.flatten(X_CFL,axis=None),ak.flatten(Y_CFL,axis=None),ak.flatten(val_cfl,axis=None),cmap="jet", shading="gouraud")
# plt.colorbar(label=r"$f$")
# plt.xlabel(r"$a_{1}$")
# plt.ylabel(r"$a_{2}$")
# plt.xlim((0,2))
# plt.ylim((0,2))
# plt.clim(0,50)
# plt.tight_layout()
# plt.savefig("case3_split_cfl.png",dpi=300)

#Analytical
fig7 = plt.figure(num=7)
plt.pcolormesh(xx,yy,analytical(xx,yy,1.0), cmap="jet",shading='gouraud')
plt.colorbar(label=r"$f$")
plt.xlabel(r"$a_{1}$")
plt.ylabel(r"$a_{2}$")
plt.clim(0,50)
plt.tight_layout()
# plt.savefig("case3_analytical.png",dpi=300)

#Analytical
fig8 = plt.figure(num=8)
plt.pcolormesh(X1,Y1,val_exact, cmap="jet",shading='gouraud')
plt.colorbar(label=r"$f$")
plt.xlabel(r"$a_{1}$")
plt.ylabel(r"$a_{2}$")
plt.clim(0,50)
plt.tight_layout()
plt.savefig("case3_scratch.png",dpi=300)


plt.show()


