from model3 import *
from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator
import awkward as ak
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

def g1fun(x,y):
    g1 = 0.25 + 0.5*(x+y)
    return g1

def g2fun(x,y):
    g2 = 0.5 + 0.25*(x+y)
    return g2

def f0_fun(x,y):
    f0 = 50.0*np.exp(-((x-0.4)**2)/0.005 -((y-0.4)**2)/0.005)
    return f0

X_1, Y_1, X_2, Y_2 = mesh_constructor([2.0,2.0],0.25,g1fun,g2fun)

f0 = f0_fun(X_2,Y_2)

print(f0[0],X_2[0],Y_2[0])
print(ak.num(f0,axis=0))

# XY_1 = zip(ak.flatten(X_1,axis=None),ak.flatten(Y_1,axis=None))
# XY_1 = list(XY_1)

# XY_2 = zip(ak.flatten(X_2,axis=None),ak.flatten(Y_2,axis=None))
# XY_2 = list(XY_2)

#Plot CFL=1 Mesh
fig6 = plt.figure(num=6)
plt.plot(ak.flatten(X_2,axis=None),ak.flatten(Y_2,axis=None),".k")
# plt.axhline(y=0)
plt.xlabel(r"$a_{1}$")
plt.ylabel(r"$a_{2}$")



# #Compute Delaunay
# tri1 = Delaunay(XY_1)
# tri2 = Delaunay(XY_2)

# interp = CloughTocher2DInterpolator(tri1,ak.flatten(f0,axis=None))

# f0_2 = interp(XY_2)

# #Plot IC on Mesh1
# fig1 = plt.figure(num=1)
# plt.tripcolor(ak.flatten(X_1,axis=None),ak.flatten(Y_1,axis=None),ak.flatten(f0,axis=None),cmap="jet", shading="gouraud")
# plt.colorbar(label=r"$f$")
# plt.xlabel(r"$a_{1}$")
# plt.ylabel(r"$a_{2}$")
# plt.xlim((0,2))
# plt.ylim((0,2))
# plt.clim(0,50)
# plt.tight_layout()

# #Plot IC on Mesh2
# fig2 = plt.figure(num=2)
# plt.tripcolor(ak.flatten(X_2,axis=None),ak.flatten(Y_2,axis=None),f0_2,cmap="jet", shading="gouraud")
# plt.colorbar(label=r"$f$")
# plt.xlabel(r"$a_{1}$")
# plt.ylabel(r"$a_{2}$")
# plt.xlim((0,2))
# plt.ylim((0,2))
# plt.clim(0,50)
# plt.tight_layout()

plt.show()




