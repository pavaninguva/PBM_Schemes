from model_1 import *
import matplotlib.pyplot as plt
import numpy as np

#formatting
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')


def f0_fun(x,y):
    f0 = 50.0*np.exp(-((x-0.4)**2)/0.005 -((y-0.4)**2)/0.005)
    return f0

def f_analytical(x,y,g_vec,t):
    g1, g2 = g_vec
    f = 50.0*np.exp(-((x-0.4-g1*t)**2)/0.005 -((y-0.4-g2*t)**2)/0.005)
    return f

val, x,y = model1_upwind(101,101,2.0,2.0,[1.0,1.0],[0.0,1.0],1.0,f0_fun)

val2, x2,y2 = model1_exact(101,2.0,2.0,[1.0,1.0],[0.0,1.0],f0_fun)

val_ana = f_analytical(x,y,[1.0,1.0],1.0)



"""
Plotting
"""

#Plot initial condition
fig1 = plt.figure(num=1)
plt.pcolormesh(x,y,val[:,:,0], cmap="jet",shading='gouraud')
plt.colorbar(label=r"$f$")
plt.xlabel(r"$a_{1}$")
plt.ylabel(r"$a_{2}$")
plt.clim(0,50)
plt.tight_layout()
# plt.savefig("case1_initial.png",dpi=300)

#Plot Analytical Solution
fig2 = plt.figure(num=2)
plt.pcolormesh(x,y,val_ana, cmap="jet",shading='gouraud')
plt.colorbar(label=r"$f$")
plt.xlabel(r"$a_{1}$")
plt.ylabel(r"$a_{2}$")
plt.clim(0,50)
plt.tight_layout()
# plt.savefig("case1_analytical.png",dpi=300)

#Plot Upwind solution
fig3 = plt.figure(num=3)
plt.pcolormesh(x,y,val[:,:,-1], cmap="jet",shading='gouraud')
plt.colorbar(label=r"$f$")
plt.xlabel(r"$a_{1}$")
plt.ylabel(r"$a_{2}$")
plt.clim(0,50)
plt.tight_layout()
# plt.savefig("case1_upwind.png",dpi=300)

#Plot exact method
fig4 = plt.figure(num=4)
plt.pcolormesh(x2,y2,val2[:,:,-1], cmap="jet",shading='gouraud')
plt.colorbar(label=r"$f$")
plt.xlabel(r"$a_{1}$")
plt.ylabel(r"$a_{2}$")
plt.clim(0,50)
plt.tight_layout()
# plt.savefig("case1_exact.png",dpi=300)





plt.show()



