import matplotlib.pyplot as plt
import numpy as np
from model2 import *

#formatting
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')

def f0_fun(x,y):
    f0 = 50.0*np.exp(-((x-0.4)**2)/0.005 -((y-0.4)**2)/0.005)
    return f0

def f_analytical(x,y,g1fun,g2fun,f0fun,t):
    g1_vals = g1fun(x)
    g2_vals = g2fun(y)

    x_shift = (x+2.0)*np.exp(-0.05*t) -2.0
    y_shift = (y+2.0)*np.exp(-0.25*t) - 2.0

    fhat = g1fun(x_shift)*g2fun(y_shift)*f0fun(x_shift,y_shift)

    f = fhat / (g1_vals*g2_vals)
    return f

def g1fun(x):
    g1 = 0.1 + 0.05*x
    return g1

def g2fun(x):
    g2 = 0.5 + 0.25*x
    return g2


#Run Simulations
val_upwind, x,y = model2_conservative_upwind([101,101],[2.0,2.0],g1fun,g2fun,[0.0,1.0],f0_fun)
val_upwind_trans, x2,y2 = model2_transformed_upwind([101,101],[2.0,2.0],g1fun,g2fun,[0.0,1.0],f0_fun)

val_con_nonuniform,x3,y3 = model2_conservative_nonuniform([0.0,2.0],[0.0,2.0],0.01,g1fun,g2fun,0.5,[0.0,1.0],f0_fun)


#Plot Analytical Solution
f_ana = f_analytical(x,y,g1fun,g2fun,f0_fun,1.0)

fig1 = plt.figure(num=1)
plt.pcolormesh(x,y,f_ana, cmap="jet",shading='gouraud')
plt.colorbar(label=r"$f$")
plt.xlabel(r"$a_{1}$")
plt.ylabel(r"$a_{2}$")
plt.clim(0,50)
plt.tight_layout()
# plt.savefig("case2_analytical.png",dpi=300)

# Plot Simulation Results
#Upwind-Con-Uniform
fig2 = plt.figure(num=2)
plt.pcolormesh(x,y,val_upwind[:,:,-1],cmap="jet",shading='gouraud')
plt.colorbar(label=r"$f$")
plt.xlabel(r"$a_{1}$")
plt.ylabel(r"$a_{2}$")
plt.clim(0,50)
plt.tight_layout()

#Upwind-Trans-Uniform
fig3 = plt.figure(num=3)
plt.pcolormesh(x2,y2,val_upwind_trans[:,:,-1],cmap="jet",shading='gouraud')
plt.colorbar(label=r"$f$")
plt.xlabel(r"$a_{1}$")
plt.ylabel(r"$a_{2}$")
plt.clim(0,50)
plt.tight_layout()

#Upwind-Con-Nonuniform
fig4 = plt.figure(num=4)
plt.pcolormesh(x3,y3,val_con_nonuniform[:,:,-1],cmap="jet",shading='gouraud')
plt.colorbar(label=r"$f$")
plt.xlabel(r"$a_{1}$")
plt.ylabel(r"$a_{2}$")
plt.clim(0,50)
plt.tight_layout()


plt.show()
