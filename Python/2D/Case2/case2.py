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

"""
Functions for Exact Scheme
"""

def a1tilde(x):
    f = 20.0*np.log(0.1+0.05*x)
    return f
def a2tilde(x):
    f = 4.0*np.log(0.5+0.25*x)
    return f
def a1(x):
    f = -2.0 + 20.0*np.exp(x/20.0)
    return f
def a2(x):
    f = -2.0 + 4.0*np.exp(x/4.0)
    return f

val_exact_ana, X6, Y6 = model2_exact_analytical([101,101],[0.0,2.0],[0.0,2.0],g1fun,g2fun,a1tilde,a2tilde,a1,a2,f0_fun,1.0)




"""
Perform Simulations
"""
#Run Simulations
val_upwind, x,y = model2_conservative_upwind([101,101],[2.0,2.0],g1fun,g2fun,[0.0,1.0],f0_fun)
val_upwind_trans, x2,y2 = model2_transformed_upwind([101,101],[2.0,2.0],g1fun,g2fun,[0.0,1.0],f0_fun)

val_con_nonuniform,x3,y3 = model2_conservative_nonuniform([0.0,2.0],[0.0,2.0],0.04,g1fun,g2fun,0.5,[0.0,1.0],f0_fun)
val_trans_nonuni,x4,y4 = model2_trans_nonuniform([0.0,2.0],[0.0,2.0],0.04,g1fun,g2fun,0.5,[0.0,1.0],f0_fun)

val_exact, X5,Y5 = model2_exact([101,101],[0.0,2.0],[0.0,2.0],g1fun,g2fun,f0_fun,[0.0,1.0])


"""
Plotting
"""

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

#Upwind-Trans-Nonuniform
fig5 = plt.figure(num=5)
plt.pcolormesh(x4,y4,val_trans_nonuni[:,:,-1],cmap="jet",shading='gouraud')
plt.colorbar(label=r"$f$")
plt.xlabel(r"$a_{1}$")
plt.ylabel(r"$a_{2}$")
plt.clim(0,50)
plt.tight_layout()

#Exact
fig6 = plt.figure(num=6)
plt.pcolormesh(X5,Y5,val_exact,cmap="jet",shading='gouraud')
plt.colorbar(label=r"$f$")
plt.xlabel(r"$a_{1}$")
plt.ylabel(r"$a_{2}$")
plt.clim(0,50)
plt.tight_layout()

#Exact with Analytical Functions
fig7 = plt.figure(num=7)
plt.pcolormesh(X6,Y6,val_exact_ana,cmap="jet",shading='gouraud')
plt.colorbar(label=r"$f$")
plt.xlabel(r"$a_{1}$")
plt.ylabel(r"$a_{2}$")
plt.clim(0,50)
plt.tight_layout()


plt.show()
