from model5 import *
import matplotlib.pyplot as plt
import numpy as np

#formatting
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')

def f0_fun(x,y):
    f0 = 50.0*np.exp(-((x-0.4)**2)/0.005 -((y-0.4)**2)/0.005)
    return f0

def lambdafun(t,x,y):
    f = x*y
    return f

def analytical(t,x,y):
    f = f0_fun(x-t,y-t)*np.exp(-(x+y)*t + t**2)
    return f

def int_lambda(x,y):
    f = x*y
    return f


"""
Simulations
"""

f_upwind, X, Y = model5_upwind([101,101],[2.0,2.0],[1.0,1.0],[0.0,1.0],1.0,lambdafun,f0_fun)

f_exact, X1,Y1 = model5_exact(101,[2.0,2.0],[1.0,1.0],[0.0,1.0],int_lambda,f0_fun)



"""
Plotting
"""

#Plot Analytical
fig1 = plt.figure(num=1)
plt.pcolormesh(X,Y,analytical(1.0,X,Y), cmap="jet",shading='gouraud')
plt.colorbar(label=r"$f$")
plt.xlabel(r"$a_{1}$")
plt.ylabel(r"$a_{2}$")
plt.clim(vmin=0.0,vmax=10)
plt.tight_layout()

#Plot Upwind
fig2 = plt.figure(num=2)
plt.pcolormesh(X,Y,f_upwind, cmap="jet",shading='gouraud')
plt.colorbar(label=r"$f$")
plt.xlabel(r"$a_{1}$")
plt.ylabel(r"$a_{2}$")
plt.clim(vmin=0.0,vmax=10)
plt.tight_layout()

#Plot exact
fig3 = plt.figure(num=3)
plt.pcolormesh(X1,Y1,f_exact, cmap="jet",shading='gouraud')
plt.colorbar(label=r"$f$")
plt.xlabel(r"$a_{1}$")
plt.ylabel(r"$a_{2}$")
plt.clim(vmin=0.0,vmax=10)
plt.tight_layout()



plt.show()