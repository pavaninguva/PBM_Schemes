import matplotlib.pyplot as plt
import numpy as np
from model3 import *

#formatting
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')

"""
Defime functions
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


"""
Solve and compute error
"""
dt_vals_exact = np.array([0.5,0.25,0.2,0.1,0.05,0.04,0.025,0.01])

exact_51_rmse = np.zeros(len(dt_vals_exact))
exact_51_mae = np.zeros(len(dt_vals_exact))

exact_101_rmse = np.zeros(len(dt_vals_exact))
exact_101_mae = np.zeros(len(dt_vals_exact))

exact_201_rmse = np.zeros(len(dt_vals_exact))
exact_201_mae = np.zeros(len(dt_vals_exact))

exact_301_rmse = np.zeros(len(dt_vals_exact))
exact_301_mae = np.zeros(len(dt_vals_exact))


for k in range(len(dt_vals_exact)):
    dt = dt_vals_exact[k]

    val_51, X51, Y51= model3_scratch([51,51],[2.0,2.0],g1fun,g2fun,[0.0,1.0],dt,f0_fun)
    ana_51 = analytical(X51,Y51,1.0)
    exact_51_rmse[k] = np.sqrt(np.mean((val_51-ana_51)**2))
    exact_51_mae[k] = np.amax(np.abs(val_51-ana_51))

    val_101, X101, Y101= model3_scratch([101,101],[2.0,2.0],g1fun,g2fun,[0.0,1.0],dt,f0_fun)
    ana_101 = analytical(X101,Y101,1.0)
    exact_101_rmse[k] = np.sqrt(np.mean((val_101-ana_101)**2))
    exact_101_mae[k] = np.amax(np.abs(val_101-ana_101))

    val_201, X201, Y201= model3_scratch([201,201],[2.0,2.0],g1fun,g2fun,[0.0,1.0],dt,f0_fun)
    ana_201 = analytical(X201,Y201,1.0)
    exact_201_rmse[k] = np.sqrt(np.mean((val_201-ana_201)**2))
    exact_201_mae[k] = np.amax(np.abs(val_201-ana_201))

    val_301, X301, Y301= model3_scratch([301,301],[2.0,2.0],g1fun,g2fun,[0.0,1.0],dt,f0_fun)
    ana_301 = analytical(X301,Y301,1.0)
    exact_301_rmse[k] = np.sqrt(np.mean((val_301-ana_301)**2))
    exact_301_mae[k] = np.amax(np.abs(val_301-ana_301))


#Plot exact scheme error
fig3 = plt.figure(num=3)
plt.loglog(dt_vals_exact,exact_51_rmse,"-ko",label=r"$N_{cell}=51$")
plt.loglog(dt_vals_exact,exact_101_rmse,"-ks",label=r"$N_{cell}=101$")
plt.loglog(dt_vals_exact,exact_201_rmse,"-k^",label=r"$N_{cell}=201$")
plt.loglog(dt_vals_exact,exact_301_rmse,"-k*",label=r"$N_{cell}=301$")
plt.xlabel(r"$\Delta t$")
plt.ylabel(r"RMSE")
plt.legend()
plt.savefig("case3_scratch_rmse.png",dpi=300)

fig4 = plt.figure(num=4)
plt.loglog(dt_vals_exact,exact_51_mae,"-ko",label=r"$N_{cell}=51$")
plt.loglog(dt_vals_exact,exact_101_mae,"-ks",label=r"$N_{cell}=101$")
plt.loglog(dt_vals_exact,exact_201_mae,"-k^",label=r"$N_{cell}=201$")
plt.loglog(dt_vals_exact,exact_301_mae,"-k*",label=r"$N_{cell}=301$")
plt.xlabel(r"$\Delta t$")
plt.ylabel(r"MAE")
plt.legend()
plt.savefig("case3_scratch_mae.png",dpi=300)

plt.show()