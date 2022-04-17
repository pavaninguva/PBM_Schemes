from model5 import *
import matplotlib.pyplot as plt
import numpy as np
from fipy import *

#formatting
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')

"""
Functions
"""

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

def model5_vanleer(nx,ny,Lx,Ly,gvec,tvec):
    #Setting up mesh with domain size Lx=Ly = 2.0
    nx = nx
    ny = ny
    dx = Lx/nx
    dy = Ly/ny
    mesh = PeriodicGrid2D(nx=nx,ny=ny,dx=dx,dy=dy)

    #Specify parameters
    convCoeff = (gvec[0],gvec[1])
    CFL = 0.25
    dt = CFL/((convCoeff[0]/dx) + (convCoeff[1]/dy))
    f1 = CellVariable(mesh=mesh, name=r"$f_{1}$")
    #Define Initial Condition
    x = mesh.cellCenters[0]
    y = mesh.cellCenters[1]
    f1.setValue(50.0*numerix.exp(-(((x-0.4)**2)/0.005) -((y-0.4)**2)/0.005))

    #Specify equation
    eq1 = TransientTerm(var=f1) + VanLeerConvectionTerm(var=f1, coeff=convCoeff) == -(x+y)*f1

    #Solve Equations
    run_time = tvec[1]
    t = tvec[0]
    while t < run_time - 1e-8:
        eq1.solve(dt=dt)
        #Update time
        t = t+dt
        print("Current Simulation Time is %s"%t)

    #Reshape to Numpy arrays
    f1_np = np.asarray(f1).reshape(nx,-1)

    #Compute analytical solution
    f_ana_ = analytical(x,y,1.0)
    f_ana = np.asarray(f_ana_).reshape(nx,-1)

    return f1_np, f_ana

"""
Run simulations
"""

n_cell_vals = np.array([11,21,41,51,81,101,161,201])

vanleer_rmse = np.zeros(len(n_cell_vals))
vanleer_mae = np.zeros(len(n_cell_vals))

upwind_rmse = np.zeros(len(n_cell_vals))
upwind_mae = np.zeros(len(n_cell_vals))

exact_rmse = np.zeros(len(n_cell_vals))
exact_mae = np.zeros(len(n_cell_vals))

for i in range(len(n_cell_vals)):
    #Extract n_cells
    n_cell = n_cell_vals[i]
    n_cell_fipy = n_cell - 1

    #Van leer and error
    val_vanleer, val_ana_fipy = model5_vanleer(n_cell_fipy,n_cell_fipy,2.0,2.0,[1.0,1.0],[0.0,1.0])
    vanleer_rmse[i] = np.sqrt(np.mean((val_ana_fipy-val_vanleer)**2))
    vanleer_mae[i] = np.amax(np.abs(val_vanleer-val_ana_fipy))

    #upwind
    val_upwind, X, Y = model5_upwind([n_cell,n_cell],[2.0,2.0],[1.0,1.0],[0.0,1.0],1.0,lambdafun,f0_fun)
    upwind_ana = analytical(1.0,X,Y)
    upwind_rmse[i] = np.sqrt(np.mean((upwind_ana-val_upwind)**2))
    upwind_mae[i] = np.amax(np.abs(upwind_ana-val_upwind))

    #Exact
    f_exact, X1,Y1 = model5_exact(101,[2.0,2.0],[1.0,1.0],[0.0,1.0],int_lambda,f0_fun)
    exact_ana = analytical(1.0,X1,Y1)
    exact_rmse[i] = np.sqrt(np.mean((exact_ana-f_exact)**2))
    exact_mae[i] = np.amax(np.abs(exact_ana-f_exact))

"""
Plotting
"""

fig1 = plt.figure(num=1)
plt.loglog(np.square(n_cell_vals),upwind_rmse,"-ko",label="Upwind",markerfacecolor="none")
plt.loglog(np.square(n_cell_vals),vanleer_rmse,"-bo",label="Van Leer",markerfacecolor="none")
plt.loglog(np.square(n_cell_vals),exact_rmse,"-ro",label="Exact",markerfacecolor="none")
plt.xlabel(r"$N_{Cells}$")
plt.ylabel(r"RMSE")
plt.legend()
plt.tight_layout()
plt.savefig("case5_rmse.png",dpi=300)


fig2 = plt.figure(num=2)
plt.loglog(np.square(n_cell_vals),upwind_mae,"-ko",label="Upwind",markerfacecolor="none")
plt.loglog(np.square(n_cell_vals),vanleer_mae,"-bo",label="Van Leer",markerfacecolor="none")
plt.loglog(np.square(n_cell_vals),exact_mae,"-ro",label="Exact",markerfacecolor="none")
plt.xlabel(r"$N_{Cells}$")
plt.ylabel(r"MAE")
plt.legend()
plt.tight_layout()
plt.savefig("case5_mae.png",dpi=300)

plt.show()


