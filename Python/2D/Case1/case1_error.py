from model_1 import *
import matplotlib.pyplot as plt
import numpy as np
from fipy import *

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

def model1_vanleer(nx,ny,Lx,Ly,g_vec,t_vec):
    #Set up mesh
    dx = Lx/nx
    dy = Ly/ny
    mesh = PeriodicGrid2D(nx=nx, ny=ny, dx=dx, dy=dy)

    #Specify Problem
    CFL = 0.5
    dt = CFL/((g_vec[0]/dx) + (g_vec[1]/dy))

    f = CellVariable(mesh=mesh)
    eq = TransientTerm(var=f) + VanLeerConvectionTerm(var=f, coeff=g_vec) == 0

    x = mesh.cellCenters[0]
    y = mesh.cellCenters[1]
    f.setValue(50.0*numerix.exp(-(((x-0.4)**2)/0.005) -((y-0.4)**2)/0.005))

    #Solve Equation
    runtime = t_vec[1]
    t = t_vec[0]

    while t < runtime -1e-8:
        eq.solve(dt=dt)
        t = t+dt

    #Reshape output
    f_np = np.asarray(f).reshape(nx,-1)

    #Compute analytical solution on mesh
    f_ana = CellVariable(mesh=mesh)
    f_ana.setValue(50.0*numerix.exp(-(((x-0.4-1.0)**2)/0.005) -((y-0.4-1.0)**2)/0.005))
    f_ana_np =  np.asarray(f_ana).reshape(nx,-1)

    return f_np, f_ana_np


n_cell_vals = np.array([11,21,41,51,81,101,201])
upwind_rmse = np.zeros(len(n_cell_vals))
upwind_mae = np.zeros(len(n_cell_vals))
vanleer_rmse = np.zeros(len(n_cell_vals))
vanleer_mae = np.zeros(len(n_cell_vals))
exact_rmse = np.zeros(len(n_cell_vals))
exact_mae = np.zeros(len(n_cell_vals))

"""
Run simulations to compute error
"""

for i in range(len(n_cell_vals)):
    #Extract n_cells
    n_cell = n_cell_vals[i]
    n_cell_fipy = n_cell - 1

    #Perform simulations
    val_upwind, x,y = model1_upwind(n_cell,n_cell,2.0,2.0,[1.0,1.0],[0.0,1.0],1.0,f0_fun)

    val_vanleer, val_ana_fipy = model1_vanleer(n_cell_fipy,n_cell_fipy,2.0,2.0,(1.0,1.0),[0.0,1.0])

    val_exact, x2,y2 = model1_exact(n_cell,2.0,2.0,[1.0,1.0],[0.0,1.0],f0_fun)

    #Compute Analytical solution
    val_ana = f_analytical(x,y,[1.0,1.0],1.0)

    #Compute Errors
    upwind_rmse[i] = np.sqrt(np.mean((val_ana-val_upwind[:,:,-1])**2))
    upwind_mae[i] = np.amax(np.abs(val_upwind[:,:,-1]-val_ana))

    exact_rmse[i] = np.sqrt(np.mean((val_ana-val_exact[:,:,-1])**2))
    exact_mae[i] = np.amax(np.abs(val_exact[:,:,-1]-val_ana))

    vanleer_rmse[i] = np.sqrt(np.mean((val_ana_fipy-val_vanleer)**2))
    vanleer_mae[i] = np.amax(np.abs(val_vanleer-val_ana_fipy))


"""
Plotting
"""

fig1 = plt.figure(num=1,figsize=(4,3))
plt.loglog(np.square(n_cell_vals),upwind_rmse,label="Upwind")
plt.loglog(np.square(n_cell_vals),vanleer_rmse,label="Van Leer")
plt.loglog(np.square(n_cell_vals),exact_rmse,label="Exact")
plt.xlabel(r"$N_{Cells}$")
plt.ylabel(r"RMSE")
plt.legend()
plt.tight_layout()
plt.savefig("case1_rmse.png",dpi=300)

fig2 = plt.figure(num=2,figsize=(4,3))
plt.loglog(np.square(n_cell_vals),upwind_mae,label="Upwind")
plt.loglog(np.square(n_cell_vals),vanleer_mae,label="Van Leer")
plt.loglog(np.square(n_cell_vals),exact_mae,label="Exact")
plt.xlabel(r"$N_{Cells}$")
plt.ylabel(r"MAE")
plt.legend()
plt.tight_layout()
plt.savefig("case1_mae.png",dpi=300)


plt.show()





