from model4 import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
import pandas as pd
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


def hfun(t,x,y):
    h = 1.0 + x*y
    return h

#Read in Analytical csv data
f_ana1 = pd.read_csv("f_ana.csv").to_numpy()
x_ana = pd.read_csv("Xvals.csv").to_numpy()
y_ana = pd.read_csv("Yvals.csv").to_numpy()




def model4_vanleer(nx,ny,Lx,Ly, gvec, tvec):
    #Setting up mesh with domain size Lx=Ly = 2.0
    nx = nx
    ny = ny
    dx = Lx/nx
    dy = Ly/ny
    mesh = Grid2D(nx=nx,ny=ny,dx=dx,dy=dy)

    #Specify Problem
    convCoeff = (gvec[0],gvec[1])
    CFL = 0.5
    dt = CFL/((convCoeff[0]/dx) + (convCoeff[1]/dy))
    # f1 = CellVariable(mesh=mesh, name=r"$f_{1}$",hasOld=True)
    f1 = CellVariable(mesh=mesh, name=r"$f_{1}$")

    #Define Initial Condition
    x = mesh.cellCenters[0]
    y = mesh.cellCenters[1]
    f1.setValue(50.0*numerix.exp(-(((x-0.4)**2)/0.005) -((y-0.4)**2)/0.005))

    #Specify equation
    eq1 = TransientTerm(var=f1) + VanLeerConvectionTerm(var=f1, coeff=convCoeff) - (1.0+x*y) == 0

    #Apply BC
    f1.faceGrad.constrain(0,where=mesh.facesTop)
    f1.faceGrad.constrain(0,where=mesh.facesRight)

    #Solve Equations
    run_time = tvec[1]
    t = tvec[0]

    while t < run_time - 1e-8:
        # f1.updateOld()
        # res = 1e8
        # while res > 1e-8:
        #     res = eq1.sweep(dt=dt,solver=LinearGMRESSolver(precon="cholesky"))
        #     print("Residual is %s"%res)
        eq1.solve(dt=dt)
        #Update time
        t = t+dt
        print("Current Simulation Time is %s"%t)

    #Reshape output
    f_np = np.asarray(f1).reshape(nx,-1)

    #Compute Analytical on FiPy mesh
    f_ana_ = griddata((x_ana.ravel(),y_ana.ravel()), f_ana1.ravel(), (x,y),method="nearest")

    f_ana = np.asarray(f_ana_).reshape(nx,-1)

    return f_np, f_ana






"""
Error Calculation
"""

n_cell_vals = np.array([11,21,41,51,81,101,201,251])
upwind_rmse = np.zeros(len(n_cell_vals))
upwind_mae = np.zeros(len(n_cell_vals))
vanleer_rmse = np.zeros(len(n_cell_vals))
vanleer_mae = np.zeros(len(n_cell_vals))
exact_rmse = np.zeros(len(n_cell_vals))
exact_mae = np.zeros(len(n_cell_vals))


for i in range(len(n_cell_vals)):
    #Extract n_cells
    n_cell = n_cell_vals[i]
    n_cell_fipy = n_cell - 1

    #Upwind
    f_upwind, X, Y = model4_upwind([n_cell,n_cell],[2.0,2.0],[1.0,1.0],[0.0,1.0],1.0,hfun,f0_fun)
    f_ana = griddata((x_ana.ravel(),y_ana.ravel()), f_ana1.ravel(), (X,Y),method="nearest")

    upwind_rmse[i] = np.sqrt(np.mean((f_upwind-f_ana)**2))
    upwind_mae[i] = np.amax(np.abs(f_upwind-f_ana))

    #Exact
    f_exact, X1,Y1 = model4_split(n_cell,[2.0,2.0],[1.0,1.0],[0.0,1.0],hfun,f0_fun)
    f_ana = griddata((x_ana.ravel(),y_ana.ravel()), f_ana1.ravel(), (X1,Y1),method="nearest")

    exact_rmse[i] = np.sqrt(np.mean((f_exact-f_ana)**2))
    exact_mae[i] = np.amax(np.abs(f_exact-f_ana))

    #Van Leer
    f_vanleer, f_ana_fipy = model4_vanleer(n_cell_fipy,n_cell_fipy,2.0,2.0,[1.0,1.0],[0.0,1.0])
    vanleer_rmse[i] = np.sqrt(np.mean((f_vanleer-f_ana_fipy)**2))
    vanleer_mae[i] = np.amax(np.abs(f_vanleer-f_ana_fipy))


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
plt.savefig("case4_rmse.png",dpi=300)


fig2 = plt.figure(num=2)
plt.loglog(np.square(n_cell_vals),upwind_mae,"-ko",label="Upwind",markerfacecolor="none")
plt.loglog(np.square(n_cell_vals),vanleer_mae,"-bo",label="Van Leer",markerfacecolor="none")
plt.loglog(np.square(n_cell_vals),exact_mae,"-ro",label="Exact",markerfacecolor="none")
plt.xlabel(r"$N_{Cells}$")
plt.ylabel(r"MAE")
plt.legend()
plt.tight_layout()
plt.savefig("case4_mae.png",dpi=300)

plt.show()

