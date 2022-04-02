import matplotlib.pyplot as plt
import numpy as np
from model3 import *
import awkward as ak
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
FiPy Function
"""

def model3_vanleer(nx,ny,Lx,Ly, g1fun,g2fun,t_vec):
    #Set up mesh
    dx = Lx/nx
    dy = Ly/ny
    mesh = PeriodicGrid2D(nx=nx, ny=ny, dx=dx, dy=dy)

    #Construct Problem
    A1 = mesh.faceCenters[0]
    A2 = mesh.faceCenters[1]

    coeff = FaceVariable(mesh=mesh,rank=1)
    coeff[0] = g1fun(A1,A2)
    coeff[1] = g2fun(A1,A2)

    f1 = CellVariable(mesh=mesh)
    eq1 = TransientTerm(var = f1) + VanLeerConvectionTerm(var=f1, coeff=coeff) == 0

    #Define Initial Condition
    x = mesh.cellCenters[0]
    y = mesh.cellCenters[1]
    f1.setValue(50.0*numerix.exp(-(((x-0.4)**2)/0.005) -((y-0.4)**2)/0.005))

    #Solve Equation
    run_time = t_vec[1]
    t = t_vec[0]

    CFL = 0.5
    dt = CFL/((max(coeff[0])/dx) + (max(coeff[1])/dy))

    while t < run_time - 1e-8:
        eq1.solve(dt=dt,solver = LinearGMRESSolver(precon="cholesky"))
        #Update time
        t = t+dt
        print("Current Simulation Time is %s"%t)

    #Reshape output
    f_np = np.asarray(f1).reshape(nx,-1)

    #Compute analytical solution on fipy mesh
    f_ana_ = analytical(x,y,1.0)
    f_ana = np.asarray(f_ana_).reshape(nx,-1)

    return f_np, f_ana

"""
Run Simulations
"""

n_cell_vals = np.array([11,21,41,51,81,101,161,201])

dt_vals = np.array([0.25,0.2,0.1,0.05,0.04,0.025,0.01])

vanleer_rmse = np.zeros(len(n_cell_vals))
vanleer_mae = np.zeros(len(n_cell_vals))

conuniform_upwind_rmse = np.zeros(len(n_cell_vals))
conuniform_upwind_mae = np.zeros(len(n_cell_vals))

split_conuniform_upwind_rmse = np.zeros(len(n_cell_vals))
split_conuniform_upwind_mae = np.zeros(len(n_cell_vals))

split_transuniform_upwind_rmse = np.zeros(len(n_cell_vals))
split_transuniform_upwind_mae = np.zeros(len(n_cell_vals))

split_transnonuni_upwind_rmse = np.zeros(len(dt_vals))
split_transnonuni_upwind_mae = np.zeros(len(dt_vals))

transuni_ncell = np.zeros(len(dt_vals))

for i in range(len(n_cell_vals)):
    #Extract n_cells
    n_cell = n_cell_vals[i]
    n_cell_fipy = n_cell - 1

    #Van leer and error
    val_vanleer, val_ana_fipy = model3_vanleer(n_cell_fipy,n_cell_fipy,2.0,2.0,g1fun,g2fun,[0.0,1.0])

    vanleer_rmse[i] = np.sqrt(np.mean((val_ana_fipy-val_vanleer)**2))
    vanleer_mae[i] = np.amax(np.abs(val_vanleer-val_ana_fipy))

    #Conuniform, upwind and error
    val_conupwind, X, Y = model3_conservative_upwind([n_cell,n_cell],[2.0,2.0],g1fun,g2fun,[0.0,1.0],f0_fun)

    val_ana = analytical(X,Y,1.0)
    conuniform_upwind_rmse[i] = np.sqrt(np.mean((val_ana-val_conupwind[:,:,-1])**2))
    conuniform_upwind_mae[i] = np.amax(np.abs(val_conupwind[:,:,-1]-val_ana))

    #Split Conuniform upwind and error
    val_split, X1, Y1 = model3_split_conservative([n_cell,n_cell],[2.0,2.0],g1fun,g2fun,[0.0,1.0],f0_fun)

    val_ana = analytical(X1,Y1,1.0)
    split_conuniform_upwind_rmse[i] = np.sqrt(np.mean((val_ana-val_split[:,:,-1])**2))
    split_conuniform_upwind_mae[i] = np.amax(np.abs(val_split[:,:,-1]-val_ana))

    #Split transuniform upwind and error
    val_split2, X2, Y2 = model3_split_transform([n_cell,n_cell],[2.0,2.0],g1fun,g2fun,[0.0,1.0],f0_fun)

    val_ana_ = analytical(X2,Y2,1.0)
    split_transuniform_upwind_rmse[i] = np.sqrt(np.mean((val_ana_-val_split2[:,:,-1])**2))
    split_transuniform_upwind_mae[i] = np.amax(np.abs(val_split2[:,:,-1]-val_ana_))

for j in range(len(dt_vals)):
    dt = dt_vals[j]

    #Solve
    val_cfl, X_CFL, Y_CFL = model3_split_transform_cfl([2.0,2.0],[0.0,1.0],dt,g1fun,g2fun,f0_fun)

    #Output ncell
    transuni_ncell[j] = ak.count(val_cfl,axis=None)

    #Compute error
    val_ana2 = analytical(X_CFL,Y_CFL,1.0) 

    split_transnonuni_upwind_rmse[j] = np.sqrt(ak.mean((val_ana2-val_cfl)**2))
    split_transnonuni_upwind_mae[j] = max(ak.max(val_ana2-val_cfl), -ak.min(val_ana2-val_cfl))


"""
Solve exact
"""

dt_vals_exact = np.array([0.5,0.25,0.2,0.1,0.05,0.04,0.025,0.01])

exact_51_rmse = np.zeros(len(dt_vals_exact))
exact_51_mae = np.zeros(len(dt_vals_exact))

exact_101_rmse = np.zeros(len(dt_vals_exact))
exact_101_mae = np.zeros(len(dt_vals_exact))

exact_201_rmse = np.zeros(len(dt_vals_exact))
exact_201_mae = np.zeros(len(dt_vals_exact))

for k in range(len(dt_vals_exact)):
    dt = dt_vals_exact[k]

    val_exact_51, X51, Y51= model3_split_exact([51,51],[2.0,2.0],g1fun,g2fun,[0.0,1.0],dt,f0_fun)
    val_ana_51 = analytical(X51,Y51,1.0)

    exact_51_rmse[k] = np.sqrt(np.mean((val_exact_51-val_ana_51)**2))
    exact_51_mae[k] = np.amax(np.abs(val_exact_51-val_ana_51))

    val_exact_101, X101, Y101= model3_split_exact([101,101],[2.0,2.0],g1fun,g2fun,[0.0,1.0],dt,f0_fun)
    val_ana_101 = analytical(X101,Y101,1.0)

    exact_101_rmse[k] = np.sqrt(np.mean((val_exact_101-val_ana_101)**2))
    exact_101_mae[k] = np.amax(np.abs(val_exact_101-val_ana_101))

    val_exact_201, X201, Y201= model3_split_exact([201,201],[2.0,2.0],g1fun,g2fun,[0.0,1.0],dt,f0_fun)
    val_ana_201 = analytical(X201,Y201,1.0)

    exact_201_rmse[k] = np.sqrt(np.mean((val_exact_201-val_ana_201)**2))
    exact_201_mae[k] = np.amax(np.abs(val_exact_201-val_ana_201))




"""
Plotting
"""

#Plot RMSE
fig1 = plt.figure(num=1)
plt.loglog(np.square(n_cell_vals),conuniform_upwind_rmse,"-ko",label="Con-Uniform,Upwind",markerfacecolor="none")
plt.loglog(np.square(n_cell_vals),split_conuniform_upwind_rmse,"--ks",label="Split-Con-Uniform,Upwind",markerfacecolor="none")
plt.loglog(np.square(n_cell_vals),split_transuniform_upwind_rmse,":k^",label="Split-Trans-Uniform,Upwind",markerfacecolor="none")
plt.loglog(np.square(n_cell_vals),vanleer_rmse,"-bo",label="Con-Uniform,Van Leer",markerfacecolor="none")
plt.loglog(transuni_ncell,split_transnonuni_upwind_rmse, "-ro", label = "Split-Trans-Uniform,Upwind",markerfacecolor="none")
plt.xlabel(r"$N_{Cells}$")
plt.ylabel(r"RMSE")
plt.legend(fontsize="medium")
plt.tight_layout()
plt.savefig("case3_rmse.png",dpi=300)

#Plot MAE
fig2 = plt.figure(num=2)
plt.loglog(np.square(n_cell_vals),conuniform_upwind_mae,"-ko",label="Con-Uniform,Upwind",markerfacecolor="none")
plt.loglog(np.square(n_cell_vals),split_conuniform_upwind_mae,"--ks",label="Split-Con-Uniform,Upwind",markerfacecolor="none")
plt.loglog(np.square(n_cell_vals),split_transuniform_upwind_mae,":k^",label="Split-Trans-Uniform,Upwind",markerfacecolor="none")
plt.loglog(np.square(n_cell_vals),vanleer_mae,"-bo",label="Con-Uniform,Van Leer",markerfacecolor="none")
plt.loglog(transuni_ncell,split_transnonuni_upwind_mae, "-ro", label = "Split-Trans-Uniform,Upwind",markerfacecolor="none")
plt.xlabel(r"$N_{Cells}$")
plt.ylabel(r"MAE")
plt.legend(fontsize="medium")
plt.tight_layout()
plt.savefig("case3_mae.png",dpi=300)


#Plot exact scheme error
fig3 = plt.figure(num=3)
plt.loglog(dt_vals_exact,exact_51_rmse,"-ko",label=r"$N_{cell}=51$")
plt.loglog(dt_vals_exact,exact_101_rmse,"-ks",label=r"$N_{cell}=101$")
plt.loglog(dt_vals_exact,exact_201_rmse,"-k^",label=r"$N_{cell}=201$")
plt.xlabel(r"$\Delta t$")
plt.ylabel(r"RMSE")
plt.legend()
plt.savefig("case3_exact_rmse.png",dpi=300)

fig4 = plt.figure(num=4)
plt.loglog(dt_vals_exact,exact_51_mae,"-ko",label=r"$N_{cell}=51$")
plt.loglog(dt_vals_exact,exact_101_mae,"-ks",label=r"$N_{cell}=101$")
plt.loglog(dt_vals_exact,exact_201_mae,"-k^",label=r"$N_{cell}=201$")
plt.xlabel(r"$\Delta t$")
plt.ylabel(r"MAE")
plt.legend()
plt.savefig("case3_exact_mae.png",dpi=300)

plt.show()