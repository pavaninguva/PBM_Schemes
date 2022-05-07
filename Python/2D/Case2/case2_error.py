from model2 import *
import matplotlib.pyplot as plt
import numpy as np
from clawpack import riemann
from clawpack import pyclaw

#formatting
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')

"""
Model Functions
"""

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

"""
FiPy Function
"""
def model2_weno(nx,ny,Lx,Ly,g1fun,g2fun,t_vec):
    #Configure Solver
    riemann_solver = riemann.vc_advection_2D
    solver = pyclaw.SharpClawSolver2D(riemann_solver)
    solver.kernel_language ="Fortran"
    solver.weno_order = 5
    solver.lim_type = 2
    solver.cfl_max = 1.0

    #Enforce BCs
    solver.bc_lower[0] = pyclaw.BC.periodic
    solver.bc_upper[0] = pyclaw.BC.periodic
    solver.bc_lower[1] = pyclaw.BC.periodic
    solver.bc_upper[1] = pyclaw.BC.periodic
    solver.aux_bc_lower[0] = pyclaw.BC.extrap
    solver.aux_bc_upper[0] = pyclaw.BC.extrap
    solver.aux_bc_lower[1] = pyclaw.BC.extrap
    solver.aux_bc_upper[1] = pyclaw.BC.extrap

    #Set mesh
    nx = nx
    ny = ny
    Lx = Lx
    Ly = Ly
    x = pyclaw.Dimension(0.0,Lx,nx,name="x")
    y = pyclaw.Dimension(0.0,Ly,ny,name="y")
    domain = pyclaw.Domain([x,y])

    #Construct Equation and Set Velocities
    num_aux = 2
    solver.num_eqn = 1
    solver.num_waves = 1
    state = pyclaw.State(domain,solver.num_eqn,num_aux)
    xe, ye = state.grid.p_nodes
    state.aux[0,:,:] = 0.1 + 0.05*xe[:-1,1:]
    state.aux[1,:,:] = 0.5 + 0.25*ye[:-1,1:]

    #Set IC
    xc, yc = domain.grid.p_centers
    state.q[0,:,:] = (50.0*np.exp(-(((xc-0.4)**2)/0.005) -((yc-0.4)**2)/0.005))*(0.1+0.05*xc)*(0.5+0.25*yc)

    #Set Controller
    claw = pyclaw.Controller()
    claw.keep_copy = True
    claw.solution = pyclaw.Solution(state, domain)
    claw.tfinal = 1.0
    claw.solver = solver

    status = claw.run()
    print(status)
    sol = claw.frames[-1].q[0,:,:]
    f_np = sol/((0.1+0.05*xc)*(0.5+0.25*yc))

    f_ana_np = f_analytical(xc,yc,g1fun,g2fun,f0_fun,1.0)

    return f_np, f_ana_np

"""
Run Simulations
"""

n_cell_vals = np.array([11,21,41,51,81,101,161,201,251])

weno_rmse = np.zeros(len(n_cell_vals))
weno_mae = np.zeros(len(n_cell_vals))

con_upwind_rmse = np.zeros(len(n_cell_vals))
con_upwind_mae = np.zeros(len(n_cell_vals))

trans_upwind_rmse = np.zeros(len(n_cell_vals))
trans_upwind_mae = np.zeros(len(n_cell_vals))

exact_rmse = np.zeros(len(n_cell_vals))
exact_mae = np.zeros(len(n_cell_vals))

exact_ana_rmse = np.zeros(len(n_cell_vals))
exact_ana_mae = np.zeros(len(n_cell_vals))

for i in range(len(n_cell_vals)):
    #Extract n_cells
    n_cell = n_cell_vals[i]
    n_cell_fipy = n_cell - 1

    #Perform Simulations
    val_weno, val_ana_fipy = model2_weno(n_cell_fipy,n_cell_fipy,2.0,2.0,g1fun,g2fun,[0.0,1.0])
    val_upwind, x,y = model2_conservative_upwind([n_cell,n_cell],[2.0,2.0],g1fun,g2fun,[0.0,1.0],f0_fun)
    val_upwind_trans, x2,y2 = model2_transformed_upwind([n_cell,n_cell],[2.0,2.0],g1fun,g2fun,[0.0,1.0],f0_fun)

    val_exact, x3,y3 = model2_exact([n_cell,n_cell],[0.0,2.0],[0.0,2.0],g1fun,g2fun,f0_fun,[0.0,1.0])

    val_exact_ana,x4,y4 = model2_exact_analytical([n_cell,n_cell],[0.0,2.0],[0.0,2.0],g1fun,g2fun,a1tilde,a2tilde,a1,a2,f0_fun,1.0)


    #Compute analytical solution
    f_ana = f_analytical(x,y,g1fun,g2fun,f0_fun,1.0)
    f_ana_exact = f_analytical(x3,y3,g1fun,g2fun,f0_fun,1.0)
    f_ana_exact_ana = f_analytical(x4,y4,g1fun,g2fun,f0_fun,1.0)

    #Compute Error
    weno_rmse[i] = np.sqrt(np.mean((val_ana_fipy-val_weno)**2))
    weno_mae[i] = np.amax(np.abs(val_weno-val_ana_fipy))

    con_upwind_rmse[i] = np.sqrt(np.mean((f_ana-val_upwind[:,:,-1])**2))
    con_upwind_mae[i] = np.amax(np.abs(f_ana-val_upwind[:,:,-1]))

    trans_upwind_rmse[i] = np.sqrt(np.mean((f_ana-val_upwind_trans[:,:,-1])**2))
    trans_upwind_mae[i] = np.amax(np.abs(f_ana-val_upwind_trans[:,:,-1]))

    exact_rmse[i] = np.sqrt(np.mean((f_ana_exact-val_exact)**2))
    exact_mae[i] = np.amax(np.abs(f_ana_exact-val_exact))

    exact_ana_rmse[i] = np.sqrt(np.mean((f_ana_exact_ana-val_exact_ana)**2))
    exact_ana_mae[i] = np.amax(np.abs(f_ana_exact_ana-val_exact_ana))

#Run nonuniform simulation
dt_vals = np.array([0.2,0.1,0.05,0.04,0.025,0.02,0.01])
n_cells_nonuni = np.zeros(len(dt_vals))

con_nonuni_rmse = np.zeros(len(dt_vals))
con_nonuni_mae = np.zeros(len(dt_vals))

trans_nonuni_rmse = np.zeros(len(dt_vals))
trans_nonuni_mae = np.zeros(len(dt_vals))

for i in range(len(dt_vals)):
    #Extract dt
    dt = dt_vals[i]

    #Run Simulations
    val_con_nonuni,x3,y3 = model2_conservative_nonuniform([0.0,2.0],[0.0,2.0],dt,g1fun,g2fun,0.5,[0.0,1.0],f0_fun)
    val_trans_nonuni,x4,y4 = model2_trans_nonuniform([0.0,2.0],[0.0,2.0],dt,g1fun,g2fun,0.5,[0.0,1.0],f0_fun)

    #Compute Analytic Solution
    f_ana_nonuni = f_analytical(x3,y3,g1fun,g2fun,f0_fun,1.0)

    #Append N_cells
    n_cells_nonuni[i] = f_ana_nonuni.size

    #Compute Error
    con_nonuni_rmse[i] = np.sqrt(np.mean((f_ana_nonuni -val_con_nonuni[:,:,-1])**2))
    con_nonuni_mae[i] = np.amax(np.abs(f_ana_nonuni-val_con_nonuni[:,:,-1]))

    trans_nonuni_rmse[i] = np.sqrt(np.mean((f_ana_nonuni-val_trans_nonuni[:,:,-1])**2))
    trans_nonuni_mae[i] = np.amax(np.abs(f_ana_nonuni-val_trans_nonuni[:,:,-1]))



"""
Plotting
"""

fig1 = plt.figure(num=1)
plt.loglog(np.square(n_cell_vals),con_upwind_rmse,"-ko",label="Con-Uniform,Upwind",markerfacecolor="none")
plt.loglog(np.square(n_cell_vals),weno_rmse,"-bo",label="Trans-Uniform,WENO",markerfacecolor="none")
plt.loglog(np.square(n_cell_vals),trans_upwind_rmse,"--ks",label="Trans-Uniform,Upwind",markerfacecolor="none")
plt.loglog(n_cells_nonuni,con_nonuni_rmse,":k^",label="Con-Nonuniform,Upwind",markerfacecolor="none")
plt.loglog(n_cells_nonuni,trans_nonuni_rmse,"-.kd",label="Trans-Nonuniform,Upwind",markerfacecolor="none")
plt.loglog(np.square(n_cell_vals),exact_rmse,"-ro",label="Exact,Numerical",markerfacecolor="none")
plt.loglog(np.square(n_cell_vals),exact_ana_rmse,"-rx",label="Exact,Analytical",markerfacecolor="none")
plt.xlabel(r"$N_{Cells}$")
plt.ylabel(r"RMSE")
plt.legend(fontsize="medium",loc="upper right",bbox_to_anchor=(0.42,0.84))
plt.tight_layout()
plt.savefig("case2_rmse.png",dpi=300)

fig2 = plt.figure(num=2)
plt.loglog(np.square(n_cell_vals),con_upwind_mae,"-ko",label="Con-Uniform,Upwind",markerfacecolor="none")
plt.loglog(np.square(n_cell_vals),weno_mae,"-bo",label="Trans-Uniform,WENO",markerfacecolor="none")
plt.loglog(np.square(n_cell_vals),trans_upwind_mae,"--ks",label="Trans-Uniform,Upwind",markerfacecolor="none")
plt.loglog(n_cells_nonuni,con_nonuni_mae,":k^",label="Con-Nonuniform,Upwind",markerfacecolor="none")
plt.loglog(n_cells_nonuni,trans_nonuni_mae,"-.kd",label="Trans-Nonuniform,Upwind",markerfacecolor="none")
plt.loglog(np.square(n_cell_vals),exact_mae,"-ro",label="Exact,Numerical",markerfacecolor="none")
plt.loglog(np.square(n_cell_vals),exact_ana_mae,"-rx",label="Exact,Analytical",markerfacecolor="none")
plt.xlabel(r"$N_{Cells}$")
plt.ylabel(r"MAE")
plt.legend(fontsize="medium",loc="upper right",bbox_to_anchor=(0.42,0.83))
plt.tight_layout()
plt.savefig("case2_mae.png",dpi=300)


plt.show()













