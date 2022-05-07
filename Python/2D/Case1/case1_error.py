from model_1 import *
import matplotlib.pyplot as plt
import numpy as np
# from fipy import *
from clawpack import riemann
from clawpack import pyclaw

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

def model1_weno(nx,ny,Lx,Ly,g_vec,t_vec):
    #Configure Solver
    riemann_solver = riemann.advection_2D
    solver = pyclaw.SharpClawSolver2D(riemann_solver)
    solver.kernel_language = "Fortran"
    solver.weno_order = 5
    solver.lim_type = 2
    solver.cfl_max = 1.0
    #Enforce BCs
    solver.bc_lower[0] = pyclaw.BC.periodic
    solver.bc_upper[0] = pyclaw.BC.periodic
    solver.bc_lower[1] = pyclaw.BC.periodic
    solver.bc_upper[1] = pyclaw.BC.periodic

    #Set mesh
    nx = nx
    ny = ny
    Lx = Lx
    Ly = Ly
    x = pyclaw.Dimension(0.0,Lx,nx,name="x")
    y = pyclaw.Dimension(0.0,Ly,ny,name="y")
    domain = pyclaw.Domain([x,y])

    #Construct Equation
    state = pyclaw.State(domain,solver.num_eqn)
    state.problem_data["u"] = g_vec[0]
    state.problem_data["v"] = g_vec[1]

    #Set IC
    xc,yc = state.grid.p_centers
    state.q[0,:,:] = 50.0*np.exp(-(((xc-0.4)**2)/0.005) -((yc-0.4)**2)/0.005)

    #Set up controller
    claw = pyclaw.Controller()
    claw.keep_copy = True
    claw.solution = pyclaw.Solution(state, domain)
    claw.tfinal = 1.0
    claw.solver = solver
    status = claw.run()

    f_np = claw.frames[-1].q[0,:,:]
    f_ana_np = 50.0*np.exp(-(((xc-1.4)**2)/0.005) -((yc-1.4)**2)/0.005)

    return f_np, f_ana_np


n_cell_vals = np.array([41,51,81,101,201,251])
upwind_rmse = np.zeros(len(n_cell_vals))
upwind_mae = np.zeros(len(n_cell_vals))
weno_rmse = np.zeros(len(n_cell_vals))
weno_mae = np.zeros(len(n_cell_vals))
exact_rmse = np.zeros(len(n_cell_vals))
exact_mae = np.zeros(len(n_cell_vals))
exact_interp_rmse = np.zeros(len(n_cell_vals))
exact_interp_mae = np.zeros(len(n_cell_vals))

"""
Run simulations to compute error
"""

for i in range(len(n_cell_vals)):
    #Extract n_cells
    n_cell = n_cell_vals[i]
    n_cell_fipy = n_cell - 1

    #Perform simulations
    val_upwind, x,y = model1_upwind(n_cell,n_cell,2.0,2.0,[1.0,1.0],[0.0,1.0],1.0,f0_fun)

    val_weno, val_ana_pyclaw = model1_weno(n_cell_fipy,n_cell_fipy,2.0,2.0,(1.0,1.0),[0.0,1.0])

    val_exact, x2,y2 = model1_exact(n_cell,2.0,2.0,[1.0,1.0],[0.0,1.0],f0_fun)

    val_exact_interp, X3,Y3 = model1_exact_interpolation(n_cell,2.0,2.0,[1.0,1.0],[0.0,1.0],f0_fun)

    #Compute Analytical solution
    val_ana = f_analytical(x,y,[1.0,1.0],1.0)

    #Compute Errors
    upwind_rmse[i] = np.sqrt(np.mean((val_ana-val_upwind[:,:,-1])**2))
    upwind_mae[i] = np.amax(np.abs(val_upwind[:,:,-1]-val_ana))

    exact_rmse[i] = np.sqrt(np.mean((val_ana-val_exact[:,:,-1])**2))
    exact_mae[i] = np.amax(np.abs(val_exact[:,:,-1]-val_ana))

    exact_interp_rmse[i] = np.sqrt(np.mean((val_ana-val_exact_interp[:,:,-1])**2))
    exact_interp_mae[i] = np.amax(np.abs(val_exact_interp[:,:,-1]-val_ana))

    weno_rmse[i] = np.sqrt(np.mean((val_ana_pyclaw-val_weno)**2))
    weno_mae[i] = np.amax(np.abs(val_weno-val_ana_pyclaw))


"""
Plotting
"""

fig1 = plt.figure(num=1)
plt.loglog(np.square(n_cell_vals),upwind_rmse,"-ko",label="Upwind",markerfacecolor="none")
plt.loglog(np.square(n_cell_vals),weno_rmse,"-bo",label="WENO",markerfacecolor="none")
plt.loglog(np.square(n_cell_vals),exact_rmse,"-ro",label="Exact",markerfacecolor="none")
plt.loglog(np.square(n_cell_vals),exact_interp_rmse,"-rx",label="Exact, Interpolation",markerfacecolor="none")
plt.xlabel(r"$N_{Cells}$")
plt.ylabel(r"RMSE")
plt.legend()
plt.tight_layout()
plt.savefig("case1_rmse.png",dpi=300)

fig2 = plt.figure(num=2)
plt.loglog(np.square(n_cell_vals),upwind_mae,"-ko",label="Upwind",markerfacecolor="none")
plt.loglog(np.square(n_cell_vals),weno_mae,"-bo",label="WENO",markerfacecolor="none")
plt.loglog(np.square(n_cell_vals),exact_mae,"-ro",label="Exact",markerfacecolor="none")
plt.loglog(np.square(n_cell_vals),exact_interp_mae,"-rx",label="Exact, Interpolation",markerfacecolor="none")
plt.xlabel(r"$N_{Cells}$")
plt.ylabel(r"MAE")
plt.legend()
plt.tight_layout()
plt.savefig("case1_mae.png",dpi=300)


plt.show()





