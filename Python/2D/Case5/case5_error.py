from model5 import *
import matplotlib.pyplot as plt
import numpy as np
from clawpack import riemann
from clawpack import pyclaw

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

def model5_weno(nx,ny,Lx,Ly,gvec,tvec):

    riemann_solver = riemann.advection_2D
    solver = pyclaw.SharpClawSolver2D(riemann_solver)
    solver.kernel_language = "Fortran"
    solver.weno_order = 5
    solver.lim_type = 2
    solver.cfl_max = 1.0

    solver.bc_lower[0] = pyclaw.BC.periodic
    solver.bc_upper[0] = pyclaw.BC.periodic
    solver.bc_lower[1] = pyclaw.BC.periodic
    solver.bc_upper[1] = pyclaw.BC.periodic

    nx = nx
    ny = ny 
    Lx = Lx 
    Ly = Ly
    x = pyclaw.Dimension(0.0,Lx,nx,name="x")
    y = pyclaw.Dimension(0.0,Ly,ny,name="y")
    domain = pyclaw.Domain([x,y])
    state = pyclaw.State(domain,solver.num_eqn)
    state.problem_data["u"] = 1.0
    state.problem_data["v"] = 1.0

    def source_step(solver,state,dt):
        x_c, y_c = state.grid.p_centers
        step = -dt*(x_c+y_c)*state.q[0,:,:]
        return step

    solver.dq_src = source_step

    xc, yc = domain.grid.p_centers
    state.q[0,:,:] = (50.0*np.exp(-(((xc-0.4)**2)/0.005) -((yc-0.4)**2)/0.005))

    claw = pyclaw.Controller()
    claw.keep_copy = True
    claw.solution = pyclaw.Solution(state, domain)
    claw.tfinal = 1.0
    claw.solver = solver

    status = claw.run()
    print(status)
    f1_np = claw.frames[-1].q[0,:,:]

    f_ana = analytical(1.0,xc,yc)
        

    return f1_np, f_ana

"""
Run simulations
"""

n_cell_vals = np.array([41,51,81,101,161,201])

weno_rmse = np.zeros(len(n_cell_vals))
weno_mae = np.zeros(len(n_cell_vals))

upwind_rmse = np.zeros(len(n_cell_vals))
upwind_mae = np.zeros(len(n_cell_vals))

exact_rmse = np.zeros(len(n_cell_vals))
exact_mae = np.zeros(len(n_cell_vals))

for i in range(len(n_cell_vals)):
    #Extract n_cells
    n_cell = n_cell_vals[i]
    n_cell_pyclaw = n_cell - 1

    #Van leer and error
    val_weno, val_ana_pyclaw = model5_weno(n_cell_pyclaw,n_cell_pyclaw,2.0,2.0,[1.0,1.0],[0.0,1.0])
    weno_rmse[i] = np.sqrt(np.mean((val_ana_pyclaw-val_weno)**2))
    weno_mae[i] = np.amax(np.abs(val_weno-val_ana_pyclaw))

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
plt.loglog(np.square(n_cell_vals),weno_rmse,"-bo",label="WENO",markerfacecolor="none")
plt.loglog(np.square(n_cell_vals),exact_rmse,"-ro",label="Exact",markerfacecolor="none")
plt.xlabel(r"$N_{Cells}$")
plt.ylabel(r"RMSE")
plt.legend()
plt.tight_layout()
plt.savefig("case5_rmse.png",dpi=300)


fig2 = plt.figure(num=2)
plt.loglog(np.square(n_cell_vals),upwind_mae,"-ko",label="Upwind",markerfacecolor="none")
plt.loglog(np.square(n_cell_vals),weno_mae,"-bo",label="WENO",markerfacecolor="none")
plt.loglog(np.square(n_cell_vals),exact_mae,"-ro",label="Exact",markerfacecolor="none")
plt.xlabel(r"$N_{Cells}$")
plt.ylabel(r"MAE")
plt.legend()
plt.tight_layout()
plt.savefig("case5_mae.png",dpi=300)

plt.show()


