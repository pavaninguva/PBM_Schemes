from model4 import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
import pandas as pd
from clawpack import riemann
from clawpack import pyclaw


#formatting
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')

"""
Functions
"""

def f0_fun(x,y):
    f0 = 10.0*np.exp(-((x-0.4)**2)/0.005 -((y-0.4)**2)/0.005)
    return f0


def hfun(t,x,y):
    h = 1.0 + x*y
    return h

#Read in Analytical csv data
f_ana1 = pd.read_csv("f_ana.csv").to_numpy()
x_ana = pd.read_csv("Xvals.csv").to_numpy()
y_ana = pd.read_csv("Yvals.csv").to_numpy()

f_anafd = pd.read_csv("f_anaFD.csv").to_numpy()
x_anafd = pd.read_csv("XvalsFD.csv").to_numpy()
y_anafd = pd.read_csv("YvalsFD.csv").to_numpy()




def model4_weno(nx,ny,Lx,Ly, gvec, tvec):
    riemann_solver = riemann.advection_2D
    solver = pyclaw.SharpClawSolver2D(riemann_solver)
    solver.kernel_language = "Fortran"
    solver.weno_order = 5
    solver.lim_type = 2
    solver.cfl_max = 1.0

    solver.bc_upper[0] = pyclaw.BC.extrap
    solver.bc_upper[1] = pyclaw.BC.extrap
    def custom_bc(state,dim,t,qbc,num_ghost):
        for i in range(num_ghost):
            qbc[0,i,:] = 0.0
    solver.bc_lower[0] = pyclaw.BC.custom
    solver.bc_lower[1] = pyclaw.BC.custom
    solver.user_bc_lower = custom_bc

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
        step = dt*(1+x_c*y_c)
        return step

    solver.dq_src = source_step

    xc, yc = domain.grid.p_centers
    state.q[0,:,:] = (10.0*np.exp(-(((xc-0.4)**2)/0.005) -((yc-0.4)**2)/0.005))

    claw = pyclaw.Controller()
    claw.keep_copy = True
    claw.solution = pyclaw.Solution(state, domain)
    claw.tfinal = 1.0
    claw.solver = solver

    status = claw.run()
    print(status)
    f_np = claw.frames[-1].q[0,:,:]

    f_ana = griddata((x_ana.ravel(),y_ana.ravel()), f_ana1.ravel(), (xc,yc),method="nearest")
    

    return f_np, f_ana






"""
Error Calculation
"""

n_cell_vals = np.array([41,51,81,101,201,321])
upwind_rmse = np.zeros(len(n_cell_vals))
upwind_mae = np.zeros(len(n_cell_vals))
weno_rmse = np.zeros(len(n_cell_vals))
weno_mae = np.zeros(len(n_cell_vals))
exact_rmse = np.zeros(len(n_cell_vals))
exact_mae = np.zeros(len(n_cell_vals))


for i in range(len(n_cell_vals)):
    #Extract n_cells
    n_cell = n_cell_vals[i]
    n_cell_pyclaw = n_cell - 1

    #Upwind
    f_upwind, X, Y = model4_upwind([n_cell,n_cell],[2.0,2.0],[1.0,1.0],[0.0,1.0],1.0,hfun,f0_fun)
    f_ana = griddata((x_anafd.ravel(),y_anafd.ravel()), f_anafd.ravel(), (X,Y),method="nearest")

    upwind_rmse[i] = np.sqrt(np.mean((f_upwind-f_ana)**2))
    upwind_mae[i] = np.amax(np.abs(f_upwind-f_ana))

    #Exact
    f_exact, X1,Y1 = model4_split(n_cell,[2.0,2.0],[1.0,1.0],[0.0,1.0],hfun,f0_fun)
    f_ana = griddata((x_anafd.ravel(),y_anafd.ravel()), f_anafd.ravel(), (X1,Y1),method="nearest")

    exact_rmse[i] = np.sqrt(np.mean((f_exact-f_ana)**2))
    exact_mae[i] = np.amax(np.abs(f_exact-f_ana))

    #Van Leer
    f_weno, f_ana_pyclaw = model4_weno(n_cell_pyclaw,n_cell_pyclaw,2.0,2.0,[1.0,1.0],[0.0,1.0])
    weno_rmse[i] = np.sqrt(np.mean((f_weno-f_ana_pyclaw)**2))
    weno_mae[i] = np.amax(np.abs(f_weno-f_ana_pyclaw))


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
plt.savefig("case4_rmse.png",dpi=300)


fig2 = plt.figure(num=2)
plt.loglog(np.square(n_cell_vals),upwind_mae,"-ko",label="Upwind",markerfacecolor="none")
plt.loglog(np.square(n_cell_vals),weno_mae,"-bo",label="WENO",markerfacecolor="none")
plt.loglog(np.square(n_cell_vals),exact_mae,"-ro",label="Exact",markerfacecolor="none")
plt.xlabel(r"$N_{Cells}$")
plt.ylabel(r"MAE")
plt.legend()
plt.tight_layout()
plt.savefig("case4_mae.png",dpi=300)

plt.show()

