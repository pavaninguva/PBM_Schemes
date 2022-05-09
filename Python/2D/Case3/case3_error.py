import matplotlib.pyplot as plt
import numpy as np
from model3 import *
import awkward as ak
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

def model3_weno(nx,ny,Lx,Ly, g1fun,g2fun,t_vec):
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
    state.aux[0,:,:] = 0.25 + 0.5*(xe[:-1,1:] + ye[:-1,1:])
    state.aux[1,:,:] = 0.5 + 0.25*(xe[:-1,1:]+ye[:-1,1:])

    #Source term
    def source_step(solver,state,dt):
        step = -dt*0.75*state.q[0,:,:]
        return step

    solver.dq_src = source_step

    #Set IC
    xc, yc = domain.grid.p_centers
    state.q[0,:,:] = (50.0*np.exp(-(((xc-0.4)**2)/0.005) -((yc-0.4)**2)/0.005))

    #Set Controller
    claw = pyclaw.Controller()
    claw.keep_copy = True
    claw.solution = pyclaw.Solution(state, domain)
    claw.tfinal = 1.0
    claw.solver = solver

    status = claw.run()
    print(status)
    f_np = claw.frames[-1].q[0,:,:]

    f_ana = analytical(xc,yc,1.0)

    return f_np, f_ana

"""
Run Simulations
"""

n_cell_vals = np.array([21,41,51,81,101,161,201])

dt_vals = np.array([0.1,0.05,0.04,0.025,0.02,0.01])

weno_rmse = np.zeros(len(n_cell_vals))
weno_mae = np.zeros(len(n_cell_vals))

conuniform_upwind_rmse = np.zeros(len(n_cell_vals))
conuniform_upwind_mae = np.zeros(len(n_cell_vals))

split_conuniform_upwind_rmse = np.zeros(len(n_cell_vals))
split_conuniform_upwind_mae = np.zeros(len(n_cell_vals))

split_transuniform_upwind_rmse = np.zeros(len(n_cell_vals))
split_transuniform_upwind_mae = np.zeros(len(n_cell_vals))

split_transnonuni_upwind_rmse = np.zeros(len(dt_vals))
split_transnonuni_upwind_mae = np.zeros(len(dt_vals))

transuni_ncell = np.zeros(len(dt_vals))

# for i in range(len(n_cell_vals)):
#     #Extract n_cells
#     n_cell = n_cell_vals[i]
#     n_cell_fipy = n_cell - 1

#     #Van leer and error
#     val_weno, val_ana_pyclaw = model3_weno(n_cell_fipy,n_cell_fipy,2.0,2.0,g1fun,g2fun,[0.0,1.0])

#     weno_rmse[i] = np.sqrt(np.mean((val_ana_pyclaw-val_weno)**2))
#     weno_mae[i] = np.amax(np.abs(val_weno-val_ana_pyclaw))

#     #Conuniform, upwind and error
#     val_conupwind, X, Y = model3_conservative_upwind([n_cell,n_cell],[2.0,2.0],g1fun,g2fun,[0.0,1.0],f0_fun)

#     val_ana = analytical(X,Y,1.0)
#     conuniform_upwind_rmse[i] = np.sqrt(np.mean((val_ana-val_conupwind[:,:,-1])**2))
#     conuniform_upwind_mae[i] = np.amax(np.abs(val_conupwind[:,:,-1]-val_ana))

#     #Split Conuniform upwind and error
#     val_split, X1, Y1 = model3_split_conservative([n_cell,n_cell],[2.0,2.0],g1fun,g2fun,[0.0,1.0],f0_fun)

#     val_ana = analytical(X1,Y1,1.0)
#     split_conuniform_upwind_rmse[i] = np.sqrt(np.mean((val_ana-val_split[:,:,-1])**2))
#     split_conuniform_upwind_mae[i] = np.amax(np.abs(val_split[:,:,-1]-val_ana))

#     #Split transuniform upwind and error
#     val_split2, X2, Y2 = model3_split_transform([n_cell,n_cell],[2.0,2.0],g1fun,g2fun,[0.0,1.0],f0_fun)

#     val_ana_ = analytical(X2,Y2,1.0)
#     split_transuniform_upwind_rmse[i] = np.sqrt(np.mean((val_ana_-val_split2[:,:,-1])**2))
#     split_transuniform_upwind_mae[i] = np.amax(np.abs(val_split2[:,:,-1]-val_ana_))

# for j in range(len(dt_vals)):
#     dt = dt_vals[j]

#     #Solve
#     val_cfl, X_CFL, Y_CFL = model3_split_transform_cfl([2.0,2.0],[0.0,1.0],dt,g1fun,g2fun,f0_fun)

#     #Output ncell
#     transuni_ncell[j] = ak.count(val_cfl,axis=None)

#     #Compute error
#     val_ana2 = analytical(X_CFL,Y_CFL,1.0) 

#     split_transnonuni_upwind_rmse[j] = np.sqrt(ak.mean((val_ana2-val_cfl)**2))
#     split_transnonuni_upwind_mae[j] = max(ak.max(val_ana2-val_cfl), -ak.min(val_ana2-val_cfl))


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
# fig1 = plt.figure(num=1)
# plt.loglog(np.square(n_cell_vals),conuniform_upwind_rmse,"-ko",label="Con-Uniform,Upwind",markerfacecolor="none")
# plt.loglog(np.square(n_cell_vals),split_conuniform_upwind_rmse,"--ks",label="Split-Con-Uniform,Upwind",markerfacecolor="none")
# plt.loglog(np.square(n_cell_vals),split_transuniform_upwind_rmse,":k^",label="Split-Trans-Uniform,Upwind",markerfacecolor="none")
# plt.loglog(np.square(n_cell_vals),weno_rmse,"-bo",label="Expanded-Uniform,WENO",markerfacecolor="none")
# plt.loglog(transuni_ncell,split_transnonuni_upwind_rmse, "-ro", label = "Split-Trans-Uniform,Upwind",markerfacecolor="none")
# plt.xlabel(r"$N_{Cells}$")
# plt.ylabel(r"RMSE")
# plt.legend(fontsize="medium")
# plt.tight_layout()
# # plt.savefig("case3_rmse.png",dpi=300)

# #Plot MAE
# fig2 = plt.figure(num=2)
# plt.loglog(np.square(n_cell_vals),conuniform_upwind_mae,"-ko",label="Con-Uniform,Upwind",markerfacecolor="none")
# plt.loglog(np.square(n_cell_vals),split_conuniform_upwind_mae,"--ks",label="Split-Con-Uniform,Upwind",markerfacecolor="none")
# plt.loglog(np.square(n_cell_vals),split_transuniform_upwind_mae,":k^",label="Split-Trans-Uniform,Upwind",markerfacecolor="none")
# plt.loglog(np.square(n_cell_vals),weno_mae,"-bo",label="Expanded-Uniform,WENO",markerfacecolor="none")
# plt.loglog(transuni_ncell,split_transnonuni_upwind_mae, "-ro", label = "Split-Trans-Uniform,Upwind",markerfacecolor="none")
# plt.xlabel(r"$N_{Cells}$")
# plt.ylabel(r"MAE")
# plt.legend(fontsize="medium")
# plt.tight_layout()
# plt.savefig("case3_mae.png",dpi=300)


#Plot exact scheme error
fig3 = plt.figure(num=3)
plt.loglog(dt_vals_exact,exact_51_rmse,"-ko",label=r"$N_{cell}=51$")
plt.loglog(dt_vals_exact,exact_101_rmse,"-ks",label=r"$N_{cell}=101$")
plt.loglog(dt_vals_exact,exact_201_rmse,"-k^",label=r"$N_{cell}=201$")
plt.xlabel(r"$\Delta t$")
plt.ylabel(r"RMSE")
plt.legend()
plt.tight_layout()
plt.savefig("case3_exact_rmse.png",dpi=300)

fig4 = plt.figure(num=4)
plt.loglog(dt_vals_exact,exact_51_mae,"-ko",label=r"$N_{cell}=51$")
plt.loglog(dt_vals_exact,exact_101_mae,"-ks",label=r"$N_{cell}=101$")
plt.loglog(dt_vals_exact,exact_201_mae,"-k^",label=r"$N_{cell}=201$")
plt.xlabel(r"$\Delta t$")
plt.ylabel(r"MAE")
plt.legend()
plt.tight_layout()
plt.savefig("case3_exact_mae.png",dpi=300)

plt.show()