import numpy as np
from clawpack import riemann
from clawpack import pyclaw
import matplotlib.pyplot as plt

#formatting
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')

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
nx = ny = 100
Lx = Ly = 2.0
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
sol = claw.frames[-1].q[0,:,:]

#Plotting
fig1 = plt.figure(num=1)
plt.pcolormesh(xc,yc,sol,cmap="jet",shading="gouraud")
plt.colorbar(label=r"$f$")
plt.xlabel(r"$a_{1}$")
plt.ylabel(r"$a_{2}$")
plt.tight_layout()
plt.clim(0,50)

plt.show()