import os
import numpy as np
import matplotlib.pyplot as plt
from dedalus import public as de

#formatting
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')

#Parameters
Lx = 2
Ly = 2
Nx = 128
Ny = 128
timestepper = de.timesteppers.RK111
stop_time = 1.0

#Domain
x_basis = de.Fourier('x', Nx, interval=(0, Lx))
y_basis = de.Fourier('y', Ny, interval=(0, Ly))
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

#Specify Equation
problem = de.IVP(domain, variables = ["f"])
problem.add_equation("dt(f) = -dx((0.25+0.5*(x+y))*f) -dy((0.5 + 0.25*(x+y))*f)")

#Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_time

#Initial Conditions
f = solver.state['f']
x = domain.grid(0)
y = domain.grid(1)
f["g"] = 50.0*np.exp(-((x-0.4)**2)/0.005 -((y-0.4)**2)/0.005)

xm, ym = np.meshgrid(x,y)
# fig, axis = plt.subplots()
# p = axis.pcolormesh(xm, ym, f['g'].T, cmap='RdBu_r')
# axis.set_xlim([0,2.])
# axis.set_ylim([0,2.])

while solver.ok:
    solver.step(5e-5)

# Plot solution
fvals = solver.state['f']

fig1 = plt.figure(num=1)
plt.pcolormesh(xm,ym,fvals["g"].T, cmap="jet",shading='gouraud')
plt.colorbar(label=r"$f$")
plt.xlabel(r"$a_{1}$")
plt.ylabel(r"$a_{2}$")
plt.clim(0,50)
plt.tight_layout()
plt.savefig("case3_dedalus.png",dpi=300)

plt.show()
