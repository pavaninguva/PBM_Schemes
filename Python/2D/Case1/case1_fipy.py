from fipy import *
import matplotlib.pyplot as plt
import numpy as np

#formatting
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')

#Setting up mesh with domain size Lx=Ly = 2.0
nx = ny = 100
dx = dy = 2.0/nx
mesh = PeriodicGrid2D(nx=nx, dx=dx, ny=ny, dy=dy)

#Specify Problem
convCoeff = (1.0,1.0)
CFL = 0.5
dt = CFL/((convCoeff[0]/dx) + (convCoeff[1]/dy))

f1 = CellVariable(mesh=mesh, name=r"$f_{1}$")
# f2 = CellVariable(mesh=mesh, name=r"$f_{2}$")
#Use VanLeer and Upwind
eq1 = TransientTerm(var=f1) + VanLeerConvectionTerm(var=f1, coeff=convCoeff) == 0
# eq2 = TransientTerm(var=f2) + ExplicitUpwindConvectionTerm(var=f2, coeff=convCoeff) == 0

#Define Initial Condition
x = mesh.cellCenters[0]
y = mesh.cellCenters[1]
f1.setValue(50.0*numerix.exp(-(((x-0.4)**2)/0.005) -((y-0.4)**2)/0.005))
# f2.setValue(50.0*numerix.exp(-(((x-0.4)**2)/0.005) -((y-0.4)**2)/0.005))

#Solve Equations
run_time = 1.0
t = 0.0

while t < run_time - 1e-8:
    eq1.solve(dt=dt)
    # eq2.solve(dt=dt)
    #Update time
    t = t+dt
    print("Current Simulation Time is %s"%t)

#Plotting
x_vals = np.linspace(0.0,2.0,nx)
y_vals = np.linspace(0.0,2.0,ny)

xx,yy = np.meshgrid(x_vals,y_vals)

#Reshape to Numpy arrays
f1_np = np.asarray(f1).reshape(nx,-1)
# f2_np = np.asarray(f2).reshape(nx,-1)


fig1 = plt.figure(num=1)
plt.pcolormesh(xx,yy,f1_np, cmap="jet",shading='gouraud')
plt.colorbar(label=r"$f$")
plt.xlabel(r"$a_{1}$")
plt.ylabel(r"$a_{2}$")
plt.clim(0,50)
plt.tight_layout()
plt.savefig("case1_vanleer.png",dpi=300)




# fig2 = plt.figure(num=2)
# plt.pcolormesh(xx,yy,f2_np, cmap="jet",shading='gouraud')
# plt.colorbar(label=r"$f$")
# plt.xlabel(r"$a_{1}$")
# plt.ylabel(r"$a_{2}$")
# plt.clim(0,50)




plt.show()

