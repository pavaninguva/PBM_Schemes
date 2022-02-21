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

#Construct Problem
A1 = mesh.faceCenters[0]
A2 = mesh.faceCenters[1]

coeff = FaceVariable(mesh=mesh,rank=1)
coeff[0] = 0.1 + 0.05*A1 
coeff[1] = 0.5 + 0.25*A2

print (max(coeff[0]))

f1 = CellVariable(mesh=mesh)
eq1 = TransientTerm(var = f1) + VanLeerConvectionTerm(var=f1, coeff=coeff) == 0

#Define Initial Condition
x = mesh.cellCenters[0]
y = mesh.cellCenters[1]
f1.setValue(50.0*numerix.exp(-(((x-0.4)**2)/0.005) -((y-0.4)**2)/0.005))

#Solve equation
run_time = 1.0
t = 0.0

CFL = 0.5
dt = CFL/((max(coeff[0])/dx) + (max(coeff[1])/dy))

while t < run_time - 1e-8:
    eq1.solve(dt=dt)
    #Update time
    t = t+dt
    print("Current Simulation Time is %s"%t)

#Compute Analytical Solution
g1_vals = 0.1 + 0.05*x
g2_vals = 0.5 + 0.25*y

x_shift = (x+2.0)*numerix.exp(-0.05*1.0) -2.0
y_shift = (y+2.0)*numerix.exp(-0.25*1.0) - 2.0

g1_shift = 0.1 + 0.05*x_shift
g2_shift = 0.5 + 0.25*y_shift
f0_shift = 50.0*numerix.exp(-(((x_shift-0.4)**2)/0.005) -(((y_shift-0.4)**2)/0.005))

f_ana = CellVariable(mesh=mesh)
f_ana.setValue((g1_shift*g2_shift*f0_shift)/(g1_vals*g2_vals))
f_ana_np = np.asarray(f_ana).reshape(nx,-1)

#Plotting
x_vals = np.linspace(0.0,2.0,nx)
y_vals = np.linspace(0.0,2.0,ny)

xx,yy = np.meshgrid(x_vals,y_vals)

#Reshape to Numpy arrays
f1_np = np.asarray(f1).reshape(nx,-1)

fig1 = plt.figure(num=1)
plt.pcolormesh(xx,yy,f1_np, cmap="jet",shading='gouraud')
plt.colorbar(label=r"$f$")
plt.xlabel(r"$a_{1}$")
plt.ylabel(r"$a_{2}$")
plt.clim(0,50)
plt.tight_layout()
plt.savefig("case2_vanleer.png",dpi=300)

fig2 = plt.figure(num=2)
plt.pcolormesh(xx,yy,f_ana_np, cmap="jet",shading='gouraud')
plt.colorbar(label=r"$f$")
plt.xlabel(r"$a_{1}$")
plt.ylabel(r"$a_{2}$")
plt.clim(0,50)
plt.tight_layout()
plt.savefig("case2_fipy_ana.png",dpi=300)

plt.show()
