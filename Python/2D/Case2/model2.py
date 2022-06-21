import numpy as np

#For exact scheme
from scipy.integrate import quad
from functools import partial

"""
Functions to solve homogeneous PBMs with a variable growth rate
that varies as a function of the intrinsic variables i.e.,
G = [G1(a1),G2(a2)]

In conservative form, the PBM is given by
df/dt + d(G1(a1)f)/da1 + d(G2(a2)f)/da2 = 0

The first variable transformation to solve this problem exactly is:
f' = G1(a1)G2(a2)f,
which transforms the PBM into 
df'/dt + G1(a1)*df'/da1 + G2(a2)*df/da2 = 0

The following solution strategies are considered:

1. Upwind Scheme applied to the conservative form on a uniform mesh
2. Upwind Scheme applied to the transformed equation on a uniform mesh
3. Upwind Scheme applied to the transformed equation on a nonuniform mesh
4. Exact Scheme incorporating all necessary transformations 
"""

def model2_conservative_upwind(n_vec, L_vec, g1fun, g2fun, t_vec,f0fun):

    #Extract mesh parameters and construct mesh
    nx, ny = n_vec
    Lx, Ly = L_vec

    dx = Lx/(nx-1)
    dy = Ly/(ny-1)

    x_vals = np.linspace(start=0,stop=Lx,num=nx)
    y_vals = np.linspace(start=0,stop=Ly,num=ny)
    X,Y = np.meshgrid(x_vals,y_vals)

    #Compute g_funs
    g1_vals = g1fun(X)
    g2_vals = g2fun(Y)

    #Compute dt using max values to ensure numerical stability
    dt = 1.0/((np.amax(g1_vals)/dx) + (np.amax(g2_vals)/dy))
    n_steps = round((t_vec[1] - t_vec[0])/dt)
    # t_vals = np.linspace(t_vec[0],t_vec[1], n_steps+1)

    #Initialize solution array and store initial conditions
    f_array = np.zeros((ny,nx, int(n_steps+1)))
    f_array[:,:,0] = f0fun(X,Y)

    #Perform Timestepping
    t = t_vec[0]

    for i in range(n_steps):
        #Initalize dummy array to be clobbered
        f_vals = np.zeros((ny,nx))
        f_old = f_array[:,:,i]
        #Iterate through each value
        for idx, f in np.ndenumerate(f_vals):
            #Top BC (a2 = Ly): d(Gf)/da2 = 0
            if idx[0] == 0:
                f_vals[idx] = f_old[idx] - (dt/dx)*((g1_vals[idx]*f_old[idx] - g1_vals[idx[0],idx[1]-1]*f_old[idx[0],idx[1]-1]))
            #Bottom BC (a2 = 0): f(ghost node) = 0
            elif idx[0] == n_vec[1] -1:
                f_vals[idx] = f_old[idx] -(dt/dx)*(g1_vals[idx]*f_old[idx] - g1_vals[idx[0],idx[1]-1]*f_old[idx[0],idx[1]-1]) -(dt/dy)*(g2_vals[idx]*f_old[idx])
            #Right BC (a1 = 0): f(ghost node) = 0
            elif idx[1] == 0:
                 f_vals[idx] = f_old[idx] -(dt/dx)*(g1_vals[idx]*f_old[idx]) -(dt/dy)*(g2_vals[idx]*f_old[idx] - g2_vals[idx[0]-1,idx[1]]*f_old[idx[0]-1,idx[1]])
             #Left BC (a1=Lx)L d(Gf)/da1 = 0
            elif idx[1] == n_vec[0]-1:
                f_vals[idx] = f_old[idx] -(dt/dy)*(g2_vals[idx]*f_old[idx] - g2_vals[idx[0]-1,idx[1]]*f_old[idx[0]-1,idx[1]])   
            #Rest 
            else:
                f_vals[idx] =  f_old[idx] -(dt/dx)*(g1_vals[idx]*f_old[idx] - g1_vals[idx[0],idx[1]-1]*f_old[idx[0],idx[1]-1]) -(dt/dy)*(g2_vals[idx]*f_old[idx] - g2_vals[idx[0]-1,idx[1]]*f_old[idx[0]-1,idx[1]])
            
        #Append arrays and update
        f_array[:,:,i+1] = f_vals
        t = t + dt
        print("Current Simulation Time is %s"%t)

    return f_array, X,Y


def model2_transformed_upwind(n_vec, L_vec, g1fun, g2fun, t_vec,f0fun):

    #Extract mesh parameters and construct mesh
    nx, ny = n_vec
    Lx, Ly = L_vec

    dx = Lx/(nx-1)
    dy = Ly/(ny-1)

    x_vals = np.linspace(start=0,stop=Lx,num=nx)
    y_vals = np.linspace(start=0,stop=Ly,num=ny)
    X,Y = np.meshgrid(x_vals,y_vals)

    #Compute g_funs
    g1_vals = g1fun(X)
    g2_vals = g2fun(Y)

    #Compute dt using max values to ensure numerical stability
    dt = 1.0/((np.amax(g1_vals)/dx) + (np.amax(g2_vals)/dy))
    n_steps = round((t_vec[1] - t_vec[0])/dt)

    #Initialize solution array and store initial conditions
    f_array = np.zeros((ny,nx, int(n_steps+1)))
    f_array[:,:,0] = f0fun(X,Y)

    #Perform Timestepping
    t = t_vec[0]

    for i in range(n_steps):
        fhat_vals = np.zeros((ny,nx))
        fhat_old = g1_vals*g2_vals*f_array[:,:,i]

        #Iterate through each value
        for idx, f in np.ndenumerate(fhat_vals):
             #Top BC (a2 = Ly): d(fhat)/da2 = 0
            if idx[0] == 0:
                fhat_vals[idx] = fhat_old[idx] - ((g1_vals[idx]*dt)/dx)*(fhat_old[idx] - fhat_old[idx[0],idx[1]-1])
            #Bottom BC (a2 = 0): fhat(ghost node) = 0
            elif idx[0] == ny -1:
                fhat_vals[idx] = fhat_old[idx] -((g1_vals[idx]*dt)/dx)*(fhat_old[idx] - fhat_old[idx[0],idx[1]-1]) -((g2_vals[idx]*dt)/dy)*(fhat_old[idx])
            #Right BC (a1 = 0): fhat(ghost node) = 0
            elif idx[1] == 0:
                 fhat_vals[idx] = fhat_old[idx] -((g1_vals[idx]*dt)/dx)*(fhat_old[idx]) -((g2_vals[idx]*dt)/dy)*(fhat_old[idx] - fhat_old[idx[0]-1,idx[1]])
            #Left BC (a1=Lx)L d(Gf)/da1 = 0
            elif idx[1] == n_vec[0]-1:
                fhat_vals[idx] = fhat_old[idx] -((g2_vals[idx]*dt)/dy)*(fhat_old[idx] - fhat_old[idx[0]-1,idx[1]])
            #Rest
            else:
                fhat_vals[idx] =  fhat_old[idx] -((g1_vals[idx]*dt)/dx)*(fhat_old[idx] - fhat_old[idx[0],idx[1]-1]) -((g2_vals[idx]*dt)/dy)*(fhat_old[idx] - fhat_old[idx[0]-1,idx[1]])

        #Append arrays and update
        f_array[:,:,i+1] = fhat_vals/(g1_vals*g2_vals)
        t = t + dt
        print("Current Simulation Time is %s"%t)

    return f_array, X,Y

def mesh_constructor(Lx,Ly, dt, g1fun,g2fun,gamma):
    #Inputs:
    #Lx: Input as a vec [Lx_0,Lx_end]
    #Ly: Input as a vec [Ly_0,Ly_end]

    Lx0,Lx_end = Lx
    Ly0,Ly_end = Ly

    #Initialize arrays
    x_vals = [Lx_end]
    y_vals = [Ly_end]

    c_x = 1
    while x_vals[-1] > Lx0:
        x_new = x_vals[c_x-1] -(1.0/gamma)*dt*g1fun(x_vals[c_x-1])
        x_vals.append(x_new)
        c_x = c_x +1 

    c_y = 1
    while y_vals[-1] > Ly0:
        y_new = y_vals[c_y-1] -(1.0/(1.0-gamma))*dt*g2fun(y_vals[c_y-1])
        y_vals.append(y_new)
        c_y = c_y +1

    x_array = np.array(x_vals[:-1][::-1])
    y_array = np.array(y_vals[:-1][::-1])

    #Compute n_cells
    nx = len(x_vals[:-1])
    ny = len(y_vals[:-1])

    #Create mesh
    X,Y = np.meshgrid(x_array,y_array)
    
    #Return last values for mesh
    X_ = x_vals[-1]
    Y_ = y_vals[-1]

    return nx,ny,X_,Y_,X,Y

def model2_conservative_nonuniform(Lx,Ly,dt,g1fun,g2fun,gamma,t_vec,f0fun):

    #Create mesh
    nx,ny,X_,Y_,X,Y = mesh_constructor(Lx,Ly,dt,g1fun,g2fun,gamma)

    #Construct gfuns
    g1_vals = g1fun(X)
    g2_vals = g2fun(Y)

    #Initialize solution array and store initial conditions
    n_steps = round((t_vec[1] - t_vec[0])/dt)

    f_array = np.zeros((ny,nx, int(n_steps+1)))
    f_array[:,:,0] = f0fun(X,Y)

    #Perform Timestepping
    t = t_vec[0]

    for i in range(n_steps):
        #Initalize dummy array to be clobbered
        f_vals = np.zeros((ny,nx))
        f_old = f_array[:,:,i]

         #Iterate through each value
        for idx, f in np.ndenumerate(f_vals):
            dx = X[idx] - X[idx[0],idx[1]-1]
            dy = Y[idx] - Y[idx[0]-1,idx[1]]
             #Top BC (a2 = Ly): d(Gf)/da2 = 0
            if idx[0] == 0:
                f_vals[idx] = f_old[idx] - (dt/dx)*((g1_vals[idx]*f_old[idx] - g1_vals[idx[0],idx[1]-1]*f_old[idx[0],idx[1]-1]))
            #Bottom BC (a2 = 0): f(ghost node) = 0
            elif idx[0] == ny -1:
                dy = Y[idx] - Y_
                f_vals[idx] = f_old[idx] -(dt/dx)*(g1_vals[idx]*f_old[idx] - g1_vals[idx[0],idx[1]-1]*f_old[idx[0],idx[1]-1]) -(dt/dy)*(g2_vals[idx]*f_old[idx])
            #Right BC (a1 = 0): f(ghost node) = 0
            elif idx[1] == 0:
                dx = X[idx] - X_
                f_vals[idx] = f_old[idx] -(dt/dx)*(g1_vals[idx]*f_old[idx]) -(dt/dy)*(g2_vals[idx]*f_old[idx] - g2_vals[idx[0]-1,idx[1]]*f_old[idx[0]-1,idx[1]])
             #Left BC (a1=Lx)L d(Gf)/da1 = 0
            elif idx[1] == nx-1:
                f_vals[idx] = f_old[idx] -(dt/dy)*(g2_vals[idx]*f_old[idx] - g2_vals[idx[0]-1,idx[1]]*f_old[idx[0]-1,idx[1]])   
            #Rest 
            else:
                f_vals[idx] =  f_old[idx] -(dt/dx)*(g1_vals[idx]*f_old[idx] - g1_vals[idx[0],idx[1]-1]*f_old[idx[0],idx[1]-1]) -(dt/dy)*(g2_vals[idx]*f_old[idx] - g2_vals[idx[0]-1,idx[1]]*f_old[idx[0]-1,idx[1]])
            
        #Append arrays and update
        f_array[:,:,i+1] = f_vals
        t = t + dt
        # print("Current Simulation Time is %s"%t)
    print(nx,ny)

    return f_array, X,Y

def model2_trans_nonuniform(Lx,Ly,dt,g1fun,g2fun,gamma,t_vec,f0fun):

    #Create mesh
    nx,ny,X_,Y_,X,Y = mesh_constructor(Lx,Ly,dt,g1fun,g2fun,gamma)

    #Construct gfuns
    g1_vals = g1fun(X)
    g2_vals = g2fun(Y)

    #Initialize solution array and store initial conditions
    n_steps = round((t_vec[1] - t_vec[0])/dt)

    f_array = np.zeros((ny,nx, int(n_steps+1)))
    f_array[:,:,0] = f0fun(X,Y)

    #Perform Timestepping
    t = t_vec[0]

    for i in range(n_steps):
        #Initalize dummy array to be clobbered
        fhat_vals = np.zeros((ny,nx))
        fhat_old = g1_vals*g2_vals*f_array[:,:,i]

        #Iterate through each value
        for idx, f in np.ndenumerate(fhat_vals):
            dx = X[idx] - X[idx[0],idx[1]-1]
            dy = Y[idx] - Y[idx[0]-1,idx[1]]
             #Top BC (a2 = Ly): d(Gf)/da2 = 0
            if idx[0] == 0:
                fhat_vals[idx] = fhat_old[idx] - (g1_vals[idx]*dt/dx)*(fhat_old[idx] - fhat_old[idx[0],idx[1]-1])
            #Bottom BC (a2 = 0): f(ghost node) = 0
            elif idx[0] == ny -1:
                dy = Y[idx] - Y_
                fhat_vals[idx] = fhat_old[idx] -(g1_vals[idx]*dt/dx)*(fhat_old[idx] - fhat_old[idx[0],idx[1]-1]) -(g2_vals[idx]*dt/dy)*(fhat_old[idx])
            #Right BC (a1 = 0): f(ghost node) = 0
            elif idx[1] == 0:
                dx = X[idx] - X_
                fhat_vals[idx] = fhat_old[idx] -(g1_vals[idx]*dt/dx)*(fhat_old[idx]) -(g2_vals[idx]*dt/dy)*(fhat_old[idx] - fhat_old[idx[0]-1,idx[1]])
             #Left BC (a1=Lx)L d(Gf)/da1 = 0
            elif idx[1] == nx-1:
                fhat_vals[idx] = fhat_old[idx] -(g2_vals[idx]*dt/dy)*(fhat_old[idx] - fhat_old[idx[0]-1,idx[1]])   
            #Rest 
            else:
                fhat_vals[idx] =  fhat_old[idx] -(g1_vals[idx]*dt/dx)*(fhat_old[idx] - fhat_old[idx[0],idx[1]-1]) -(g2_vals[idx]*dt/dy)*(fhat_old[idx] - fhat_old[idx[0]-1,idx[1]])
            
        #Append arrays and update
        f_array[:,:,i+1] = fhat_vals/(g1_vals*g2_vals)
        t = t + dt
        # print("Current Simulation Time is %s"%t)
    print(nx,ny)

    return f_array, X,Y


def model2_exact(n_vec, Lx,Ly, g1fun,g2fun, f0fun, t_vec):

    def recip(x,fun):
        f = 1/(fun(x))
        return f

    #Compute Variable Transformations
    def x_transfun(x):
        result = np.array(list(map(partial(quad,(lambda x:recip(x,g1fun)),0,epsabs=1e-15,epsrel=1e-15),x)))[:,0]
        return result

    def y_transfun(y):
        result = np.array(list(map(partial(quad,(lambda x:recip(x,g2fun)),0,epsabs=1e-15,epsrel=1e-15),y)))[:,0]
        return result

    #Create Interpolation Items for Computing Inverse
    x_vals = np.linspace(Lx[0],Lx[1],20001)
    y_vals = np.linspace(Ly[0],Ly[1],20001)

    xtilde_vals = x_transfun(x_vals)
    ytilde_vals = y_transfun(y_vals)

    #Create mesh in transformed variable space
    nx,ny = n_vec
    
    xtilde_range = x_transfun(Lx)
    ytilde_range = y_transfun(Ly)
    xtilde_mesh_vals = np.linspace(xtilde_range[0],xtilde_range[1],nx)
    ytilde_mesh_vals = np.linspace(ytilde_range[0],ytilde_range[1],ny)

    Xtilde, Ytilde = np.meshgrid(xtilde_mesh_vals,ytilde_mesh_vals)

    #Create mesh in original variables
    x_mesh_vals = np.interp(xtilde_mesh_vals,xtilde_vals,x_vals)
    y_mesh_vals = np.interp(ytilde_mesh_vals,ytilde_vals,y_vals)

    X,Y = np.meshgrid(x_mesh_vals,y_mesh_vals)

    t_end = t_vec[1]
    #Compute Solution
    fhat = np.zeros((ny,nx))
    for idx, f in np.ndenumerate(fhat):
        if Xtilde[idx] < xtilde_vals[0] + t_end or Ytilde[idx] < ytilde_vals[0] + t_end:
            fhat[idx] = 0.0
            # pass
        else:
            Xtilde_shift = np.interp(Xtilde[idx] - t_end,xtilde_vals,x_vals)
            Ytilde_shift =  np.interp(Ytilde[idx] - t_end,ytilde_vals,y_vals)
            fhat[idx] = g1fun(Xtilde_shift)*g2fun(Ytilde_shift)*f0fun(Xtilde_shift,Ytilde_shift)

    f = fhat/(g1fun(X)*g2fun(Y))

    return f, X, Y

def model2_exact_analytical(n_vec,Lx,Ly,g1fun,g2fun,tildea1_fun,tildea2_fun,a1fun,a2fun,f0fun,t_end):
    #Unpack
    nx,ny = n_vec
    Lx0,Lxend = Lx
    Ly0,Lyend = Ly

    xtilde0 = tildea1_fun(Lx0)
    xtildeend = tildea1_fun(Lxend)

    ytilde0 = tildea2_fun(Ly0)
    ytildeend = tildea2_fun(Lyend)

    #Create mesh
    xtilde_vals = np.linspace(xtilde0,xtildeend,nx)
    ytilde_vals = np.linspace(ytilde0,ytildeend,ny)
    Xtilde, Ytilde = np.meshgrid(xtilde_vals,ytilde_vals)

    x_vals = a1fun(xtilde_vals)
    y_vals = a2fun(ytilde_vals)
    X,Y = np.meshgrid(x_vals,y_vals)

    fhat = np.zeros((ny,nx))
    for idx, f in np.ndenumerate(fhat):
        if Xtilde[idx] < xtilde_vals[0] + t_end or Ytilde[idx] < ytilde_vals[0] + t_end:
            fhat[idx] = 0.0
            # pass
        else:
            Xtilde_shift = a1fun(Xtilde[idx] - t_end)
            Ytilde_shift = a2fun(Ytilde[idx] - t_end)
            fhat[idx] = g1fun(Xtilde_shift)*g2fun(Ytilde_shift)*f0fun(Xtilde_shift,Ytilde_shift)

    f = fhat/(g1fun(X)*g2fun(Y))

    return f, X,Y



    








    











    



