import numpy as np


def model5_upwind (nvec, Lvec, g_vec, t_vec, CFL, lambdafun, f0fun):

    #Define mesh and timevec
    nx,ny = nvec
    Lx,Ly = Lvec

    dx = Lx/(nx-1)
    dy = Ly/(ny-1)
    dt = CFL/((g_vec[0]/dx) + (g_vec[1]/dy))
    n_steps = round((t_vec[1] - t_vec[0])/dt)

    x_vals = np.linspace(start=0,stop=Lx,num=nx)
    y_vals = np.linspace(start=0,stop=Ly,num=ny)
    X,Y = np.meshgrid(x_vals,y_vals)

    #Initialize solution array and store initial conditions
    f_old = f0fun(X,Y)

    #Perform timestepping
    alpha = (g_vec[0]*dt)/dx
    beta = (g_vec[1]*dt)/dy

    t = t_vec[0]
    for i in range(n_steps):
        lambda_vals = lambdafun(t, X, Y)
         #Initalize dummy array to be clobbered
        f_vals = np.zeros((nx,ny))
        #Iterate through each value
        for idx, f in np.ndenumerate(f_vals):
            #Impose BCs
            #Top BC (a2 = Ly): df/da2 = 0
            if idx[0] == ny-1:
                f_vals[idx] = f_old[idx] -alpha*(f_old[idx[0],idx[1]] - f_old[idx[0],idx[1]-1]) - dt*lambda_vals[idx]*f_old[idx]
            #Bottom BC (a2 = 0): f(a2=0) = 0
            elif idx[0] == 0:
                f_vals[idx] = 0.0
            #Right BC (a1 = 0): f(a1=0) = 0
            elif idx[1] == 0:
                f_vals[idx] = 0.0
            #Left BC (a1=Lx)L df/da1 = 0
            elif idx[1] == nx-1:
                f_vals[idx] = f_old[idx] -beta*(f_old[idx[0],idx[1]] - f_old[idx[0]-1,idx[1]]) - dt*lambda_vals[idx]*f_old[idx]
            else:
                f_vals[idx] =  f_old[idx] -alpha*(f_old[idx[0],idx[1]] - f_old[idx[0],idx[1]-1]) -beta*(f_old[idx[0],idx[1]] - f_old[idx[0]-1,idx[1]]) - dt*lambda_vals[idx]*f_old[idx]


        #update
        f_old = f_vals
        t = t + dt
        print("Current Simulation Time is %s"%t)

    return f_old, X,Y


def model5_exact(nx, Lvec, gvec, t_vec, int_lambda, f0fun):
    #Define mesh and timevec
    Lx,Ly = Lvec
    dx = Lx/(nx-1)
    dt = 1.0/((gvec[0]/dx))
    #Compute dy and ny using dt
    dy = gvec[0]*dt
    ny = int((Ly/dy) + 1)
    n_steps = round((t_vec[1] - t_vec[0])/dt)

    x_vals = np.linspace(start=0,stop=Lx,num=nx)
    y_vals = np.linspace(start=0,stop=Ly,num=ny)
    X,Y = np.meshgrid(x_vals,y_vals)

    #Initialize functions 
    f0_vals = f0fun(X,Y)
    #Transform
    mu_vals = np.exp(int_lambda(X,Y))

    fhat_old = f0_vals*mu_vals

    #Perform timestepping
    t = t_vec[0]
    for i in range(n_steps):
        #Initalize dummy array to be clobbered
        fhat_vals = np.zeros((ny,nx))
        fhat_vals_ = np.zeros((ny,nx))

        #Solve in x-direction first
        for idx, f in np.ndenumerate(fhat_vals):
            #Impose BCs
            #Left BC (a1 = 0): f(ghost node) = 0
            if idx[1] == 0:
                fhat_vals[idx] = 0.0 
            #No Need to Impose No BC on RHS
            else:
                fhat_vals[idx] = fhat_old[idx[0],idx[1]-1]

        #Solve in y-direction 
        for idx_, f_ in np.ndenumerate(fhat_vals_):
            #Impose BCs
            #Bottom BC
            if idx_[0] == 0:
                fhat_vals_[idx_] = 0.0
            else:
                fhat_vals_[idx_] = fhat_vals[idx_[0]-1, idx_[1]]
    
        #Update
        fhat_old = fhat_vals_
        t = t + dt
        print("Current Simulation Time is %s"%t)

    #Transform back to f_vals
    f_vals = fhat_old/mu_vals

    return f_vals, X,Y



