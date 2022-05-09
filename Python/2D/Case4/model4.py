import numpy as np

"""
Functions to solve homogeneous constant-growth PBMs of the form:
df/dt + g1*df/a1 + g2*df/da2 = h(t,a1,a2)
g1,g2 > 0.
The upwind scheme and the exact scheme based on dimensional splitting
are implmemented here
"""

def model4_upwind(nvec, Lvec, g_vec, t_vec, CFL, hfun,f0fun):

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
        h_vals = hfun(t, X, Y)
        #Initalize dummy array to be clobbered
        f_vals = np.zeros((nx,ny))
        #Iterate through each value
        for idx, f in np.ndenumerate(f_vals):
            #Impose BCs
            #Top BC (a2 = Ly): df/da2 = 0
            if idx[0] == ny-1:
                f_vals[idx] = f_old[idx] -alpha*(f_old[idx[0],idx[1]] - f_old[idx[0],idx[1]-1]) + dt*h_vals[idx]
            #Bottom BC (a2 = 0): f(ghost node) = 0
            elif idx[0] == 0:
                f_vals[idx] = f_old[idx] -alpha*(f_old[idx[0],idx[1]] - f_old[idx[0],idx[1]-1] ) -beta*(f_old[idx[0],idx[1]]) + dt*h_vals[idx]
            #Right BC (a1 = 0): f(ghost node) = 0
            elif idx[1] == 0:
                f_vals[idx] = f_old[idx] -alpha*(f_old[idx[0],idx[1]]) -beta*(f_old[idx[0],idx[1]] - f_old[idx[0]-1,idx[1]]) + dt*h_vals[idx]
            #Left BC (a1=Lx)L df/da1 = 0
            elif idx[1] == nx-1:
                f_vals[idx] = f_old[idx] -beta*(f_old[idx[0],idx[1]] - f_old[idx[0]-1,idx[1]]) + dt*h_vals[idx]
            else:
                f_vals[idx] =  f_old[idx] -alpha*(f_old[idx[0],idx[1]] - f_old[idx[0],idx[1]-1]) -beta*(f_old[idx[0],idx[1]] - f_old[idx[0]-1,idx[1]]) + dt*h_vals[idx]


        #update
        f_old = f_vals
        t = t + dt
        print("Current Simulation Time is %s"%t)

    return f_old, X,Y


# def model4_split(nx, Lvec, g_vec, t_vec, hfun, f0fun):

#     #Define mesh and timevec
#     Lx,Ly = Lvec
#     dx = Lx/(nx-1)
#     dt = 1.0/((g_vec[0]/dx))
#     #Compute dy and ny using dt
#     dy = g_vec[0]*dt
#     ny = int((Ly/dy) + 1)
#     n_steps = round((t_vec[1] - t_vec[0])/dt)

#     x_vals = np.linspace(start=0,stop=Lx,num=nx)
#     y_vals = np.linspace(start=0,stop=Ly,num=ny)
#     X,Y = np.meshgrid(x_vals,y_vals)

#     #Initialize array
#     f_old = f0fun(X,Y)

#     #Perform timestepping
#     t = t_vec[0]
#     for i in range(n_steps):
#         #Initalize dummy array to be clobbered
#         f_vals = np.zeros((ny,nx))
#         f_vals_ = np.zeros((ny,nx))

#         h_vals = hfun(t, X, Y)

#         #Solve in x-direction first
#         for idx, f in np.ndenumerate(f_vals):
#             #Impose BCs
#             #Left BC (a1 = 0): f(ghost node) = 0
#             if idx[1] == 0:
#                 f_vals[idx] = 0.0 
#             #No Need to Impose No BC on RHS
#             else:
#                 f_vals[idx] = f_old[idx[0],idx[1]-1]

#         #Solve in y-direction 
#         for idx_, f_ in np.ndenumerate(f_vals_):
#             #Impose BCs
#             #Bottom BC
#             if idx_[0] == 0:
#                 f_vals_[idx_] = dt*h_vals[idx]
#             else:
#                 f_vals_[idx_] = f_vals[idx_[0]-1, idx_[1]] + dt*h_vals[idx]

#         #Update
#         f_old = f_vals_
#         t = t + dt
#         print("Current Simulation Time is %s"%t)

#     return f_old, X,Y


def model4_split(nx, Lvec, g_vec, t_vec, hfun, f0fun):

    #Define mesh and timevec
    Lx,Ly = Lvec
    dx = Lx/(nx-1)
    dt = 1.0/((g_vec[0]/dx))
    #Compute dy and ny using dt
    dy = g_vec[0]*dt
    ny = int((Ly/dy) + 1)
    n_steps = round((t_vec[1] - t_vec[0])/dt)

    x_vals = np.linspace(start=0,stop=Lx,num=nx)
    y_vals = np.linspace(start=0,stop=Ly,num=ny)
    X,Y = np.meshgrid(x_vals,y_vals)

    #Initialize array
    f_old = f0fun(X,Y)

    #Perform timestepping
    t = t_vec[0]
    for i in range(n_steps):
        #Initalize dummy array to be clobbered
        f_vals = np.zeros((ny,nx))
        f_vals_ = np.zeros((ny,nx))
        f_vals__ = np.zeros((ny,nx))

        #Solve in x-direction first
        for idx, f in np.ndenumerate(f_vals):
            #Impose BCs
            #Left BC (a1 = 0): f(ghost node) = 0
            if idx[1] == 0:
                f_vals[idx] = 0.0 
            #No Need to Impose No BC on RHS
            else:
                f_vals[idx] = f_old[idx[0],idx[1]-1]

        #Solve in y-direction 
        for idx_, f_ in np.ndenumerate(f_vals_):
            #Impose BCs
            #Bottom BC
            if idx_[0] == 0:
                f_vals_[idx_] = 0.0
            else:
                f_vals_[idx_] = f_vals[idx_[0]-1, idx_[1]]

        #Solve non-homogeneous
        for idx__, f__ in np.ndenumerate(f_vals__):
            f_vals__[idx__] = f_vals_[idx__] + dt*hfun(t,X[idx__],Y[idx__])

        #Update
        f_old = f_vals__
        t = t + dt
        print("Current Simulation Time is %s"%t)

    return f_old, X,Y





