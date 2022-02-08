import numpy as np

"""
Functions to solve homogeneous constant-growth PBMs of the form:
df/dt + g1*df/a1 + g2*df/da2 = 0,
g1,g2 > 0.

The upwind scheme and the exact scheme based on dimensional splitting
are implmemented here
"""

def model_1_upwind(nx, ny, Lx, Ly, g_vec, t_vec, CFL, f0fun):
    #Definitions of inputs
    # nx: Number of points in the x-axis
    # ny: Number of points in the y-axis
    # Lx: Length of domain on x-axis
    # Ly: Length of domain on y-axis
    # g_vec: Input as a list, g_vec[i] = g_i
    # t_vec: Input as a list, t_vec[0] = t0, t_vec[1] = t_end
    # dt: timestep
    # f0fun: function defining the initial condition

    #Define mesh and timevec
    dx = Lx/(nx-1)
    dy = Ly/(ny-1)
    dt = CFL/((g_vec[0]/dx) + (g_vec[1]/dy))
    n_steps = round((t_vec[1] - t_vec[0])/dt)
    t_vals = np.linspace(t_vec[0],t_vec[1], n_steps+1)

    x_vals = np.linspace(start=0,stop=Lx,num=nx)
    y_vals = np.linspace(start=0,stop=Ly,num=ny)
    X,Y = np.meshgrid(x_vals,y_vals)

    #Initialize solution array and store initial conditions
    f_array = np.zeros((nx,ny, int(n_steps+1)))
    f_array[:,:,0] = f0fun(X,Y)

    #Perform timestepping
    alpha = (g_vec[0]*dt)/dx
    beta = (g_vec[1]*dt)/dy

    counter = 0
    t = t_vec[0]

    for i in range(n_steps):
        #Initalize dummy array to be clobbered
        f_vals = np.zeros((nx,ny))
        f_old = f_array[:,:,i]
        #Iterate through each value
        for idx, f in np.ndenumerate(f_vals):
            #Impose BCs
            #Top BC (a2 = Ly): df/da2 = 0
            if idx[0] == 0:
                f_vals[idx] = f_old[idx] -alpha*(f_old[idx[0],idx[1]] - f_old[idx[0],idx[1]-1])
            #Bottom BC (a2 = 0): f(ghost node) = 0
            elif idx[0] == ny -1:
                f_vals[idx] = f_old[idx] -alpha*(f_old[idx[0],idx[1]]) -beta*(f_old[idx[0],idx[1]] - f_old[idx[0],idx[1]-1])
            #Right BC (a1 = 0): f(ghost node) = 0
            elif idx[1] == 0:
                f_vals[idx] = f_old[idx] -alpha*(f_old[idx[0],idx[1]] - f_old[idx[0]-1,idx[1]]) -beta*(f_old[idx[0],idx[1]])
            #Left BC (a1=Lx)L df/da1 = 0
            elif idx[1] == nx-1:
                f_vals[idx] = f_old[idx] -beta*(f_old[idx[0],idx[1]] - f_old[idx[0]-1,idx[1]])
            else:
                f_vals[idx] =  f_old[idx] -alpha*(f_old[idx[0],idx[1]] - f_old[idx[0]-1,idx[1]]) -beta*(f_old[idx[0],idx[1]] - f_old[idx[0],idx[1]-1])


        #Append array and update
        f_array[:,:,i+1] = f_vals
        t = t + dt
        print("Current Simulation Time is %s"%t)
    









    return f_array, X,Y