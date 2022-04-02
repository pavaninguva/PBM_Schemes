import numpy as np

#For exact scheme
from scipy.integrate import quad
from functools import partial
from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator
from scipy.spatial import Delaunay
import awkward as ak
import time 

"""
Functions to solve homogeneous PBMs with a variable growth rate
that varies as a function of the intrinsic variables i.e.,
G = [G1(a1,a2),G2(a1,a2)]

In conservative form, the PBM is given by
df/dt + d(G1(a1,a2)f)/da1 + d(G2(a1,a2)f)/da2 = 0

The following schemes are implemented:

1. Upwind scheme applied to conservative form on a uniform mesh
"""


def model3_conservative_upwind(n_vec, L_vec, g1fun, g2fun, t_vec,f0fun):

    #Extract mesh parameters and construct mesh
    nx, ny = n_vec
    Lx, Ly = L_vec

    dx = Lx/(nx-1)
    dy = Ly/(ny-1)

    x_vals = np.linspace(start=0,stop=Lx,num=nx)
    y_vals = np.linspace(start=0,stop=Ly,num=ny)
    X,Y = np.meshgrid(x_vals,y_vals)

    #Compute g_funs
    g1_vals = g1fun(X,Y)
    g2_vals = g2fun(X,Y)

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


def model3_split_conservative(n_vec,L_vec,  g1fun, g2fun, t_vec, f0fun):

    #Extract mesh parameters and construct mesh
    nx, ny = n_vec
    Lx, Ly = L_vec

    dx = Lx/(nx-1)
    dy = Ly/(ny-1)

    x_vals = np.linspace(start=0,stop=Lx,num=nx)
    y_vals = np.linspace(start=0,stop=Ly,num=ny)
    X,Y = np.meshgrid(x_vals,y_vals)

    #Compute g_funs
    g1_vals = g1fun(X,Y)
    g2_vals = g2fun(X,Y)

    #Compute dt using max values to ensure numerical stability
    dt = min(1.0/(np.amax(g1_vals)/dx), 1.0/(np.amax(g2_vals)/dy))
    n_steps = round((t_vec[1] - t_vec[0])/dt)

    #Initiaze solution array and ICs
    f_array = np.zeros((ny,nx, int(n_steps+1)))
    f_array[:,:,0] = f0fun(X,Y)

    #Solve 
    t = t_vec[0]

    for i in range(n_steps):
        #Create dummy arrays to be clobbered
        f_vals = np.zeros((ny,nx))
        f_vals_ = np.zeros((ny,nx))
        f_old = f_array[:,:,i]

        #Solve first subproblem: df*/dt + d(Gf*)/dx = 0
        for idx, f in np.ndenumerate(f_vals):
            #Right BC (a1 = 0): f(ghost node) = 0
            if idx[1] == 0:
                f_vals[idx] = f_old[idx] -(dt/dx)*(g1_vals[idx]*f_old[idx])
            else:
                f_vals[idx] =  f_old[idx] -(dt/dx)*(g1_vals[idx]*f_old[idx] - g1_vals[idx[0],idx[1]-1]*f_old[idx[0],idx[1]-1])

        #Solve second subproblem: df**/dt + d(Gf**)/dy
        for idx_, f_ in np.ndenumerate(f_vals_):
            if idx_[0] == 0:
                f_vals_[idx_] = f_vals[idx_] -(dt/dy)*(g2_vals[idx_]*f_vals[idx_])
            else:
                f_vals_[idx_] =  f_vals[idx_] -(dt/dy)*(g2_vals[idx_]*f_vals[idx_] - g2_vals[idx_[0]-1,idx_[1]]*f_vals[idx_[0]-1,idx_[1]])

        #Append arrays and update
        f_array[:,:,i+1] = f_vals_
        t = t+ dt
        print("Current Simulation Time is %s"%t)

    return f_array, X,Y


def model3_split_transform(n_vec,L_vec,  g1fun, g2fun, t_vec, f0fun):

    #Extract mesh parameters and construct mesh
    nx, ny = n_vec
    Lx, Ly = L_vec

    dx = Lx/(nx-1)
    dy = Ly/(ny-1)

    x_vals = np.linspace(start=0,stop=Lx,num=nx)
    y_vals = np.linspace(start=0,stop=Ly,num=ny)
    X,Y = np.meshgrid(x_vals,y_vals)

    #Compute g_funs
    g1_vals = g1fun(X,Y)
    g2_vals = g2fun(X,Y)

    #Compute dt using max values to ensure numerical stability
    dt = min(1.0/(np.amax(g1_vals)/dx), 1.0/(np.amax(g2_vals)/dy))
    n_steps = round((t_vec[1] - t_vec[0])/dt)

    #Initiaze solution array and ICs
    f_array = np.zeros((ny,nx, int(n_steps+1)))
    f_array[:,:,0] = f0fun(X,Y)

    #Solve 
    t = t_vec[0]

    for i in range(n_steps):
        #Create dummy arrays to be clobbered
        fhat_vals = np.zeros((ny,nx))
        fhat_vals_ = np.zeros((ny,nx))
        f_old = f_array[:,:,i]

        #Compute Transform for 1st subproblem
        fhat_old = f_old*g1_vals
        #Solve first subproblem: dfhat*/dt + G1*d(fhat*)/dx = 0
        for idx, f in np.ndenumerate(fhat_vals):
            #Right BC (a1 = 0): f(ghost node) = 0
            if idx[1] == 0:
                fhat_vals[idx] = fhat_old[idx] -(dt/dx)*(g1_vals[idx])*fhat_old[idx]
            else:
                fhat_vals[idx] =  fhat_old[idx] -(dt/dx)*(g1_vals[idx])*(fhat_old[idx] - fhat_old[idx[0],idx[1]-1])
        #Compute transform for 2nd subproblem
        f_star = (fhat_vals/g1_vals)*g2_vals
        #Solve second subproblem: df**/dt + d(Gf**)/dy
        for idx_, f_ in np.ndenumerate(fhat_vals_):
            if idx_[0] == 0:
                fhat_vals_[idx_] = f_star[idx_] -(dt/dy)*(g2_vals[idx_])*f_star[idx_]
            else:
                fhat_vals_[idx_] =  f_star[idx_] -(dt/dy)*(g2_vals[idx_])*(f_star[idx_] - f_star[idx_[0]-1,idx_[1]])

        #Append arrays and update
        f_array[:,:,i+1] = fhat_vals_/g2_vals
        t = t+ dt
        print("Current Simulation Time is %s"%t)

    return f_array, X,Y


def mesh_constructor(L_vec,dt,g1fun,g2fun):
    #Unpack values
    Lx,Ly = L_vec

    xend_vals = [Lx]
    c_xend = 0
    while xend_vals[-1] > 0.0:
        xend_new = xend_vals[c_xend] - dt*g1fun(xend_vals[c_xend],Ly)
        xend_vals.append(xend_new)
        c_xend = c_xend + 1
    #Remove last negative value
    xend_vals = xend_vals[:-1]
    
    yend_vals = [Ly]
    c_yend = 0
    while yend_vals[-1] > 0.0:
        yend_new = yend_vals[c_yend] - dt*g2fun(2.0,yend_vals[c_yend])
        yend_vals.append(yend_new)
        c_yend = c_yend + 1
    #remove last negative value
    yend_vals = yend_vals[:-1]

    #Compute mesh for 1st subproblem
    # x_1_vals = [xend_vals[::-1]]
    # y_1_vals = [[Ly]*len(xend_vals[::-1])]
    x_1_vals = []
    y_1_vals = []
    for i in range(len(yend_vals)):
        y_val = yend_vals[i]
        #Create dummy var
        x = [Lx]
        c_x = 0
        while x[-1] > 0.0:
            x_new = x[c_x] - dt*g1fun(x[c_x],y_val)
            x.append(x_new)
            c_x = c_x +1 
        #Append to x_vals
        x = x[:-1]
        x_1_vals.append(x[::-1])
        #Create y vals list
        y = [y_val]*len(x)
        # print(y)
        y_1_vals.append(y)
    #Generate X_1 and Y_1  array
    X_1 = ak.Array(x_1_vals)
    Y_1 = ak.Array(y_1_vals)

    #Compute mesh for 2nd subproblem
    # y_2_vals = [yend_vals[::-1]]
    # x_2_vals = [[Lx]*len(yend_vals[::-1])]
    y_2_vals = []
    x_2_vals = []
    for i in range(len(xend_vals)):
        x_val = xend_vals[i]
        y_2 = [Ly]
        c_y = 0
        while y_2[-1] > 0.0:
            y_new = y_2[c_y] - dt*g2fun(x_val,y_2[c_y])
            y_2.append(y_new)
            c_y = c_y +1
        #Append to y_vals
        y_2 = y_2[:-1]
        y_2_vals.append(y_2[::-1])
        #Append  x_vals
        x_2 = [x_val]*len(y_2)
        x_2_vals.append(x_2)

    #Generate X_2 and Y_2 arrays
    X_2 = ak.Array(x_2_vals)
    Y_2 = ak.Array(y_2_vals)    

    return X_1,Y_1, X_2, Y_2

def model3_split_transform_cfl(L_vec, t_vec, dt, g1fun, g2fun, f0_fun):
    #Generate mesh
    X_1, Y_1, X_2, Y_2 = mesh_constructor(L_vec,dt,g1fun,g2fun)

    mesh1_shape = ak.num(X_1)
    mesh2_shape = ak.num(X_2)

    #Compute interpolation things outside loop
    XY_1 = list(zip(ak.flatten(X_1,axis=None),ak.flatten(Y_1,axis=None)))
    XY_2 = list(zip(ak.flatten(X_2,axis=None),ak.flatten(Y_2,axis=None)))

    tri1 = Delaunay(XY_1)
    tri2 = Delaunay(XY_2)

    #Generate ICs and Transformation
    f0 = f0_fun(X_1,Y_1)
    fhat0 = f0*g1fun(X_1,Y_1)

    #Timestepping
    n_steps = round((t_vec[1]-t_vec[0])/dt)
    t = t_vec[0]

    for i in range(n_steps):
        if t == t_vec[0]:
            #Initialize arrays
            fhat1_old = fhat0
            fhat1_new_list = []
            #Solve first subproblem dfhat1/dt + G1*dfhat1/a1 = 0
            #iterate through the rows: 
            for row in range(ak.num(fhat1_old,axis=0)):
                row_vals = []
                for col in range(len(fhat1_old[row])):
                    #Enforce BC
                    if col == 0:
                        row_vals.append(0.0)
                    else:
                        if np.isnan(fhat1_old[row][col-1]) == False:
                            row_vals.append(fhat1_old[row][col-1])
                        else:
                            row_vals.append(0.0)
                fhat1_new_list.append(row_vals)
            fhat1_new = ak.Array(fhat1_new_list)
        else:
            #Interpolate f_new back to mesh1
            interp_2 = LinearNDInterpolator(tri2,ak.flatten(f_new,axis=None))
            f1_old_ = interp_2(XY_1)
            #Reshape
            f1_old = ak.unflatten(f1_old_,mesh1_shape)
            fhat1_old = f1_old*g1fun(X_1,Y_1)
            fhat1_new_list = []
            for row_1 in range(ak.num(fhat1_old,axis=0)):
                row_vals = []
                for col_1 in range(len(fhat1_old[row_1])):
                    #Enforce BC
                    if col_1 == 0:
                        row_vals.append(0.0)
                    else:
                        if np.isnan(fhat1_old[row_1][col_1-1]) == False:
                            row_vals.append(fhat1_old[row_1][col_1-1])
                        else: 
                            row_vals.append(0.0)
                fhat1_new_list.append(row_vals)
            fhat1_new = ak.Array(fhat1_new_list)

        #Recompute f from fhat1_new
        f1_new = fhat1_new/g1fun(X_1,Y_1)
        #Interpolate to mesh2
        interp_1 = LinearNDInterpolator(tri1,ak.flatten(f1_new,axis=None))
        f2_old_ = interp_1(XY_2)
        #Reshape to mesh2 shape
        f2_old = ak.unflatten(f2_old_,mesh2_shape)
        #Transform
        fhat2_old = f2_old*g2fun(X_2,Y_2)
        fhat2_new_list = []
        #Solve second subproblem
        for row_2 in range(ak.num(fhat2_old,axis=0)):
            row2_vals = []
            for col_2 in range(len(fhat2_old[row_2])):
                #Enforce BC
                if col_2 == 0:
                    row2_vals.append(0.0)
                else:
                    if np.isnan(fhat2_old[row_2][col_2-1]) == False:
                        row2_vals.append(fhat2_old[row_2][col_2-1])
                    else: 
                        row2_vals.append(0.0)
            fhat2_new_list.append(row2_vals)
        fhat2_new = ak.Array(fhat2_new_list)

        #Recompute f from fhat2_new
        f_new = fhat2_new/g2fun(X_2,Y_2)

        #Update time
        t = t + dt
        print("Current Simulation Time is %s"%t)

    return f_new, X_2, Y_2


def model3_scratch(n_vec,L_vec,g1fun,g2fun,t_vec,dt,f0fun):
    #Generate mesh
    nx, ny = n_vec
    Lx, Ly = L_vec

    x_vals = np.linspace(start=0,stop=Lx,num=nx)
    y_vals = np.linspace(start=0,stop=Ly,num=ny)
    X,Y = np.meshgrid(x_vals,y_vals)

    #Compute Delauny for interpolation
    points = np.dstack((X,Y)).reshape(-1,2)
    tri = Delaunay(points)

    #Compute initial conditions
    f_old = f0fun(X,Y)

    #Compute g1 and g2
    g1_vals = g1fun(X,Y)
    g2_vals = g2fun(X,Y)

    #Set up time stepping
    n_steps = round((t_vec[1] - t_vec[0])/dt)
    print(n_steps)
    t = t_vec[0]
    start = time.time()
    for i in range(n_steps):
        #Solve first sub-problem
        #transform to fhat_1
        fhat_old_1 = f_old*g1_vals
        #Initalize new fhat_1
        fhat_new_1 = np.zeros((ny,nx))
        #Create interpolatable
        interp1 = LinearNDInterpolator(tri, fhat_old_1.copy().flatten())
        #Compute solution for each point
        for idx, f in np.ndenumerate(fhat_new_1):
            x_shift = np.exp(-dt/2)*(0.5+X[idx]+Y[idx]) - 0.5 - Y[idx]
            if x_shift < 0.0:
                fhat_new_1[idx] = 0.0
            else:
                coord_1 = np.array([x_shift, Y[idx]])
                val = interp1(coord_1)
                if np.isnan(val) == True:
                    fhat_new_1[idx] = 0.0
                else:
                    fhat_new_1[idx] = val

        #Recompute f_new_1
        f_new_1 = fhat_new_1/g1_vals

        #Solve 2nd sub-problem
        fhat_old_2 = f_new_1*g2_vals
        #Initialize new fhat_new_2
        fhat_new_2 = np.zeros((ny,nx))
        #Create interpolatable
        interp2 = LinearNDInterpolator(tri, fhat_old_2.copy().flatten())
        #Solve at each point
        for idx_, f_ in np.ndenumerate(fhat_new_2):
            y_shift = np.exp(-dt/4)*(2+X[idx_]+Y[idx_]) - 2 - X[idx_]
            if y_shift < 0.0:
                fhat_new_2[idx_] = 0.0
            else:
                coord_2 = np.array([X[idx_], y_shift])
                val2 = interp2(coord_2)
                if np.isnan(val2) == True:
                    fhat_new_2[idx_] = 0.0
                else:
                    fhat_new_2[idx_] = val2

        #Recompute and clobber f_old
        f_old = fhat_new_2/g2_vals
        #Update time
        t = t + dt
        print("Current Simulation Time is %s"%t)
    end = time.time()
    print("Time Taken for Simulation is %s"%(end-start))

    return f_old, X, Y


def model3_split_exact(n_vec,L_vec,g1fun,g2fun,t_vec,dt,f0fun):
    #Generate mesh
    nx, ny = n_vec
    Lx, Ly = L_vec

    def recip(x,y,fun):
        f = 1/(fun(x,y))
        return f

    #Define transformation functions
    def x_transfun(x,y):
        result = np.array(list(map(partial(quad,(lambda x:recip(x,y,g1fun)),0.0),x)))[:,0]
        return result

    def y_transfun(x,y):
        result = np.array(list(map(partial(quad,(lambda y:recip(x,y,g2fun)),0.0),y)))[:,0]
        return result

    #Construct mesh for 1st subproblem
    y_vals_1 = np.linspace(0.0,Ly,ny)
    Y_1 = np.zeros((ny,nx))
    for i in range(len(y_vals_1)):
        Y_1[i,:] = y_vals_1[i]

    #Compute ends of a1 using x_transfun and get XTilde_1
    XTilde_1 = np.zeros((ny,nx))
    for i in range(len(y_vals_1)):
        xtilde_end_1 = x_transfun([Lx],y_vals_1[i])
        XTilde_1[i,:]  = np.linspace(0.0,xtilde_end_1[0],nx)

    #Use Interpolation to obtain X_1
    X_1 = np.zeros((ny,nx))
    for i in range(len(y_vals_1)):
        XTilde_Row_1 = XTilde_1[i,:]
        y_val_1 = y_vals_1[i]
        #Generate interpolatable
        x_interp_1 = np.linspace(0.0,Lx,5001)
        xTilde_1_interp = x_transfun(x_interp_1,y_val_1)
        #Interpolate
        X_1[i,:] = np.interp(XTilde_Row_1,xTilde_1_interp,x_interp_1)

    #Construct mesh for 2nd subproblem
    x_vals_2 = np.linspace(0.0,Lx,nx)
    X_2 = np.zeros((ny,nx))
    for i in range(len(x_vals_2)):
        X_2[:,i] = x_vals_2[i]

    #Compute ends of a2 using y_transfun and get YTilde_2
    YTilde_2 = np.zeros((ny,nx))
    for i in range(len(x_vals_2)):
        ytilde_end_2 = y_transfun(x_vals_2[i],[Ly])
        YTilde_2[:,i]  = np.linspace(0.0,ytilde_end_2[0],ny)

    #Use Interpolation to obtain Y_2
    Y_2 = np.zeros((ny,nx))
    for i in range(len(x_vals_2)):
        YTilde_Col_2 = YTilde_2[:,i]
        x_val_2 = x_vals_2[i]
        #Generate interpolatable
        y_interp_2 = np.linspace(0.0,Ly,5001)
        yTilde_2_interp = y_transfun(x_val_2, y_interp_2)
        #Interpolate
        Y_2[:,i] = np.interp(YTilde_Col_2,yTilde_2_interp,y_interp_2)

    #Compute Delauny for interpolation
    points_1 = np.dstack((XTilde_1,Y_1)).reshape(-1,2)
    tri_1 = Delaunay(points_1)

    points_1_ = np.dstack((X_1,Y_1)).reshape(-1,2)
    tri_1_ = Delaunay(points_1_)

    points_2 = np.dstack((X_2,YTilde_2)).reshape(-1,2)
    tri_2 = Delaunay(points_2)

    points_2_ = np.dstack((X_2,Y_2)).reshape(-1,2)
    tri_2_ = Delaunay(points_2_)

    print("Mesh Generation Completed")

    #Compute initial conditions
    f_old = f0fun(X_1,Y_1)

    #Set up time stepping
    n_steps = round((t_vec[1] - t_vec[0])/dt)
    print(n_steps)
    t = t_vec[0]

    start = time.time()
    for i in range(n_steps):
        #Solve first sub-problem
        #transform to fhat_1
        fhat_old_1 = f_old*g1fun(X_1,Y_1)
        #Initalize new fhat_1
        fhat_new_1 = np.zeros((ny,nx))
        #Create interpolatable
        interp1 = LinearNDInterpolator(tri_1, fhat_old_1.copy().flatten())
        #Compute solution for each point
        for idx, f in np.ndenumerate(fhat_new_1):
            x_shift = XTilde_1[idx] - dt
            if x_shift < 0.0:
                fhat_new_1[idx] = 0.0
            else:
                coord_1 = np.array([x_shift, Y_1[idx]])
                val = interp1(coord_1)
                if np.isnan(val) == True:
                    fhat_new_1[idx] = 0.0
                else:
                    fhat_new_1[idx] = val

        #Recompute f_new_1 on mesh1
        f_new_1_mesh1 = fhat_new_1/g1fun(X_1,Y_1)
        #Interpolate to mesh2
        f_interp_1 = LinearNDInterpolator(tri_1_,f_new_1_mesh1.copy().flatten())
        #Compute f_new_1 on mesh2
        f_new_1_mesh2 = f_interp_1(X_2,Y_2)

        #Solve 2nd sub-problem
        fhat_old_2 = f_new_1_mesh2*g2fun(X_2,Y_2)
        #Initialize new fhat_new_2
        fhat_new_2 = np.zeros((ny,nx))
        #Create interpolatable
        interp2 = LinearNDInterpolator(tri_2, fhat_old_2.copy().flatten())
        #Solve at each point
        for idx_, f_ in np.ndenumerate(fhat_new_2):
            y_shift = YTilde_2[idx_] - dt
            if y_shift < 0.0:
                fhat_new_2[idx_] = 0.0
            else:
                coord_2 = np.array([X_2[idx_], y_shift])
                val2 = interp2(coord_2)
                if np.isnan(val2) == True:
                    fhat_new_2[idx_] = 0.0
                else:
                    fhat_new_2[idx_] = val2

        #Recompute and clobber f_old
        f_old_mesh2 = fhat_new_2/g2fun(X_2,Y_2)

        f_interp_2 = LinearNDInterpolator(tri_2_,f_old_mesh2.copy().flatten())
        f_old = f_interp_2(X_1,Y_1)
        #Update time
        t = t + dt
        print("Current Simulation Time is %s"%t)
    
    end = time.time()
    print("Time Taken for Simulation is %s"%(end-start))

    return f_old, X_1, Y_1
        
            






    
        


        











    


    











    


    

    


    

    




