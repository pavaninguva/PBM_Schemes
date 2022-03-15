import numpy as np

#For exact scheme
from scipy.integrate import quad
from functools import partial
from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator
from scipy.spatial import Delaunay
import awkward as ak

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


def model3_split_transform(n_vec,L_vec,  g1fun, g2fun, t_vec, dt, f0fun):

    #Extract mesh parameters
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

    print("Mesh Generation is Complete")

    #Solve
    
    #Compute ICs on X_1, Y_1
    f0 = f0fun(X_1,Y_1)

    #Compute triangulation to speed up interpolation
    # points_1 = np.dstack((X_1,Y_1)).reshape(-1,2)
    # points_2 = np.dstack((X_2,Y_2)).reshape(-1,2)

    points_1 = np.dstack((XTilde_1,Y_1)).reshape(-1,2)
    points_2 = np.dstack((X_2,YTilde_2)).reshape(-1,2)

    tri_1 = Delaunay(points_1)
    tri_2 = Delaunay(points_2)

    print("Delaunay Computation is Complete")


    t = t_vec[0]
    while t < t_vec[1] -1e-8:
        if t == t_vec[0]:
            fhat_1_old = f0*g1fun(X_1,Y_1)
            fhat_1_new = np.zeros((ny,nx))
            fhat_1_interp = LinearNDInterpolator(tri_1,fhat_1_old.copy().flatten())
            # fhat_1_interp = CloughTocher2DInterpolator(tri_1,fhat_1_old.copy().flatten())
            for idx, f in np.ndenumerate(fhat_1_new):
                if XTilde_1[idx] < t + dt:
                    fhat_1_new[idx] = 0.0
                else:
                    # x_shift = np.interp(XTilde_1[idx] -dt ,xTilde_1_interp,x_interp_1)
                    fhat_1_new[idx] = fhat_1_interp(XTilde_1[idx] -dt,Y_1[idx])
        else:
            #Interpolate f_old back to mesh_1
            f_1_old_interp = CloughTocher2DInterpolator(tri_2,f_old.copy().flatten())
            f_old_mesh1 = f_1_old_interp(XTilde_1,Y_1)
            fhat_1_old = f_old_mesh1*g1fun(X_1,Y_1)
            fhat_1_interp = LinearNDInterpolator(tri_1,fhat_1_old.copy().flatten())
            # fhat_1_interp = CloughTocher2DInterpolator(tri_1,fhat_1_old.copy().flatten())
            for idx, f in np.ndenumerate(fhat_1_new):
                if XTilde_1[idx] < t + dt:
                    fhat_1_new[idx] = 0.0
                else:
                    # x_shift = np.interp(XTilde_1[idx] -(t+dt),xTilde_1_interp,x_interp_1)
                    fhat_1_new[idx] = fhat_1_interp(XTilde_1[idx]-dt,Y_1[idx]) 

        #Retransform to obtain f on mesh_1
        f_2_old_mesh1 = fhat_1_new/g1fun(X_1,Y_1)

        #Interpolate to mesh_2
        f_2_old_interp = CloughTocher2DInterpolator(tri_1,f_2_old_mesh1.copy().flatten())
        f_2_old_mesh2 = f_2_old_interp(X_2,YTilde_2)


        #Solve second subproblem
        fhat_2_old = f_2_old_mesh2*g2fun(X_2,Y_2)
        fhat_2_new = np.zeros((ny,nx))
        fhat_2_interp = LinearNDInterpolator(tri_2,fhat_2_old.copy().flatten())
        # fhat_2_interp = CloughTocher2DInterpolator(tri_2,fhat_2_old.copy().flatten())
        for idx_,f in np.ndenumerate(fhat_2_old):
            if YTilde_2[idx_] < t + dt:
                fhat_2_new[idx_] = 0.0
            else:
                # y_shift = np.interp(YTilde_2[idx_]-dt,yTilde_2_interp,y_interp_2)
                fhat_2_new[idx_] = fhat_2_interp(X_2[idx_],YTilde_2[idx]-dt)

        f_old = fhat_2_new/g2fun(X_2,Y_2)
        t = t + dt
        print("Current Simulation Time is %s"%t)
        


    return X_2, Y_2, f_old, f_2_old_mesh1


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
    x_1_vals = [xend_vals[::-1]]
    y_1_vals = [[Ly]*len(xend_vals[::-1])]
    for i in range(len(yend_vals)):
        y_val = yend_vals[i]
        #Create dummy var
        x = [Lx]
        c_x = 0
        while x[-1] > 0.0:
            x_new = x[c_x-1] - dt*g1fun(x[c_x-1],y_val)
            x.append(x_new)
            c_x = c_x +1 
        #Append to x_vals
        x = x[:-1]
        x_1_vals.append(x[::-1])
        #Create y vals list
        y = [y_val]*len(x)
        y_1_vals.append(y)
    #Generate X_1 and Y_1  array
    X_1 = ak.Array(x_1_vals)
    Y_1 = ak.Array(y_1_vals)

    #Compute mesh for 2nd subproblem
    y_2_vals = [yend_vals[::-1]]
    x_2_vals = [[Lx]*len(yend_vals[::-1])]
    for i in range(len(xend_vals)):
        x_val = xend_vals[i]
        y_2 = [Ly]
        c_y = 0
        while y_2[-1] > 0.0:
            y_new = y_2[c_y-1] - dt*g2fun(x_val,y_2[c_y-1])
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

    #Generate ICs and Transformation
    f0 = f0_fun(X_1,X_2)
    fhat0 = f0*g1fun(X_1,X_2)

    #Timestepping
    n_steps = round((t_vec[1]-t_vec[0])/dt)
    t = t_vec[0]

    for i in range(n_steps):
        




    
        


        











    


    











    


    

    


    

    




