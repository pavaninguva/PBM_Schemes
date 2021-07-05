% Script for employing finite differencing to solve PBMs
%
% This function is able to execute and run a variety of finite difference
% schemes for solving problems of the form: 
% df/dt + d(u(x)f)/dx = 0 in 1-D subject to the following boundary conditions.

% On the left end of the domain, a modified Dirichlet B.C. is enforced
% which sets the value of the Ghost node at 0 i.e. f(0) = 0

%This scheme employs two variable transforms to solve the PBM exactly

function [f, mesh, varargout] = model_2_exact(N_cells, f0fun, ufun, zfun, xfun, x_vec, t_vec, varargin)

%% Description

%INPUTS
%N_cells: specify the number of cells for the mesh
%f0fun: initial profile as a function handle
%ufun: input u(x) as a function handle
%zfun: z(x) is defined as int_{0}^{x} (1/(u(k)) dk. Input as a function
%handle
%xfun: compute the inverse of zfun i.e. x(z) offline and input as a function handle
%x_vec: array containing the limits of the domain in x
%t_Vec: array containing the start and stop time

%varargin: This is to mainly deal with the different forms of outputs. Type
%in "output_style" followed by one of the following options: 
%"all": Outputs f containing the solution at each timestep
%"final": only outputs the final timestep
%"stride": outputs the solution at every nth time. Specify n as the next
%argument

%OUTPUTS
%f is the array with the solution. Depending on the output style, there
%would be different formats
%mesh is the computed mesh. 
%varargout would only contain the list of times outputted in the event
%stride is used.

%% Code

%Construct mesh:
z_mesh = linspace(zfun(x_vec(1)), zfun(x_vec(2)), N_cells);
x_mesh = xfun(z_mesh);

%Construct f0 
f0_untransformed = f0fun(x_mesh);
u_vals = ufun(x_mesh);
%Transformed f0 = f0(x).*u(x)
f0_trans = f0_untransformed.*u_vals;

%compute dz and dt
dz = z_mesh(2) - z_mesh(1);
dt = dz;

%Initialize counter etc
f_trans_old = f0_trans;
t = t_vec(1);
counter = 1;

%perform timestepping using while loop

while t < t_vec(2)
    %Update first node: 
    f_trans_new(1) = 0;
    %update the rest
    for i = 2:N_cells
       f_trans_new(i) = f_trans_old(i-1); 
    end
    
    %update counters etc
    counter = counter +1
    t = t + dt;
    f_trans_old = f_trans_new;
    
    %outputs
    if varargin{2} == "all"
        f(counter,:) = f_trans_new./u_vals;
    elseif varargin{2} == "stride"
        if mod((counter -1), varargin{3}) == 0
           f((((counter-1)/varargin{3}) +1),:) = f_trans_new./u_vals;
           stride_vec((((counter-1)/varargin{3}) +1)) = t;
        end
    end
end

%output 
if varargin{2} == "all" | varargin{2} == "stride"
    f(1,:) = f0_untransformed;
else
    f = f_trans_new./u_vals;
end
stride_vec(1) = t_vec(1);
varargout{1} = stride_vec;
mesh = x_mesh;

end