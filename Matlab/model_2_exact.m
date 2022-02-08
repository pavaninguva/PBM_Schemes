% Script for employing finite differencing to solve PBMs
%
% This function is able to execute and run a variety of finite difference
% schemes for solving problems of the form: 
% df/dt + d(u(x)f)/dx = 0 in 1-D subject to the following boundary conditions.

% On the left end of the domain, a modified Dirichlet B.C. is enforced
% which sets the value of the Ghost node at 0 i.e. f(0) = 0

%This scheme employs two variable transforms to solve the PBM exactly and
%uses a function evaluation for the transformed PBM. 

function [f, mesh] = model_2_exact(N_cells, f0_transfun, ufun, zfun, xfun, x_vec, t_end)

%% Description

%INPUTS
%N_cells: specify the number of cells for the mesh
%f0_transfun: Input f0(x).*u(x) as a function handle 
%ufun: Input u(x) as a function handle
%zfun: z(x) is defined as int_{0}^{x} (1/(u(k)) dk. Input as a function
%handle
%xfun: compute the inverse of zfun i.e. x(z) offline and input as a function handle
%x_vec: array containing the limits of the domain in x
%t_end: Specify the ending time of the simulation.

%OUTPUTS
%f is the array with the solution. Depending on the output style, there
%would be different formats
%mesh is the computed mesh. 

%% Code

%Construct mesh:
z_mesh = linspace(zfun(x_vec(1)), zfun(x_vec(2)), N_cells);
x_mesh = xfun(z_mesh);

%compute functions
u_vals = ufun(x_mesh);

%construct exact solution
for i = 1:length(x_mesh)
    if z_mesh(i) < z_mesh(1) + t_end
        f_trans(i) = 0;
    else
        f_trans(i) = f0_transfun(xfun(z_mesh(i) -t_end));
    end
end


%Outputs
mesh = x_mesh;
f = f_trans./u_vals;

        
end




