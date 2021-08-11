% Script for employing exact finite differencing to solve PBMs
%
% This function is able to execute and run a variety of finite difference
% schemes for solving problems of the form: 
% df/dt + df/dx = k(x)f in 1-D subject to the following boundary conditions.

% On the left end of the domain (x = 0 typically), a boundary condition suited for such
% models is enforced. f(x=0,t) = \int_{0}^{\infty} b(x)f(x,t) dx
% On the right end of the domain, df/dx = 0 is enforced.

%The left boundary condition would thus require a numerical quadrature
%scheme. 

function [f,varargout] = model_4_exact(N_cells, f0fun, intkfun, bfun, t_vec, x_vec, varargin)
%% Description

%The inputs and outputs are defined as follows: 
%INPUTS
%N_cells: Number of cells in the mesh
%f0: Initial profile in the form of a function handle 
%intkfun: Provide the nonhomogeneous function term after performing integration 
%as a function handle.
%bfun: Provide the expression for b(x) for the boundary condition as a
%function handle
%CFL: specify the CFL number, typically between 0-1. 
%scheme: Three schemes are implemented: Upwind, Lax Wendroff and Leapfrog. 
%t_vec: Array containing the start and stop time e.g. [0,1] seconds
%x_vec: Array containing the coordinates of the start and end of the
%domain.

%varargin: This is to mainly deal with the different forms of outputs. Type
%in "output_style" followed by one of the following options: 
%"all": Outputs f containing the solution at each timestep
%"final": only outputs the final timestep
%"stride": outputs the solution at every nth time. Specify n as the next
%argument

%OUTPUTS
%f is the array with the solution. Depending on the output style, there
%would be different formats

%varargout would only contain the list of times outputted in the event
%stride is used. 

%% Code
%compute dx and dt using mesh and CFL:
mesh = linspace(x_vec(1),x_vec(2),N_cells);
dx = (x_vec(2) - x_vec(1))/(N_cells-1);
dt = dx;

%initialize the functions and counter
f0 = f0fun(mesh);
intk = intkfun(mesh);
f0_trans = f0.*exp(intk);

b = bfun(mesh);
f_trans_old = f0_trans;
t = t_vec(1);
counter = 1;

while t < t_vec(2) - dt*1e-3
   %update first node
   f_trans_new(1) = trapz(mesh, (f_trans_old.*b)./(exp(intk)));
   %upadate est
   for i = 2:length(mesh)
       f_trans_new(i) = f_trans_old(i-1);
   end
   
   %update counters etc
   counter = counter +1;
   t = t + dt;
   f_trans_old = f_trans_new;
   
   %outputs
    if varargin{2} == "all"
        f(counter,:) = f_trans_new./exp(intk);
    elseif varargin{2} == "stride"
        if mod((counter -1), varargin{3}) == 0
           f((((counter-1)/varargin{3}) +1),:) = f_trans_new./exp(intk);
           stride_vec((((counter-1)/varargin{3}) +1)) = t;
        end
    end
end

%output 
if varargin{2} == "all" | varargin{2} == "stride"
    f(1,:) = f0;
else
    f = f_trans_new./exp(intk);
end
stride_vec(1) = t_vec(1);
varargout{1} = stride_vec;



end