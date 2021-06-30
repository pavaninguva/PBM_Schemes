% Script for employing naive finite differencing to solve PBMs
%
% This function is able to execute and run a variety of finite difference
% schemes for solving problems of the form: 
% df/dt + df/dx = k(x)f in 1-D subject to the following boundary conditions.

% On the left end of the domain (x = 0 typically), a boundary condition suited for such
% models is enforced. f(x=0,t) = \int_{0}^{\infty} b(x)f(x,t) dx
% On the right end of the domain, df/dx = 0 is enforced.

%The left boundary condition would thus require a numerical quadrature
%scheme. 

function [f, varargout] = model_4_naive(N_cells, f0fun, kfun, kprimefun, bfun, CFL, t_vec, x_vec, scheme, varargin)
%% Description

%The inputs and outputs are defined as follows: 
%INPUTS
%N_cells: Number of cells in the mesh
%f0: Initial profile in the form of a function handle
%u: this is the coefficient of the spatial term in the PDE. 
%kfun: Provide the nonhomogeneous function term as a function handle.
%kprimefun: Provide dk/dx as a function handle. For the Lax Wendroff scheme
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
dt = CFL*(dx);

%initialize the functions and counter
f0 = f0fun(mesh);
k = kfun(mesh);
if length(kprimefun(mesh)) == 1
    kprime = kprimefun(mesh)*ones(1,length(mesh));
else
    kprime = kprimefun(mesh);
end
b = bfun(mesh);
f_old = f0;%f_old corresponds to the solution from the previous timestep
f_old_old = f0;%f_old_old corresponds to the solution from 2nd prior step
t = t_vec(1);
counter = 1;

while t < t_vec(2) - dt*1e-3
   if scheme == "Upwind"
       %update first node using Trapezium rule
       f_new(1) = trapz(mesh,f_old.*b);
       %Update the rest
       for i = 2:length(mesh)
          f_new(i) = f_old(i) - CFL*(f_old(i) - f_old(i-1)) -dt*k(i)*f_old(i); 
       end
        
   elseif scheme == "Lax Wendroff"
       %update first node
       f_new(1) = trapz(mesh,f_old.*b);
       %update the rest
       for i = 2:length(mesh) -1
           f_new(i) = f_old(i) ...
                      - 0.5*(dt/dx)*(f_old(i+1) - f_old(i-1)) ... 
                      - dt*k(i)*f_old(i) ...
                      + 0.5*(CFL^2)*(f_old(i+1) -2*f_old(i) + f_old(i-1)) ...
                      + 0.5*dt*CFL*k(i)*(f_old(i+1) - f_old(i-1)) ...
                      + 0.5*(dt^2)*((k(i)^2) + kprime(i))*f_old(i);
       end
       %update last node
       f_new(length(mesh)) = f_old(end) - dt*k(end)*f_old(end) ...
                             +(CFL^2)*(f_old(end-1) - f_old(end)) ...
                             + 0.5*(dt^2)*((k(end)^2) + kprime(end))*f_old(end);
                         
   elseif scheme == "Leapfrog"
       %use upwind to compute the first time step
       if counter == 1
           %update first node using Trapezium rule
           f_new(1) = trapz(mesh,f_old.*b);
           %Update the rest
           for i = 2:length(mesh)
              f_new(i) = f_old(i) - CFL*(f_old(i) - f_old(i-1)) -dt*k(i)*f_old(i); 
           end
       else
          %update first node
          f_new(1) = trapz(mesh,f_old.*b);
          %update the rest
          for i = 2:length(mesh)-1
             f_new(i) =  f_old_old(i) - CFL*(f_old(i+1) - f_old(i-1)) ...
                         - 2*dt*k(i)*f_old(i);
          end
          %update last node
          f_new(length(mesh)) = f_old_old(end) - 2*dt*k(i)*f_old(end);
           
       end
       
       
   end
   
   %update counters etc
   counter = counter +1;
   t = t + dt;
   f_old_old = f_old;
   f_old = f_new;
   
   %outputs
   if varargin{2} == "all"
       f(counter,:) = f_new;
   elseif varargin{2} == "stride"
       if mod((counter -1), varargin{3}) == 0
           f((((counter-1)/varargin{3}) +1),:) = f_new;
           stride_vec((((counter-1)/varargin{3}) +1)) = t;
       end
   end    
end

%output 
if varargin{2} == "all" | varargin{2} == "stride"
    f(1,:) = f0;
else
    f = f_new;
end

stride_vec(1) = t_vec(1);
varargout{1} = stride_vec;
end