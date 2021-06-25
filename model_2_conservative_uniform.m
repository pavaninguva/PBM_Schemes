% Script for employing finite differencing to solve PBMs
%
% This function is able to execute and run a variety of finite difference
% schemes for solving problems of the form: 
% df/dt + d(u(x)f)/dx = 0 in 1-D subject to the following boundary conditions.

% On the left end of the domain, a modified Dirichlet B.C. is enforced
% which sets the value of the Ghost node at 0 i.e. f(0) = 0
% On the right end of the domain, df/dx = 0 is enforced. 



function [f,varargout] = model_2_conservative_uniform(mesh, f0, ufun, uprimefun, CFL, scheme, t_vec, varargin)

%% Description

%INPUTS
%mesh: input the mesh as an array
%f0: initial profile which should be the same size as mesh
%u: input u(x) as a function handle
%uprime: input u'(x) as a function handle. Used for the
%Lax-Wendroff scheme. 
%CFL: specify CFL, typically between 0-1
%scheme: Three schemes are implemented: Upwind, Lax-Wendroff and Leapfrog
%t_vec: Array containing the start and stop time. 

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
n_cells = length(mesh) -1;
dx = (mesh(end) - mesh(1))/n_cells;
u = ufun(mesh);
uprime = uprimefun(mesh);
dt = CFL*(dx)/(max(u));

%for lax wendroff and leapfrog
u_n1 = ufun(mesh(end)+dx);

%Initialize, f, t and counter
f_old = f0;
f_old_old = f0;
t = t_vec(1);
counter = 1;

%start simulation
while t < t_vec(2) + dt*1e-3
   if scheme == "Upwind"
       %update first node
       f_new(1) = f_old(1) - (dt/dx)*(u(1)*f_old(1));
       %use for loop to update the rest
       for i = 2:length(mesh)
          f_new(i) = f_old(i) - (dt/dx)*(u(i)*f_old(i) - u(i-1)*f_old(i-1));
       end
       
   elseif scheme == "Lax Wendroff"
       %update first node
       f_new(1) = f_old(1) + (0.5*(dt^2)*uprime(1) - dt)*(u(2)*f_old(2))/(2*dx) ...
                  + 0.5*((dt/dx)^2)*u(1)*(u(2)*f_old(2) - 2*u(1)*f_old(1));
       %Update the rest 
       for i = 2:length(mesh) - 1
          f_new(i) = f_old(i) ...
                     + (0.5*(dt^2)*uprime(i)- dt)*(u(i+1)*f_old(i+1) - u(i-1)*f_old(i-1))/(2*dx) ...
                     +(0.5*((dt/dx)^2)*u(i))*(u(i+1)*f_old(i+1) - 2*u(i)*f_old(i) + u(i-1)*f_old(i-1));
       end
       %update last node:
       f_new(length(mesh)) = f_old(end) ...
                             + (0.5*(dt^2)*uprime(end)- dt)*(u_n1 - u(end-1))*f_old(end-1)/(2*dx) ...
                             +(0.5*((dt/dx)^2)*u(end))*(u_n1*f_old(end-1) - 2*u(end)*f_old(end) + u(end-1)*f_old(end-1));
   
   elseif scheme == "Leapfrog"
       %Use upwind to compute first timestep
       if counter == 1
           %update first node:
           f_new(1) = f_old(1) - (dt/dx)*(u(1)*f_old(1));
           %update nodes with for loop
           for i = 2:length(mesh)
                f_new(i) = f_old(i) - (dt/dx)*(u(i)*f_old(i) - u(i-1)*f_old(i-1));
           end
       else
           %update first node
           f_new(1) = f_old_old(1) - (dt/dx)*(u(2)*f_old(2));
           %update nodes with for loop
           for i = 2:length(mesh)-1
              f_new(i) = f_old_old(i) - (dt/dx)*(u(i+1)*f_old(i+1) - u(i-1)*f_old(i-1)); 
           end
           %update last node
           f_new(length(mesh)) = f_old_old(end) - (dt/dx)*(u_n1 - u(end-1)*f_old(end-1)); 
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