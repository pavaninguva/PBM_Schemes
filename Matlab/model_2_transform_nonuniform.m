% Script for employing finite differencing to solve PBMs
%
% This function is able to execute and run a variety of finite difference
% schemes for solving problems of the form: 
% df/dt + d(u(x)f)/dx = 0 in 1-D subject to the following boundary conditions.

% On the left end of the domain, a modified Dirichlet B.C. is enforced
% which sets the value of the Ghost node at 0 i.e. f(0) = 0
% On the right end of the domain, df/dx = 0 is enforced. 

% A variable transform of the form fhat = u(x)f is applied which transforms
% the equation to: dfhat/dt + u(x)dfhat/dx = 0. 

%Instead of inputting a mesh, the function precomputes the mesh to enforce
%CFL = 1.0 at each point. Therefore, input a desired value of dt.

function [f,mesh,varargout] = model_2_transform_nonuniform(dt, f0fun, ufun, uprimefun, scheme, t_vec, x_vec, varargin)

%% Description

%INPUTS
%dt: input the desired value of dt
%f0fun: initial profile as a function handle
%ufun: input u(x) as a function handle
%uprimefun: input u'(x) as a function handle. Used for the
%Lax-Wendroff scheme. 
%scheme: Three schemes are implemented: Upwind, Lax-Wendroff and Leapfrog
%t_vec: Array containing the start and stop time. 
%x_vec: Array containing the limits of the domain e.g. [0,1]

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

%% Compute mesh

%start from end of the domain and initialize x and counter
x(1) = x_vec(2);
mesh_counter = 1;

%while loop to compute the mesh
while x(end) > x_vec(1)
    %compute
    x(mesh_counter +1) = x(mesh_counter) - ufun(x(mesh_counter))*dt;
    %update counter
    mesh_counter = mesh_counter +1;
end

%flip array to get mesh
mesh = flip(x(1:end-1));

%compute ghost node values for Lax Wendroff & Leapfrog
x_0 = x(end);
x_n1 = mesh(end) + ufun(mesh(end))*dt;

%% Simulation

%precompute various things and intialize 
u = ufun(mesh);
uprime = uprimefun(mesh);
f0 = f0fun(mesh);
t = t_vec(1);
counter = 1;

%variable transformation:
fhat_old = f0.*u;
fhat_old_old = f0.*u;

%for Lax Wendroff
u_n1 = ufun(x_n1);

%start simulation

while t < t_vec(2) - dt*1e-3
   if scheme == "Upwind" 
       %update first node
       fhat_new(1) = fhat_old(1) - (u(1)*dt)*fhat_old(1)/(mesh(1) - x_0);
       %update rest
       for i = 2:length(mesh)
          fhat_new(i) = fhat_old(i) - (u(i)*dt)*(fhat_old(i) - fhat_old(i-1))/(mesh(i) - mesh(i-1)); 
       end
       
   elseif scheme == "Lax Wendroff"
       %update first node
       fhat_new(1) = fhat_old(1) + (0.5*(dt^2)*u(1)*uprime(1) - dt*u(1))*(fhat_old(2))/(mesh(2) - x_0) ...
                     + (0.5*(dt^2)*(u(1)^2))*(fhat_old(2) - 2*fhat_old(1))/((mesh(2) - mesh(1))*(mesh(1) - x_0));
       %update the rest
       for i = 2:length(mesh)-1
           fhat_new(i) = fhat_old(i) + (0.5*(dt^2)*u(i)*uprime(i) - dt*u(i))*(fhat_old(i+1) - fhat_old(i-1))/(mesh(i+1) - mesh(i-1)) ...
                     + (0.5*(dt^2)*(u(i)^2))*(fhat_old(i+1) - 2*fhat_old(i) + fhat_old(i-1))/((mesh(i+1) - mesh(i))*(mesh(i) - mesh(i-1)));
       end
       %update last node
       %compute the value of fhat^{N+1}_{i} using the ghost cell approach
       fhat_n1 = (uprime(end)/u(end))*(x_n1 - mesh(end-1))*fhat_old(end) + fhat_old(end-1);
       %perform the update
       fhat_new(length(mesh)) = fhat_old(end) + (0.5*(dt^2)*u(end)*uprime(end) - dt*u(end))*(fhat_n1 - fhat_old(end-1))/(x_n1 - mesh(end-1)) ...
                                + (0.5*(dt^2)*(u(end)^2))*(fhat_n1 - 2*fhat_old(end) + fhat_old(end-1))/((x_n1 - mesh(end))*(mesh(end) - mesh(end-1)));
                            
   elseif scheme == "Leapfrog"
       if counter == 1
          %update first node
           fhat_new(1) = fhat_old(1) - (u(1)*dt)*fhat_old(1)/(mesh(1) - x_0);
           %update rest
           for i = 2:length(mesh)
              fhat_new(i) = fhat_old(i) - (u(i)*dt)*(fhat_old(i) - fhat_old(i-1))/(mesh(i) - mesh(i-1)); 
           end 
       else
           %update first node
           fhat_new(1) = fhat_old_old(1) - (2*dt*u(1))*(fhat_old(2))/(mesh(2) - x_0);
           %update rest
           for i = 2:length(mesh) -1
               fhat_new(i) = fhat_old_old(i) - (2*dt*u(i))*(fhat_old(i+1) - fhat_old(i-1))/(mesh(i+1) - mesh(i-1));
           end
           %update last node
           fhat_n1 = (uprime(end)/u(end))*(x_n1 - mesh(end-1))*fhat_old(end) + fhat_old(end-1);
           fhat_new(length(mesh)) = fhat_old_old(end) - (2*dt*u(end))*(fhat_n1 - fhat_old(end-1))/(x_n1 - mesh(end-1));
       end
               
           
   end
   
   %update counters etc
   counter = counter +1;
   t = t + dt;
   fhat_old_old = fhat_old;
   fhat_old = fhat_new;
   
   %outputs
   if varargin{2} == "all"
       f(counter,:) = fhat_new./u;
   elseif varargin{2} == "stride"
       if mod((counter -1), varargin{3}) == 0
           f((((counter-1)/varargin{3}) +1),:) = fhat_new./u;
           stride_vec((((counter-1)/varargin{3}) +1)) = t;
       end
   end
end

%output 
if varargin{2} == "all" | varargin{2} == "stride"
    f(1,:) = f0;
else
    f = fhat_new./u;
end
stride_vec(1) = t_vec(1);
varargout{1} = stride_vec;

end

