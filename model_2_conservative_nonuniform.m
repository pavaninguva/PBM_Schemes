% Script for employing finite differencing to solve PBMs
%
% This function is able to execute and run a variety of finite difference
% schemes for solving problems of the form: 
% df/dt + d(u(x)f)/dx = 0 in 1-D subject to the following boundary conditions.

% On the left end of the domain, a modified Dirichlet B.C. is enforced
% which sets the value of the Ghost node at 0 i.e. f(0) = 0
% On the right end of the domain, df/dx = 0 is enforced. 

%Instead of inputting a mesh, the function precomputes the mesh to enforce
%CFL = 1.0 at each point. Therefore, input a desired value of dt. 

function [f,mesh,varargout] = model_2_conservative_nonuniform(dt, f0fun, ufun, uprimefun, scheme, t_vec, x_vec, varargin)

%% Description

%INPUTS
%dt: input the desired value of dt
%f0fun: initial profile which should be the same size as mesh
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
f_old = f0;
f_old_old = f0;
t = t_vec(1);
counter = 1;

length(u)
length(uprime)
length(mesh)

%for Lax Wendroff
u_n1 = ufun(x_n1);

%start simulation

while t < t_vec(2) - dt*1e-3
   if scheme == "Upwind" 
       %update first node
       f_new(1) = f_old(1) - (dt/(mesh(1) - x_0))*(u(1)*f_old(1));
       %update rest of the nodes using for loop
       for i = 2:length(mesh)
           f_new(i) = f_old(i) - (dt/(mesh(i) - mesh(i-1)))*(u(i)*f_old(i) - u(i-1)*f_old(i-1));
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