% Script for employing finite differencing to solve PBMs
%
% This function is able to execute and run a variety of finite difference
% schemes for solving problems of the form: 
% df/dt + d(u(x)f)/dx = 0 in 1-D subject to the following boundary conditions.

% On the left end of the domain, a modified Dirichlet B.C. is enforced
% which sets the value of the Ghost node at 0 i.e. f(0) = 0
% On the right end of the domain, df/dx = 0 is enforced. 



function [f,varargout] = model_2_conservative_uniform(mesh, f0, u, uprime, CFL, scheme, t_vec, varargin)

%% Description

%INPUTS
%mesh: input the mesh as an array
%f0: initial profile which should be the same size as mesh
%u: input u(x) as an array, should be same size as mesh
%uprime: input u'(x) as an array as the same size as the mesh. Used for the
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
dt = CFL*(dx)/(max(u));

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