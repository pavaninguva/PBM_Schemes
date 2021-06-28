% Script for employing finite differencing to solve PBMs
%
% This function is able to execute and run a variety of finite difference
% schemes for solving problems of the form: 
% df/dt + u*df/dx = g(x) in 1-D subject to the following boundary conditions.

% On the left end of the domain, a modified Dirichlet B.C. is enforced
% which sets the value of the Ghost node at 0 i.e. f(0) = 0
% On the right end of the domain, df/dx = 0 is enforced.

function [f, varargout] = model_3(mesh, f0, u, gfun, CFL, scheme, t_vec, varargin)
%% Description

%The inputs and outputs are defined as follows: 
%INPUTS
%mesh: input the mesh in the form of an array e.g. linspace
%f0: Initial profile also in an array which should be the same length as
%the mesh
%u: this is the coefficient of the spatial term in the PDE. 
%gfun: Provide the nonhomogeneous term as a function handle. 
%CFL: specify the CFL number, typically between 0-1. 
%scheme: Three schemes are implemented: Upwind, Lax Wendroff and Leapfrog. 
%t_vec: Array containing the start and stop time e.g. [0,1] seconds

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




end