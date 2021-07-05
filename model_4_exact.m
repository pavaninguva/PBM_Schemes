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

function [f,varargout] = model_4_exact(N_cells, f0fun, intkfun, bfun, CFL, t_vec, x_vec, varargin)
%% Description

%The inputs and outputs are defined as follows: 
%INPUTS
%N_cells: Number of cells in the mesh
%f0: Initial profile in the form of a function handle
%u: this is the coefficient of the spatial term in the PDE. 
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
dt = CFL*(dx);

%initialize the functions and counter
f0 = f0fun(mesh);
intk = intkfun(mesh);




end