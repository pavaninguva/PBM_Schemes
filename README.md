# PBM_Schemes
This repository contains scripts that can be used to explore the numerical solution of a variety of population balance models. Principally, three numerical schemes are considered: the Upwind, Lax-Wendroff and Leapfrog schemes. Further information can be found in the primary manuscript and supplementary information. 

## Model 1

We consider a population balance model of the form: 
$$
\frac{\partial f}{\partial t} + g\frac{\partial f}{\partial a} = 0 \ , \ f(t=0,a) = f_{0},
$$
where $f$ is the number density, $t$ is time, $a$ can potentially be the age or length  and $g$ is a constant and positive growth rate. A Neumann boundary condition is imposed on the right boundary of the domain and a modified Dirichlet boundary condition on the left domain which specifies that cell at the ghost node on the left domain (i.e. $a < 0$) is 0. Refer to the main text for a more detailed explanation. 

## Model 2

We consider a population balance model with a variable growth rate as follows: 
$$
\frac{\partial f}{\partial t} + \frac{\partial (g(a)f)}{\partial a} = 0 \ , \ f(t=0,a) = f_{0}.
$$
The boundaries of the domain are treated in the same manner as model 1. 

## Model 3

A non-homogeneous population balance model of the form is considered: 
$$
\frac{\partial f}{\partial t} + g\frac{\partial f}{\partial a} = \alpha(a)
$$
 

