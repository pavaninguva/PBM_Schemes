# PBM_Schemes
This repository contains scripts that can be used to explore the numerical solution of a variety of population balance models. Principally, three numerical schemes are considered: the Upwind, Lax-Wendroff and Leapfrog schemes. Further information can be found in the primary manuscript and supplementary information. 

## Model 1

We consider a population balance model of the form: 

<img src="https://render.githubusercontent.com/render/math?math=\frac{\partial f}{\partial t} %2B g\frac{\partial f}{\partial a} = 0 \ , \ f(t=0,a) = f_{0},">

where <img src="https://render.githubusercontent.com/render/math?math=f"> is the number density, <img src="https://render.githubusercontent.com/render/math?math=t">  is time, <img src="https://render.githubusercontent.com/render/math?math=a"> can potentially be the age or length or any other appropriate variable and <img src="https://render.githubusercontent.com/render/math?math=g"> is a constant and positive growth rate. A Neumann boundary condition is imposed on the right boundary of the domain and a modified Dirichlet boundary condition on the left domain which specifies that cell at the ghost node on the left domain (i.e. <img src="https://render.githubusercontent.com/render/math?math=a<0"> ) is 0. Refer to the main text for a more detailed explanation. 

On a separate script, the function can be called as follows: 

```
[f, stride_vec] = model_1(mesh,f0,0.5,0.5,"Upwind",[0,1],"output_style","stride", 20);
```

The various input and output arguments are outlined in the comments within the function script. The mesh and the values of <img src="https://render.githubusercontent.com/render/math?math=f_{0}">should be precomputed and supplied accordingly. 

An example plot can be seen below: 

<img src="./Figures/model1.png" alt="model1" style="zoom:50%;" />

## Model 2

We consider a population balance model with a variable growth rate as follows: 

<img src="https://render.githubusercontent.com/render/math?math=\frac{\partial f}{\partial t} %2B \frac{\partial ( g(a) f)}{\partial a} = 0 \ , \ f(t=0,a) = f_{0},">

The boundaries of the domain are treated in the same manner as model 1. 

As described in the manuscript, three model formulations were considered: 

1. Applying finite differencing directly on the equation on a uniform mesh (`model_2_conservative_uniform.m`)
2. Applying finite differencing on the conservative model formulation using a non-uniform mesh to enforce CFL = 1.0 at each point (`model_2_conservative_nonuniform.m`)
3. Applying finite differencing on the transformed model formulation with the non-uniform mesh from the second approach (`model_2_transformed_nonuniform.m`)

### Conservative, Uniform Grid

The first formulation can be employed as follows: 

```matlab
[f2,stride_vec2] = model_2_conservative_uniform(mesh,f0_2, @(x)growth_rate(x), @(x)dudx(x), 1.0, "Lax Wendroff", [0,1],"output_style","stride",20);
```

with the following associated functions for <img src="https://render.githubusercontent.com/render/math?math=g(a)"> and <img src="https://render.githubusercontent.com/render/math?math=g'(a)">: 

```matlab
function u = growth_rate(mesh)
u = 0.1*4.34 + 0.06*4.34*mesh;
end

function uprime = dudx(mesh)
uprime = 0.06*4.34*ones(length(mesh),1);
end
```

### Conservative, Non-uniform Grid

### Transformed, Non-uniform Grid

### Sample Plot

A sample plot can be seen below:

<img src="./Figures/model2.png" alt="model2" style="zoom: 25%;" />

## Model 3

A non-homogeneous population balance model of the form is considered: 
$$
\frac{\partial f}{\partial t} + g\frac{\partial f}{\partial a} = \alpha(a)
$$


