%EXAMPLE SCRIPT 

%This script is meant to showcase the implementation of the various models

%% Model 1
%define mesh
mesh = linspace(0,1,201);

%define f0
f0 = (1/0.01)*exp(-mesh./0.01);

%rum simulation using model_1 function
[f, stride_vec] = model_1(mesh,f0,0.5,1.0,"Leapfrog",[0,1],"output_style","stride",20);

%simple plotting
figure(1)
plot(mesh,f)
ylabel("f")
xlabel("x")
legend(string(stride_vec))

%% Model 2

%Using model_2_conservative_uniform.m
u = growth_rate(mesh);
uprime = dudx(mesh);
f0_2 = 50*exp(-((mesh-0.2).^2)/0.0005);
[f2,stride_vec2] = model_2_conservative_uniform(mesh,f0_2, u, uprime, 1.0, "Upwind", [0,1],"output_style","stride",20);

%simple plotting
figure(2)
plot(mesh,f2)
ylabel("f")
xlabel("x")
legend(string(stride_vec2))





%% functions

%functions for model 2

function u = growth_rate(mesh)
u = 0.1*4.34 + 0.06*4.34*mesh;
end

function uprime = dudx(mesh)
uprime = 0.06*4.34*ones(length(mesh),1);
end
