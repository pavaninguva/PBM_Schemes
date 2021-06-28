%EXAMPLE SCRIPT 

%This script is meant to showcase the implementation of the various models

%% Model 1

%rum simulation using model_1 function
[f, stride_vec] = model_1(101,@(x)(1/0.01)*exp(-x./0.01),0.5,1.0,"Leapfrog",[0,1],[0,1],"output_style","stride",10);

%simple plotting
figure(1)
plot(mesh,f)
ylabel("f")
xlabel("x")
legend(string(stride_vec))

%% Model 2

%Using model_2_conservative_uniform.m
f0_2 = 50*exp(-((mesh-0.2).^2)/0.0005);
[f2,stride_vec2] = model_2_conservative_uniform(mesh,f0_2, @(x)growth_rate(x), @(x)dudx(x), 1.0, "Lax Wendroff", [0,1],"output_style","final");

%simple plotting
figure(2)
plot(mesh,f2)
ylabel("f")
xlabel("x")
legend(string(stride_vec2))

%using model_2_conservative_nonuniform.m
[f3, mesh3, stride_vec3] = model_2_conservative_nonuniform(0.01,@(x)50*exp(-((x-0.2).^2)/0.0005),@(x)growth_rate(x), @(x)dudx(x), "Lax Wendroff", [0,1],[0,1], "output_style","stride",20);

%simple plotting
figure(3)
plot(mesh3,f3)
ylabel("f")
xlabel("x")
legend(string(stride_vec3))

%using model_2_transform_nonuniform.m

[f4, mesh4, stride_vec4] = model_2_transform_nonuniform(0.005,@(x)50*exp(-((x-0.2).^2)/0.0005),@(x)growth_rate(x), @(x)dudx(x), "Leapfrog", [0,1],[0,1], "output_style","stride",20);

%simple plotting
figure(4)
plot(mesh4,f4)
ylabel("f")
xlabel("x")
legend(string(stride_vec4))




%% functions

%functions for model 2

function u = growth_rate(mesh)
u = 0.1*4.34 + 0.06*4.34*mesh;
end

function uprime = dudx(mesh)
uprime = 0.06*4.34*ones(length(mesh),1);
end
