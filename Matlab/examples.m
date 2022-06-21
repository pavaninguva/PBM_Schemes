%EXAMPLE SCRIPT 

%This script is meant to showcase the implementation of the various models

%% Basic Parameters


%% Model 1

%rum simulation using model_1 function
[f, stride_vec] = model_1(101,@(x)(1/0.01)*exp(-x./0.01),0.5,1.0,"Leapfrog",[0,1],[0,1],"output_style","stride",10);

%simple plotting
figure(1)
plot(linspace(0,1,101),f)
ylabel("f")
xlabel("x")
legend(string(stride_vec))

%% Model 2

%Using model_2_conservative_uniform.m
[f2,stride_vec2] = model_2_conservative_uniform(101,@(x)50*exp(-((x-0.2).^2)/0.0005), @(x)growth_rate(x), @(x)dudx(x), 1.0, "Upwind",[0,1],[0,1],"output_style","stride",20);
%simple plotting
figure(2)
plot(linspace(0,1,101),f2)
ylabel("f")
xlabel("x")
legend(string(stride_vec2))

%using model_2_conservative_nonuniform.m
[f3,mesh3, stride_vec3] = model_2_conservative_nonuniform(0.02,@(x)50*exp(-((x-0.2).^2)/0.0005),@(x)growth_rate(x),@(x)dudx(x), "Upwind", [0,1],[0,1],"output_stytle","stride",20);
%simple plotting
figure(3)
plot(mesh3,f3)
ylabel("f")
xlabel("x")
legend(string(stride_vec3))

%using model_2_transform_nonuniform.m
[f4,mesh4, stride_vec4] = model_2_transform_nonuniform(0.02,@(x)50*exp(-((x-0.2).^2)/0.0005),@(x)growth_rate(x),@(x)dudx(x), "Upwind", [0,1],[0,1],"output_stytle","stride",20);

%simple plotting
figure(4)
plot(mesh4,f4)
ylabel("f")
xlabel("x")
legend(string(stride_vec4))

%using model_2_exact.m
[f5,mesh5] = model_2_exact(101, @(x)f0_transfun(x), @(x)growth_rate(x), @(x)(50/(3*4.34))*log(5+3*x), @(x)(exp((3*4.34/50)*x) -5)/3, [0,1], 1);

%simple plotting
figure(5)
plot(mesh5,f5)
ylabel("f")
xlabel("x")


%% Model 3
[f6,stride_vec6] = model_3(201,@(x)50*exp(-((x-0.2).^2)/0.0005),0.5,@(x)(1+(0.1*x)+(0.1*x.^2)),@(x)(0.1 + 0.2.*x),1.0,"Lax Wendroff",[0,1],[0,1],"output_style","stride",20);

%simple plotting
figure(6)
plot(linspace(0,1,201),f6)
ylabel("f")
xlabel("x")
legend(string(stride_vec6))

%% Model 4

%using model_4_naive.m
[f7,stride_vec7] = model_4_naive(501,@(x)50*exp(-((x-0.2).^2)/0.0005), @(x)x, @(x)1,@(x)(0), 1.0, [0,1],[0,2], "Upwind", "output_style","stride",40);

%simple plotting
figure(7)
plot(linspace(0,2,501), f7)
ylabel("f")
xlabel("x")
legend(string(stride_vec7))

%using model_4_exact.m
[f8,stride_vec8] = model_4_exact(101,@(x)50*exp(-((x-0.2).^2)/0.0005),@(x)intk(x), @(x)(0), [0,1], [0,2],"output_style","stride",20);

%simple plotting
figure(8)
plot(linspace(0,1,101),f8)
ylabel("f")
xlabel("x")
legend(string(stride_vec8))



%% functions

%functions for model 2

function u = growth_rate(mesh)
u = 0.1*4.34 + 0.06*4.34*mesh;
end

function uprime = dudx(mesh)
uprime = 0.06*4.34*ones(length(mesh),1);
end

function f = f0_transfun(x)
f = (0.434 + 0.2604*x).*(50*exp(-((x-0.2).^2)/0.0005));
end

%function for model 4
function f = intk(x)
f = 0.5*(x.^2);
end
