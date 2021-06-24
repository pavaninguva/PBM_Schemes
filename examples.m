%EXAMPLE SCRIPT 

%This script is meant to showcase the implementation of the various models

%% Case 1
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