%EXAMPLE SCRIPT 

%This script is meant to showcase the implementation of the various models

%% Case 1

mesh = linspace(0,1,201);

f0 = (1/0.01)*exp(-mesh./0.01);

[f, stride_vec] = model_1(mesh,f0,0.5,0.5,"Upwind",[0,1],"output_style","stride", 20);

figure(1)
plot(mesh,f0);
hold on
plot(mesh,f)
hold off
ylabel("f")
xlabel("x")
legend(string(stride_vec))