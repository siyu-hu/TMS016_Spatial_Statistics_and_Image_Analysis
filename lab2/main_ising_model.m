%-----------------------
% TMS016 - Exercise 2: Ising Model
%-----------------------
clear; clc;
m = 50; 
n = 50; 
K = 2;
alpha =0;
nsim =100;
plotflag= 2;


% Define 4-neighbor structure
N = [0 1 0;
     1 0 1;
     0 1 0];

% Create a random one-hot encoded initial state z0
z0 = rand(m, n, K);
z0_new = zeros(m,n,K);
for i =1: m
    for j = 1:n
        [~, kmax] =  max(z0(i,j,:));
        z0_new(i, j, kmax) = 1; 
    end
end

z0 = z0_new;
% a better way to create random one-hot matrix : z0 = z0 == max(z0, [], 3); 

beta_values = [-1, -0.5, 0.1, 0.3, 0.4,0.42, 0.44, 0.5, 0.9, 2];% 

for i = 1:length(beta_values)
    beta_val = beta_values(i);
    beta = beta_val * eye(K);  % diagonal matrix

    fprintf('Running simulation for beta = %.2f\n', beta_val);
    [z, Mz, ll] = mrf_sim(z0, N, alpha, beta, nsim, plotflag);

    sgtitle(['\beta = ', num2str(beta_val)], 'FontSize', 16);
    pause;  % Wait for key press to continue
end