%-------------------------------
% TMS016 -  Potts Model
%-------------------------------
clear; clc;

m = 50;
n = 50;
K = 4;               % Number of classes
nsim = 100;
plotflag = 2;

% One-hot random initialization
z0 = rand(m, n, K);
z0 = z0 == max(z0, [], 3);

% 4-neighbor structure
N = [0 1 0;
     1 0 1;
     0 1 0];

%% Part 1: Try different beta values with diagonal matrix
fprintf('\n---- Potts Model: Varying beta (diagonal) ----\n');

beta_values = [0.1, 0.3, 0.5, 0.9, 1.5];
alpha = 0;

for i = 1:length(beta_values)
    beta_val = beta_values(i);
    beta = beta_val * eye(K);  % class prefers same-class neighbors

    fprintf('Simulating Potts model with beta = %.2f\n', beta_val);
    [z, Mz, ll] = mrf_sim(z0, N, alpha, beta, nsim, plotflag);
    
    sgtitle(['Potts model (K = ', num2str(K), '), \beta = ', num2str(beta_val)]);
    pause;
end

%% Part 2: Try custom full beta matrix
fprintf('\n---- Potts Model: Custom beta matrix ----\n');

% Class 1 strongly prefers 1, dislikes 2 & 3 & 4
beta = [ 3  -1  -1  -1;
        -1   3  -1  -1;
        -1  -1   3  -1;
        -1  -1  -1   3 ];


fprintf('Simulating Potts model with full beta matrix\n');
[z, Mz, ll] = mrf_sim(z0, N, alpha, beta, nsim, plotflag);
sgtitle('Potts model (K = 4) with custom beta matrix');

