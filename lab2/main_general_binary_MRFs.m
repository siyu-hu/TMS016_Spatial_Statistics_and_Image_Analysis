%-------------------------------
% TMS016 - General Binary MRFs
%-------------------------------
clear; clc;

% Parameters
m = 50;
n = 50;
K = 2;
nsim = 100;
plotflag = 2;

% Initial random one-hot state
z0 = rand(m, n, K);
z0 = z0 == max(z0, [], 3);  % one-hot

%% Part 1: Varying alpha, beta = 0
fprintf('\n PART 1 \n');

alpha_list = [
    0   0;
    1  -1;
   -1   1;
    0.5 -0.5;
];

N = [0 1 0; 1 0 1; 0 1 0];  % 4-neighbor
beta = 0 * eye(K);          % no spatial interaction

for i = 1:size(alpha_list, 1)
    alpha = alpha_list(i, :);
    fprintf('\nSimulating with alpha = [%g %g], beta = 0\n', alpha(1), alpha(2));
    
    [z, Mz, ll] = mrf_sim(z0, N, alpha, beta, nsim, plotflag);
    
    sgtitle(['\alpha = [', num2str(alpha), '], \beta = 0']);
    saveas(gcf, sprintf('part1_alpha_%g_%g.png', alpha(1), alpha(2)));
    pause;
end

%% Part 2: Varying neighborhood structure N, fixed alpha=0 and beta=0.5
fprintf('\n PART 2: Varying neighborhood N, alpha = 0, beta = 0.5 \n');

N_list = {
    [0 1 0;
     1 0 1;
     0 1 0],  % 4-neighbor

    [1 1 1;
     1 0 1;
     1 1 1],  % 8-neighbor

    [1 1 0;
     1 0 1;
     0 1 1],  % diagonal-like

    [0 0 0 1 1;
     0 0 1 1 0;
     0 1 0 1 0;
     0 1 1 0 0;
     1 1 0 0 0]  % long-range
};

alpha = 0;
beta = 0.5 * eye(K);  % spatial smoothing

for i = 1:length(N_list)
    N = N_list{i};
    fprintf('\nSimulating with neighborhood structure #%d\n', i);
    
    [z, Mz, ll] = mrf_sim(z0, N, alpha, beta, nsim, plotflag);
    
    sgtitle(['N structure #', num2str(i), ', \beta = 0.5']);
    saveas(gcf, sprintf('part2_Nstruct_%d.png', i));
    pause;
end
