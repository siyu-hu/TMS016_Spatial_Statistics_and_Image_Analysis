% Project 2.1 - Image Segmentation using K-means, GMM, and MRF-GMM
clear; clc;

function p = my_normpdf(x, mu, sigma)
    p = (1 / sqrt(2 * pi * sigma^2)) * exp(-(x - mu).^2 / (2 * sigma^2));
end


% ----------- Step 1: Load and Display Original Data -------------
load('permeability.mat');  
figure;
imagesc(Y); axis image; colormap(gray); colorbar;
title('Original Permeability Image');

% Flatten the image data to a vector
data = Y(:);

%% ----------- Step 2: K-means (K=2) Segmentation ----------------------

data = Y(:);            % Flatten image
K = 2;                     % Two clusters
max_iter = 10;             % Number of iterations
centroids = rand(K,1);     % Randomly initialize two centroids

for iter = 1:max_iter
    % Step 1: Assign each pixel to the closest centroid
    dist = abs(data - centroids');     % Euclidean distance
    [~, labels_kmeans] = min(dist, [], 2);

    % Step 2: Update centroids
    for k = 1:K
        if any(labels_kmeans == k)
            centroids(k) = mean(data(labels_kmeans == k));
        end
    end
end

% Reshape result into image form
seg_kmeans = reshape(labels_kmeans, size(Y));

figure;
imagesc(seg_kmeans); axis image; colormap(gray);
title('Manual K-means Segmentation');


%% ----------- Step 3: Gaussian Mixture Model (GMM) using EM (1D, K=2) --------------

data = Y(:);             % Flatten the image
N = length(data);
K = 2;                   % Number of Gaussian components

% Initialization
mu = [min(data); max(data)];            
sigma2 = [var(data); var(data)];        
pi_k = [0.5; 0.5];                      
max_iter = 20;
responsibility = zeros(N, K);

for iter = 1:max_iter
    for k = 1:K
        % Manual normal PDF calculation
        p(:,k) = pi_k(k) * (1 / sqrt(2 * pi * sigma2(k))) * ...
            exp(-(data - mu(k)).^2 / (2 * sigma2(k)));
    end
    sum_p = sum(p, 2);
    responsibility = p ./ sum_p;

    Nk = sum(responsibility, 1);
    for k = 1:K
        mu(k) = sum(responsibility(:,k) .* data) / Nk(k);
        sigma2(k) = sum(responsibility(:,k) .* (data - mu(k)).^2) / Nk(k);
        pi_k(k) = Nk(k) / N;
    end
end

% Assign hard labels
[~, labels_gmm] = max(responsibility, [], 2);
seg_gmm = reshape(labels_gmm, size(Y));

% Check what classes were assigned
disp('Unique labels:');
disp(unique(labels_gmm));  % Should show [1; 2]

figure;
imagesc(seg_gmm); axis image; colormap(gray);
title('Manual GMM Segmentation');
drawnow;


%% ----------- Step 4: Markov Random Field Mixture Model(MRF-GMM via ICM) --------
labels = reshape(labels_gmm, size(Y));  % Use GMM result as init
[rows, cols] = size(Y);
num_iter = 5;  % Number of ICM iterations
beta = 1;      % Smoothness weight

% Reuse the same GMM parameters for likelihood estimation
for iter = 1:num_iter
    for i = 2:rows-1
        for j = 2:cols-1
            p = Y(i,j);
            neighbors = [labels(i-1,j), labels(i+1,j), labels(i,j-1), labels(i,j+1)];

            % Compute log-likelihoods manually
            loglike1 = -log(1 / sqrt(2 * pi * sigma2(1))) - (p - mu(1))^2 / (2 * sigma2(1));
            loglike2 = -log(1 / sqrt(2 * pi * sigma2(2))) - (p - mu(2))^2 / (2 * sigma2(2));

            e1 = -loglike1 + beta * sum(neighbors ~= 1);
            e2 = -loglike2 + beta * sum(neighbors ~= 2);

            labels(i,j) = 1 + (e2 < e1);  % Choose lower energy
        end
    end
end

figure;
imagesc(labels); axis image; colormap(gray);
title('MRF-GMM Segmentation (ICM)');
