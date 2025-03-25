%--------------------------
% Load image and colormap
%--------------------------
x = imread('chalmersplatsen.jpg');
[m,n,d] = size(x);% m,n is # of pixels, d is color componets.
x = double(x)/255;

% Red channel
figure;  
imagesc(x(:,:,1)); 
colormap hot; % built in color map.
colorbar;
axis image;

% Green channel
figure;  
imagesc(x(:,:,2)); 
colormap summer;
colorbar;
axis image;

% Blue channel
figure;  
imagesc(x(:,:,3)); 
colormap winter;
colorbar;
axis image

% % define red colormap: black - red - yellow - white
% imagesc(x(:,:,1)); 
% n_temp = 256;  % # of color in this colormap
% n1 = floor(n_temp /3);
% n2 = floor(n_temp /3);
% n3 = n_temp  - n1 - n2; 
% r = linspace(0, 1, n_temp )'; 
% g = [zeros(n1,1); linspace(0,1,n2+n3)'];  
% b = [zeros(n1+n2,1); linspace(0,1,n3)'];  
% map = [r g b]; 
% figure;
% colormap(map);
% axis image;


%% 
%--------------------------
% Color manipulations
%--------------------------

x_new = x(:,:,[2 1 3]);
imshow(x_new);

figure;
y1 = mean(x,3);
y2 = rgb2gray(x);
imshow(y1);

figure;
subplot(1,3,1); imagesc(y1); title('Average Gray'); axis image; colormap gray;
subplot(1,3,2); imagesc(y2); title('rgb2gray'); axis image; colormap gray;
subplot(1,3,3); imagesc(abs(y1 - y2)); title('Difference'); axis image; colormap gray;

figure;
imhist(y2);title('Distribution of pixel values ');

%%
%--------------------------
% Intensity-free representation
%--------------------------

% Simple method
%  G / (R + G + B)
figure;
relativeGreen = x(:,:,2) ./ (x(:,:,1) + x(:,:,2) + x(:,:,3)); 
imagesc(relativeGreen); colormap(summer); colorbar;title('Green Relative Value');

% R / (R + G + B)
figure;
relativeRed = x(:,:,1)./ (x(:,:,1) + x(:,:,2) + x(:,:,3)); 
imagesc(relativeRed); colormap(hot); colorbar;title('Red Relative Value');

% B / (R + G + B)
figure;
relativeBlue = x(:,:,3)./ (x(:,:,1) + x(:,:,2) + x(:,:,3)); 
imagesc(relativeBlue); colormap(winter); colorbar;title('Blue Relative Value');

%%
% Using LAB color space
lab = rgb2lab(x);
subplot(1,3,1); imagesc(lab(:,:,1)); title('L - Lightness'); colorbar;axis image;
subplot(1,3,2); imagesc(lab(:,:,2)); title('A - Green <-> Red'); colorbar;axis image;
subplot(1,3,3); imagesc(lab(:,:,3)); title('B - Blue <-> Yellow'); colorbar;axis image;
%%
%--------------------------
% K = 4  classfication
%--------------------------

xrel = zeros(size(lab));
for i = 1:3 % normalization 
    xi = lab(:,:,i);
    xrel(:,:,i) = (xi - min(xi(:))) / (max(xi(:)) - min(xi(:)));
end

z(:,:,1) = xrel(:,:,2) > 0.4;
z(:,:,2) = xrel(:,:,2) < 0.4 & xrel(:,:,1) < 0.3;
z(:,:,3) = xrel(:,:,2) < 0.4 & xrel(:,:,1) > 0.3 & xrel(:,:,3) < 0.3;
z(:,:,4) = xrel(:,:,2) < 0.4 & xrel(:,:,1) > 0.3 & xrel(:,:,3) > 0.3;


% 4 class color / red/ green / blue /yellow
RGB = [1 0 0; 0 1 0; 0 0 1; 1 1 0]; 
I = classification2rgb(z, RGB);
figure;
imshow(I);  
title('Classified Image K = 4');

function I = classification2rgb(z, RGB)
    [m, n, K] = size(z);
    I = zeros(m, n, 3);  
    for k = 1:K
        for c = 1:3  % R, G, B - 3 channels
            I(:,:,c) = I(:,:,c) + z(:,:,k) * RGB(k, c);
        end
    end
end
