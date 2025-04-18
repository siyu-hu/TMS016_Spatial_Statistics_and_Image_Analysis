function y = classification2rgb(x,RGB)

% RGBIMAGE Make an RGB image from several weight images.
%          Specially suited for visualisation of classification
%          probabilities.
%
%  y=rgbimage(x)
%  y=rgbimage(x,RGB)
%
%  x: the weight images, as an m-n-d matrix
%     0 <= x(i,j,k) <= 1, for all i,j and k
%     0 <= sum(x(i,j,:),3) <= 1, for all i,j
%  y: the RGB image, m-n-3 matrix
%  RGB: color definitions, default = 
%       [0 0 1;0 1 0;1 0 0;...
%        0 1 1;1 0 1;1 1 0;...
%        1 1 1;0 0 0];
%
%  y(i,j,k) = x(i,j,1)*RGB(1,k) + x(i,j,2)*RGB(2,k) + etc.
%
%  Note the order of the default colors; blue, green, red, etc.

if (nargin<2)
    RGB=[0    0.4470    0.7410;
    0.8500    0.3250    0.0980;
    0.9290    0.6940    0.1250;
    0.4940    0.1840    0.5560;
    0.4660    0.6740    0.1880;
    0.3010    0.7450    0.9330;
    0.6350    0.0780    0.1840];
end

y=zeros(size(x,1),size(x,2),3);
for k=1:size(x,3)
  y(:,:,1) = y(:,:,1) + x(:,:,k)*RGB(k,1);
  y(:,:,2) = y(:,:,2) + x(:,:,k)*RGB(k,2);
  y(:,:,3) = y(:,:,3) + x(:,:,k)*RGB(k,3);
end
y=max(min(y,1),0);
