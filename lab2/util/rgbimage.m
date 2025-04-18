function y=rgbimage(x,RGB)

% Internal function that makes an RGB image from several weight images.

n_col = size(x,3);
if (nargin<2)
  RGB = jet(n_col);
end

y=zeros(size(x,1),size(x,2),3);
for k=1:n_col
  y(:,:,1) = y(:,:,1) + x(:,:,k)*RGB(k,1);
  y(:,:,2) = y(:,:,2) + x(:,:,k)*RGB(k,2);
  y(:,:,3) = y(:,:,3) + x(:,:,k)*RGB(k,3);
end
