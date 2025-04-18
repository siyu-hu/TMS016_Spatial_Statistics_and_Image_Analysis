function [z,Mz,ll]=mrf_sim(z0,N,alpha,beta,nsim,plotflag)
% z=mrf_sim(z0,N,alpha,beta,nsim) Samples a Markov random fields using
% Gibbs sampling. The field x is supposed to be on an m x n lattice, and is
% represented by a matrix z = [m n K] where z(i,j,k) = 1 if x_(i,j) = k and
% z(i,j,l) = 0 for l != 0. The inputs are:
%   z0    : starting value of size [m n K]
%   N     : The stencil for the neighborhood structure. 
%             Example:
%              N = [0 1 0;
%                   1 0 1;
%                   0 1 0];
%   alpha : The alpha parameters on the form
%             - a single value (for alpha_k = alpha)
%             - a vector of length K with the alpha_k
%             - a matrix of size [m*n K] with alpha_k for each pixel.
%   beta  : The beta parameters on the form:
%             - a single value (beta_kk = beta and beta_kl = 0 for k!=l).
%             - a Kx1 vector   (beta_kk = beta_k and beta_kl = 0 for k!=l).
%             - a KxK matrix with the values beta_kl. 
%   nsim  : The number of iterations in the Gibbs sampler. 
%
% [z,Mz] = mrf_sim(z0,N,alpha,beta,nsim) also returns the mean of the nsim
% simulations for each pixel. 
%
% [z,Mz,ll] = mrf_sim(..,plotflag) specifies whether to plot results during
% the simulations. plotflag = 0 gives no plots, plotflag = 1 will plot the
% field each iteration, and plotflag = 2 will also plot the log-liklihood
% track. 
%
% David Bolin (david.bolin@chalmers.se) 2018

if nargin < 6; plotflag = 0; end
if nargin < 5; nsim = 100; end
if nargin < 4; error('Too few input parameters'); end

[m,n,K] = size(z0);

if m == 1 || n == 1 || K == 1
  error('z0 should be an mxnxK matrix.')
end

%compute neighbors for each pixels based on N
[W, a, b] = build_W(N,[m n]);


%Make alpha an [m*n K] matrix
if numel(alpha) == 1
  alpha = alpha*ones([m*n K]);
elseif numel(alpha) == K
  alpha = repmat(alpha(:)',[m*n,1]);
else 
  if size(alpha,1) ~= m*n || size(alpha,2) ~= K
    error('wrong size of alpha.')
  end
end

%check size of beta 
if numel(beta) == 1
  beta = beta*eye(K);
elseif numel(beta)==K
  beta = diag(beta);
elseif size(beta,1) ~= K || size(beta,2) ~=K
  error('wrong size of beta.')
end

%compute the indices for the different groups
ij_ = [kron(1:a,ones(1,b)); kron(ones(1,a),1:b)];
ijm = false(m*n,a*b);
for k=1:a*b
    Ind = false(m,n);
    Ind(ij_(1,k):a:m,ij_(2,k):a:n) = true;
    ijm(:,k) = Ind(:);
end


%precompute things for the different groups 
Wij = cell(1,a*b);
alphaij = cell(1,a*b);
KKij = cell(1,a*b); 
Mzc = cell(1,a*b);
KK = repmat(1:K,[m*n,1]);
for k=1:a*b
  Wij{k} = W(ijm(:,k),:);
  alphaij{k} = alpha(ijm(:,k),:);
  KKij{k} = KK(ijm(:,k),:);
  Mzc{k} = zeros(sum(ijm(:,k)),K);
end

z = reshape(z0,[m*n K]);
if nargout == 3
  ll = zeros(1,nsim+1);
  f = W*z;
  ll(1) = sum(sum(z.*alpha + 0.5*(f.*z)*beta));
end
for i=1:nsim
  if plotflag > 0
    z_im = reshape(z,[m n K]);
    y = classification2rgb(z_im);
    if plotflag == 1
      figure(1)
      imagesc(y)
      axis off;axis image
    else
      figure(1)
      subplot(211)
      imagesc(y)
      axis off;axis image
      subplot(212)
      plot(ll(1:max(i-1,1)))
    end
    drawnow
  end
  for k=randperm(a*b)
    f = Wij{k}*z;
    Mz_cond = alphaij{k}+f*beta;
    Mz_cond = exp( bsxfun( @minus, Mz_cond, max(Mz_cond, [], 2) ) );
    Msum = sum(Mz_cond,2);
    Mz_cond = bsxfun(@rdivide,Mz_cond, Msum);
    if nargout >1
      Mzc{k} = Mzc{k}+Mz_cond;
    end
      
    e = rand(sum(ijm(:,k)),1);
    x = 1 + sum(bsxfun(@lt,cumsum(Mz_cond,2),e), 2);
    z(ijm(:,k),:) = bsxfun(@eq,x,KKij{k})*1;
  end
  if nargout == 3 || plotflag == 2
    f = W*z;
    ll(i+1) = sum(sum(z.*alpha + 0.5*(f.*z)*beta));
  end
end

if nargout > 1
  Mz = zeros(m*n,K);
	for k=1:2
		Mz(ijm(:,k),:) = Mzc{k};
  end
  Mz = Mz/nsim;
  Mz = reshape(Mz,[m n K]);
end
z = reshape(z,[m n K]);

function [W,a,b] = build_W(N,sz)

if sum(N-rot90(N,2))
  error('The neighbourhood must have reflective symmetry.')
end
[a,b] = size(N);
if ((mod(a,2)~=1) || (mod(b,2)~=1))
  error('The neighbourhood must have odd width and height.')
end
a = ceil(a/2);
b = ceil(b/2);
if N(a,b)~=0
  error('The pixel cannot be neighbor with itself.')
end

II = [];
KK = [];
JJ_I = [];
JJ_J = [];

[I,J] = ndgrid(1:sz(1),1:sz(2));
I = I(:); J = J(:);
for i=1:size(N,1)
  for j=1:size(N,2)
    if (N(i,j) ~= 0)
      II = [II;I+sz(1)*(J-1)];
      JJ_I = [JJ_I;I+i-(size(N,1)+1)/2];
      JJ_J = [JJ_J;J+j-(size(N,2)+1)/2];
      KK = [KK; N(i,j)*ones(prod(sz),1)];
    end
  end
end
JJ = JJ_I+sz(1)*(JJ_J-1);
ok = (JJ_I>=1) & (JJ_I<=sz(1)) & (JJ_J>=1) & (JJ_J<=sz(2));
II(~ok) = []; JJ(~ok) = []; KK(~ok) = [];
W = sparse(II,JJ,KK,prod(sz),prod(sz));
