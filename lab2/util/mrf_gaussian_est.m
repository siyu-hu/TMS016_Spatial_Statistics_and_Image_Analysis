function theta = mrf_gaussian_est(W,y,opt,obs_ind)

%Internal function that estimates the parameters of the Gaussian distributions
%for a fixed classification.

if nargin < 3; opt = [0 0]; end
if iscell(y);
	nrep = length(y);
else
	nrep = 1;
	y  = mat2cell(y);
	W = mat2cell(W);
end
K = size(W{1},2);

theta_hat = cell(nrep,K);
for rep = 1:nrep
	szW = size(W{rep});
	szy = size(y{rep});
	d = size(y{rep},2);
	for k=1:K
  	  sW = sum(W{rep}(obs_ind{rep},k));
  	  theta_hat{rep,k}.mu = (W{rep}(obs_ind{rep},k)'*y{rep})/sW;
  	  yc = bsxfun(@minus, y{rep},theta_hat{rep,k}.mu);
  	  if opt(1) == 0; %no restriction on sigma
	  	theta_hat{rep,k}.Sigma = (yc'*(bsxfun(@times,W{rep}(obs_ind{rep},k),yc)))/sW;
	  elseif opt(1) == 1 %diagonal sigma
		theta_hat{rep,k}.Sigma = diag(sum(bsxfun(@times,yc,W{rep}(obs_ind{rep},k)).*yc)/sW);
	  elseif opt(1) == 2 %diagonal sigma with common variance
		Wyc = bsxfun(@times,W{rep}(obs_ind{rep},k),yc);
		theta_hat{rep,k}.Sigma = eye(d)*sum(yc(:).*Wyc(:))/sW/d;
	  end
	end
end

theta = cell(1,K);
for k = 1:K
	theta{k}.mu = 0;
	theta{k}.Sigma = 0;
	for rep = 1:nrep
		theta{k}.mu = theta{k}.mu + theta_hat{rep,k}.mu/nrep;
		theta{k}.Sigma = theta{k}.Sigma +  theta_hat{rep,k}.Sigma/nrep;
	end
end

if opt(2) == 1; %common sigma for all classes
	Sigma = 0;
	for k = 1:K; Sigma = Sigma + theta{k}.Sigma/K; end
	for k = 1:K; theta{k}.Sigma = Sigma; end
end