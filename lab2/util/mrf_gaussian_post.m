function alpha_post=mrf_gaussian_post(alpha,theta,y,opt,obs_ind,mask)
% alpha_post=mrf_gaussian_post(alpha,theta,y,opt,obs_ind,mask)
% Computes the posterior alpha-parameters in a MRF with Gaussian data
%
% Inputs:
%  alpha : 1x1 or 1xK or m*nxK, the prior alpha-parameters.
%          See mrf_sim for the model specification.
%  theta : cell array of length K specifying the expectation vector and
%          covariance matrix for each k=1,...,K, as
%          theta{k}.mu and theta{k}.Sigma
%  y     : m*nxd, data image with d-dimensional Gaussian vector data.
%  alpha_post : m*nxK, the posterior alpha-parameters.

% Formula:
%   alpha_post_ik = alpha_ik + log(p(y_i|theta_k))

if nargin < 5; obs_ind = []; end
if nargin < 4; opt = [0 0]; end
if iscell(y);
  nrep = length(y);
	if ~iscell(alpha);
		alphac = cell(1,nrep);
		for rep = 1:nrep
			alphac{rep} = alpha;
		end
		alpha = alphac;
		clear alphac;
	end
else
	nrep = 1;
	y  = {y};
	if ~iscell(alpha);
		alphac = cell(1,nrep);
		for rep = 1:nrep
			alphac{rep} = alpha;
		end
		alpha = alphac;
	end
	clear alphac;
end
if isempty(obs_ind)
  obs_ind = cell(1,nrep);
  for rep = 1:nrep
    obs_ind{rep} = ones(1,size(y{rep},1));
  end
end

K = length(theta);
%parfor rep=1:nrep
for rep=1:nrep
	sz = size(y{rep});
	d = sz(2);
	sz = [sum(mask{rep}(:)) K];
	alpha_post{rep} = make_K_im(alpha{rep},sz);
	for k=1:K
  		ycm = bsxfun(@minus,y{rep},theta{k}.mu(:)');
  		if opt(1) == 0
	  		alpha_post{rep}(obs_ind{rep},k) = alpha_post{rep}(obs_ind{rep},k) -d/2*log(2*pi)...
	  					-1/2*log(det(theta{k}.Sigma))...
	  					-0.5*sum(ycm.*(ycm*inv(theta{k}.Sigma)),2);
		elseif opt(1)==1
			Sigmai = diag(1./diag(theta{k}.Sigma));
	  		alpha_post{rep}(obs_ind{rep},k) = alpha_post{rep}(obs_ind{rep},k) -d/2*log(2*pi)...
      					-1/2*sum(log(diag(theta{k}.Sigma))) ...
      					-0.5*sum(ycm.*(ycm*Sigmai),2);
		elseif opt(1) == 2
			alpha_post{rep}(obs_ind{rep},k) = alpha_post{rep}(obs_ind{rep},k) -d/2*log(2*pi)...
      					-d/2*log(theta{k}.Sigma(1)) ...
      					-0.5*sum(ycm.*ycm,2)/theta{k}.Sigma(1);
		end
	end
end

function im=make_K_im(v,sz)
if (length(v)==1) %alpha = 1x1
  im = v*ones(sz);
elseif (size(v,1)==sz(1)) && (size(v,2)==sz(2))%alpha = m*n x K
    im = v;
else % alpha = 1xK
  im = repmat(v,[sz(1),1]);
end
