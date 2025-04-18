function [theta,alpha,beta,cl,p]=mrf_sgd(y,K,opts)

% [theta,z,p,converged,alpha,beta,timings]=mrf_sgd(y,K,opts)
%	Function that estimates parameters in MRF mixture models using stochastic
%	gradient descent.
%
% INPUTS:
% y    : data of size [m n d] where m x n is the size of the image and d is
%	       the dimension. y can also be a cell with several images. Missing
%	       values can be indicated by NaN values.
% K    : The number of classes.
% opts : structure with options (all options have default values):
%       common_beta : 0/1. If 1 (default), the beta matrix is on the form
%                     beta*eye(K), otherwise diag(beta_1,...,beta_K).
%                     Default is 1.
%       alpha_type  : 0/1. If 1 (default), each class has an alpha parameter.
%                     If 0, no alpha parameter are used.
%       sigma_type  : 0/1/2 determines the structure of the covariance
%                     matrix for the Gaussian components:
%            					 0 : no restriction on the matrices (default)
%                      1 : assume diagonal covariance matrices
%             				 2 : assume Sigma_k = sigma_k*eye(d)
%       common_sigma : 0/1 determines if a common covariance matrix should be
%                   	 used for all Gaussian components, with structure
%               			 determined by sigma_type. Default 0.
%        plot_flag : determines what should be plotted:
%                     0 : no plots (default)
%                 		>0: plot classification
%                  		>1: also plot some parameter tracks.
%                 		>2: also plot posterior probability differences.
%         alpha0,beta0,p0,theta0 : starting values for parameters (optional).
%         tol          : Tolerance for convergence, the optimization is
%                        stopped when the step size is below the tolerance
%                        (default 1e-7).
%         N             : The neighborhood structure. Default is
%                         N = [0 1 0;1 0 1;0 1 0].
%         iter       : Max number of iterations for the estimation (default 100)
%         gibbs_iter : Number of iterations for the Gibbs sampler used to
%                      estimate the gradients (default 10).
%         class_iter : Number of iterations for the Gibbs sampler used for
%                      the final classification (default 1000).
% OUTPUTS:
% theta : A cell containing estimates of mean and covariance matrix
%         for the mixture components.
% alpha : estimates of the alpha parameters.
% beta  : estimates of the beta parameters.
% cl    : MAP classification of the pixels.
% z     : posterior estimate of the indicator field.
%
% David Bolin (david.bolin@chalmers.se) 2018

if nargin < 3; opts = []; end

if isempty(opts)
  iter = 100;
  alpha_type = 1;
  common_beta = 1;
  sigma_type = 0;
  common_sigma = 0;
  plotflag = 0;
  tol = 1e-7;
  gibbs_iter = 10;
  class_iter = 1000;
  beta0 = [];
  alpha0 = [];
  p0 = [];
  theta0 = [];
  N = [0 1 0; 1 0 1; 0 1 0];
else
  names = fieldnames(opts);
  if sum(strcmp(names,'iter'))>0
    iter = opts.iter;
  else
    iter = 100;
  end
  if sum(strcmp(names,'class_iter'))>0
    class_iter = opts.class_iter;
  else
    class_iter = 1000;
  end

  if sum(strcmp(names,'N'))>0
    N = opts.N;
  else
    N = [0 1 0; 1 0 1; 0 1 0];
  end

  if sum(strcmp(names,'alpha_type'))>0
    alpha_type = opts.alpha_type;
  else
    alpha_type = 1;
  end
  if sum(strcmp(names,'plot'))>0
    plotflag = opts.plot;
  else
    plotflag = 1;
  end
  if sum(strcmp(names,'sigma_type'))>0
    sigma_type = opts.sigma_type;
  else
    sigma_type = 0;
  end
  if sum(strcmp(names,'common_beta'))>0
    common_beta = opts.common_beta;
  else
    common_beta = 1;
  end
  if sum(strcmp(names,'common_sigma'))>0
    common_sigma = opts.common_sigma;
  else
    common_sigma = 0;
  end
  if sum(strcmp(names,'beta0'))>0
    beta0 = opts.beta0;
  else
    beta0 = [];
  end
  if sum(strcmp(names,'alpha0'))>0
    alpha0 = opts.alpha0;
  else
    alpha0 = [];
  end
  if sum(strcmp(names,'beta0'))>0
    beta0 = opts.beta0;
  else
    beta0 = [];
  end
  if sum(strcmp(names,'p0'))>0
    p0 = opts.p0;
  else
    p0 = [];
  end
  if sum(strcmp(names,'theta0'))>0
    theta0 = opts.theta0;
  else
    theta0 = [];
  end
  if sum(strcmp(names,'gibbs_iter'))>0
    gibbs_iter = opts.gibbs_iter;
  else
    gibbs_iter = 10;
  end
  if sum(strcmp(names,'tol'))>0
    tol = opts.tol;
  else
    tol = 1e-7;
  end
end

if isempty(iter), iter = 50; end
opt = [alpha_type common_beta==0 sigma_type common_sigma];
sigma_opt = opt(3:4);
tic;

if iscell(y);
	nrep = length(y);
else
	nrep = 1;
	y  = {y};
end

ntot = 0;
mask = cell(1,nrep);
obs_ind = cell(1,nrep);
for rep = 1:nrep
  mask{rep} = true(size(y{rep},1),size(y{rep},2));
  [m{rep}, n{rep}] = size(mask{rep});
  y{rep} = reshape(y{rep},[m{rep}*n{rep} size(y{rep},3)]);
  ys = sum(y{rep},3);
  obs_ind{rep} = true(m{rep}*n{rep},1);
	obs_ind{rep}(isnan(ys(:))) = false;
  mn{rep} = m{rep}*n{rep};
	sz{rep} = size(y{rep});
	ntot = ntot + mn{rep};
	d{rep} = sz{rep}(end);
	if length(sz{rep}) == 3; y{rep} = y{rep}(:); end


	if islogical(mask{rep})==0; mask{rep} = logical(mask{rep}); end
	if issparse(mask{rep})==0; mask{rep} = sparse(mask{rep}); end

	mc{rep} = mask{rep}(:);

  [W{rep} a_mrf b_mrf] = build_W(N,mask{rep});
end

%obtain initial estimates of theta and z
ystacked = [];
for rep = 1:nrep; ystacked = [ystacked; y{rep}]; end
if isempty(alpha0) || isempty(beta0) || isempty(p0) || isempty(theta0)

  disp('Compute initial classification.')

  [theta,prior, pstacked,ll] = normmix_em(ystacked,K,50,0);
  alpha=log(prior); beta = 0.5*ones(1,K);
else
	alpha = alpha0; beta = beta0;
	pstacked = p0; theta = theta0;
end
if opt(1) == 0
	alpha = log(ones(1,K)/K);
end
[tmp,clstacked]=max(pstacked,[],2);
if plotflag;
  pstack = zeros(prod(cell2mat(mn)),K);
end
for rep = 1:nrep
  p{rep} = zeros(mn{rep},K);
  cl{rep} = zeros(mn{rep},1);

  ps = pstacked(1:sz{rep}(1),:);
	cs = clstacked(1:sz{rep}(1));

	z0 = zeros(sz{rep}(1),K);
	for i=1:sz{rep}(1); z0(i,clstacked(i))=1; end
  z{rep} = zeros(mn{rep},K);
  for i=1:sz{rep}(1)
    z{rep}(obs_ind{rep}(i),:) = z0(i,:);
    p{rep}(obs_ind{rep}(i),:) = ps(i,:);
    cl{rep}(obs_ind{rep}(i)) = cs(i,:);
  end
  pstacked = pstacked(sz{rep}(1)+1:end,:);
  clstacked = clstacked(sz{rep}(1)+1:end);
  %draw random starting values for z at unobserved pixels
  for i=1:mn{rep}
    if sum(z{rep}(i,:))==0; z{rep}(i,randi(K,1)) = 1; end
  end

end
clear ystacked pstacked clstacked


%% plot %%
if plotflag
	figure(1),clf;
	if plotflag>1; figure(2);clf; end
	if plotflag>2; figure(3);clf; end
  	alpha_history(1,:) = exp(alpha);
    beta_history(1,:) = beta;
    for k=1:K
    	%beta_history(k,:,1) = beta;
  		mu_history(k,:,1) = theta{k}.mu;
  		if sigma_opt(1) == 0
	  		Sigma_history(:,:,k,1) = theta{k}.Sigma;
		elseif sigma_opt(1) == 1
			Sigma_history(:,k,1) = diag(theta{k}.Sigma);
		else
			Sigma_history(k,1) = theta{k}.Sigma(1);
		end
	end
	figure(1)
	for rep = 1:nrep
		if nrep < 6
			subplot(1,nrep,rep)
		else
			subplot(floor(sqrt(nrep)), ceil(nrep/floor(sqrt(nrep))), rep)
		end
    clplot = zeros(m{rep}*n{rep},1);
    clplot(mc{rep}) = cl{rep};
    imagesc(reshape(clplot,[m{rep} n{rep}])); axis off
    drawnow
	end
end

loop = 0;
done = 0;
disp('Estimate parameters.')
reverseStr = '';
grad_gauss = struct;
grad_gauss.alpha = 1;
grad_gauss.grad_prev = cell(1,K);
grad_gauss.d_prev    = cell(1,K);
for j=1:K
	grad_gauss.grad_prev{j} = [];
	grad_gauss.d_prev{j} = [];
end
grad_gauss.stepsize = 1;

grad_mrf = struct;
grad_mrf.alpha = 1;
grad_mrf.grad_prev = [];
grad_mrf.d_prev    = [];
grad_mrf.stepsize = 1;
grad_mrf.opt = opt(1);
grad_mrf.n = ntot;
stepsize = 1;
timings = toc;
while (~done)
	tic;
	loop = loop+1;
	if plotflag; p_old = pstack; end

	msg = sprintf('iteration: %d %s %d %s %d %s', loop,' (max ', iter,', stepsize', stepsize,')');
	fprintf([reverseStr, msg]);
    reverseStr = repmat(sprintf('\b'), 1, length(msg));
  	%disp('alpha post')
	alpha_post=mrf_gaussian_post(alpha,theta,y,sigma_opt,obs_ind,mask);

    if iscell(alpha_post) == 0
        alpha_post = {alpha_post};
    end
    for rep = 1:nrep
        alpha_post{rep} = alpha_post{rep} - max(alpha_post{rep},[],2)*ones(1,K);
    end

	update_mrf = 1;
	if loop>10
		if grad_mrf.stepsize < tol
			update_mrf = 0;
		end
		if mod(loop,10)==0
			update_mrf = 1;
		end
	end

	if update_mrf
    [z p da db H]=mrf_sim_grad(z,N,alpha_post,beta,gibbs_iter,mask,W,alpha,opt,a_mrf,b_mrf);
    [alpha beta grad_mrf] = mrf_take_step(alpha,beta,da,db,H,grad_mrf);
  end
  if loop<10
      theta=mrf_gaussian_est(p,y,sigma_opt,obs_ind);
  else
	  [theta, grad_gauss]=mrf_gaussian_grad(p,y,theta, grad_gauss,obs_ind);
	end

	%if loop>5; grad_gauss.alpha=1; end

  	% The stopping criterion:
  	%[grad_mrf.stepsize grad_gauss.stepsize]
  	stepsize = abs(grad_mrf.stepsize + grad_gauss.stepsize);
  	converged = (loop>1) & (stepsize<tol);
  	done = converged | (loop>=iter);
	timings = [timings;toc];
  	%% plot %%
  	if plotflag
  		pstack = []; for rep = 1:nrep; pstack = [pstack;p{rep}]; end
 		p_diff(loop,1) = max(abs((p_old(:)-pstack(:))));

  		alpha_history(loop+1,:) = exp(alpha);
  		beta_history(loop+1,:) = beta;
    	for k=1:K
    		%beta_history(k,:,loop+1) = beta;
      		mu_history(k,:,loop+1) = theta{k}.mu;
      		if sigma_opt(1) == 0
	  			Sigma_history(:,:,k,loop+1) = theta{k}.Sigma;
			elseif sigma_opt(1) == 1
				Sigma_history(:,k,loop+1) = diag(theta{k}.Sigma);
			else
				Sigma_history(k,loop+1) = theta{k}.Sigma(1);
			end
    	end
    	set(0,'CurrentFigure',1)
    	for rep = 1:nrep
    		if nrep < 6
				subplot(1,nrep,rep)
			else
				subplot(floor(sqrt(nrep)), ceil(nrep/floor(sqrt(nrep))), rep)
			end
	    	[tmp,cl] = max(p{rep},[],2);
    		clplot = zeros(m{rep}*n{rep},K); clplot(mc{rep},:) = p{rep};
    		imagesc(rgbimage(reshape(clplot,[m{rep} n{rep} K]),jet(K)))
    		axis image; axis off
    	end
    	if plotflag>1;
    		set(0,'CurrentFigure',2)
    		subplot(221); plot(alpha_history); title('alpha')
    		subplot(222); plot(squeeze(beta_history)); title('beta')
    		subplot(223); plot(squeeze(mu_history(:,1,:))'); title('\mu_1')
    		subplot(224);
    		if sigma_opt(1) == 0
	  			plot(squeeze(Sigma_history(1,1,:,:))');title('\Sigma_{11}')
			elseif sigma_opt(1) == 1
				plot(squeeze(Sigma_history(1,:,:))');title('\Sigma_{11}')
			else
				plot(squeeze(Sigma_history(:,:))');title('\Sigma_{11}')
			end
    	end
    	if plotflag>2;
    		set(0,'CurrentFigure',3)
    		subplot(211);
    		lim = p_diff(loop);
          	p_diff_n = histc(p_old(:)-pstack(:),...
          					 linspace(-lim,lim,100))*100/length(pstack(:));
          	bar(linspace(-lim,lim,100),p_diff_n.^0.25,'histc');
          	axis([-lim,lim,0,max(p_diff_n).^0.25])
    		subplot(212); plot(log10(p_diff(:,2)));
    	end
    	title(loop)
    	drawnow
	end
end
fprintf('\n');
%based on the final parameter values, classify image
disp('Classify image.')
[z p]=mrf_sim_grad(z,N,alpha_post,beta,class_iter,mask,W,alpha,opt,a_mrf,b_mrf);
cl = cell(1,nrep);
for rep = 1:nrep
  [tmp,cl{rep}] = max(p{rep},[],2);
  cl{rep} = reshape(cl{rep},[m{rep} n{rep}]);
  p{rep} = reshape(p{rep},[m{rep} n{rep} K]);
end

if nrep == 1
 p = p{1};
 cl = cl{1};
end


function [W,a,b] = build_W(N,mask)
sz = size(mask);

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
W = W(mask(:)==1,mask(:)==1);
