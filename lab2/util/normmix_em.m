function [theta,prior,p,ll]=normmix_em(x,K,convergence,plotflag,prior0)

%Internal function that does ML estimation of a GMM using the EM algorithm.

tic;
% Parse input parameters
if nargin<3, convergence = []; end
if nargin<4, plotflag = []; end
if nargin<5, prior0 = []; end
if isempty(convergence), convergence = 200; end
if (length(convergence)<2), convergence(2) = 5e-4; end
if isempty(plotflag), plotflag = 0; end

%% plot %%
if plotflag
  if plotflag>=4, figure(3),clf, end
  if plotflag>=3, figure(2),clf, end
  figure(1),clf
end
%% plot %%

if iscell(K)  % Initial estimates were supplied.
  theta0 = K;
  K = length(theta0);
  if isempty(prior0), prior0 = ones(1,K)/K; end
else % No initial theta estimates available.
  if isempty(prior0)
    [theta0,prior0] = normmix_km(x,K,1);
  else
    theta0 = normmix_km(x,K,1);
  end
end

[n,d] = size(x);
prior = prior0;
theta = theta0;

% Perform the E-step:
p = E_step(x,theta,prior);

for k=1:K
  prior_history(1,k,1) = prior(k);
  mu_history(k,:,1) = theta{k}.mu;
  Sigma_history(:,:,k,1) = theta{k}.Sigma;
end

ond = ones(1,d);
onn = ones(n,1);

reverseStr = '';

timings = toc;
loop = 0;
done = 0;
while (~done)
  tic;
  loop = loop+1;
  %if loop>1
  %msg = sprintf('iteration: %d %s %d %s %d %s', loop,' (max ', convergence(1),', pdiff =', p_diff(loop-1),')');
  %else
  %msg = sprintf('iteration: %d %s %d %s', loop,' (max ', convergence(1),')');
  %end
  %fprintf([reverseStr, msg]);
  %reverseStr = repmat(sprintf('\b'), 1, length(msg));
  % For the stopping criterion:
  p_old = p;

  % The E-step:
  %%%%%%%%%%%%%%%%%%%
  if (loop>1) % Work not already done above.
    p = E_step(x,theta,prior);
  end

  % The M-step:
  % New pi-estimates:
  prior = sum(p,1)/n;

  % New mu and Sigma estimates:
  for k=1:K
  	pk = p(:,k);
  	ps = sum(pk);
    theta{k}.mu = sum(bsxfun(@times,x,pk))/ps;
    y = bsxfun(@minus,x,theta{k}.mu);
    theta{k}.Sigma = y'*bsxfun(@times,y,pk)/ps;
  end

  % The stopping criterion:
  p_diff(loop) = max(abs((p_old(:)-p(:))));
  converged = (loop>1) & (p_diff(loop)<convergence(2));
  done = converged | (loop>=convergence(1));
  timings = [timings;toc];
  %% plot %%
  if plotflag
    for k=1:K
      prior_history(1,k,loop+1) = prior(k);
      mu_history(k,:,loop+1) = theta{k}.mu;
      Sigma_history(:,:,k,loop+1) = theta{k}.Sigma;
    end

    figure(1)
    subplot(311)
    plot(0:loop,squeeze(prior_history)')
    title('\pi')
    subplot(312)
    plot(0:loop,squeeze(mu_history(:,1,:))')
    title('\mu_1')
    subplot(313)
    plot(0:loop,squeeze(Sigma_history(1,1,:,:))')
    title('\Sigma_{11}')
    if (plotflag>=2)
        if (loop>1)
          figure(2)
          subplot(211)
          semilogy(1:loop,p_diff)
          title('Maximal p-difference')
          subplot(212)
          lim = p_diff(loop);
          p_diff_n = histc(p_old(:)-p(:),...
                           linspace(-lim,lim,100))*100/length(p(:));
          bar(linspace(-lim,lim,100),p_diff_n.^0.25,'histc');
          axis([-lim,lim,0,max(p_diff_n).^0.25])
          title('p-difference histogram')
        end
    end
  drawnow
  end
end
%fprintf('\n');
%calculate log-like
pp = zeros(n,1);

spd = 1;
for k=1:K
  if min(eig(theta{k}.Sigma))>0
    pp = pp + prior(k)*mvnpdf(x,theta{k}.mu,theta{k}.Sigma);
  else
    spd = 0;
  end
end
if spd == 1
  ll = sum(log(pp));
else
  ll = -Inf;
end



function p=E_step(x,theta,prior)
[n d] = size(x);
K = length(theta);
p = zeros(n,K);
for k=1:K
  y = bsxfun(@minus,x,theta{k}.mu);
  p(:,k) = exp(-0.5*sum( ((y*inv(theta{k}.Sigma)).*y) ,2) ) / ...
	   ((2*pi)^(d/2)*det(theta{k}.Sigma)^0.5);
end
p = p*diag(prior);
p = p./repmat(sum(p,2),[1,K]);


function [theta0,prior0,p,ll]=normmix_km(x,K,maxiter)
% Parse input parameters
if nargin<3, maxiter = []; end
if nargin<4, plotflag = []; end
if isempty(maxiter), maxiter = 1; end

% Use some steps of K-means for rough initial estimates:
[cl,theta0] = kmeans(x,K,maxiter);
% pi-estimates (use uniform pi):
prior0 = ones(1,K)/K;
% Sigma-estimates (use common, isotropic Sigma):
n = size(x,1);
d = size(x,2);
Sigma = 0;
for k=1:K
  y = x(cl==k,:)-repmat(theta0{k}.mu,[sum(cl==k),1]);
  Sigma = Sigma + sum(sum(y.*y,1),2);
end
Sigma = eye(d)*Sigma/n/d;
for k=1:K
  theta0{k}.Sigma = Sigma;
end

%calculate log-like
pp = zeros(n,1);

spd = 1;
for k=1:K
  if min(eig(theta0{k}.Sigma))>0
    pp = pp + (1/K)*mvnpdf(x,theta0{k}.mu,theta0{k}.Sigma);
  else
    spd = 0;
  end
end
if spd == 1
  ll = sum(log(pp));
else
  ll = -Inf;
end

%calculate probabilities
p = zeros(n,K);
for k=1:K
  y = bsxfun(@minus,x,theta0{k}.mu);
  p(:,k) = exp(-0.5*sum( ((y*inv(theta0{k}.Sigma)).*y) ,2) ) / ...
	   ((2*pi)^(d/2)*det(theta0{k}.Sigma)^0.5);
end
p = p*diag(ones(1,K)/K);
p = p./repmat(sum(p,2),[1,K]);



function [cl,theta]=kmeans(x,K,maxiter)

if nargin<3, maxiter = []; end
if nargin<4, plotflag = []; end
if isempty(maxiter), maxiter = inf; end

[n,d] = size(x);

% Find unique starting points:
if (n<K), error('Not enough data!'); end
start_idx = ceil(rand(K,1)*n);
mu = x(start_idx,:);
while (size(unique(mu,'rows'),1)<K)
  start_idx = ceil(rand(K,1)*n);
  mu = x(start_idx,:);
end

cl_old = ones(n,1);
cl = zeros(n,1);

loop = 0;
mu_history(:,:,loop+1) = mu;
while (~all(cl==cl_old)) & (loop<maxiter)
  loop = loop+1;
  cl_old = cl;
  % Squared distances:
  for i=1:K
    %dist2(:,i) = sum((x-ones(n,1)*mu(i,:)).^2,2);
    dist2(:,i) = sum((bsxfun(@minus,x,mu(i,:))).^2,2);
  end
  % Classify:
  [tmp,cl] = min(dist2,[],2);
  % New mu estimates
  for i=1:K
    mu(i,:) = mean(x(cl==i,:),1);
  end
end

% Collect output:
for k=1:K
  theta{k}.mu = mu(k,:);
end
