function [pars_new, grad_gauss] = mrf_gaussian_grad(Mz,y,pars, grad_gauss,obs_ind)

% Internal function that computes the gradient of the Gaussian parameters.

if iscell(y);
	nrep = length(y);
else
	nrep = 1;
	y  = {y};
	W = {Mz};
end
K = size(Mz{1},2);
d = size(y{1},2);
pars_new = cell(1,K);
D = duplicatematrix(d);
tmp = ones(d,d);
ttmp = triu(tmp);
stepsize = 0;

for j=1:K
	Qd = inv(pars{j}.Sigma);
  	mu = pars{j}.mu;
  	nabla = 0;
  	H = 0;
  	pik = 0;
	for rep = 1:nrep
  		zk = Mz{rep}(obs_ind{rep},j);
	  	sk = sum(zk);
  		pik = pik + sk;
  		yc = bsxfun(@minus,y{rep},mu);
  		yc_z = bsxfun(@times,zk,yc);

   	 	yc_zs = sum(yc_z,1);
   	 	dmu   = yc_zs * Qd;
  		d2mu  = -sk*Qd;
   	 	dQ    = -(yc_z'*yc)/2;%-(yc_z/2)' * yc;
    	dQ    =  dQ + (sk/2 * pars{j}.Sigma);
   		dQ = dQ(:)' * D;
	    ddQ   = -(sk/2) * D' * kron(pars{j}.Sigma ,pars{j}.Sigma) * D;
    	dmuQ  = -  kron(yc_zs , eye(d)) * D;

 	 	H = H+[d2mu dmuQ;dmuQ' ddQ];
  		nabla = nabla + [dmu';dQ(:)];
  	end

	%if the condition number is low, add "a" so that the condition number
	%becomes "b".
    H = (H+H')/2;
    if sum(isnan(H(:)))> 0 || sum(isinf(H(:))) > 0
		H
	end
	e = eig(H);
	b = 1e-13;
  	if e(end)/e(1) < b
        H = diag(diag(H));
		if max(diag(H))>0
  			H = H - (1+1e-6)*max(diag(H));
		end
	end

	dH = diag(1./diag(H));
    step = - (dH*H)\ (dH * nabla);

    if ~isempty(grad_gauss.grad_prev{j})
       	beta_k = norm(nabla)^2 / norm(grad_gauss.grad_prev{j})^2;
       	if beta_k > 1
           	beta_k = 0;
       	end
       	step = step + beta_k * grad_gauss.d_prev{j};
    end
    step = grad_gauss.alpha*step;
  	Qd_old = Qd;
    k = 0;
    p = 1;
    while p > 0
        Qd(:) = Qd_old(:);
        Qd(:) = Qd(:) + 10^(-k)*(D*step(d+1:end));
        Qd    = (Qd + Qd')/2;
        k = k +1;
        [~ , p] = chol(Qd);
    end
  	pars_new{j}.mu = mu + step(1:d)';
  	pars_new{j}.Sigma = inv(Qd);

    grad_gauss.grad_prev{j} = nabla;
    grad_gauss.d_prev{j} = step;
    stepsize = stepsize + nabla'*step/pik;
end
grad_gauss.stepsize = stepsize;
if stepsize < 10
	if stepsize < 1
		grad_gauss.alpha = 0.95*grad_gauss.alpha;
	else
		grad_gauss.alpha = 0.99*grad_gauss.alpha;
	end
end



