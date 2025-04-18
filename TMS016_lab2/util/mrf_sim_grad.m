function [z Mz da db H]=mrf_sim_grad(z,N,alpha,beta,iter,mask,W,alpha0,opt,a,b)

%Internal function that estimates the gradient of the MRF parameter using
% Gibbs sampling.

if nargin < 9; opt = []; end
if nargin < 8; alpha0 = []; end
if nargin < 7; error('must supply W'); end
if isempty(opt) && nargout > 2
	error('must supply opt if calculating gradients')
end
if isempty(alpha0) && nargout > 2
	error('must supply alpha if calculating gradients')
end
if isempty(opt); opt = [0 0 0]; end
if isempty(alpha0); alpha0 = alpha{1}; end

if iscell(z);
	nrep = length(z);
	if ~iscell(mask); mask = {mask}; end
	if ~iscell(W); W = {W}; end
	if ~iscell(alpha); alpha = {alpha}; end
else
	nrep = 1;
	z  = {z};
	if ~iscell(mask); mask = {mask}; end
	if ~iscell(W); W = {W}; end
	if ~iscell(alpha); alpha = {alpha}; end
end

for rep = 1:nrep
  [m{rep} n{rep}] = size(mask{rep});

	cmask{rep} = mask{rep}(:);
	[mn{rep},K] = size(z{rep});

	alpha{rep} = make_K_im(alpha{rep},[mn{rep},K]);
	a0{rep} = make_K_im(exp(alpha0),[mn{rep} K]);
	b0{rep} = make_K_im(beta,[mn{rep} K]);
	Mz{rep} = zeros(mn{rep},K);
	KK{rep} = repmat(reshape(1:K,[1,K]),[mn{rep},1]);
end

on = ones(1,K);

da = 0;
db = 0;
H = 0;
if opt(1) == 1
	da = zeros(1,K);
	if opt(2)==1 %alpha vector and beta vector
		H = zeros(2*K,2*K);
		db = zeros(1,K);
	else %alpha vector and common beta
		H = zeros(K+1,K+1);
	end
else
	if opt(2) == 1 %no alpha and beta vector
		H = zeros(K,K);
		db = zeros(1,K);
	end
end

ij_ = [kron(1:a,ones(1,b)); kron(ones(1,a),1:b)];
ijm = cell(nrep,1);
for rep = 1:nrep
  ijm{rep} = false(m{rep}*n{rep},a*b);
	ealpha{rep} = exp(alpha{rep});
	%precalculate masks:
  for k=1:a*b
    Ind = false(m{rep},n{rep});
    Ind(ij_(1,k):a:m{rep},ij_(2,k):a:n{rep}) = true;
    Ind = Ind(:);
    Ind = Ind(cmask{rep});
    ijm{rep}(:,k) = Ind(:);
end
end






Wij = cell(nrep,a*b);
b0ij = cell(nrep,a*b);
eaij = cell(nrep,a*b);
ea0ij = cell(nrep,a*b);
KKij = cell(nrep,a*b);
ijmf = cell(nrep,a*b);
Mzc = cell(nrep);
for rep = 1:nrep
	Mzc{rep} = cell(1,2);
	for k=1:a*b
		Wij{rep,k} = W{rep}(ijm{rep}(:,k),:);
		b0ij{rep,k} = b0{rep}(ijm{rep}(:,k),:);
		eaij{rep,k} = ealpha{rep}(ijm{rep}(:,k),:);
		ea0ij{rep,k} = a0{rep}(ijm{rep}(:,k),:);
		KKij{rep,k} = KK{rep}(ijm{rep}(:,k),:);
		Mzc{rep}{k} = zeros(sum(ijm{rep}(:,k)),K);
	end
end

for loop=1:iter
	for rep=1:nrep
 		for k=randperm(a*b)
  		ijmask = ijm{rep}(:,k);
   	 	f = Wij{rep,k}*z{rep};
    	efb = exp(f.*b0ij{rep,k});
			Mz_cond = eaij{rep,k}.*efb;
			Msum = sum(Mz_cond,2);
			Mz_cond = bsxfun(@rdivide,Mz_cond, Msum);
			if nargout >1
				Mzc{rep}{k} = Mzc{rep}{k}+Mz_cond;
			end
			if nargout > 2
    		% Calculate gradients
    		expterm = ea0ij{rep,k}.*efb;
				expsum = sum(expterm,2);
    		expterm = bsxfun(@rdivide,expterm,expsum);
    		esum = sum(expterm);
    		fe = f.*expterm;
    		efsum = sum(fe);
    		zij = z{rep}(ijmask,:);

    		db = db + sum(f.*zij)- efsum;
				da = da + sum(zij) - esum;
    		%calculate hessian
    		if opt(1) ==1
	    		d2a = -diag(esum) + expterm'*expterm;
    		else
    			d2a = [];
    		end

    		if opt(2)==1
 					d2b = -diag(sum(f.*fe)) + fe'*fe;
 					if opt(1) == 1
     				dab = -diag(efsum) + fe'*expterm;
     			else
     				dab = [];
     			end
   			else
   				efsum2 = sum(fe,2);
   				d2b = sum(efsum2.^2 - sum(f.*fe,2));
   				if opt(1) == 1
	   				dab = -efsum + sum(bsxfun(@times,expterm,efsum2));
   				else
   					dab = [];
   				end
   			end
   			H = H - [d2a dab';dab d2b];
			end
   		e = rand(sum(ijmask),1);
   		x = 1 + sum( bsxfun(@lt,cumsum(Mz_cond,2),e), 2);
		  z{rep}(ijmask,:) = bsxfun(@eq,x,KKij{rep,k})*1;
		end
	end
end

Mz = cell(nrep,1);
for rep = 1:nrep
	Mz{rep} = zeros(mn{rep},K);
	for k=1:a*b
		Mz{rep}(ijm{rep}(:,k),:) = Mzc{rep}{k};
	end
end

if (nargout>1)
	db = db/iter;
	if opt(1) == 1
	  	da = da/iter;
  	else
  		da = [];
  	end
  	H = H/iter;
	for rep = 1:nrep
		Mz{rep} = Mz{rep}/iter;
  	end
    if opt(2)==0
  		db = sum(db);
  	end
end

function im=make_K_im(v,sz)
if (length(v)==1) %v = 1x1
  im = v*ones(sz);
elseif ((size(v,1)==sz(1)) && (size(v,2)==sz(2))) %v=mn x K, do nothing
    im = v;
else %v = 1xK
  im = repmat(v,[sz(1),1]);
end
