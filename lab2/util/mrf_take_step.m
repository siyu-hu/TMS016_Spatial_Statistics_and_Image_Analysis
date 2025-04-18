function [alpha beta grad_mrf] = mrf_take_step(alpha,beta,da,db,H,grad_mrf)

%Internal function that updates the MRF parameter given the gradient.

if grad_mrf.opt == 1
	nabla = [da(2:end) db]';
	H = H(2:end,2:end);
else %no alpha parameters
	nabla = db';
end

step = H\nabla;

if ~isempty(grad_mrf.grad_prev)
	beta_k = norm(nabla)^2 / norm(grad_mrf.grad_prev)^2;
	if beta_k > 1
		beta_k = 0;
    end
    step = step + beta_k * grad_mrf.d_prev;
end

step = grad_mrf.alpha*step;

if grad_mrf.opt == 1
    if length(db) == 1
        Pnew = [alpha(2:end)';beta(1)] + step;
    else
        Pnew = [alpha(2:end)';beta'] + step;
    end
    alpha = [0;Pnew(1:length(alpha)-1)]';
	beta = Pnew(length(alpha):end)';
else
	beta = (beta + step');
end

grad_mrf.grad_prev = nabla;
grad_mrf.d_prev = step;
grad_mrf.stepsize = nabla'*step/grad_mrf.n;

if grad_mrf.stepsize < 10
	if grad_mrf.stepsize < 1
		grad_mrf.alpha = 0.95*grad_mrf.alpha;
	else
		grad_mrf.alpha = 0.99*grad_mrf.alpha;
	end
end