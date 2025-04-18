function vechs_Q = vechs(Q)

% Creates the vector vech*(Q) such that for a symmetric matrix one has
% D * vech*(Q) = Q
% where vech*(Q) = [q_11 q_22 q_nn q_21 q 31 ... q_n-1n]
% thus first the diagonal entries, then the lower triangular entries
% comlunmn stacked

n = length(Q);
vechs_Q = [Q(logical(eye(n))); Q(logical(tril(ones(n),-1)))];