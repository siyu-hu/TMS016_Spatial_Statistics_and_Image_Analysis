function D = duplicatematrix(n)

% Creates the matrix D such that for a symmetric matrix one has
% D * vech*(A) = A
% where vech*(A) = [a_11 a_22 a_nn a_21 a 31 ... a_n-1n]
% thus first the diagonal entries, then the lower triangular entries
% comlunmn stacked

I = find(eye(n));
I2 = find(tril(ones(n,n),-1));
I3 = find(triu(ones(n,n), 1));
n_I = n;
n_I2 = n*(n-1)/2;
D = sparse([I;I2;I3],[1:n_I,n_I + (1:n_I2),n_I + (1:n_I2)]',1,n*n,n_I+n_I2);
