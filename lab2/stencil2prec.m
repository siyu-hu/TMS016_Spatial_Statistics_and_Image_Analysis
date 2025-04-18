function Q=stencil2prec(sz,q)
% Q = stencil2prec(sz,q) constructs the precision matrix Q for a GMRF on a
% regular m x n grid based on the stencil q. sz = [m n] specifies the size
% of the image. 
%
% Example:
%  Q = stencil2prec([100 100],[0 -1 0;-1 5 -1; 0 -1 0]);
%
% Note that the function does not modify the precision at the grid borders.
% This is equivalent to assuming that the field is equal to its expected
% value outside the grid.
%
% David Bolin (david.bolin@chalmers.se) 2018.

II = [];
KK = [];
JJ_I = [];
JJ_J = [];

[I,J] = ndgrid(1:sz(1),1:sz(2));
I = I(:); J = J(:);
for i=1:size(q,1)
  for j=1:size(q,2)
    if (q(i,j) ~= 0)
      II = [II;I+sz(1)*(J-1)];
      JJ_I = [JJ_I;I+i-(size(q,1)+1)/2];
      JJ_J = [JJ_J;J+j-(size(q,2)+1)/2];
      KK = [KK; q(i,j)*ones(prod(sz),1)];
    end
  end
end
JJ = JJ_I+sz(1)*(JJ_J-1);
ok = (JJ_I>=1) & (JJ_I<=sz(1)) & (JJ_J>=1) & (JJ_J<=sz(2));
II(~ok) = []; JJ(~ok) = []; KK(~ok) = [];
Q = sparse(II,JJ,KK,prod(sz),prod(sz));