function [WHataux, WHat, V] = cov_ahat_gamma(p, Xtot, Z, eta, lags)
% Estimates the asymptotic covariance of [vec(A)', vech(Sigma)', Gamma']'.
%
% Inputs:
%   p     - VAR lag order
%   Xtot  - (T x np+m) right-hand side variables (post-transformation, with constant)
%   Z     - (T x k) external instrument aligned with Xtot
%   eta   - (n x T) VAR residuals
%   lags  - Newey-West bandwidth
%
% Outputs:
%   WHataux - asy. var of [vec(A)', vech(Sigma)', Gamma']'
%   WHat    - asy. var of [vec(A)', Gamma']'  (drops Sigma block)
%   V       - selector s.t. vech(Sigma) = V vec(Sigma)

n       = size(eta, 1);
k       = size(Z, 2);
m       = size(Xtot, 2) - n*p;
T1aux   = size(eta, 2);
T2aux   = n*m + n^2*p + n^2 + n*k;

etaaux     = reshape(eta, [n,1,T1aux]);
matagg     = [Xtot, eta', Z]';
mataggaux  = permute(reshape(matagg, [size(matagg,1),1,T1aux]), [2,1,3]);
auxeta     = bsxfun(@plus, bsxfun(@times,etaaux,mataggaux), ...
                   -mean(bsxfun(@times,etaaux,mataggaux), 3));
vecAss1    = reshape(auxeta, [T2aux,1,T1aux]);
AuxHAC2    = reshape(vecAss1, [T2aux, T1aux])';
WhatAss1   = nw_hac(AuxHAC2, lags);

% Selector matrix V: vech(Sigma) = V * vec(Sigma)
I = eye(n);
V = kron(I(1,:), I);
for i_vars = 2:n
    V = [V; kron(I(i_vars,:), I(i_vars:end,:))]; %#ok<AGROW>
end

Q1   = Xtot' * Xtot / T1aux;
Q2   = Z' * Xtot / T1aux;
Shat = [kron([zeros(n*p,m), eye(n*p)] * (Q1^(-1)), eye(n)), zeros(n^2*p, n^2 + k*n); ...
        zeros(n*(n+1)/2, n^2*p + n*m), V, zeros(n*(n+1)/2, k*n); ...
        -kron(Q2*(Q1^(-1)), eye(n)), zeros(k*n, n^2), eye(k*n)];

WHataux = Shat * WhatAss1 * Shat';

% Extract the vec(A) and Gamma blocks (drop vech(Sigma))
idx_A     = 1:(n^2)*p;
idx_Gamma = (n^2)*p + n*(n+1)/2 + 1:size(WHataux,1);
WHat = [WHataux(idx_A,idx_A),       WHataux(idx_A,idx_Gamma); ...
        WHataux(idx_Gamma,idx_A),   WHataux(idx_Gamma,idx_Gamma)];
end
