function [lb, ub, irf_out, WHat] = ar_ci_raw(B, AL, Xtot, Z, eta, T, n, p, horizons, nvar, scale, critval, newey_lags)
% Low-level MSW Anderson-Rubin confidence intervals.
% Direct equivalent of the original AR_CI function, using +pvar private helpers.
%
% INPUTS
%   B          [n x 1]  structural vector  (use Gamma = (eta*Z)/T for unit normalisation)
%   AL         [n x np] VAR coefficient matrix  (= est.B')
%   Xtot       [T x K]  RHS variables with constant (= est.X from estimate_panel_var)
%   Z          [T x 1]  instrument aligned with Xtot / eta
%   eta        [n x T]  VAR residuals  (= est.resid')
%   T          scalar   number of observations
%   n          scalar   number of variables
%   p          scalar   VAR lag order
%   horizons   scalar   number of horizons (excluding impact)
%   nvar       scalar   normalising variable index
%   scale      scalar   shock scale  (use 1 for unit normalisation)
%   critval    scalar   chi2 critical value  (= norminv(1-(1-conf)/2)^2)
%   newey_lags scalar   Newey-West bandwidth
%
% OUTPUT
%   lb, ub     [n x horizons+1]  CI bounds
%   irf_out    [n x horizons+1]  point IRF estimates
%   WHat       asymptotic covariance of [vec(A)', Gamma']'

e = eye(n);

% MA representation: [I, C_1, ..., C_H]  →  (n x n x H+1)
Caux = [eye(n), ma_rep(AL, p, horizons)];
C    = reshape(Caux, [n, n, horizons+1]);

% G matrices (derivatives of MA coefficients wrt vec(A))
[G, ~] = g_matrices(AL, ma_rep(AL, p, horizons), p, horizons, n);

% Asymptotic covariance of (vec(A), Gamma)
[~, WHat, ~] = cov_ahat_gamma(p, Xtot, Z, eta, newey_lags);

W1  = WHat(1:n^2*p,       1:n^2*p);
W12 = WHat(1:n^2*p,       n^2*p+1:end);
W2  = WHat(n^2*p+1:end,   n^2*p+1:end);

% MSW quadratic formula
lb        = zeros(n, horizons+1);
ub        = zeros(n, horizons+1);
irf_out   = zeros(n, horizons+1);

for j = 1:n
    for ih = 1:horizons+1
        a = T*B(nvar)^2 - critval*W2(nvar,nvar);

        b = -2*T*scale*(e(:,j)'*C(:,:,ih)*B)*B(nvar) ...
            + 2*critval*scale*(kron(B',e(:,j)')*G(:,:,ih)*W12(:,nvar)) ...
            + 2*critval*scale*e(:,j)'*C(:,:,ih)*W2(:,nvar);

        c = (T^.5*scale*e(:,j)'*C(:,:,ih)*B)^2 ...
            - critval*scale^2*(kron(B',e(:,j)')*G(:,:,ih))*W1*(kron(B',e(:,j)')*G(:,:,ih))' ...
            - 2*critval*scale^2*(kron(B',e(:,j)')*G(:,:,ih))*W12*C(:,:,ih)'*e(:,j) ...
            - critval*scale^2*e(:,j)'*C(:,:,ih)*W2*C(:,:,ih)'*e(:,j);

        Delta = b^2 - 4*a*c;

        if     a > 0 && Delta > 0
            lb(j,ih) = (-b - sqrt(Delta)) / (2*a);
            ub(j,ih) = (-b + sqrt(Delta)) / (2*a);
        elseif a < 0 && Delta > 0
            lb(j,ih) = (-b + sqrt(Delta)) / (2*a);
            ub(j,ih) = (-b - sqrt(Delta)) / (2*a);
        elseif a > 0 && Delta < 0
            lb(j,ih) = NaN;   ub(j,ih) = NaN;
        else
            lb(j,ih) = -Inf;  ub(j,ih) = Inf;
        end

        irf_out(j,ih) = scale * e(:,j)' * C(:,:,ih) * B / B(nvar);
    end
end

% Fix impact bound of normalising variable
lb(nvar,1) = scale;
ub(nvar,1) = scale;
end
