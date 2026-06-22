function [lb, ub, lb_cum, ub_cum, casedummy, casedummy_cum] = compute_msw_ci(est, iv, nsteps, confidence)
% Computes Montiel-Olea–Stock–Watson (2021) weak-IV robust confidence sets
% for SVAR-IV impulse responses, for both non-cumulative and cumulative IRFs.
%
% INPUTS
%   est         struct from estimate_panel_var
%   iv          struct from identify_iv
%   nsteps      scalar  horizon (excluding impact)
%   confidence  scalar  confidence level (default: 0.95)
%
% OUTPUT
%   lb, ub           [K x nsteps+1]  non-cumulative CI bounds
%   lb_cum, ub_cum   [K x nsteps+1]  cumulative CI bounds
%   casedummy        [K x nsteps+1]  case classification (1-4)
%   casedummy_cum    [K x nsteps+1]  cumulative case classification

if nargin < 4, confidence = 0.95; end

n      = iv.K;
p      = iv.p;
AL     = iv.AL;
B      = iv.B;
scale  = iv.scale;
nvar   = iv.nvar;
T      = iv.T;
WHat   = iv.WHat;

critval = norminv(1 - (1-confidence)/2)^2;

% Partition WHat into blocks for vec(A) and Gamma
W1  = WHat(1:(n^2)*p,       1:(n^2)*p);
W12 = WHat(1:(n^2)*p,       (n^2)*p+1:end);
W2  = WHat((n^2)*p+1:end,   (n^2)*p+1:end);

% MA representation (n x n x nsteps+1)
Caux  = [eye(n), ma_rep(AL, p, nsteps)];
C     = reshape(Caux, [n, n, nsteps+1]);
Ccum  = cumsum(C, 3);

% G matrices: derivatives of vec(C) wrt vec(A)
[G, Gcum] = g_matrices(AL, ma_rep(AL, p, nsteps), p, nsteps, n);

e = eye(n);

% --- Non-cumulative MSW CI ---
[lb, ub, casedummy] = msw_bounds(C, G, B, W1, W12, W2, e, n, nsteps, T, scale, nvar, critval);

% --- Cumulative MSW CI ---
[lb_cum, ub_cum, casedummy_cum] = msw_bounds(Ccum, Gcum, B, W1, W12, W2, e, n, nsteps, T, scale, nvar, critval);

% Set impact bound to scale for normalising variable
lb(nvar, 1)     = scale;
ub(nvar, 1)     = scale;
lb_cum(nvar, 1) = scale;
ub_cum(nvar, 1) = scale;
end

% ─────────────────────────────────────────────────────────────────────────────
function [lb, ub, casedummy] = msw_bounds(C, G, B, W1, W12, W2, e, n, nsteps, T, scale, nvar, critval)
% Applies the MSW quadratic formula for each variable j and horizon h.
% Returns lb, ub, casedummy — each (n x nsteps+1).

lb        = zeros(n, nsteps+1);
ub        = zeros(n, nsteps+1);
casedummy = zeros(n, nsteps+1);

for j = 1:n
    for ih = 1:nsteps+1
        a = T * B(nvar)^2  -  critval * W2(nvar, nvar);

        b = -2*T*scale * (e(:,j)'*C(:,:,ih)*B) * B(nvar) ...
            + 2*critval*scale * (kron(B', e(:,j)') * G(:,:,ih) * W12(:,nvar)) ...
            + 2*critval*scale * e(:,j)' * C(:,:,ih) * W2(:,nvar);

        c = (T^.5 * scale * e(:,j)' * C(:,:,ih) * B)^2 ...
            - critval*scale^2 * (kron(B',e(:,j)') * G(:,:,ih)) * W1 * (kron(B',e(:,j)') * G(:,:,ih))' ...
            - 2*critval*scale^2 * (kron(B',e(:,j)') * G(:,:,ih)) * W12 * C(:,:,ih)' * e(:,j) ...
            - critval*scale^2 * e(:,j)' * C(:,:,ih) * W2 * C(:,:,ih)' * e(:,j);

        Delta = b^2 - 4*a*c;

        if     a > 0 && Delta > 0
            casedummy(j,ih) = 1;
            lb(j,ih) = (-b - sqrt(Delta)) / (2*a);
            ub(j,ih) = (-b + sqrt(Delta)) / (2*a);
        elseif a < 0 && Delta > 0
            casedummy(j,ih) = 2;
            lb(j,ih) = (-b + sqrt(Delta)) / (2*a);
            ub(j,ih) = (-b - sqrt(Delta)) / (2*a);
        elseif a > 0 && Delta < 0
            casedummy(j,ih) = 3;
            lb(j,ih) = NaN;
            ub(j,ih) = NaN;
        else
            casedummy(j,ih) = 4;
            lb(j,ih) = -Inf;
            ub(j,ih) =  Inf;
        end
    end
end
end
