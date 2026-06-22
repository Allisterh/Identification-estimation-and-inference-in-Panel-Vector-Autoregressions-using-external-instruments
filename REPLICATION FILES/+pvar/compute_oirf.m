function [irf, irf_cum] = compute_oirf(B, Sigma_u, nsteps, K, p)
% Computes orthogonalised IRFs (Cholesky) for all K shocks.
%
% OUTPUT
%   irf      [K x K x nsteps]  impulse responses
%   irf_cum  [K x K x nsteps]  cumulative IRFs

P       = chol(Sigma_u, 'lower');
F       = pvar.compute_companion(B, K, p);
irf     = zeros(K, K, nsteps);
irf_cum = zeros(K, K, nsteps);
irf(:,:,1)     = P;
irf_cum(:,:,1) = P;
F_power = eye(size(F));
for h = 2:nsteps
    F_power        = F_power * F;
    Theta_h        = F_power(1:K, 1:K);
    irf(:,:,h)     = Theta_h * P;
    irf_cum(:,:,h) = irf_cum(:,:,h-1) + irf(:,:,h);
end
end
