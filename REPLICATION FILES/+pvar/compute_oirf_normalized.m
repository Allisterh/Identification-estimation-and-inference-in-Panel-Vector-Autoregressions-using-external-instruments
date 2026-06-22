function [irf, irf_cum] = compute_oirf_normalized(B, Sigma_u, nsteps, K, p)
% Computes the Cholesky IRF of the first shock, normalised to a unit shock.
%
% OUTPUT
%   irf      [K x 1 x nsteps]
%   irf_cum  [K x 1 x nsteps]  cumulative

P             = chol(Sigma_u, 'lower');
F             = pvar.compute_companion(B, K, p);
irf           = zeros(K, 1, nsteps);
irf_cum       = zeros(K, 1, nsteps);
initial_impact = P(:, 1);
normalization  = P(1, 1);

if abs(normalization) > 1e-10
    irf(:,1,1) = initial_impact / normalization;
else
    irf(:,1,1) = initial_impact;
end
irf_cum(:,1,1) = irf(:,1,1);

F_power = eye(size(F));
for h = 2:nsteps
    F_power    = F_power * F;
    Theta_h    = F_power(1:K, 1:K);
    if abs(normalization) > 1e-10
        irf(:,1,h) = (Theta_h * initial_impact) / normalization;
    else
        irf(:,1,h) = Theta_h * initial_impact;
    end
    irf_cum(:,1,h) = irf_cum(:,1,h-1) + irf(:,1,h);
end
end
