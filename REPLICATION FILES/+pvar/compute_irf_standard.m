function [irf_lower, irf_upper] = compute_irf_standard(est, nsteps, alpha)
% Analytic confidence bands using a normal approximation (±z * |IRF| / sqrt(N)).
% Faster than bootstrap but less reliable in small samples.
%
% INPUTS
%   est     output struct from estimate_panel_var or estimate_pvar_spillover_fod
%   nsteps  IRF horizon
%   alpha   significance level (e.g. 0.10)
%
% OUTPUT
%   irf_lower, irf_upper   [K x K x nsteps]

K = est.K; p = est.p;
F = pvar.compute_companion(est.B, K, p);
P = chol(est.Sigma_u, 'lower');

se_band = zeros(K, K, nsteps);
for h = 1:nsteps
    if h == 1
        Theta = eye(K);
    else
        Theta = (F^(h-1))(1:K, 1:K);
    end
    irf_h = Theta * P;
    se_band(:,:,h) = abs(irf_h) / sqrt(est.nobs);
end

z_crit = norminv(1 - alpha/2);
[irf, ~] = pvar.compute_oirf(est.B, est.Sigma_u, nsteps, K, p);
irf_lower = irf - z_crit * se_band;
irf_upper = irf + z_crit * se_band;
end
