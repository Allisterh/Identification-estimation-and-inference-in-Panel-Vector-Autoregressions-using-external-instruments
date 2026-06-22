function [irf_lower, irf_upper] = compute_irf_bootstrap(Y, id, time, S, X_exog, p, inst, nsteps, nboot, alpha)
% Panel bootstrap (resample units) for the normalised IRF of the first shock.
%
% INPUTS
%   Y, id, time, S, X_exog   same as estimate_pvar_spillover_fod
%   p        lag order
%   inst     instrument lag vector
%   nsteps   IRF horizon
%   nboot    number of bootstrap replications
%   alpha    significance level (e.g. 0.10 for 90% CI)
%
% OUTPUT
%   irf_lower, irf_upper   [K x 1 x nsteps]  at alpha/2 and 1-alpha/2 quantiles

[~, K]   = size(Y);
panels   = unique(id);
n_panels = length(panels);
irf_boot = zeros(K, 1, nsteps, nboot);

for b = 1:nboot
    if mod(b, 100) == 0
        fprintf('  Bootstrap %d / %d\n', b, nboot);
    end
    boot_idx = panels(randi(n_panels, n_panels, 1));
    Y_b=[]; id_b=[]; t_b=[]; S_b=[]; X_b=[];
    for i = 1:n_panels
        idx  = find(id == boot_idx(i));
        Y_b  = [Y_b;  Y(idx,:)];    %#ok<AGROW>
        id_b = [id_b; i*ones(length(idx),1)]; %#ok<AGROW>
        t_b  = [t_b;  time(idx)];   %#ok<AGROW>
        S_b  = [S_b;  S(idx)];      %#ok<AGROW>
        if ~isempty(X_exog)
            X_b = [X_b; X_exog(idx,:)]; %#ok<AGROW>
        end
    end
    try
        est_b = pvar.estimate_pvar_spillover_fod(Y_b, id_b, t_b, p, S_b, inst, X_b);
        [irf_b, ~] = pvar.compute_oirf_normalized(est_b.B, est_b.Sigma_u, nsteps, est_b.K, p);
        irf_boot(:,:,:,b) = irf_b;
    catch
        irf_boot(:,:,:,b) = nan;
    end
end

valid = ~isnan(squeeze(irf_boot(1,1,1,:)));
fprintf('Bootstrap complete: %d / %d successful\n', sum(valid), nboot);
irf_boot_clean = irf_boot(:,:,:,valid);
irf_lower = quantile(irf_boot_clean, alpha/2,   4);
irf_upper = quantile(irf_boot_clean, 1-alpha/2, 4);
end
