function [irf_lower_cum, irf_upper_cum] = compute_irf_bootstrap_cumulative(Y, id, time, S, X_exog, p, inst, nsteps, nboot, alpha)
% Like compute_irf_bootstrap but returns CIs on cumulative IRFs.
%
% Useful when the response of interest is the cumulative (permanent) effect.
% X_exog rows are time-demeaned within each bootstrap draw.

[~, K]   = size(Y);
panels   = unique(id);
n_panels = length(panels);
irf_boot_cum = zeros(K, 1, nsteps, nboot);

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
    % time-demean Y, S, X within bootstrap draw
    for tv = unique(t_b)'
        tidx = (t_b == tv);
        if sum(tidx) > 1
            Y_b(tidx,:)  = Y_b(tidx,:)  - mean(Y_b(tidx,:),  1);
            S_b(tidx)    = S_b(tidx)    - mean(S_b(tidx));
            if ~isempty(X_b)
                X_b(tidx,:) = X_b(tidx,:) - mean(X_b(tidx,:), 1);
            end
        end
    end
    try
        est_b = pvar.estimate_pvar_spillover_fod(Y_b, id_b, t_b, p, S_b, inst, X_b);
        [irf_b, ~] = pvar.compute_oirf_normalized(est_b.B, est_b.Sigma_u, nsteps, est_b.K, p);
        irf_boot_cum(:,:,:,b) = cumsum(irf_b, 3);
    catch
        irf_boot_cum(:,:,:,b) = nan;
    end
end

valid = ~isnan(squeeze(irf_boot_cum(1,1,1,:)));
fprintf('Bootstrap complete: %d / %d successful\n', sum(valid), nboot);
c = irf_boot_cum(:,:,:,valid);
irf_lower_cum = quantile(c, alpha/2,   4);
irf_upper_cum = quantile(c, 1-alpha/2, 4);
end
