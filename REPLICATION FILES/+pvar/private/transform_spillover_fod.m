function S_fod = transform_spillover_fod(S, id, time, p)
data   = sortrows([id, time, S], [1 2]);
panels = unique(data(:,1));
S_fod  = [];
for i = 1:length(panels)
    panel_data = data(data(:,1) == panels(i), :);
    Sn = panel_data(:, 3);
    Tn = length(Sn);
    if Tn <= p+1, continue; end
    S_dep   = Sn(p+1:end);
    T_dep   = length(S_dep);
    S_dep_fod = nan(T_dep-1, 1);
    for t = 1:T_dep-1
        n_future   = T_dep - t;
        future_mean = mean(S_dep(t+1:T_dep));
        weight     = sqrt(n_future / (n_future + 1));
        S_dep_fod(t) = weight * (S_dep(t) - future_mean);
    end
    S_fod = [S_fod; S_dep_fod]; %#ok<AGROW>
end
end
