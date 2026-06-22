function X_out = transform_exog_fod(X_exog, id, time, p)
data   = sortrows([id, time, X_exog], [1 2]);
panels = unique(data(:,1));
X_out  = [];
for i = 1:length(panels)
    panel_data = data(data(:,1) == panels(i), :);
    Xn = panel_data(:, 3:end);
    Tn = size(Xn, 1);
    if Tn <= p+1, continue; end
    X_dep = Xn(p+1:end, :);
    X_out = [X_out; X_dep(1:end-1, :)]; %#ok<AGROW>
end
end
