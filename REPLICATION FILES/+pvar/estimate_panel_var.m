function results = estimate_panel_var(Y, id, year, p, varargin)
% Estimates a panel VAR via GMM with FOD or first-difference transformation.
%
% INPUTS
%   Y            [N x K]  endogenous variables (panel-stacked, one row per obs)
%   id           [N x 1]  unit identifier
%   year         [N x 1]  time identifier
%   p            scalar   lag order
%
% NAME-VALUE OPTIONS
%   'transformation'  'fod' (default) | 'fd'
%   'instlags'        vector of instrument lag indices (default: empty = OLS)
%   'varNames'        cell array of K names
%
% OUTPUT  struct with fields:
%   .B            [K*p x K]  VAR coefficients
%   .Sigma_u      [K x K]    residual covariance
%   .resid                   residuals
%   .MAIC .MBIC .MQIC        lag selection criteria
%   .J_stat .J_pvalue .J_df  Hansen J test
%   .nobs .npanels

p_input = inputParser;
addRequired(p_input, 'Y');
addRequired(p_input, 'id');
addRequired(p_input, 'year');
addRequired(p_input, 'p');
addParameter(p_input, 'transformation', 'fod');
addParameter(p_input, 'varNames', []);
addParameter(p_input, 'instlags', []);
addParameter(p_input, 'display', false);
parse(p_input, Y, id, year, p, varargin{:});

transformation = lower(p_input.Results.transformation);
varNames       = p_input.Results.varNames;
instlags       = p_input.Results.instlags;

[~, K] = size(Y);
if isempty(varNames)
    varNames = arrayfun(@(i) sprintf('Var%d', i), 1:K, 'UniformOutput', false);
end

data    = sortrows([id, year, Y], [1 2]);
panels  = unique(data(:,1));
npanels = length(panels);

Y_dep = []; X_reg = []; Z_instr = [];
id_dep_out = []; time_dep_out = [];
npanels_used = 0;

for i_panel = 1:npanels
    panel_data  = data(data(:,1) == panels(i_panel), :);
    Yn          = panel_data(:, 3:end);
    Tn          = size(Yn, 1);
    if Tn <= p+1, continue; end

    panel_times_i = panel_data(:, 2);

    X_i = [];
    for lag = 1:p
        X_i = [X_i, Yn(p+1-lag:end-lag, :)]; %#ok<AGROW>
    end
    Y_dep_i = Yn(p+1:end, :);
    time_i  = panel_times_i(p+1:end);

    if strcmp(transformation, 'fod')
        Y_dep_i = apply_fod(Y_dep_i);
        X_i     = apply_fod(X_i);
        time_i  = time_i(1:end-1);
    else
        Y_dep_i = diff(Y_dep_i);
        X_i     = diff(X_i);
        time_i  = time_i(2:end);
    end

    minRows = min(size(Y_dep_i,1), size(X_i,1));
    Y_dep_i = Y_dep_i(1:minRows, :);
    X_i     = X_i(1:minRows, :);
    time_i  = time_i(1:minRows);

    Z_i = nan(minRows, length(instlags) * K);
    for t = 1:minRows
        col = 1;
        for lag = instlags
            src_row = p + t - lag;
            if src_row >= 1 && src_row <= Tn
                Z_i(t, col:col+K-1) = Yn(src_row, :);
            end
            col = col + K;
        end
    end

    nanRows = any(isnan(Y_dep_i),2) | any(isnan(X_i),2) | any(isnan(Z_i),2);
    Y_dep_i(nanRows,:) = []; X_i(nanRows,:) = [];
    Z_i(nanRows,:)     = []; time_i(nanRows) = [];
    if isempty(Y_dep_i), continue; end

    Y_dep        = [Y_dep; Y_dep_i];           %#ok<AGROW>
    X_reg        = [X_reg; X_i];               %#ok<AGROW>
    Z_instr      = [Z_instr; Z_i];             %#ok<AGROW>
    id_dep_out   = [id_dep_out;   repmat(panels(i_panel), length(time_i), 1)]; %#ok<AGROW>
    time_dep_out = [time_dep_out; time_i];     %#ok<AGROW>
    npanels_used = npanels_used + 1;
end

if isempty(Y_dep), error('pvar:noObs', 'No valid observations after transformation.'); end

intercept = ones(size(X_reg, 1), 1);
X_reg   = [intercept, X_reg];
Z_instr = [intercept, Z_instr];

[Q, ~] = qr(Z_instr, 0);
B_all  = (X_reg' * Q * (Q' * X_reg)) \ (X_reg' * Q * (Q' * Y_dep));

intercept_coef = B_all(1, :);
B      = B_all(2:end, :);
resid  = Y_dep - X_reg * B_all;
Sigma_u = cov(resid, 1);

n_obs  = size(Y_dep, 1);
n_inst = size(Z_instr, 2);

moments = zeros(n_obs, n_inst * K);
for i = 1:n_obs
    moments(i, :) = kron(Z_instr(i, :), resid(i, :));
end
g_bar   = mean(moments, 1)';
Omega   = moments' * moments / n_obs;
W       = inv(Omega + 1e-8 * eye(size(Omega)));
J_stat  = n_obs * (g_bar' * W * g_bar);
J_df    = n_inst * K - size(X_reg, 2) * K;
J_pvalue = NaN;
if J_df > 0, J_pvalue = 1 - chi2cdf(J_stat, J_df); end

MBIC = NaN; MAIC = NaN; MQIC = NaN;
if J_df >= 0
    MBIC = J_stat - J_df * log(n_obs);
    MAIC = J_stat - 2 * J_df;
    MQIC = J_stat - 2 * log(log(n_obs)) * J_df;
end

XtZ     = X_reg' * Z_instr;
ZtZ_inv = inv(Z_instr' * Z_instr + 1e-8 * eye(n_inst));
V       = XtZ * ZtZ_inv * XtZ';
V_inv   = inv(V + 1e-8 * eye(size(V)));

B_se = zeros(size(B_all)); B_tstat = zeros(size(B_all)); B_pval = zeros(size(B_all));
for k = 1:K
    sigma2_k = sum(resid(:,k).^2) / n_obs;
    se_k     = sqrt(diag(sigma2_k * V_inv));
    B_se(:,k)    = se_k;
    B_tstat(:,k) = B_all(:,k) ./ se_k;
    B_pval(:,k)  = 2 * (1 - tcdf(abs(B_tstat(:,k)), n_obs - size(X_reg,2)));
end

results.B              = B;
results.intercept      = intercept_coef;
results.B_se           = B_se(2:end, :);
results.B_tstat        = B_tstat(2:end, :);
results.B_pval         = B_pval(2:end, :);
results.intercept_se   = B_se(1, :);
results.intercept_tstat = B_tstat(1, :);
results.intercept_pval = B_pval(1, :);
results.resid          = resid;
results.Sigma_u        = Sigma_u;
results.nobs           = n_obs;
results.npanels        = npanels_used;
results.transformation = upper(transformation);
results.varNames       = varNames;
results.p              = p;
results.K              = K;
results.instlags       = instlags;
results.Y              = Y_dep;
results.X              = X_reg;
results.Z              = Z_instr;
results.Z_instr        = Z_instr;
results.J_stat         = J_stat;
results.J_pvalue       = J_pvalue;
results.J_df           = J_df;
results.MBIC           = MBIC;
results.MAIC           = MAIC;
results.MQIC           = MQIC;
results.id_dep         = id_dep_out;
results.time_dep       = time_dep_out;
end
