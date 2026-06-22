function results = estimate_pvar_spillover_fod(Y, id, time, p, S_spillover, inst, X_exog)
% Estimates a panel VAR with FOD transformation and a cross-sectional spillover.
%
% INPUTS
%   Y            [N x K]  endogenous variables
%   id           [N x 1]  unit identifier
%   time         [N x 1]  time identifier
%   p            scalar   lag order
%   S_spillover  [N x 1]  spillover variable (e.g. neighbour-weighted outcomes)
%   inst         vector   instrument lag indices, e.g. (p+1):14
%   X_exog       [N x M]  exogenous controls (optional; pass [] to omit)
%
% OUTPUT  struct — same fields as estimate_panel_var plus:
%   .spillover_coefs .spillover_se .spillover_tstat .spillover_pval
%   .exog_coefs .exog_se .exog_tstat .exog_pval

if nargin < 7, X_exog = []; end

[~, K] = size(Y);

base = pvar.estimate_panel_var(Y, id, time, p, ...
    'transformation', 'fod', 'instlags', inst, 'display', false);

Y_fod    = base.Y;
X_fod    = base.X;
id_fod   = base.id_dep;
time_fod = base.time_dep;

S_fod = transform_spillover_fod(S_spillover, id, time, p);

if ~isempty(X_exog)
    X_exog_fod = transform_exog_fod(X_exog, id, time, p);
    minRows    = min([size(Y_fod,1), size(X_fod,1), length(S_fod), size(X_exog_fod,1)]);
    X_exog_fod = X_exog_fod(1:minRows, :);
else
    minRows    = min([size(Y_fod,1), size(X_fod,1), length(S_fod)]);
    X_exog_fod = [];
end

Y_fod    = Y_fod(1:minRows, :);
X_fod    = X_fod(1:minRows, :);
S_fod    = S_fod(1:minRows);
id_fod   = id_fod(1:minRows);
time_fod = time_fod(1:minRows);

n_exog = size(X_exog_fod, 2);
B_all  = zeros(size(X_fod,2) + 1 + n_exog, K);
resid  = zeros(minRows, K);
lambda = 1e-6;

for k = 1:K
    if k == 1
        X_k = [X_fod, X_exog_fod];
    else
        X_k = [X_fod, S_fod, X_exog_fod];
    end
    B_k = (X_k'*X_k + lambda*eye(size(X_k,2))) \ (X_k'*Y_fod(:,k));
    if k == 1
        B_all(1:size(X_k,2), k) = B_k;
        B_all(size(X_k,2)+1,  k) = 0;
    else
        B_all(:, k) = B_k;
    end
    resid(:, k) = Y_fod(:,k) - X_k*B_k;
end

B               = B_all(2:K*p+1, :);
intercept_coefs = B_all(1, :);
spillover_coefs = B_all(K*p+2, 2);
exog_coefs      = [];
if n_exog > 0, exog_coefs = B_all(K*p+3:end, :); end

Sigma_u = cov(resid, 1);

B_se = zeros(size(B_all)); B_tstat = B_se; B_pval = B_se;
for k = 1:K
    if k == 1, X_k = [X_fod, X_exog_fod]; else, X_k = [X_fod, S_fod, X_exog_fod]; end
    sigma2_k  = sum(resid(:,k).^2) / minRows;
    se_k_full = sqrt(diag(sigma2_k * inv(X_k'*X_k + lambda*eye(size(X_k,2)))));
    if k == 1
        B_se(1:size(X_k,2), k) = se_k_full;
        B_se(K*p+2, k) = NaN;
    else
        B_se(1:size(X_k,2), k) = se_k_full;
    end
    B_tstat(:,k) = B_all(:,k) ./ B_se(:,k);
    B_pval(:,k)  = 2*(1 - tcdf(abs(B_tstat(:,k)), minRows - size(X_k,2)));
end

results.B                = B;
results.intercept        = intercept_coefs;
results.B_se             = B_se(2:K*p+1, :);
results.B_tstat          = B_tstat(2:K*p+1, :);
results.B_pval           = B_pval(2:K*p+1, :);
results.intercept_se     = B_se(1, :);
results.intercept_tstat  = B_tstat(1, :);
results.intercept_pval   = B_pval(1, :);
results.Sigma_u          = Sigma_u;
results.K                = K;
results.p                = p;
results.spillover_coefs  = spillover_coefs;
results.spillover_se     = B_se(K*p+2, 2);
results.spillover_tstat  = B_tstat(K*p+2, 2);
results.spillover_pval   = B_pval(K*p+2, 2);
results.exog_coefs       = exog_coefs;
results.exog_se          = B_se(K*p+3:end, :);
results.exog_tstat       = B_tstat(K*p+3:end, :);
results.exog_pval        = B_pval(K*p+3:end, :);
results.Y                = Y_fod;
results.X                = X_fod;
results.S                = S_fod;
results.resid            = resid;
results.nobs             = minRows;
results.id_resid         = id_fod;
results.time_resid       = time_fod;
results.J_stat           = base.J_stat;
results.J_df             = base.J_df;
results.J_pvalue         = base.J_pvalue;
results.MBIC             = base.MBIC;
results.MAIC             = base.MAIC;
results.MQIC             = base.MQIC;
end
