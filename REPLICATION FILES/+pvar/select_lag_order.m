function results = select_lag_order(Y, id, time, p_max, varargin)
% Selects lag order by looping p = 1 … p_max and comparing MAIC / MBIC / MQIC.
%
% INPUTS
%   Y        [N x K]  endogenous variables
%   id       [N x 1]  unit identifier
%   time     [N x 1]  time identifier
%   p_max    scalar   maximum lag to test
%
% NAME-VALUE OPTIONS
%   'spillover'   [N x 1]  spillover variable  (omit or [] for no-spillover model)
%   'exog'        [N x M]  exogenous controls  (omit or [] to exclude)
%   'max_inst'    scalar   maximum instrument lag (default: 2*p_max)
%
% OUTPUT  struct with fields:
%   .table           matlab table with one row per lag
%   .optimal_MAIC .optimal_MBIC .optimal_MQIC   chosen lag numbers

p_in = inputParser;
addParameter(p_in, 'spillover', []);
addParameter(p_in, 'exog',      []);
addParameter(p_in, 'max_inst',  2*p_max);
parse(p_in, varargin{:});

S        = p_in.Results.spillover;
X_exog   = p_in.Results.exog;
max_inst = p_in.Results.max_inst;

MAIC = nan(p_max,1); MBIC = nan(p_max,1); MQIC = nan(p_max,1);
J_vec = nan(p_max,1); r_vec = nan(p_max,1);
CD    = nan(p_max,1); n_obs_vec = nan(p_max,1);

for p = 1:p_max
    fprintf('Testing p = %d ...\n', p);
    inst = (p+1):max_inst;
    try
        if isempty(S)
            est = pvar.estimate_panel_var(Y, id, time, p, ...
                'transformation','fod','instlags',inst);
        else
            est = pvar.estimate_pvar_spillover_fod(Y, id, time, p, S, inst, X_exog);
        end
        n   = size(est.resid, 1);
        J   = est.J_stat;
        r   = est.J_df;
        MAIC(p) = J - 2*r;
        MBIC(p) = J - r*log(n);
        MQIC(p) = J - 2*r*log(log(n));
        TSS = sum(var(est.Y,0,1)) * n;
        RSS = sum(est.resid(:).^2);
        CD(p)       = 1 - RSS/TSS;
        J_vec(p)    = J;
        r_vec(p)    = r;
        n_obs_vec(p) = n;
    catch ME
        fprintf('  ERROR: %s\n', ME.message);
    end
end

valid = ~isnan(MBIC);
vi    = find(valid);
[~, i_maic] = min(MAIC(valid)); opt_MAIC = vi(i_maic);
[~, i_mbic] = min(MBIC(valid)); opt_MBIC = vi(i_mbic);
[~, i_mqic] = min(MQIC(valid)); opt_MQIC = vi(i_mqic);

lag_order = (1:p_max)';
results.table        = table(lag_order, n_obs_vec, r_vec, J_vec, MAIC, MBIC, MQIC, CD, ...
    'VariableNames', {'Lag','N_obs','r','J_stat','MAIC','MBIC','MQIC','CD'});
results.optimal_MAIC = opt_MAIC;
results.optimal_MBIC = opt_MBIC;
results.optimal_MQIC = opt_MQIC;
results.MAIC         = MAIC;
results.MBIC         = MBIC;
results.MQIC         = MQIC;

fprintf('\n  Optimal — MAIC: p=%d   MBIC: p=%d   MQIC: p=%d\n', ...
    opt_MAIC, opt_MBIC, opt_MQIC);
end
