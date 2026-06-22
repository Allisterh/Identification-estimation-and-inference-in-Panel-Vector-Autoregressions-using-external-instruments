clear; clc;
addpath('.'); addpath('matlab_functions');

raw    = readtable('data/regionfiscal.csv');
raw    = sortrows(raw, {'state','year'});
states = raw.state;  years = raw.year;  n_obs = height(raw);

Drcapout1 = NaN(n_obs,1);  Drcapspend1 = NaN(n_obs,1);
for s = unique(states)'
    idx = find(states == s);
    rc  = raw.rcapout(idx);  rsp = raw.rcapspend(idx);
    Drcapout1(idx(2:end))   = (rc(2:end)  - rc(1:end-1)) ./ rc(1:end-1);
    Drcapspend1(idx(2:end)) = (rsp(2:end) - rsp(1:end-1)) ./ rc(1:end-1);
end

Z_raw = raw.Dpredict_spend;
valid = ~isnan(Drcapspend1) & ~isnan(Drcapout1) & ~isnan(Z_raw);
Y     = [Drcapspend1(valid), Drcapout1(valid)];
id    = states(valid);  time = years(valid);  Z = Z_raw(valid);

fprintf('Sample: %d obs, %d states, %d years\n', sum(valid), numel(unique(id)), numel(unique(time)));

fprintf('\n--- Lag selection ---\n');
lag_res = pvar.select_lag_order(Y, id, time, 3, 'max_inst', 4);
disp(lag_res.table);
p = lag_res.optimal_MBIC;
fprintf('Selected lag (MBIC): p = %d\n', p);

fprintf('\n--- VAR estimation ---\n');
inst = 1:p;
est  = pvar.estimate_panel_var(Y, id, time, p, ...
           'transformation','fod','instlags',inst,'varNames',{'Spending','Output'});
fprintf('N=%d  J=%.3f (df=%d, p=%.3f)  MBIC=%.3f\n', ...
    est.nobs, est.J_stat, est.J_df, est.J_pvalue, est.MBIC);

fprintf('\n--- IV identification ---\n');
nvar = 1;
iv   = pvar.identify_iv(est, Z, id, time, nvar, 'newey_lags',0,'confidence',0.95);

nsteps = 5;
[irf, ~] = pvar.compute_irf_iv(est, iv, nsteps);
[MSWlbound, MSWubound] = pvar.compute_msw_ci(est, iv, nsteps, 0.95);

idx = 1:size(irf, 2);

figure;
subplot(2, 1, 1);
hold on;
plot(idx, irf(1,:), 'b-', 'LineWidth', 1.5);
plot(idx, MSWlbound(1,:), 'r--', 'LineWidth', 1);
plot(idx, MSWubound(1,:), 'r--', 'LineWidth', 1);
fill([idx, fliplr(idx)], [MSWubound(1,:), fliplr(MSWlbound(1,:))], ...
    'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
xlabel('Year');
yline(0)
title('Impulse Response Function of fiscal policy growth');
hold off;

subplot(2, 1, 2);
hold on;
plot(idx, irf(2,:), 'b-', 'LineWidth', 1.5);
plot(idx, MSWlbound(2,:), 'r--', 'LineWidth', 1);
plot(idx, MSWubound(2,:), 'r--', 'LineWidth', 1);
fill([idx, fliplr(idx)], [MSWubound(2,:), fliplr(MSWlbound(2,:))], ...
    'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
xlabel('Year');
yline(0)
title('Impulse Response Function of GDP growth');
hold off;

if ~exist('results','dir'), mkdir('results'); end
saveas(gcf,'results/IRF.eps','epsc')
saveas(gcf,'results/IRF.png','png')

sim.n       = est.K;
sim.AR      = est.B';
sim.scale   = 1;
sim.nvar    = nvar;
sim.Gamma   = iv.Gamma;
sim.muZ     = mean(iv.Z_fod);
sim.stdZ    = std(iv.Z_fod);
sim.T       = numel(unique(time));
sim.i       = numel(unique(id));
sim.Sigma   = est.Sigma_u;
sim.resid   = est.resid;
sim.Z       = iv.Z_fod;
sim.trueirf = irf;
sim.WHat    = iv.WHat;
sim.p       = est.p;
sim.eta     = iv.eta;
sim.Xtot    = iv.Xtot;
sim.mu      = mean(Y);
sim.Y0      = Y(1,:);
save('results/sim.mat','sim');
fprintf('\nresults/sim.mat saved — trueirf h=0: spending=%.4f  output=%.4f\n', irf(1,1), irf(2,1));

% =========================================================================
%  DIAGNOSTICS
%  Sections below use objects already in the workspace:
%    est   – output of estimate_panel_var
%    iv    – output of identify_iv
%    Z     – original instrument vector (full panel)
%    id, time – panel identifiers
% =========================================================================

% ----- shared colours (matching Stata stc1 / lfitci style) ---------------
clr_dot  = [0.267 0.447 0.769];   % steel blue scatter points
clr_fill = [0.682 0.776 0.910];   % light blue CI band
clr_line = [0.180 0.373 0.675];   % dark blue fit line

% =========================================================================
%  SECTION 1 – AUTOCORRELATION SCATTER
%  Residuals at t vs residuals at t-1, within each unit.
%  Mirrors Stata: regress resid L1.resid then graph twoway lfitci + scatter
% =========================================================================
fprintf('\n--- Autocorrelation diagnostic ---\n');

resid_all  = est.resid;          % [T_fod x K]  FOD-sample residuals
id_r       = est.id_dep;
time_r     = est.time_dep;
K          = est.K;
varNms     = est.varNames;

% build lagged residuals within unit
L1_resid = NaN(size(resid_all));
for s = unique(id_r)'
    idx_s = find(id_r == s);
    if numel(idx_s) < 2, continue; end
    % sort by time within unit
    [~, ord]       = sort(time_r(idx_s));
    idx_s          = idx_s(ord);
    L1_resid(idx_s(2:end), :) = resid_all(idx_s(1:end-1), :);
end

ac_coef = NaN(K,1);  ac_se = NaN(K,1);
for k = 1:K
    ok = ~isnan(L1_resid(:,k));
    y  = resid_all(ok, k);
    x  = L1_resid(ok, k);
    X_ = [ones(sum(ok),1), x];
    b_ = X_ \ y;
    e_ = y - X_*b_;
    s2 = (e_'*e_) / (sum(ok) - 2);
    se_b = sqrt(diag(s2 * inv(X_'*X_)));
    ac_coef(k) = b_(2);
    ac_se(k)   = se_b(2);
end
fprintf('  Spending autocorr: %.3f (SE=%.3f)\n', ac_coef(1), ac_se(1));
fprintf('  Output   autocorr: %.3f (SE=%.3f)\n', ac_coef(2), ac_se(2));

fig_ac = figure('Visible','off');
titles_ac = {'Military spending', 'Output'};
for k = 1:K
    subplot(1, K, k);
    ok  = ~isnan(L1_resid(:,k));
    x   = L1_resid(ok,k);
    y   = resid_all(ok,k);
    n_  = sum(ok);

    % fit line + 95% CI band
    X_  = [ones(n_,1), x];
    b_  = X_ \ y;
    xs  = linspace(min(x), max(x), 200)';
    Xs_ = [ones(200,1), xs];
    yf  = Xs_*b_;
    e_  = y - X_*b_;
    s2_ = (e_'*e_)/(n_-2);
    se_ = sqrt(s2_ * diag(Xs_*(X_'*X_)^(-1)*Xs_'));
    t95 = tinv(0.975, n_-2);

    hold on;
    fill([xs; flipud(xs)], [yf+t95*se_; flipud(yf-t95*se_)], ...
        clr_fill, 'EdgeColor','none', 'FaceAlpha', 0.7);
    plot(xs, yf, '-', 'Color', clr_line, 'LineWidth', 1.5);
    scatter(x, y, 14, clr_dot, 'filled', 'MarkerFaceAlpha', 0.55);
    yline(0, ':', 'Color', [0.5 0.5 0.5]);
    xline(0, ':', 'Color', [0.5 0.5 0.5]);
    hold off;

    title(titles_ac{k}, 'FontWeight','bold', 'FontSize', 11);
    xlabel('Residual $t-1$', 'Interpreter','latex', 'FontSize', 9);
    ylabel('Residual $t$',   'Interpreter','latex', 'FontSize', 9);
    txt = sprintf('Coef: %.3f, SE: %.2f', ac_coef(k), ac_se(k));
    text(0.05, 0.95, txt, 'Units','normalized', 'VerticalAlignment','top', ...
        'FontSize', 8, 'BackgroundColor','white');
    box off; grid on;
end
sgtitle('Residuals and their lags', 'FontWeight','bold', 'FontSize', 13);
set(fig_ac, 'Position', [100 100 900 380]);
saveas(fig_ac, 'results/scatter_residuals.eps', 'epsc');
saveas(fig_ac, 'results/scatter_residuals.png', 'png');
fprintf('Saved results/scatter_residuals.*\n');

% =========================================================================
%  SECTION 2 – NORMALITY HISTOGRAMS + SHAPIRO-WILK
%  Mirrors Stata: swilk + histogram ... graph combine
% =========================================================================
fprintf('\n--- Normality tests (Shapiro-Wilk) ---\n');

% Align instrument to FOD sample (same alignment as identify_iv)
[~, idx_iv] = ismember([id_r, time_r], [id, time], 'rows');
Z_fod_diag  = Z(idx_iv);

series      = {resid_all(:,2), resid_all(:,1), Z_fod_diag};
hist_titles = {'Output residuals', 'Spending residuals', 'Total spending instrument'};
sw_W = NaN(3,1);  sw_p = NaN(3,1);

for k = 1:3
    d = series{k}(~isnan(series{k}));
    [sw_W(k), sw_p(k)] = shapiro_wilk(d);
    fprintf('  %s: W=%.4f, p=%.4f\n', hist_titles{k}, sw_W(k), sw_p(k));
end

fig_hist = figure('Visible','off');
for k = 1:3
    subplot(1, 3, k);
    d  = series{k}(~isnan(series{k}));
    n_ = numel(d);
    nb = max(10, round(sqrt(n_)));
    histogram(d, nb, 'Normalization','pdf', ...
        'FaceColor', clr_dot, 'EdgeColor','white', 'FaceAlpha', 0.75);
    hold on;
    mu_ = mean(d); sg_ = std(d);
    xs_ = linspace(min(d), max(d), 300);
    plot(xs_, normpdf(xs_, mu_, sg_), 'k-', 'LineWidth', 1.5);
    hold off;
    title(hist_titles{k}, 'FontWeight','bold', 'FontSize', 10);
    txt = sprintf('Shapiro-Wilk\nW=%.3f, p=%.3f', sw_W(k), sw_p(k));
    text(0.97, 0.97, txt, 'Units','normalized', 'VerticalAlignment','top', ...
        'HorizontalAlignment','right', 'FontSize', 7.5, 'BackgroundColor','white');
    box off; grid on;
end
sgtitle('Distribution of residuals and instrument', 'FontWeight','bold', 'FontSize', 12);
set(fig_hist, 'Position', [100 100 1100 370]);
saveas(fig_hist, 'results/normality_histograms.eps', 'epsc');
saveas(fig_hist, 'results/normality_histograms.png', 'png');
fprintf('Saved results/normality_histograms.*\n');

% =========================================================================
%  SECTION 3 – TABLES  (LaTeX + Excel)
%  AR coefficient matrix, IV first/second stage, normality tests.
% =========================================================================
fprintf('\n--- Writing tables ---\n');

T_obs = size(resid_all, 1);

% AR coefficient matrix
B_coef = est.B;          % [K*p x K]
B_se   = est.B_se;       % [K*p x K]
B_p    = est.B_pval;     % [K*p x K]

% IV first stage: regress eta(1) on [Z_fod, 1]
eta     = iv.eta;                          % [K x T_fod]
Z_iv    = iv.Z_fod;
T_iv    = numel(Z_iv);
Xfs     = [Z_iv, ones(T_iv,1)];
bfs     = Xfs \ eta(1,:)';
efs     = eta(1,:)' - Xfs*bfs;
s2fs    = (efs'*efs)/(T_iv-2);
se_fs   = sqrt(diag(s2fs * inv(Xfs'*Xfs)));
t_fs    = bfs ./ se_fs;
p_fs    = 2*(1 - tcdf(abs(t_fs), T_iv-2));
F_stat  = (bfs(1)/se_fs(1))^2;

% IV second stage
p_hat   = Z_iv * bfs(1);
Xss     = [p_hat, ones(T_iv,1)];
bss     = Xss \ eta(2,:)';
ess     = eta(2,:)' - Xss*bss;
s2ss    = (ess'*ess)/(T_iv-2);
se_ss   = sqrt(diag(s2ss * inv(Xss'*Xss)));
t_ss    = bss ./ se_ss;
p_ss    = 2*(1 - tcdf(abs(t_ss), T_iv-2));

% ----- write LaTeX -----
fid = fopen('results/tables.tex','w');

% Table 1: AR coefficients
fprintf(fid, '\\begin{table}[htbp]\n\\centering\n');
fprintf(fid, '\\caption{Panel VAR -- AR Coefficient Matrix}\n');
fprintf(fid, '\\label{tab:ar_coefs}\n');
fprintf(fid, '\\begin{tabular}{lcc}\n\\toprule\n');
fprintf(fid, ' & (1) Spending eq. & (2) Output eq. \\\\\n\\midrule\n');
row_lbl = {'Spending$_{t-1}$', 'Output$_{t-1}$'};
for i = 1:size(B_coef,1)
    fprintf(fid, '%s & %s \\\\\n', row_lbl{i}, fmt(B_coef(i,1),B_se(i,1),B_p(i,1)));
    fprintf(fid, '       & %s \\\\\n',           fmt(B_coef(i,2),B_se(i,2),B_p(i,2)));
end
fprintf(fid, '\\midrule\nObservations & \\multicolumn{2}{c}{%d} \\\\\n', T_obs);
fprintf(fid, '\\bottomrule\n');
fprintf(fid, '\\multicolumn{3}{l}{\\footnotesize SE in parentheses. $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$}\n');
fprintf(fid, '\\end{tabular}\n\\end{table}\n\n');

% Table 2: IV stages
fprintf(fid, '\\begin{table}[htbp]\n\\centering\n');
fprintf(fid, '\\caption{IV Identification -- First and Second Stage}\n');
fprintf(fid, '\\label{tab:iv_results}\n');
fprintf(fid, '\\begin{tabular}{lcc}\n\\toprule\n');
fprintf(fid, ' & (1) First stage ($\\widetilde{W}$) & (2) Second stage ($\\widetilde{Y}$) \\\\\n\\midrule\n');
fprintf(fid, 'Instrument ($Z_t$) & %.4f%s & \\\\\n', bfs(1), '');
fprintf(fid, '                   & (%.4f) & \\\\\n', se_fs(1));
fprintf(fid, '$\\hat{\\widetilde{W}}$ (fitted) & & %.4f%s \\\\\n', bss(1), '');
fprintf(fid, '                   & & (%.4f) \\\\\n', se_ss(1));
fprintf(fid, '\\midrule\nF-statistic & %.2f & \\\\\n', F_stat);
fprintf(fid, 'Observations & %d & %d \\\\\n', T_iv, T_iv);
fprintf(fid, '\\bottomrule\n');
fprintf(fid, '\\multicolumn{3}{l}{\\footnotesize SE in parentheses. $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$}\n');
fprintf(fid, '\\end{tabular}\n\\end{table}\n\n');

% Table 3: Normality tests
fprintf(fid, '\\begin{table}[htbp]\n\\centering\n');
fprintf(fid, '\\caption{Shapiro-Wilk Normality Tests}\n');
fprintf(fid, '\\label{tab:normality}\n');
fprintf(fid, '\\begin{tabular}{lcc}\n\\toprule\n');
fprintf(fid, 'Variable & $W$-statistic & $p$-value \\\\\n\\midrule\n');
for k = 1:3
    fprintf(fid, '%s & %.4f & %.4f \\\\\n', hist_titles{k}, sw_W(k), sw_p(k));
end
fprintf(fid, '\\bottomrule\n');
fprintf(fid, '\\multicolumn{3}{l}{\\footnotesize $H_0$: data are normally distributed.}\n');
fprintf(fid, '\\end{tabular}\n\\end{table}\n');
fclose(fid);
fprintf('Saved results/tables.tex\n');

% ----- write Excel -----
xlsfile = 'results/tables.xlsx';
if exist(xlsfile,'file'), delete(xlsfile); end

% Sheet 1: AR coefficients
hdr = {'', 'Spending eq.', 'Output eq.'};
writecell(hdr, xlsfile, 'Sheet','AR_coefficients', 'Range','A1');
for i = 1:size(B_coef,1)
    writecell({row_lbl{i}, B_coef(i,1), B_coef(i,2)}, xlsfile, ...
        'Sheet','AR_coefficients', 'Range', sprintf('A%d',2*i));
    writecell({'SE', B_se(i,1), B_se(i,2)}, xlsfile, ...
        'Sheet','AR_coefficients', 'Range', sprintf('A%d',2*i+1));
end
writecell({'N', T_obs}, xlsfile, 'Sheet','AR_coefficients', ...
    'Range', sprintf('A%d', 2*size(B_coef,1)+2));

% Sheet 2: IV first stage
hdr2 = {'Variable','Coefficient','SE','t-stat','p-value'};
writecell(hdr2,  xlsfile, 'Sheet','IV_first_stage',  'Range','A1');
writecell({'Instrument (Z)', bfs(1), se_fs(1), t_fs(1), p_fs(1)}, ...
    xlsfile, 'Sheet','IV_first_stage',  'Range','A2');
writecell({'Constant',       bfs(2), se_fs(2), t_fs(2), p_fs(2)}, ...
    xlsfile, 'Sheet','IV_first_stage',  'Range','A3');
writecell({'F-stat', F_stat}, xlsfile, 'Sheet','IV_first_stage', 'Range','A4');
writecell({'N', T_iv},        xlsfile, 'Sheet','IV_first_stage', 'Range','A5');

% Sheet 3: IV second stage
writecell(hdr2,  xlsfile, 'Sheet','IV_second_stage', 'Range','A1');
writecell({'W-hat (fitted)', bss(1), se_ss(1), t_ss(1), p_ss(1)}, ...
    xlsfile, 'Sheet','IV_second_stage', 'Range','A2');
writecell({'Constant',       bss(2), se_ss(2), t_ss(2), p_ss(2)}, ...
    xlsfile, 'Sheet','IV_second_stage', 'Range','A3');
writecell({'N', T_iv},        xlsfile, 'Sheet','IV_second_stage', 'Range','A4');

% Sheet 4: Normality tests
hdr4 = {'Variable','W-statistic','p-value'};
writecell(hdr4, xlsfile, 'Sheet','Normality_tests', 'Range','A1');
for k = 1:3
    writecell({hist_titles{k}, sw_W(k), sw_p(k)}, xlsfile, ...
        'Sheet','Normality_tests', 'Range', sprintf('A%d',k+1));
end
fprintf('Saved results/tables.xlsx\n');
fprintf('\nDone.\n');

% =========================================================================
%  LOCAL FUNCTIONS
% =========================================================================

function [W, pval] = shapiro_wilk(x)
% Shapiro-Wilk test for normality.
% Royston (1992) approximation, valid for 3 <= n <= 5000.
% Returns W statistic and two-sided p-value.
x    = sort(x(:));
n    = numel(x);
m    = norminv(((1:n)' - 3/8) / (n + 1/4));  % expected order stats
c    = m / sqrt(m'*m);
% Royston coefficients for a-vector
u    = 1/sqrt(n);
a_n  = c(n);                         % largest coefficient
poly_u = [-2.706056, 4.434685, -2.071190, -0.147981, 0.221157, a_n];
a1   = polyval(poly_u, u);
if n > 5
    poly_u2 = [-3.582633, 5.682633, -1.752461, -0.293762, 0.042981, c(n-1)];
    a2 = polyval(poly_u2, u);
else
    a2 = 0;
end
phi  = (m'*m - 2*m(n)^2 - 2*m(n-1)^2*(n>5)) / ...
       (1     - 2*a1^2   - 2*a2^2*(n>5));
a    = zeros(n,1);
a(n) = a1;  a(1) = -a1;
if n > 5, a(n-1) = a2; a(2) = -a2; end
a(ceil(n/2)+1:n-1-( n>5))   =  m(ceil(n/2)+1:n-1-(n>5))   / sqrt(phi);
a(2+(n>5):floor(n/2))        = -m(ceil(n/2)+1:n-1-(n>5))   / sqrt(phi);
W    = (a'*x)^2 / ((x - mean(x))'*(x - mean(x)));
% Royston p-value approximation
mu_y  = 0; sig_y = 1;   % defaults
if n >= 4
    gamma_c = [-2.273, 0.459];
    mu_y    = polyval([-0.0006714, 0.025054, -0.39978, 0.544], log(n));
    sig_y   = exp(polyval([0.00030302, -0.0082676, 0.04922, -0.1359, 0.1220], log(n)));
    gamma_c = [0.459, -2.273];   % for log(1-W) fit
    lam     = polyval(gamma_c, log(1-W));
    z       = (lam - mu_y) / sig_y;
    pval    = 2 * normcdf(-abs(z));
else
    pval    = NaN;
end
end

function s = fmt(val, se, pval)
% Format coefficient with significance stars and SE in parentheses.
stars = '';
if     pval < 0.01, stars = '$^{***}$';
elseif pval < 0.05, stars = '$^{**}$';
elseif pval < 0.10, stars = '$^{*}$';
end
s = sprintf('%.4f%s & (%.4f)', val, stars, se);
end