clear; clc;
addpath('.'); addpath('matlab_functions');

raw    = readtable('data/regionfiscal_appendix.csv');
raw    = sortrows(raw, {'state','year'});
states = raw.state;  years = raw.year;

Y  = [raw.Drcapspend2, raw.Drcapout2];
Z  = raw.Dpredict_spend2;
id = states;  time = years;

valid = ~isnan(Y(:,1)) & ~isnan(Y(:,2)) & ~isnan(Z);
Y = Y(valid,:);  id = id(valid);  time = time(valid);  Z = Z(valid);
fprintf('Sample: %d obs, %d states, %d years\n', sum(valid), numel(unique(id)), numel(unique(time)));

fprintf('\n--- Lag selection ---\n');
lag_res = pvar.select_lag_order(Y, id, time, 4, 'max_inst', 6);
disp(lag_res.table);
fprintf('MBIC: p=%d  (Stata specification: p=2)\n', lag_res.optimal_MBIC);

fprintf('\n--- VAR estimation (p=2) ---\n');
p = 2;  inst = 1:2;
est = pvar.estimate_panel_var(Y, id, time, p, ...
          'transformation','fod','instlags',inst,'varNames',{'Spending (2-yr)','Output (2-yr)'});
fprintf('N=%d  J=%.3f (df=%d, p=%.3f)  MBIC=%.3f\n', ...
    est.nobs, est.J_stat, est.J_df, est.J_pvalue, est.MBIC);

fprintf('\n--- IV identification ---\n');
nvar = 1;
iv   = pvar.identify_iv(est, Z, id, time, nvar, 'newey_lags',0,'confidence',0.95);

nsteps = 5;
[irf, ~]               = pvar.compute_irf_iv(est, iv, nsteps);
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
saveas(gcf,'results/IRF_2.eps','epsc')
saveas(gcf,'results/IRF_2.png','png')
