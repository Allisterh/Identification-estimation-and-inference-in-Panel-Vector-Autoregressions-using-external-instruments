function iv = identify_iv(est, Z, id_orig, time_orig, nvar, varargin)
% Identifies the structural shock using an external instrument (SVAR-IV).
%
% INPUTS
%   est        struct from estimate_panel_var
%   Z          [N x 1]  instrument (full panel, same length as id_orig/time_orig)
%   id_orig    [N x 1]  unit identifiers for Z
%   time_orig  [N x 1]  time identifiers for Z
%   nvar       scalar   normalizing variable index (typically 1 for shock variable)
%
% NAME-VALUE OPTIONS
%   'newey_lags'  scalar  Newey-West bandwidth (default: 0)
%   'confidence'  scalar  for Wald test display (default: 0.95)
%
% OUTPUT  struct with fields:
%   .B          [K x 1]  structural impact vector
%   .Gamma      [K x 1]  first-stage covariance E[Z eta']
%   .scale      scalar   normalization scale (impact of shock on variable 1)
%   .WHat       asymptotic covariance of [vec(A)', Gamma']'
%   .Wald       Wald statistic for instrument strength
%   .critval    chi2 critical value at the given confidence level
%   .Z_fod      [T x 1]  instrument aligned with FOD sample
%   .eta        [K x T]  VAR residuals
%   .AL         [K x Kp] VAR coefficient matrix for IRF functions

p_in = inputParser;
addParameter(p_in, 'newey_lags', 0);
addParameter(p_in, 'confidence', 0.95);
parse(p_in, varargin{:});

newey_lags = p_in.Results.newey_lags;
confidence = p_in.Results.confidence;
critval    = norminv(1 - (1-confidence)/2)^2;

K  = est.K;
p  = est.p;
T  = size(est.resid, 1);

% --- Align instrument to FOD sample ---
[~, idx] = ismember([est.id_dep, est.time_dep], [id_orig, time_orig], 'rows');
if any(idx == 0)
    error('pvar:identify_iv:alignment', ...
          'Some FOD-sample (id,time) pairs not found in id_orig/time_orig.');
end
Z_fod = Z(idx, :);

eta = est.resid';   % (K x T)
Xtot = est.X;       % (T x 1+Kp)

% --- First stage: regress eta(1,:)' on [Z_fod, const] ---
FS = regress(eta(1,:)', [Z_fod, ones(T,1)]);
p_hat = Z_fod * FS(1:size(Z_fod,2));   % (T x 1) fitted values

% --- Second stage: for each other variable regress eta(k,:)' on [p_hat, const] ---
sqsp = zeros(K-1, 1);
for k = 2:K
    SS = regress(eta(k,:)', [p_hat, ones(T,1)]);
    sqsp(k-1) = SS(1);
end

% --- Structural impact vector (for reference only) ---
Biv     = [1; sqsp];
sigma_b = cov(est.resid, 1);
s21s11  = sqsp;
S11     = sigma_b(1,1);
S21     = sigma_b(2:end, 1);
S22     = sigma_b(2:end, 2:end);
Q_mat   = s21s11*S11*s21s11' - (S21*s21s11' + s21s11*S21') + S22;
sp      = sqrt(max(0, S11 - (S21 - s21s11*S11)' * (Q_mat \ (S21 - s21s11*S11))));
B_structural = Biv * sp;   % kept for reference; not used in IRF/CI

% --- First-stage covariance Gamma = E[Z * eta'] ---
% This is the correct B vector for the MSW CI (matches MSWfunction convention).
% Unit normalisation: irf(nvar, 0) = scale * Gamma(nvar) / Gamma(nvar) = 1.
Gamma = (eta * Z_fod) / T;   % (K x 1)
scale = 1;                    % unit shock at variable nvar

% --- Asymptotic covariance of (vec(A), Gamma) ---
[~, WHat, ~] = cov_ahat_gamma(p, Xtot, Z_fod, eta, newey_lags);

% --- Wald statistic: T * Gamma(nvar)^2 / Var(Gamma(nvar)) ---
W2   = WHat(K^2*p+1:end, K^2*p+1:end);
Wald = T * Gamma(nvar)^2 / W2(nvar, nvar);

% --- AL matrix (K x Kp) for MA/IRF computation ---
AL = est.B';   % (K x Kp)

% --- Output ---
iv.B             = Gamma;          % Gamma used as B in IRF and MSW CI
iv.Gamma         = Gamma;          % alias
iv.B_structural  = B_structural;   % Biv*sp, kept for reference
iv.scale         = scale;          % 1 → unit normalization at nvar
iv.sqsp          = sqsp;           % IV multiplier: Gamma(2)/Gamma(1) ≈ sqsp
iv.sp            = sp;
iv.WHat          = WHat;
iv.W2            = W2;
iv.Wald          = Wald;
iv.critval       = critval;
iv.nvar          = nvar;
iv.Z_fod         = Z_fod;
iv.eta           = eta;
iv.AL            = AL;
iv.T             = T;
iv.K             = K;
iv.p             = p;
iv.Xtot          = Xtot;

fprintf('IV identification complete.\n');
fprintf('  Gamma (first-stage cov):  Gamma(1)=%.6f  Gamma(2)=%.6f\n', Gamma(1), Gamma(2));
fprintf('  Impact multiplier (sqsp): %.4f  (output per unit spending at h=0)\n', sqsp(1));
fprintf('  Wald statistic: %.3f  |  Critical value (%.0f%%): %.3f  ', Wald, 100*confidence, critval);
if Wald > critval
    fprintf('[strong instrument]\n');
else
    fprintf('[WARNING: weak instrument]\n');
end
fprintf('  Unit-normalised IRF: irf(%d,0) = 1 by construction.\n', nvar);
end
