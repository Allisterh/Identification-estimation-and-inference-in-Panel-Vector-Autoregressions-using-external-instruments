function [irf, irf_cum] = compute_irf_iv(est, iv, nsteps)
% Computes impulse response functions for an externally identified SVAR-IV.
%
% INPUTS
%   est    struct from estimate_panel_var
%   iv     struct from identify_iv
%   nsteps scalar  number of horizons (excluding impact)
%
% OUTPUT
%   irf      [K x nsteps+1]  non-cumulative IRFs
%   irf_cum  [K x nsteps+1]  cumulative IRFs

n    = iv.K;
AL   = iv.AL;
B    = iv.B;
scale = iv.scale;
nvar  = iv.nvar;

% Build MA representation [I, C_1, C_2, ..., C_nsteps] — shape: (n x n x nsteps+1)
Caux  = [eye(n), ma_rep(AL, iv.p, nsteps)];
C     = reshape(Caux, [n, n, nsteps+1]);
Ccum  = cumsum(C, 3);

% Structural impact: B1 = scale * B / B(nvar)
B1 = scale * B ./ B(nvar, 1);

% IRF: e_j' * C_h * B1 for each variable j and horizon h
irf     = reshape(sum(bsxfun(@times, C,    B1'), 2), [n, nsteps+1]);
irf_cum = reshape(sum(bsxfun(@times, Ccum, B1'), 2), [n, nsteps+1]);
end
