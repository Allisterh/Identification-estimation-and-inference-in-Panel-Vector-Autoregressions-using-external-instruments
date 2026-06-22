function Sigma = nw_hac(vars, lags)
% Newey-West HAC covariance estimator.
% vars: (T x k) data matrix, lags: number of NW lags.
Sigma0    = (1/size(vars,1)) * (vars' * vars);
Sigma_cov = @(k) (1/size(vars,1)) * (vars(1:end-k,:))' * (vars(1+k:end,:));
Sigma = Sigma0;
for n = 1:lags
    Sigma = Sigma + (1 - n/(lags+1)) * (Sigma_cov(n) + Sigma_cov(n)');
end
end
