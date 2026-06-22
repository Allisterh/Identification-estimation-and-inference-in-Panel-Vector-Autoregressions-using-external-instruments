function F = compute_companion(B, K, p)
% Builds the [K*p x K*p] companion matrix from coefficient matrix B [K*p x K].
Kp = K * p;
F  = zeros(Kp, Kp);
for lag = 1:p
    F(1:K, (lag-1)*K+1:lag*K) = B((lag-1)*K+1:lag*K, :)';
end
if p > 1
    F(K+1:end, 1:Kp-K) = eye(Kp-K);
end
end
