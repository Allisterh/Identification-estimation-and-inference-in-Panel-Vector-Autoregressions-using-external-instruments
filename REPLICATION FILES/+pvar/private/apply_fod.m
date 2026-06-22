function Y_fod = apply_fod(Y)
[T, K] = size(Y);
Y_fod = nan(T-1, K);
for t = 1:T-1
    n_future   = T - t;
    future_mean = mean(Y(t+1:T, :), 1);
    weight     = sqrt(n_future / (n_future + 1));
    Y_fod(t, :) = weight * (Y(t, :) - future_mean);
end
end
