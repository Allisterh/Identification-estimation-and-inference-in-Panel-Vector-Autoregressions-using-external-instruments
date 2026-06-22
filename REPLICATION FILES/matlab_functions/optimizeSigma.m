function [x_opt, fval] = optimizeSigma(theta0, Sigma_star)
    % theta0 - 2x1 vector
    % Sigma_star - 2x2 target matrix

    % Initial guess for x1 and x2
    x0 = [0; 0];

    % Objective function
objFunc = @(x) computeError(x, theta0, Sigma_star);
options = optimoptions('simulannealbnd', 'Display', 'iter', 'MaxIterations', 10000000);
options.FunctionTolerance = 1e-100;
[x_opt, fval] = simulannealbnd(objFunc, x0, [], [], options);
end
function error = computeError(x, theta0, Sigma_star)
    % x - 2x1 vector containing x1 and x2
    % theta0 - 2x1 vector
    % Sigma_star - 2x2 target matrix

    % Construct Theta0 matrix
    Theta0 = [theta0, x(:)];  % x(:) ensures x is treated as a 2x1 column vector

    % Calculate SigmaX
    SigmaX = Theta0 * Theta0';

    % Calculate Frobenius norm of the difference
    error = norm(SigmaX - Sigma_star, 'fro');
end