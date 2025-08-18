% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha = 1.5;                                                                                       %
gamma = 1.5;                                                                                       %
B1 = 10;                                                                                           %
B2 = 10;                                                                                           %
tau_max = 3.5; % Maximum value for tau                                                             %                                              
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function data_cp = generate_recurrent_data_cp(n, tau_max, alpha, gamma)
    % Distribution: V ~ Weibull(1.5, 1.5), Z1, Z2, Z3 ~ Binomial(n, p)
    Z1 = binornd(1, 0.2, [n, 1]);
    Z2 = binornd(1, 0.5, [n, 1]);
    Z3 = binornd(1, 0.8, [n, 1]);
    tau = unifrnd(0, tau_max, [n, 1]);

    % Preallocate estimated size (overallocate then trim at end)
    max_events = ceil(tau_max / (mean(exp(0.3*Z1 + 0.18*Z2 - 0.68*Z3)))); % Rough estimate
    data_all = cell(n,1);

    for i = 1:n
        current_time = 0;
        event_times = [];
        gap_times = [];

        while true
            U = rand; % Uniform(0,1)
            V = (-log(U) / alpha)^(1 / gamma);
            gap_time = V * exp(0.3 * Z1(i) + 0.18*Z2(i) - 0.68*Z3(i));
            next_event_time = current_time + gap_time;

            if next_event_time > tau(i)
                break;
            end

            event_times = [event_times; next_event_time];
            gap_times = [gap_times; gap_time];
            current_time = next_event_time;
        end

        if ~isempty(event_times)
            last_event_time = event_times(end);
            censoring_time = tau(i);
            censoring_gap_time = tau(i) - last_event_time;
        else
            censoring_time = tau(i);
            censoring_gap_time = tau(i);
        end

        nrow = length(event_times) + 1;
        % Store as a matrix for speed
        id = i * ones(nrow,1);
        time = [event_times; censoring_time];
        gap_time = [gap_times; censoring_gap_time];
        event = [ones(length(event_times), 1); 0];
        Z1i = Z1(i) * ones(nrow,1);
        Z2i = Z2(i) * ones(nrow,1);
        Z3i = Z3(i) * ones(nrow,1);
        taui = tau(i) * ones(nrow,1);

        data_all{i} = [id, time, gap_time, event, Z1i, Z2i, Z3i, taui];
    end

    data_mat = vertcat(data_all{:});
    % Convert to table at the end
    data_cp = array2table(data_mat, ...
        'VariableNames', {'id', 'time', 'gap_time', 'event', 'Z1', 'Z2', 'Z3', 'tau'});
    data_cp = sortrows(data_cp, {'id', 'time'});
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GEHAN WEIGHT                                                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% AMI MODEL

% ALGORITHM 1

% Accessory Functions

% Step 1

% Estimate Theta_G and Sigma_G

function obj_value = objective_function(theta, data_cp)
    % theta: a vector of length 3 (for Z1, Z2, Z3)
    % Efficient, vectorized computation
    
    ids = unique(data_cp.id);
    n = numel(ids);

    % Pre-extract tau and covariates for all subjects (one per id)
    [~, ia] = unique(data_cp.id, 'first');
    tau_vec = data_cp.tau(ia);
    Z_mat = [data_cp.Z1(ia), data_cp.Z2(ia), data_cp.Z3(ia)]; % n x 3
    
    % Precompute log(tau) for all subjects
    log_tau = log(tau_vec); % n x 1

    total_sum = 0;

    for idx_i = 1:n
        i = ids(idx_i);
        i_rows = (data_cp.id == i);
        S_ij_vec = data_cp.time(i_rows);         % All event/censor times for subject i
        Z_i = Z_mat(idx_i, :)';                  % 3 x 1 column vector

        % For each event/censoring time of subject i
        for S_ij = S_ij_vec'
            % Compute covariate effects for all l at once
            Z_diff = Z_i' - Z_mat; % n x 3, each row is Z_i - Z_l
            cov_effect = Z_diff * theta(:); % n x 1

            % Vectorized contribution for all l
            contribution_vec = log_tau - log(S_ij) - cov_effect;
            total_sum = total_sum + sum(max(contribution_vec, 0));
        end
    end

    obj_value = total_sum / (n^2);
end

function estimated_theta = optimize_theta(data_cp)
    % Efficient optimization for vector theta (length 3 for Z1, Z2, Z3)
    initial_theta = [0.2; 0.1; 0.7]; % initial guess for theta

    options = optimset('fminsearch');
    options.Display = 'off';

    [estimated_theta, ~] = fminsearch(@(theta) ...
        objective_function(theta, data_cp), initial_theta, options);
end

function indicator = indicator_function(tau_l, S_ij, theta, Z_i, Z_l)
    % Efficient vectorized indicator function for three covariates
    % tau_l, S_ij: scalars or vectors of equal length
    % theta: vector of length 3
    % Z_i, Z_l: vectors of length 3 or matrices (n x 3) for batch

    % Ensure column vectors for correct broadcasting
    log_tau_l = log(tau_l);
    log_S_ij = log(S_ij);

    % theta' * (Z_i - Z_l) for each comparison (vectorized)
    theta = theta(:); % ensure theta is column
    Z_diff = Z_i - Z_l; % size: [n x 3] if batch, or [1 x 3] if scalar
    theta_Z_diff = Z_diff * theta; % [n x 1] or scalar

    indicator = (log_tau_l - log_S_ij) >= theta_Z_diff;
end


function S_0 = S_0_func(S_ij, theta, Z_i, data_cp)
    % Efficient vectorized version for three covariates
    % Calculates S0 for a single S_ij, Z_i, and theta

    ids = unique(data_cp.id);
    n = numel(ids);

    % Extract tau and covariates for all subjects
    [~, ia] = unique(data_cp.id, 'first');
    tau_vec = data_cp.tau(ia);
    Z_mat = [data_cp.Z1(ia), data_cp.Z2(ia), data_cp.Z3(ia)]; % n x 3

    % Broadcast Z_i for all l
    Z_i_mat = repmat(Z_i(:)', n, 1); % n x 3

    % Call vectorized indicator function
    Y_vec = indicator_function(S_ij, tau_vec, theta, Z_i_mat, Z_mat); % returns n x 1 logical

    S_0 = mean(Y_vec); % (1/n) * sum(Y_vec)
end

function S_1 = S_1_func(S_ij, theta, Z_i, data_cp)
    % Efficient vectorized version for three covariates
    % Calculates S1 for a single S_ij, Z_i, and theta

    ids = unique(data_cp.id);
    n = numel(ids);

    % Extract tau and covariates for all subjects
    [~, ia] = unique(data_cp.id, 'first');
    tau_vec = data_cp.tau(ia);                        % n x 1
    Z_mat = [data_cp.Z1(ia), data_cp.Z2(ia), data_cp.Z3(ia)]; % n x 3

    % Broadcast Z_i for all l
    Z_i_mat = repmat(Z_i(:)', n, 1);                  % n x 3

    % Vectorized at-risk indicator
    Y_vec = indicator_function(S_ij, tau_vec, theta, Z_i_mat, Z_mat); % n x 1 logical

    % Multiply at-risk indicator by Z_l (row-wise)
    S_1 = mean(Z_mat .* Y_vec); % returns 1 x 3 vector

    % If you want S_1 as a column vector:
    S_1 = S_1(:);
end

function S_2 = S_2_func(S_ij, theta, Z_i, data_cp)
    % Efficient vectorized version for three covariates
    % Calculates S2 for a single S_ij, Z_i, and theta
    % Returns a 3x3 matrix: mean of Z_l * Z_l' for all l at risk

    % Ensure column vectors
    theta = theta(:);
    Z_i = Z_i(:);

    ids = unique(data_cp.id);
    n = numel(ids);

    % Extract tau and covariates for all subjects
    [~, ia] = unique(data_cp.id, 'first');
    tau_vec = data_cp.tau(ia);                             % n x 1
    Z_mat = [data_cp.Z1(ia), data_cp.Z2(ia), data_cp.Z3(ia)]; % n x 3

    % Broadcast Z_i for all l
    Z_i_mat = repmat(Z_i', n, 1);                          % n x 3

    % Vectorized at-risk indicator
    Y_vec = indicator_function(tau_vec, S_ij, theta, Z_i_mat, Z_mat); % n x 1 logical

    % Initialize total_sum
    total_sum = zeros(3, 3);

    % Accumulate Z_l * Z_l' for each at-risk subject
    for k = 1:n
        if Y_vec(k)
            Z_l = Z_mat(k, :)'; % 3x1 column vector
            total_sum = total_sum + (Z_l * Z_l');
        end
    end

    S_2 = (1/n) * total_sum;
end

function phi_w = phi_function(S_ij, theta, Z_i, data_cp)
    % Efficient computation of phi(w) for three covariates

    S_0 = S_0_func(S_ij, theta, Z_i, data_cp);   % scalar
    S_1 = S_1_func(S_ij, theta, Z_i, data_cp);   % 3x1 vector
    S_2 = S_2_func(S_ij, theta, Z_i, data_cp);   % 3x3 matrix

    % Calculate phi(w) -- for vector covariates
    % S_0 * S_2 is a 3x3 matrix, (S_1 * S_1') is the outer product, also 3x3
    phi_w = S_0 * S_2 - (S_1 * S_1');
end

function Sigma_G = calculate_sigma(data_cp, theta_hat)
    % Efficient version for three covariates
    % Returns a 3x3 matrix

    ids = unique(data_cp.id);
    n = numel(ids);

    % Preallocate sum for 3x3 matrices
    total_sum = zeros(3,3);

    % Pre-extract covariates for all subjects
    [~, ia] = unique(data_cp.id, 'first');
    Z_mat = [data_cp.Z1(ia), data_cp.Z2(ia), data_cp.Z3(ia)]; % n x 3

    for idx_i = 1:n
        i = ids(idx_i);
        i_rows = (data_cp.id == i);
        i_data = data_cp(i_rows, :);
        K_i = height(i_data);

        Z_i = Z_mat(idx_i, :)'; % 3x1 column vector

        % Vectorized over all events/censoring times for subject i
        S_ij_vec = i_data.time;
        for j = 1:K_i
            S_ij = S_ij_vec(j);
            phi_w = phi_function(S_ij, theta_hat, Z_i, data_cp); % 3x3 matrix
            total_sum = total_sum + phi_w;
        end
    end

    Sigma_G = (1/n) * total_sum; % 3x3 matrix
end

% Step 2

% Estimate Theta_tilde_G

function S_M_G = objective_function_tilde(theta, data_cp, sigma)
    % Efficient implementation for three covariates (marginal case)
    % Returns the norm of (average sum - scaled sigma), matching the _k version pattern

    % Ensure theta is column vector
    theta = theta(:);

    n = length(unique(data_cp.id));
    total_sum = zeros(3,1);  % 3x1 column for three covariates

    % Pre-extract tau and covariates for all subjects
    [~, ia] = unique(data_cp.id, 'first');
    tau_vec = data_cp.tau(ia);                         % n x 1
    Z_mat = [data_cp.Z1(ia), data_cp.Z2(ia), data_cp.Z3(ia)]; % n x 3

    for idx_i = 1:n
        i = ia(idx_i);
        i_data = data_cp(data_cp.id == i, :);
        S_ij_vec = i_data.time;     % All event/censor times for subject i
        Z_i = Z_mat(idx_i, :)';     % 3 x 1 column

        for S_ij = S_ij_vec'
            % Vectorized difference with all subjects
            Z_diff = Z_i' - Z_mat;   % 1 x 3 minus n x 3 → n x 3
            % Indicator for all l
            Y_l = indicator_function(tau_vec, S_ij, theta, repmat(Z_i', n, 1), Z_mat); % n x 1 logical
            % Vectorized contribution for all l
            contribution = Z_diff .* Y_l;    % n x 3
            total_sum = total_sum + sum(contribution, 1)'; % add as column
        end
    end

    % If sigma is a matrix, use its diagonal. Assume sigma is variance vector.
    if ismatrix(sigma) && all(size(sigma) == [3, 3])
        sigma_vec = sqrt((1/n) * diag(sigma));         % 3 x 1 column
    else
        sigma_vec = sqrt((1/n) * sigma(:));            % 3 x 1 column
    end

    S_M_G = norm((total_sum / (n^2)) - sigma_vec);     % returns scalar
end


function estimated_theta_tilde = optimize_theta_tilde(data_cp, sigma)
    % Optimize theta for each column of sqrtm(sigma)
    estimated_theta_tilde = zeros(3, 3);

    % Optimization options
    options = optimset('fminsearch');
    options.Display = 'off'; % Suppress output during optimization

    % Calculate the square root of the 3x3 matrix sigma
    sigma_sqrt = sqrtm(sigma);

    % Loop over each column of the square root matrix
    for k = 1:3
        % Extract the k-th column of the square root of sigma
        sigma_k = sigma_sqrt(:, k); % 3x1 column vector

        % Initial guess for theta (3-dimensional vector)
        initial_theta_tilde_k = [0.2; 0.1; 0.6];

        % Optimize theta by minimizing the objective function
        estimated_theta_tilde(:, k) = fminsearch(@(theta) ...
            objective_function_tilde(theta, data_cp, sigma_k), ...
            initial_theta_tilde_k, options);
    end
end


% Estimating Lambda_G_M

% Lambda_G_M Function
function lambda_hat = Lambda_G_M(theta_G_M, t, data_cp)
    % Efficient, vectorized estimation for three covariates

    ids = unique(data_cp.id);
    n = numel(ids);

    % Pre-extract tau and covariates for all subjects
    [~, ia] = unique(data_cp.id, 'first');
    tau_vec = data_cp.tau(ia);                             % n x 1
    Z_mat = [data_cp.Z1(ia), data_cp.Z2(ia), data_cp.Z3(ia)]; % n x 3

    total_sum = 0;

    % Ensure theta_G_M is a column vector
    theta_G_M = theta_G_M(:);

    for idx_i = 1:n
        i = ids(idx_i);
        i_rows = (data_cp.id == i);
        individual_data = data_cp(i_rows, :);
        Z_i = Z_mat(idx_i, :)';      % 3 x 1 column vector
        S_ij_vec = individual_data.time; % all event/censor times for subject i

        for S_ij = S_ij_vec'
            % Numerator: log(t) - log(S_ij) >= theta_G_M' * Z_i (scalar)
            numerator = (log(t) - log(S_ij)) >= (theta_G_M' * Z_i);

            if numerator % Only one scalar condition
                % Denominator: sum over all l
                Z_i_mat = repmat(Z_i', n, 1); % n x 3
                Z_diff = Z_i_mat - Z_mat;     % n x 3
                den_vec = (log(tau_vec) - log(S_ij)) >= (Z_diff * theta_G_M); % n x 1 logical
                weight_denominator = sum(den_vec);

                if weight_denominator > 0
                    total_sum = total_sum + (1 / weight_denominator);
                end
            end
        end
    end

    lambda_hat = total_sum;
end

function [lambda_hat_t1, lambda_hat_t2] = evaluate_Lambda_G_M(theta_G_M, data_cp)
    % Evaluate Lambda_G_M at t = 1 and t = 3 for three covariates

    % Time points for evaluation
    t1 = 1;
    t2 = 3;

    % Compute Lambda_G_M at t1 and t2
    lambda_hat_t1 = Lambda_G_M(theta_G_M, t1, data_cp);
    lambda_hat_t2 = Lambda_G_M(theta_G_M, t2, data_cp);
end


% Estimating Sigma_Lambda_2_G_M
function first_term = First_Term_G_M(theta_G_M, t, data_cp)
    % Efficient, vectorized estimation for three covariates
    % Returns n * sum over (1/denominator^2) for eligible event times

    ids = unique(data_cp.id);
    n = numel(ids);

    % Pre-extract tau and covariates for all subjects
    [~, ia] = unique(data_cp.id, 'first');
    tau_vec = data_cp.tau(ia);                                 % n x 1
    Z_mat = [data_cp.Z1(ia), data_cp.Z2(ia), data_cp.Z3(ia)];  % n x 3

    total_sum = 0;

    % Ensure theta_G_M is a column vector
    theta_G_M = theta_G_M(:);

    for idx_i = 1:n
        i = ids(idx_i);
        i_rows = (data_cp.id == i);
        individual_data = data_cp(i_rows, :);
        Z_i = Z_mat(idx_i, :)';      % 3 x 1 column vector
        S_ij_vec = individual_data.time; % all event/censor times for subject i

        for S_ij = S_ij_vec'
            % Numerator: scalar condition
            numerator = (log(t) - log(S_ij)) >= (theta_G_M' * Z_i);

            if numerator % Only one scalar condition
                % Denominator: sum over all l
                Z_i_mat = repmat(Z_i', n, 1); % n x 3
                Z_diff = Z_i_mat - Z_mat;     % n x 3
                den_vec = (log(tau_vec) - log(S_ij)) >= (Z_diff * theta_G_M); % n x 1 logical
                weight_denominator = sum(den_vec);

                if weight_denominator > 0
                    total_sum = total_sum + (1 / (weight_denominator)^2);
                end
            end
        end
    end

    first_term = n * total_sum;
end


% Estimate A_hat
function A_hat_B_tilde = estimate_A_hat_M(theta_G_M, Gamma_hat_G, t, data_cp, B)
    % Efficient vectorized estimation for three covariates using multivariate bootstrap

    n = length(unique(data_cp.id));  % Number of samples

    % Covariance matrix: ensure it's 3x3
    cov_matrix = (1/n) * Gamma_hat_G;  % 3x3 covariance matrix

    % Bootstrap samples of theta
    theta_tilde = mvnrnd(theta_G_M', cov_matrix, B)';  % 3 x B matrix

    % Compute Lambda_G_M for all bootstrap samples and the original estimate
    lambda_hat_theta_G = Lambda_G_M(theta_G_M, t, data_cp); % scalar
    lambda_hat_tilde = zeros(1, B);

    for k = 1:B
        lambda_hat_tilde(k) = Lambda_G_M(theta_tilde(:, k), t, data_cp);
    end

    % Construct Z and Y for regression
    Z = [ones(B, 1), (theta_G_M' - theta_tilde')]; % B x 4
    Y = (lambda_hat_tilde' - lambda_hat_theta_G);  % B x 1

    % OLS regression: (Z'Z)^(-1) Z'Y
    A_hat_B_tilde_full = (Z' * Z) \ (Z' * Y);      % 4 x 1

    % Remove intercept row (first row)
    A_hat_B_tilde = A_hat_B_tilde_full(2:end, :);  % 3 x 1

    % Symmetrize (for vector, this is a no-op, but included for consistency)
    A_hat_B_tilde = (1/2) * (A_hat_B_tilde + A_hat_B_tilde);

end

function [Sigma_Lambda_2_G_M_t1, Sigma_Lambda_2_G_M_t2] = evaluate_Sigma_Lambda_2_G_M(theta_G_M, Gamma_hat_G, data_cp, B)
    % Evaluate Sigma_Lambda_2_G_M at t = 1 and t = 3 for three covariates

    % Evaluate at t = 1
    t1 = 1;
    first_term_t1 = First_Term_G_M(theta_G_M, t1, data_cp);
    A_hat_t1 = estimate_A_hat_M(theta_G_M, Gamma_hat_G, t1, data_cp, B);
    Sigma_Lambda_2_G_M_t1 = first_term_t1 + (A_hat_t1' * Gamma_hat_G * A_hat_t1);

    % Evaluate at t = 3
    t2 = 3;
    first_term_t2 = First_Term_G_M(theta_G_M, t2, data_cp);
    A_hat_t2 = estimate_A_hat_M(theta_G_M, Gamma_hat_G, t2, data_cp, B);
    Sigma_Lambda_2_G_M_t2 = first_term_t2 + (A_hat_t2' * Gamma_hat_G * A_hat_t2);
end


% API MODEL

% ALGORITHM 1

% Accessory Functions

% Step 1

% Estimate Theta_G and Sigma_G
function X_lr_tilde = X_lr_tilde_func(T_lr, tau_l, S_l_kl, theta, Z_l, r, K_l)
    % Function to calculate \widetilde{X}_{lr}(\boldsymbol{\theta})
    exp_theta_Z_l = exp(theta' * Z_l); % for vector Z_l, theta must match

    if r <= K_l
        X_lr_tilde = T_lr * exp_theta_Z_l;
    else
        % r = K_l+1
        X_lr_tilde = (tau_l - S_l_kl) * exp_theta_Z_l;
    end
end

function X_ij_tilde = X_ij_tilde_func(T_ij, theta, Z_i)
    % Function to calculate \widetilde{X}_{ij}(\boldsymbol{\theta})

    exp_theta_Z_i = exp(theta' * Z_i); % For vector Z_i, theta must match
    X_ij_tilde = T_ij * exp_theta_Z_i;
end

function L_P_G = objective_function_p(theta, data_cp)
    % Calculates the objective function L_P_G for vector-valued theta and covariates
    % Supports multiple covariates: Z1, Z2, Z3 in data_cp

    ids = unique(data_cp.id);
    n = numel(ids);

    total_sum = 0;

    for idx_i = 1:n
        i = ids(idx_i);
        i_data = data_cp(data_cp.id == i, :);
        K_i = height(i_data);
        % Extract vector covariates Z1, Z2, Z3 for individual i
        Z_i = [i_data.Z1(1); i_data.Z2(1); i_data.Z3(1)]; % Use column vector for robustness

        for j = 1:K_i
            T_ij = i_data.gap_time(j);

            for idx_l = 1:n
                l = ids(idx_l);
                l_data = data_cp(data_cp.id == l, :);
                K_l = height(l_data);
                tau_l = l_data.tau(1);
                S_l_kl = l_data.time(K_l);
                % Extract vector covariates Z1, Z2, Z3 for individual l
                Z_l = [l_data.Z1(1); l_data.Z2(1); l_data.Z3(1)]; % Use column vector for robustness

                for r = 1:(K_l + 1)
                    if r <= K_l
                        T_lr = l_data.gap_time(r);
                    else
                        T_lr = (tau_l - S_l_kl);
                    end

                    % Pass inputs as column vectors for compatibility
                    X_lr_tilde = X_lr_tilde_func(T_lr, tau_l, S_l_kl, theta, Z_l, r, K_l);
                    X_ij_tilde = X_ij_tilde_func(T_ij, theta, Z_i);

                    log_diff = log(X_lr_tilde) - log(X_ij_tilde);
                    log_diff_pos = max(log_diff, 0);

                    total_sum = total_sum + log_diff_pos;
                end
            end
        end
    end

    L_P_G = (1 / n^2) * total_sum;
end

function estimated_theta_p = optimize_theta_p(data_cp)
    % Optimizes theta for objective_function_p with support for vector-valued theta

    initial_theta_p = [0.2; 0.1; 0.7]; % Initial guess for multiple covariates

    options = optimset('fminsearch');
    options.Display = 'off';

    [estimated_theta_p, ~] = fminsearch(@(theta) ...
        objective_function_p(theta, data_cp), initial_theta_p, options);
end


% Generalized at-risk process Y_l
function indicator = indicator_tilde_func(T_lr, tau_l, S_l_kl, T_ij, theta, Z_i, Z_l, r, K_l)
    % Generalized at-risk process indicator \widetilde{Y}_{l} for multiple covariates

    X_ij_tilde = X_ij_tilde_func(T_ij, theta, Z_i);
    X_lr_tilde = X_lr_tilde_func(T_lr, tau_l, S_l_kl, theta, Z_l, r, K_l);

    indicator = (X_ij_tilde <= X_lr_tilde);
end

function Y_l_P = Y_l_P_func(tau_l, S_l_kl, T_ij, theta, Z_i, Z_l, K_l, l_data)
    % Function to calculate Y_{l}^P(\widetilde{X}_{ij}(\boldsymbol{\theta}) | \boldsymbol{\theta})
    % Inputs:
    %   tau_l     - censoring/event time for individual l
    %   S_l_kl    - last observed event time for l
    %   T_ij      - time or interval for i,j
    %   theta     - parameter vector
    %   Z_i       - covariate(s) for individual i (column vector)
    %   Z_l       - covariate(s) for individual l (column vector)
    %   K_l       - number of intervals/events for l
    %   l_data    - table for individual l (contains gap_time, etc.)
    %
    % Output:
    %   Y_l_P     - sum of indicator_tilde_func over r

    sum_indicator = 0;
    for r = 1:(K_l + 1)
        if r <= K_l
            T_lr = l_data.gap_time(r); % r-th gap time for individual l
        else
            T_lr = (tau_l - S_l_kl);
        end

        indicator_value = indicator_tilde_func(T_lr, tau_l, S_l_kl, T_ij, theta, Z_i, Z_l, r, K_l);
        sum_indicator = sum_indicator + indicator_value;
    end

    Y_l_P = sum_indicator;
end

function S_0 = S_0_func_p(T_ij, theta, Z_i, data_cp)
    % Calculate \mathbb{S}^{(0)}(T_ij e^{theta' Z_i} | theta)
    % Inputs:
    %   T_ij    - time or interval for i,j
    %   theta   - parameter vector
    %   Z_i     - covariate(s) for individual i (column vector)
    %   data_cp - dataset with individuals' info
    %
    % Output:
    %   S_0     - average sum over all individuals

    unique_ids = unique(data_cp.id);
    n = numel(unique_ids);
    total_sum = 0;

    for l = unique_ids'
        l_data = data_cp(data_cp.id == l, :);  % Subset for individual l
        if ~isempty(l_data)
            tau_l = l_data.tau(1);
            Z_l = [l_data.Z1(1); l_data.Z2(1); l_data.Z3(1)]; % Use Z1, Z2, Z3 as column vector for robustness
            K_l = height(l_data);
            S_l_kl = l_data.time(K_l); % Last observed event time for individual l
            Y_l = Y_l_P_func(tau_l, S_l_kl, T_ij, theta, Z_i, Z_l, K_l, l_data);
            total_sum = total_sum + Y_l;
        end
    end

    S_0 = (1/n) * total_sum;
end

% S1
function S_1 = S_1_func_p(T_ij, theta, Z_i, data_cp)
    % Calculate \mathbb{S}^{(1)}(T_ij e^{theta' Z_i} | theta)
    % Returns a 3x1 vector (for three covariates)

    unique_ids = unique(data_cp.id);
    n = numel(unique_ids);
    total_sum = zeros(3, 1); % For three covariates Z1, Z2, Z3 as column vector

    for l = unique_ids'
        l_data = data_cp(data_cp.id == l, :);  % Subset for individual l
        if ~isempty(l_data)
            tau_l = l_data.tau(1);
            Z_l = [l_data.Z1(1); l_data.Z2(1); l_data.Z3(1)]; % Use Z1, Z2, Z3 as column vector
            K_l = height(l_data);
            S_l_kl = l_data.time(K_l); % Last observed event time for individual l
            Y_l = Y_l_P_func(tau_l, S_l_kl, T_ij, theta, Z_i, Z_l, K_l, l_data);
            total_sum = total_sum + Z_l * Y_l; % Vector multiplication, result is 3x1
        end
    end

    S_1 = (1/n) * total_sum;
end


% S2
function S_2 = S_2_func_p(T_ij, theta, Z_i, data_cp)
    % Calculate \mathbb{S}^{(2)}(T_ij e^{theta' Z_i} | theta)
    % Returns a 3x3 matrix (for three covariates)

    unique_ids = unique(data_cp.id);
    n = numel(unique_ids);
    total_sum = zeros(3, 3); % For three covariates Z1, Z2, Z3 (squared matrix)

    for l = unique_ids'
        l_data = data_cp(data_cp.id == l, :);  % Subset for individual l
        if ~isempty(l_data)
            tau_l = l_data.tau(1);
            Z_l = [l_data.Z1(1); l_data.Z2(1); l_data.Z3(1)]; % Use Z1, Z2, Z3 as column vector
            K_l = height(l_data);
            S_l_kl = l_data.time(K_l); % Last observed event time for individual l
            Y_l = Y_l_P_func(tau_l, S_l_kl, T_ij, theta, Z_i, Z_l, K_l, l_data);
            total_sum = total_sum + (Z_l * Z_l') * Y_l; % Outer product (3x3) scaled by Y_l
        end
    end

    S_2 = (1/n) * total_sum;
end

% Function to compute \varphi(w)
function phi_w = phi_function_p(T_ij, theta, Z_i, data_cp)
    % Compute S^0, S^1, and S^2 for the given parameters and data
    % S_0: scalar
    % S_1: 3x1 vector
    % S_2: 3x3 matrix

    S_0 = S_0_func_p(T_ij, theta, Z_i, data_cp);   % scalar
    S_1 = S_1_func_p(T_ij, theta, Z_i, data_cp);   % 3x1 vector
    S_2 = S_2_func_p(T_ij, theta, Z_i, data_cp);   % 3x3 matrix

    % Efficient calculation: phi(w) = S_0 * S_2 - (S_1 * S_1')
    phi_w = S_0 * S_2 - (S_1 * S_1');   % Result is a 3x3 matrix
end

% Function to compute \widehat{\Sigma}_G
function Sigma_G = calculate_sigma_p(data_cp, theta_hat)
    % Efficient computation of \widehat{\Sigma}_G for vector covariates Z1, Z2, Z3
    % Returns a 3x3 matrix (covariate count = 3)

    unique_ids = unique(data_cp.id);
    n = numel(unique_ids);
    total_sum = zeros(3, 3); % For three covariates

    for i = unique_ids'
        i_data = data_cp(data_cp.id == i, :);  % Subsetting data for individual i
        K_i = height(i_data);
        Z_i = [i_data.Z1(1); i_data.Z2(1); i_data.Z3(1)]; % Use Z1, Z2, Z3 as column vector

        for j = 1:K_i
            T_ij = i_data.gap_time(j);  % T_ij for event j
            % Compute \varphi(T_{ij} e^{theta' Z_i})
            phi_w = phi_function_p(T_ij, theta_hat, Z_i, data_cp); % 3x3 matrix
            total_sum = total_sum + phi_w;
        end
    end

    Sigma_G = (1/n) * total_sum;
end

% Step 2

% Estimate Theta_tilde_G

% Objective function_tilde_p

function S_P_G = objective_function_tilde_p(theta, data_cp, sigma)
    % Efficient and vectorized calculation of the objective function S_P_G
    % for vector-valued covariates Z1, Z2, Z3

    unique_ids = unique(data_cp.id);
    n = numel(unique_ids);
    total_sum = zeros(3, 1); % Column vector for three covariates

    for idx_i = 1:n
        i = unique_ids(idx_i);
        i_data = data_cp(data_cp.id == i, :);
        K_i = height(i_data);
        Z_i = [i_data.Z1(1); i_data.Z2(1); i_data.Z3(1)]; % Column vector

        for j = 1:K_i
            T_ij = i_data.gap_time(j);

            for idx_l = 1:n
                l = unique_ids(idx_l);
                l_data = data_cp(data_cp.id == l, :);
                K_l = height(l_data);
                tau_l = l_data.tau(1);
                S_l_kl = l_data.time(K_l);
                Z_l = [l_data.Z1(1); l_data.Z2(1); l_data.Z3(1)]; % Column vector
                Y_l = Y_l_P_func(tau_l, S_l_kl, T_ij, theta, Z_i, Z_l, K_l, l_data);

                Z_diff = Z_i - Z_l; % Column vector
                total_sum = total_sum + Y_l * Z_diff;
            end
        end
    end

    % Ensure sigma is a column vector for robust math
    S_P_G = norm((total_sum / (n^2)) - sqrt(1/n) * sigma);
end

% Optimization to find the parameter theta
function estimated_theta_tilde_p = optimize_theta_tilde_p(data_cp, sigma)
 % Initialize a 3x3 matrix to store the estimated theta values
    estimated_theta_tilde_p = zeros(3, 3);
    
    % Optimization options
    options = optimset('fminsearch');
    options.Display = 'off'; % Suppress output during optimization
    
    % Calculate the square root of the 3x3 matrix sigma
    sigma_sqrt = sqrtm(sigma);
    
    % Loop over each column of the square root matrix
    for k = 1:3
        % Extract the k-th column of the square root of sigma
        sigma_k = sigma_sqrt(:, k);

        % Initial guess for theta (3-dimensional vector)
        initial_theta_tilde_p = [0.2; 0.1; 0.7]; % Assuming all elements start at 0.25
        
        % Optimize theta by minimizing the objective function
        estimated_theta_tilde_p(:, k) = fminsearch(@(theta) ...
            objective_function_tilde_p(theta, data_cp, sigma_k), ...
            initial_theta_tilde_p, options);
    end
end


% Estimating Lambda_G_P

% Lambda_G_P Function
function Lambda_0_t = Lambda_G_P(theta, data_cp, t)
    % Efficient estimation of Lambda_G_P (cumulative baseline hazard)
    % Uses vector-valued covariates Z1, Z2, Z3

    unique_ids = unique(data_cp.id);
    n = numel(unique_ids);
    total_sum = 0;

    for idx_i = 1:n
        i = unique_ids(idx_i);
        i_data = data_cp(data_cp.id == i, :);
        K_i = height(i_data);
        Z_i = [i_data.Z1(1); i_data.Z2(1); i_data.Z3(1)]; % Column vector

        for j = 1:K_i
            T_ij = i_data.gap_time(j);

            % Compute sum of Y_l across all l
            Y_l_sum = 0;
            for idx_l = 1:n
                l = unique_ids(idx_l);
                l_data = data_cp(data_cp.id == l, :);
                K_l = height(l_data);
                tau_l = l_data.tau(1);
                S_l = l_data.time(K_l);
                Z_l = [l_data.Z1(1); l_data.Z2(1); l_data.Z3(1)]; % Column vector

                Y_l = Y_l_P_func(tau_l, S_l, T_ij, theta, Z_i, Z_l, K_l, l_data);
                Y_l_sum = Y_l_sum + Y_l;
            end

            % Calculate the indicator function
            X_ij = T_ij * exp(theta' * Z_i); % Column vectors for robust math
            indicator = (X_ij <= t);

            % Only update total_sum if Y_l_sum > 0
            if Y_l_sum > 0
                total_sum = total_sum + (indicator / Y_l_sum);
            end
        end
    end

    Lambda_0_t = total_sum;
end


% Function to evaluate Lambda_G_P at t = 1 and t = 3
function [lambda_hat_t1, lambda_hat_t2] = evaluate_Lambda_G_P(theta_G_P, data_cp)
    % Efficiently evaluate Lambda_G_P at t = 1 and t = 3
    t1 = 1;
    lambda_hat_t1 = Lambda_G_P(theta_G_P, data_cp, t1);

    t2 = 3;
    lambda_hat_t2 = Lambda_G_P(theta_G_P, data_cp, t2);
end


% Estimating Sigma_Lambda_2_G_P
function first_term = First_Term_G_P(theta, data_cp, t)
    % Efficient estimation of first term for Sigma_Lambda_2_G_P
    % Uses vector-valued covariates Z1, Z2, Z3

    unique_ids = unique(data_cp.id);
    n = numel(unique_ids);
    total_sum = 0;

    for idx_i = 1:n
        i = unique_ids(idx_i);
        i_data = data_cp(data_cp.id == i, :);
        K_i = height(i_data);
        Z_i = [i_data.Z1(1); i_data.Z2(1); i_data.Z3(1)]; % Column vector

        for j = 1:K_i
            T_ij = i_data.gap_time(j);

            Y_l_sum = 0;
            for idx_l = 1:n
                l = unique_ids(idx_l);
                l_data = data_cp(data_cp.id == l, :);
                K_l = height(l_data);
                tau_l = l_data.tau(1);
                S_l = l_data.time(K_l);
                Z_l = [l_data.Z1(1); l_data.Z2(1); l_data.Z3(1)]; % Column vector

                Y_l = Y_l_P_func(tau_l, S_l, T_ij, theta, Z_i, Z_l, K_l, l_data);
                Y_l_sum = Y_l_sum + Y_l;
            end

            % Use matrix product for vector-valued theta/Z_i
            X_ij = T_ij * exp(theta' * Z_i); % Column vectors for robust math
            indicator = (X_ij <= t);

            % Only update total_sum if Y_l_sum > 0
            if Y_l_sum > 0
                total_sum = total_sum + (indicator / (Y_l_sum)^2);
            end
        end
    end

    first_term = n * total_sum;
end

% Estimate A_hat
function A_hat_B_tilde = estimate_A_hat_P(theta_G_P, Gamma_hat_G, t, data_cp, B)
    % Efficient estimation of A_hat with vector-valued theta_G_P (length 3)
    % Gamma_hat_G should be a 3x3 covariance matrix

    n = numel(unique(data_cp.id));
    cov_matrix = (1/n) * Gamma_hat_G;  % 3x3 covariance matrix

    % Bootstrap samples of theta: B samples, each is a 1x3 vector
    theta_tilde = mvnrnd(theta_G_P, cov_matrix, B);  % B x 3

    % Compute Lambda_G_P for theta_G_P and for each bootstrap theta_tilde
    lambda_hat_theta_G = Lambda_G_P(theta_G_P, data_cp, t);
    lambda_hat_tilde = zeros(B, 1);

    for k = 1:B
        lambda_hat_tilde(k) = Lambda_G_P(theta_tilde(k, :)', data_cp, t); % column vector input
    end

    % Construct Z and Y for regression
    Z = [ones(B, 1), repmat(theta_G_P(:)', B, 1) - theta_tilde];  % B x 4 (if theta length is 3)
    Y = lambda_hat_tilde - lambda_hat_theta_G;  % B x 1

    % Linear regression: solve (Z'Z) * beta = Z'Y
    beta = (Z' * Z) \ (Z' * Y);

    % Remove intercept (first element)
    A_hat_B_tilde = beta(2:end);

    % Output is a vector, no symmetrization needed
end

% Function to evaluate Sigma_Lambda_2_G_M at t = 1 and t = 3
function [Sigma_Lambda_2_G_P_t1, Sigma_Lambda_2_G_P_t2] = evaluate_Sigma_Lambda_2_G_P(theta_G_P, Gamma_hat_G, data_cp, B)
    % Evaluate Sigma_Lambda_2_G_P at t = 1 and t = 3 for vector-valued theta
    t1 = 1;
    first_term_t1 = First_Term_G_P(theta_G_P, data_cp, t1);
    A_hat_t1 = estimate_A_hat_P(theta_G_P, Gamma_hat_G, t1, data_cp, B);
    Sigma_Lambda_2_G_P_t1 = first_term_t1 + (A_hat_t1' * Gamma_hat_G * A_hat_t1);

    t2 = 3;
    first_term_t2 = First_Term_G_P(theta_G_P, data_cp, t2);
    A_hat_t2 = estimate_A_hat_P(theta_G_P, Gamma_hat_G, t2, data_cp, B);
    Sigma_Lambda_2_G_P_t2 = first_term_t2 + (A_hat_t2' * Gamma_hat_G * A_hat_t2);
end

% Plot Objective Function

function plot_objective_function_g(data_cp)
    % Define a grid of theta values for theta(1) and theta(2)
    theta1_values = linspace(-1, 1, 10); % Range for theta(1)
    theta2_values = linspace(-1, 1, 10); % Range for theta(2)
    
    % Initialize matrix to store objective function values
    obj_values = zeros(length(theta1_values), length(theta2_values));
    
    % Fix theta(3) to a constant value (for example, 0.6)
    theta3_fixed = 0.6;
    
    % Loop through values of theta(1) and theta(2) to compute the objective function
    for i = 1:length(theta1_values)
        for j = 1:length(theta2_values)
            theta = [theta1_values(i); theta2_values(j); theta3_fixed]; % Create 3x1 theta vector
            % Compute the objective function value for each (theta1, theta2)
            obj_values(i, j) = objective_function_p(theta, data_cp);
        end
    end
    
    % Create a surface plot of the objective function
    figure;
    surf(theta1_values, theta2_values, obj_values');
    xlabel('Theta 1');
    ylabel('Theta 2');
    zlabel('Objective Function Value');
    title('Objective Function vs Theta(1) and Theta(2)');
    colorbar; % Add colorbar to visualize objective function values
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LOG-RANK WEIGHT                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% AMI MODEL

% ALGORITHM 2

% Accessory Functions

% Score Function S_LR_M
function obj_value2 = S_LR_M(theta, data_cp)
    % Efficient generalized score function for three covariates: Z1, Z2, Z3
    % theta: column vector (3x1)
    unique_ids = unique(data_cp.id);
    n = numel(unique_ids);
    total_sum = zeros(3, 1); % For three covariates as a column vector

    for idx_i = 1:n
        i = unique_ids(idx_i);
        individual_data = data_cp(data_cp.id == i, :);
        K_i = height(individual_data);
        Z_i = [individual_data.Z1(1); individual_data.Z2(1); individual_data.Z3(1)]; % column vector

        for j = 1:K_i
            S_ij = individual_data.time(j);

            for idx_l = 1:n
                l = unique_ids(idx_l);
                l_data = data_cp(data_cp.id == l, :);
                tau_l = l_data.tau(1);
                Z_l = [l_data.Z1(1); l_data.Z2(1); l_data.Z3(1)]; % column vector

                indicator = (log(tau_l) - log(S_ij)) >= (theta' * (Z_i - Z_l));

                if indicator
                    weight_denominator = 0;
                    for idx_l2 = 1:n
                        l2 = unique_ids(idx_l2);
                        l2_data = data_cp(data_cp.id == l2, :);
                        tau_l2 = l2_data.tau(1);
                        Z_l2 = [l2_data.Z1(1); l2_data.Z2(1); l2_data.Z3(1)]; % column vector

                        weight_denominator = weight_denominator + ...
                            ((log(tau_l2) - log(S_ij)) >= (theta' * (Z_i - Z_l2)));
                    end

                    if weight_denominator > 0
                        contribution = (Z_i - Z_l) / weight_denominator;
                        total_sum = total_sum + contribution;
                    end
                end
            end
        end
    end

    obj_value2 = total_sum / n; % Returns a 3x1 column vector
end

function D_LR_B = estimate_D_LR_B(theta_G, Gamma_hat_G, data_cp, B)
    % Parameters
    n = length(unique(data_cp.id));  % Number of samples
    cov_matrix = (1/n) * Gamma_hat_G;  % Covariance matrix

    % Bootstrap samples of theta (size n x B)
    theta_tilde = mvnrnd(theta_G', cov_matrix, B)';

    % Compute S_LR_K for theta_G (3x1 vector) and for bootstrap samples (3xB matrix)
    S_LR_M_theta_G = S_LR_M(theta_G, data_cp);  % 3x1 vector
    S_LR_M_theta_tilde = zeros(3, B);           % 3xB matrix for bootstrap samples

    % Loop over B to compute S_LR_K for each bootstrap sample
    for k = 1:B
        S_LR_M_theta_tilde(:, k) = S_LR_M(theta_tilde(:, k), data_cp);  % Store 3x1 vector
    end

    % Construct Z (B x n+1) and Y (B x 3)
    Z = [ones(B, 1), (theta_tilde - theta_G)'];  % Z is Bx(n+1)
    Y = (S_LR_M_theta_tilde - S_LR_M_theta_G)';  % Y is Bx3, where each row is a 3x1 difference

    % Solve the least-squares system Z'D_LR_B_tilde = Z'Y
    D_LR_B_tilde = (Z' * Z) \ (Z' * Y);          % D_LR_B_tilde will be (n+1)x3
    D_LR_B_tilde = D_LR_B_tilde(2:end, :);       % Remove the first row to get nx3

    % Symmetrize the result (if applicable)
    D_LR_B = (1/2) * (D_LR_B_tilde + D_LR_B_tilde');
end

function d_hat_lr_b_inv = test_finite_d_hat_lr_b_inv(d_hat_lr_b)
    % Function to compute the matrix inverse of d_hat_lr_b
    % If the inverse is not finite or the matrix is singular, return NaN
    
    % Try to compute the matrix inverse
    try
        d_hat_lr_b_inv = inv(d_hat_lr_b);
        
        % Check if the result is finite
        if any(~isfinite(d_hat_lr_b_inv), 'all')
            d_hat_lr_b_inv = NaN;
        end
    catch
        % If the matrix is singular or not invertible, return NaN
        d_hat_lr_b_inv = NaN;
    end
end

function phi_w = phi_function_lr(S_ij, theta, Z_i, data_cp)
    % Compute S^0, S^1, and S^2 (marginal version)
    S_0 = S_0_func(S_ij, theta, Z_i, data_cp);
    S_1 = S_1_func(S_ij, theta, Z_i, data_cp);
    S_2 = S_2_func(S_ij, theta, Z_i, data_cp);

    % Check if S_0 is greater than 0
    if S_0 > 0
        % Calculate phi(w) for vector-valued covariates
        phi_w = (S_2 / S_0) - ((S_1 / S_0) * (S_1 / S_0)');
    else
        % Handle the case where S_0 <= 0
        warning('S_0 is less than or equal to zero, returning NaN');
        phi_w = NaN;
    end
end


% Function to compute \widehat{\Sigma}_LR
function Sigma_LR = calculate_sigma_lr(data_cp, theta_hat)
    % Generalized calculation of Sigma_LR for three covariates Z1, Z2, Z3
    n = length(unique(data_cp.id));
    total_sum = 0;
    unique_ids = unique(data_cp.id);

    for i = unique_ids'
        i_data = data_cp(data_cp.id == i, :);  % Subsetting data for individual i
        K_i = height(i_data);
        Z_i = [i_data.Z1(1); i_data.Z2(1); i_data.Z3(1)]; % Column vector of covariates
        for j = 1:K_i
            S_ij = i_data.time(j);  % Event/censor time for individual i at j
            % Compute \varphi(w) for marginal model
            phi_w = phi_function_lr(S_ij, theta_hat, Z_i, data_cp);
            % Only add phi_w if it is not NaN
            if ~isnan(phi_w)
                total_sum = total_sum + phi_w;
            end
        end
    end

    Sigma_LR = (1/n) * total_sum;
end


% Objective function_tilde_lr_m
function obj_value1 = objective_function_tilde_lr_m(theta, data_cp, sigma)
    % Efficient and generalized objective function for three covariates Z1, Z2, Z3 
    unique_ids = unique(data_cp.id);
    n = numel(unique_ids);
    total_sum = zeros(3, 1); % for three covariates (column vector)

    for idx_i = 1:n
        i_data = data_cp(data_cp.id == unique_ids(idx_i), :);
        K_i = height(i_data);
        Z_i = [i_data.Z1(1); i_data.Z2(1); i_data.Z3(1)]; % Column vector

        for j = 1:K_i
            S_ij = i_data.time(j);

            % Compute indicator and denominator for all individuals 
            indicator_vec = false(n, 1);
            denominator = 0;
            Z_l_mat = zeros(3, n); % 3 x n for vectorized covariates

            for idx_l = 1:n
                l_data = data_cp(data_cp.id == unique_ids(idx_l), :);
                tau_l = l_data.tau(1);
                Z_l = [l_data.Z1(1); l_data.Z2(1); l_data.Z3(1)];
                indicator = log(tau_l) - log(S_ij) >= dot(theta, Z_i - Z_l);
                indicator_vec(idx_l) = indicator;
                Z_l_mat(:, idx_l) = Z_l;

                if indicator
                    denominator = denominator + 1;
                end
            end

            % Add contributions for all individuals l where indicator is true
            if denominator > 0
                for idx_l = 1:n
                    if indicator_vec(idx_l)
                        Z_l = Z_l_mat(:, idx_l);
                        contribution = (Z_i - Z_l) / denominator;
                        total_sum = total_sum + contribution;
                    end
                end
            end
        end
    end

    % Ensure sigma is column vector
    sigma_vec = sqrt((1/n) * sigma(:));
    obj_value1 = norm(total_sum / n - sigma_vec); % returns scalar
end

% Optimization to find the parameter theta
function estimated_theta_tilde_lr_m = optimize_theta_tilde_lr_m(data_cp, sigma)
estimated_theta_tilde_lr_m = zeros(3, 3);

% Optimization options
options = optimset('fminsearch');
options.Display = 'off'; % Suppress output during optimization

% Calculate the square root of the 3x3 matrix sigma
sigma_sqrt = sqrtm(sigma);

% Loop over each column of the square root matrix
for k = 1:3
    % Extract the k-th column of the square root of sigma
    sigma_k = sigma_sqrt(:, k);

    % Initial guess for theta (3-dimensional vector)
    initial_theta_tilde_m = [0.2; 0.1; 0.7];

    % Optimize theta by minimizing the objective function
    estimated_theta_tilde_lr_m(:, k) = fminsearch(@(theta) ...
        objective_function_tilde_lr_m(theta, data_cp, sigma_k), ...
        initial_theta_tilde_m, options);
end
end


% Estimating Lambda_LR_M
function lambda_hat = Lambda_LR_M(theta_LR_M, t, data_cp)
    % Vectorized Lambda_LR_M estimation for three covariates Z1, Z2, Z3
    % theta_LR_M: column vector (3x1)
    % Z_i: column vector (3x1)

    unique_ids = unique(data_cp.id);
    n = numel(unique_ids);

    % Precompute covariates and tau for all individuals
    Z_mat = zeros(3, n); % 3 x n matrix, each column is Z for one individual
    tau_vec = zeros(1, n); % tau for each individual
    for idx = 1:n
        subj_data = data_cp(data_cp.id == unique_ids(idx), :);
        Z_mat(:, idx) = [subj_data.Z1(1); subj_data.Z2(1); subj_data.Z3(1)]; % column vector
        tau_vec(idx) = subj_data.tau(1);
    end

    total_sum = 0;

    % Loop over individuals/events (outer loop only for observed events)
    for idx_i = 1:n
        i_data = data_cp(data_cp.id == unique_ids(idx_i), :);
        K_i = height(i_data);
        Z_i = Z_mat(:, idx_i); % column vector

        S_vec = i_data.time(:)'; % 1 x K_i

        % Loop over all events for subject i
        for jj = 1:K_i
            S_ij = S_vec(jj);

            % Numerator condition: log(t) - log(S_ij) >= theta' * Z_i
            numerator = (log(t) - log(S_ij)) >= (theta_LR_M' * Z_i);

            if numerator
                % Denominator: vectorized over all subjects
                Z_diff_mat = Z_i - Z_mat; % 3 x n matrix
                theta_dot_diff = theta_LR_M' * Z_diff_mat; % 1 x n row vector
                denom_indicator = (log(tau_vec) - log(S_ij)) >= theta_dot_diff; % 1 x n logical
                weight_denominator = sum(denom_indicator);

                if weight_denominator > 0
                    total_sum = total_sum + (1 / weight_denominator);
                end
            end
        end
    end

    lambda_hat = total_sum;
end

% Function to evaluate Lambda_LR_M at t = 1 and t = 3
function [lambda_hat_t1, lambda_hat_t2] = evaluate_Lambda_LR_M(theta_LR_M, data_cp)
    % Generalized evaluation for Lambda_LR_M at two time points
    t1 = 1;
    t2 = 3;

    lambda_hat_t1 = Lambda_LR_M(theta_LR_M, t1, data_cp);
    lambda_hat_t2 = Lambda_LR_M(theta_LR_M, t2, data_cp);
end


% Estimating Sigma_Lambda_2_LR_M
function first_term = First_Term_LR_M(theta_LR_M, t, data_cp)
    % Vectorized First_Term_LR_M for three covariates Z1, Z2, Z3
    unique_ids = unique(data_cp.id);
    n = numel(unique_ids);

    % Precompute covariates and tau for all individuals
    Z_mat = zeros(3, n); % 3 x n matrix
    tau_vec = zeros(1, n); % 1 x n vector
    for idx = 1:n
        subj_data = data_cp(data_cp.id == unique_ids(idx), :);
        Z_mat(:, idx) = [subj_data.Z1(1); subj_data.Z2(1); subj_data.Z3(1)];
        tau_vec(idx) = subj_data.tau(1);
    end

    total_sum = 0;

    for idx_i = 1:n
        i_data = data_cp(data_cp.id == unique_ids(idx_i), :);
        K_i = height(i_data);
        Z_i = Z_mat(:, idx_i);

        S_vec = i_data.time(:)'; % 1 x K_i

        for jj = 1:K_i
            S_ij = S_vec(jj);

            % Numerator condition
            numerator = log(t) - log(S_ij) >= theta_LR_M(:)' * Z_i;

            if numerator
                % Vectorize denominator calculation
                Z_diff_mat = Z_i - Z_mat; % 3 x n
                theta_dot_diff = theta_LR_M(:)' * Z_diff_mat; % 1 x n
                denom_indicator = (log(tau_vec) - log(S_ij)) >= theta_dot_diff; % 1 x n logical
                weight_denominator = sum(denom_indicator);

                if weight_denominator > 0
                    total_sum = total_sum + 1 / (weight_denominator^2);
                end
            end
        end
    end

    first_term = n * total_sum;
end

% Estimate A_hat
function A_hat_B_tilde = estimate_A_hat_M_LR(theta_LR_M, Gamma_hat_LR, t, data_cp, B)
    n = numel(unique(data_cp.id));
    cov_matrix = (1/n) * Gamma_hat_LR;  % 3x3 covariance matrix

    % Bootstrap samples of theta: each row is a 1x3 vector
    theta_tilde = mvnrnd(theta_LR_M', cov_matrix, B);  % B x 3

    % Compute Lambda_LR_M for theta_LR_M and for each bootstrap theta_tilde
    lambda_hat_theta_LR = Lambda_LR_M(theta_LR_M, t, data_cp);    % scalar
    lambda_hat_tilde = zeros(1, B);

    for k = 1:B
        lambda_hat_tilde(k) = Lambda_LR_M(theta_tilde(:,k), t, data_cp);   % scalar
    end

    % Construct Z and Y for regression
    Z = [ones(B, 1), (theta_LR_M' - theta_tilde')]; % B x 4 (if theta has 3 elements)
    Y = lambda_hat_tilde' - lambda_hat_theta_LR; % B x 1

    % Linear regression (multivariate theta, scalar Y)
    A_hat_B_tilde = (Z' * Z) \ (Z' * Y); % (4 x B) * (B x 1) = 4 x 1

    % Remove intercept (first row), keep only coefficient part
    A_hat_B_tilde = A_hat_B_tilde(2:end, :); % 3 x 1
    A_hat_B_tilde = (1/2) * (A_hat_B_tilde + A_hat_B_tilde);
end

% Function to evaluate Sigma_Lambda_2_LR_M at t = 1 and t = 3
function [Sigma_Lambda_2_LR_M_t1, Sigma_Lambda_2_LR_M_t2] = evaluate_Sigma_Lambda_2_LR_M(theta_LR_M, Gamma_hat_LR, data_cp, B)
    % Generalized evaluation of Sigma_Lambda_2_LR_M for vector-valued theta (three covariates)
    % Evaluates at t = 1 and t = 3

    % Evaluate at t = 1
    t1 = 1;
    first_term_t1 = First_Term_LR_M(theta_LR_M, t1, data_cp);
    A_hat_t1 = estimate_A_hat_M_LR(theta_LR_M, Gamma_hat_LR, t1, data_cp, B); % 3x1 vector
    Sigma_Lambda_2_LR_M_t1 = first_term_t1 + (A_hat_t1' * Gamma_hat_LR * A_hat_t1); % scalar

    % Evaluate at t = 3
    t2 = 3;
    first_term_t2 = First_Term_LR_M(theta_LR_M, t2, data_cp);
    A_hat_t2 = estimate_A_hat_M_LR(theta_LR_M, Gamma_hat_LR, t2, data_cp, B); % 3x1 vector
    Sigma_Lambda_2_LR_M_t2 = first_term_t2 + (A_hat_t2' * Gamma_hat_LR * A_hat_t2); % scalar
end

% API MODEL

% ALGORITHM 2

% Accessory Functions

% Score Function S_LR_P
function S_LR_P = S_LR_P(theta, data_cp)
    % Score function S_LR_P for ALGORITHM 2 (API Model)
    % Handles vector-valued covariates as column vectors (e.g., [Z1; Z2; Z3])

    unique_ids = unique(data_cp.id);
    n = numel(unique_ids);

    total_sum_vec = zeros(3, 1); % For three covariates as a column vector

    for idx_i = 1:n
        i = unique_ids(idx_i);
        i_data = data_cp(data_cp.id == i, :);
        K_i = height(i_data);
        Z_i = [i_data.Z1(1); i_data.Z2(1); i_data.Z3(1)]; % Column vector for covariates

        for j = 1:K_i
            T_ij = i_data.gap_time(j);

            % Compute Y_l for all individuals
            Y_l_values = zeros(n, 1);
            Y_l_sum = 0;
            for idx_l_inner = 1:n
                l_inner = unique_ids(idx_l_inner);
                l_inner_data = data_cp(data_cp.id == l_inner, :);
                K_l_inner = height(l_inner_data);
                tau_l_inner = l_inner_data.tau(1);
                S_l_kl_inner = l_inner_data.time(K_l_inner);
                Z_l_inner = [l_inner_data.Z1(1); l_inner_data.Z2(1); l_inner_data.Z3(1)]; % Column vector
                Y_l_inner = Y_l_P_func(tau_l_inner, S_l_kl_inner, T_ij, theta, Z_i, Z_l_inner, K_l_inner, l_inner_data);
                Y_l_values(idx_l_inner) = Y_l_inner;
                Y_l_sum = Y_l_sum + Y_l_inner;
            end

            for idx_l = 1:n
                l = unique_ids(idx_l);
                l_data = data_cp(data_cp.id == l, :);
                Z_l = [l_data.Z1(1); l_data.Z2(1); l_data.Z3(1)]; % Column vector

                if Y_l_sum > 0
                    weight = Y_l_values(idx_l) / Y_l_sum;
                else
                    weight = 0;
                end

                Z_diff = Z_i - Z_l; % Column vector difference
                total_sum_vec = total_sum_vec + weight * Z_diff;
            end
        end
    end
    S_LR_P = total_sum_vec / n; % Returns 3x1 column vector (for three covariates)
end

% Function to compute \varphi(w)
function phi_w = phi_function_lr_p(T_ij, theta, Z_i, data_cp)
    % Generalized phi(w) for API model with vector-valued covariates Z1, Z2, Z3
    % Ensure Z_i and theta are column vectors of the same length

    S_0 = S_0_func_p(T_ij, theta, Z_i, data_cp);
    S_1 = S_1_func_p(T_ij, theta, Z_i, data_cp);
    S_2 = S_2_func_p(T_ij, theta, Z_i, data_cp);

    if S_0 > 0
        % S_1/S_0 is a column vector; want outer product not element-wise square
        phi_w = (S_2 / S_0) - (S_1 / S_0) * (S_1 / S_0)';
    else
        warning('S_0 is less than or equal to zero, returning NaN');
        phi_w = NaN;
    end
end


% Function to compute \widehat{\Sigma}_LR
function Sigma_LR = calculate_sigma_lr_p(data_cp, theta_hat)
    % Generalized calculation of Sigma_LR for API model with multiple covariates (e.g., Z1, Z2, Z3)
    unique_ids = unique(data_cp.id);
    n = numel(unique_ids);
    total_sum = zeros(length(theta_hat), length(theta_hat)); % Matrix for vector-valued covariates

    for idx_i = 1:n
        i = unique_ids(idx_i);
        i_data = data_cp(data_cp.id == i, :);  % Subsetting data for individual i
        K_i = height(i_data);
        Z_i = [i_data.Z1(1); i_data.Z2(1); i_data.Z3(1)]; % Column vector covariate

        for j = 1:K_i
            T_ij = i_data.gap_time(j);  
            phi_w = phi_function_lr_p(T_ij, theta_hat, Z_i, data_cp); 
            if ~isnan(phi_w)
                total_sum = total_sum + phi_w;
            end
        end
    end

    Sigma_LR = (1/n) * total_sum;
end


% Objective function_tilde_lr_p
function obj_value1 = objective_function_tilde_lr_p(theta, data_cp, sigma)
    % Generalized for vector-valued covariates (e.g., [Z1; Z2; Z3])
    unique_ids = unique(data_cp.id);
    n = numel(unique_ids);

    total_sum_vec = zeros(3, 1); % for three covariates as column vector

    for idx_i = 1:n
        i = unique_ids(idx_i);
        i_data = data_cp(data_cp.id == i, :);
        K_i = height(i_data); % number of events for individual i
        Z_i = [i_data.Z1(1); i_data.Z2(1); i_data.Z3(1)]; % column vector covariate

        for j = 1:K_i
            T_ij = i_data.gap_time(j);

            Y_l_values = zeros(n, 1);
            Y_l_sum = 0;
            for idx_l_inner = 1:n
                l_inner = unique_ids(idx_l_inner);
                l_inner_data = data_cp(data_cp.id == l_inner, :);
                K_l_inner = height(l_inner_data);
                tau_l_inner = l_inner_data.tau(1);
                S_l_kl_inner = l_inner_data.time(K_l_inner);
                Z_l_inner = [l_inner_data.Z1(1); l_inner_data.Z2(1); l_inner_data.Z3(1)]; % column vector
                Y_l_inner = Y_l_P_func(tau_l_inner, S_l_kl_inner, T_ij, theta, Z_i, Z_l_inner, K_l_inner, l_inner_data);
                Y_l_values(idx_l_inner) = Y_l_inner;
                Y_l_sum = Y_l_sum + Y_l_inner;
            end

            for idx_l = 1:n
                l = unique_ids(idx_l);
                l_data = data_cp(data_cp.id == l, :);
                Z_l = [l_data.Z1(1); l_data.Z2(1); l_data.Z3(1)]; % column vector
                if Y_l_sum > 0
                    Y_l = Y_l_values(idx_l) / Y_l_sum;
                else
                    Y_l = 0;
                end
                Z_diff = Z_i - Z_l;
                total_sum_vec = total_sum_vec + Y_l * Z_diff;
            end
        end
    end
    % sigma is assumed to be a column vector of length 3
    obj_value1 = norm(total_sum_vec / n - sqrt((1/n) * sigma));
end


% Optimization to find the parameter theta
function estimated_theta_tilde_lr_p = optimize_theta_tilde_lr_p(data_cp, sigma)
estimated_theta_tilde_lr_p = zeros(3, 3);

% Optimization options
options = optimset('fminsearch');
options.Display = 'off'; % Suppress output during optimization

% Calculate the square root of the 3x3 matrix sigma
sigma_sqrt = sqrtm(sigma);

% Loop over each column of the square root matrix
for k = 1:3
    % Extract the k-th column of the square root of sigma
    sigma_k = sigma_sqrt(:, k);

    % Initial guess for theta (3-dimensional vector)
    initial_theta_tilde_p = [0.2; 0.1; 0.7];

    % Optimize theta by minimizing the objective function
    estimated_theta_tilde_lr_p(:, k) = fminsearch(@(theta) ...
        objective_function_tilde_lr_p(theta, data_cp, sigma_k), ...
        initial_theta_tilde_p, options);

end
end

% Estimating Lambda_LR_P

% Lambda_LR_P Function
function Lambda_0_t = Lambda_LR_P(theta, data_cp, t)
    % Generalized Lambda_LR_P for API model with vector-valued covariates (e.g., [Z1; Z2; Z3])
    unique_ids = unique(data_cp.id);
    n = numel(unique_ids);

    total_sum = 0;

    for idx_i = 1:n
        i = unique_ids(idx_i);
        i_data = data_cp(data_cp.id == i, :);
        K_i = height(i_data); % Number of observed events for individual i
        Z_i = [i_data.Z1(1); i_data.Z2(1); i_data.Z3(1)]; % column vector covariate

        for j = 1:K_i
            T_ij = i_data.gap_time(j);

            Y_l_sum = 0;
            for idx_l = 1:n
                l = unique_ids(idx_l);
                l_data = data_cp(data_cp.id == l, :);
                K_l = height(l_data);
                tau_l = l_data.tau(1);
                S_l = l_data.time(K_l);
                Z_l = [l_data.Z1(1); l_data.Z2(1); l_data.Z3(1)]; % column vector
                Y_l = Y_l_P_func(tau_l, S_l, T_ij, theta, Z_i, Z_l, K_l, l_data);
                Y_l_sum = Y_l_sum + Y_l;
            end

            % Calculate the indicator function for vector-valued theta and Z_i
            X_ij = i_data.gap_time(j) * exp(theta' * Z_i); % theta' * Z_i for column vectors
            indicator = X_ij <= t;

            if Y_l_sum > 0
                total_sum = total_sum + (indicator / Y_l_sum);
            end
        end
    end

    Lambda_0_t = total_sum;
end

% Function to evaluate Lambda_LR_P at t = 1 and t = 3
function [lambda_hat_t1, lambda_hat_t2] = evaluate_Lambda_LR_P(theta_LR_P, data_cp)
    % Evaluate Lambda_LR_P at two time points: t = 1 and t = 3
    t1 = 1;
    lambda_hat_t1 = Lambda_LR_P(theta_LR_P, data_cp, t1);

    t2 = 3;
    lambda_hat_t2 = Lambda_LR_P(theta_LR_P, data_cp, t2);
end

% Estimating Sigma_Lambda_2_LR_P

function first_term = First_Term_LR_P(theta, data_cp, t)
    % Generalized First_Term_LR_P for vector-valued covariates (e.g., [Z1; Z2; Z3])
    unique_ids = unique(data_cp.id);
    n = numel(unique_ids);

    total_sum = 0;
    
    for idx_i = 1:n
        i = unique_ids(idx_i);
        i_data = data_cp(data_cp.id == i, :);
        K_i = height(i_data);
        Z_i = [i_data.Z1(1); i_data.Z2(1); i_data.Z3(1)]; % column vector covariate
        
        for j = 1:K_i
            T_ij = i_data.gap_time(j);

            Y_l_sum = 0;
            for idx_l = 1:n
                l = unique_ids(idx_l);
                l_data = data_cp(data_cp.id == l, :);
                K_l = height(l_data);
                tau_l = l_data.tau(1);
                S_l = l_data.time(K_l);
                Z_l = [l_data.Z1(1); l_data.Z2(1); l_data.Z3(1)]; % column vector
                Y_l = Y_l_P_func(tau_l, S_l, T_ij, theta, Z_i, Z_l, K_l, l_data);
                Y_l_sum = Y_l_sum + Y_l;
            end

            % Use theta' * Z_i for column vectors
            X_ij = i_data.gap_time(j) * exp(theta' * Z_i);
            indicator = X_ij <= t;
            
            if Y_l_sum > 0
                total_sum = total_sum + (indicator / (Y_l_sum)^2);
            end
        end
    end

    first_term = n * total_sum;
end


% Estimate A_hat
function A_hat_B_tilde = estimate_A_hat_P_LR(theta_LR_P, Gamma_hat_LR, t, data_cp, B)
    % Estimate A_hat for API LR model via parametric bootstrap
    % theta_LR_P: column vector (p x 1), Gamma_hat_LR: pxp covariance matrix

    n = numel(unique(data_cp.id));
    cov_matrix = (1/n) * Gamma_hat_LR;  % pxp covariance matrix

    % Ensure theta_LR_P is a row for mvnrnd, but column for Lambda_LR_P
    theta_LR_P_row = theta_LR_P(:)'; % 1 x p

    % Bootstrap samples of theta (B x p)
    theta_tilde = mvnrnd(theta_LR_P_row, cov_matrix, B);

    % Compute Lambda_LR_P for theta_LR_P and for each bootstrap theta_tilde
    lambda_hat_theta_LR = Lambda_LR_P(theta_LR_P(:), data_cp, t);    % scalar
    lambda_hat_tilde = zeros(B, 1);

    for k = 1:B
        lambda_hat_tilde(k) = Lambda_LR_P(theta_tilde(k, :)', data_cp, t);   % Pass as column vector
    end

    % Construct Z and Y for regression
    Z = [ones(B, 1), bsxfun(@minus, theta_tilde, theta_LR_P_row)]; % B x (p+1)
    Y = lambda_hat_tilde - lambda_hat_theta_LR; % B x 1

    % Linear regression (multivariate theta, scalar Y)
    A_hat_B_tilde = (Z' * Z) \ (Z' * Y); % (p+1 x B) * (B x 1) = (p+1) x 1

    % Remove intercept (first row), keep only coefficient part
    A_hat_B_tilde = A_hat_B_tilde(2:end, :); % px1

    % Symmetrize (for vector, this is a no-op, for consistency)
    % A_hat_B_tilde = (1/2) * (A_hat_B_tilde + A_hat_B_tilde);
end


% Function to evaluate Sigma_Lambda_2_LR_M at t = 0.5 and t = 1.5
function [Sigma_Lambda_2_LR_P_t1, Sigma_Lambda_2_LR_P_t2] = evaluate_Sigma_Lambda_2_LR_P(theta_LR_P, Gamma_hat_LR, data_cp, B)
    % Generalized evaluation of Sigma_Lambda_2_LR_P for vector-valued theta (multi-covariate API model)
    % Evaluates at t = 1 and t = 3

    % Evaluate at t = 1
    t1 = 1;
    first_term_t1 = First_Term_LR_P(theta_LR_P, data_cp, t1);
    A_hat_t1 = estimate_A_hat_P_LR(theta_LR_P, Gamma_hat_LR, t1, data_cp, B); % px1 vector
    Sigma_Lambda_2_LR_P_t1 = first_term_t1 + (A_hat_t1' * Gamma_hat_LR * A_hat_t1); % scalar

    % Evaluate at t = 3
    t2 = 3;
    first_term_t2 = First_Term_LR_P(theta_LR_P, data_cp, t2);
    A_hat_t2 = estimate_A_hat_P_LR(theta_LR_P, Gamma_hat_LR, t2, data_cp, B); % px1 vector
    Sigma_Lambda_2_LR_P_t2 = first_term_t2 + (A_hat_t2' * Gamma_hat_LR * A_hat_t2); % scalar
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  MAIN FUNCTIONS                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Read the CSV file into a table or matrix
data_cp = readtable('data_cp.csv');  % For table format

% Generate data
%n = 50;
%data_cp = generate_recurrent_data_cp(n, tau_max, alpha, gamma);

% Display the first few rows of the data
disp(head(data_cp));

% Assuming 'data_cp' is your table variable and 'id' is the column name
unique_ids = unique(data_cp.id); % Get unique IDs
num_unique_ids = length(unique_ids); % Count the number of unique IDs
n = num_unique_ids;

% Display the number of unique IDs
disp(['Number of unique IDs in the data: ', num2str(num_unique_ids)]);

% Plot Objective Function
plot_objective_function_g(data_cp);

% Display results Gehan Weight
initial_theta_p = [-0.5; -0.20; 0.01];
disp(objective_function_p(initial_theta_p, data_cp));


% Estimate for theta_g_p
estimated_theta_g_p = optimize_theta_p(data_cp);

% Calculate Sigma_G for theta_g_p
estimated_sigma_g_p = calculate_sigma_p(data_cp, estimated_theta_g_p);

% Optimize theta_tilde_g_p
estimated_theta_tilde_g_p = optimize_theta_tilde_p(data_cp, estimated_sigma_g_p);
    
            
% Assuming estimated_theta_tilde_g_p is a 3x3 matrix
% and estimated_theta_g_p is a 3x1 column vector

% Initialize the result matrix
result_subtract = zeros(3, 3);

% Loop over each column
for i = 1:3
    result_subtract(:, i) = estimated_theta_tilde_g_p(:, i) - estimated_theta_g_p;
end

% Calculate Gamma for
% theta_g_k
estimated_gamma_g_p = num_unique_ids * (result_subtract)*(result_subtract)';

% Calculate Lambda_hat_g_p
[Lambda_hat_g_p_t1, Lambda_hat_g_p_t2] = evaluate_Lambda_G_P(estimated_theta_g_p, data_cp);
            
% Calcualte Sigma_Lambda_2_G_P
[Sigma_Lambda_2_g_p_t1, Sigma_Lambda_2_g_p_t2] = evaluate_Sigma_Lambda_2_G_P(estimated_theta_g_p, estimated_gamma_g_p , data_cp, B2);


fprintf('Estimated Theta (g_p): \n');
disp(estimated_theta_g_p);

fprintf('Estimated Sigma (g_p): %.4f\n', estimated_sigma_g_p);

fprintf('Estimated Theta Tilde (g_p): \n');
disp(estimated_theta_tilde_g_p);

fprintf('Estimated Gamma (g_p): %.4f\n', estimated_gamma_g_p);

fprintf('Lambda Hat g_p at t1: %.4f\n', Lambda_hat_g_p_t1);
fprintf('Lambda Hat g_p at t2: %.4f\n', Lambda_hat_g_p_t2);

fprintf('Sigma Lambda 2 g_p at t1: %.4f\n', Sigma_Lambda_2_g_p_t1);
fprintf('Sigma Lambda 2 g_p at t2: %.4f\n', Sigma_Lambda_2_g_p_t2);

% Display results Log-Rank Weight

% Calculate theta_lr_p            
score_lr_p = S_LR_P(estimated_theta_g_p, data_cp); 
d_hat_lr_b_p = estimate_D_LR_B(estimated_theta_g_p, estimated_gamma_g_p, data_cp, B1);            
d_hat_lr_b_p_inv = test_finite_d_hat_lr_b_inv(d_hat_lr_b_p);            
estimated_theta_lr_p = estimated_theta_g_p - (d_hat_lr_b_p_inv * score_lr_p);    

% Calculate Sigma_LR for theta_lr_p using the function calculate_sigma_lr_p
estimated_sigma_lr_p = calculate_sigma_lr_p(data_cp, estimated_theta_lr_p);

% Calculate Gamma for theta_lr_p
d_hat_lr_b_p_inv = test_finite_d_hat_lr_b_inv(d_hat_lr_b_p);
estimated_gamma_lr_p = d_hat_lr_b_p_inv' * estimated_sigma_lr_p * d_hat_lr_b_p_inv;

% Optimize theta_tilde_lr_p
estimated_theta_tilde_lr_p = optimize_theta_tilde_lr_p(data_cp, estimated_sigma_lr_p);

% Calculate GammaHuang for theta_lr_p 
estimated_gammahuang_lr_p = (sqrt(n) * (estimated_theta_tilde_lr_p - estimated_theta_lr_p))^2;
            
% Calculate Lambda_hat_lr_p            
[Lambda_hat_lr_p_t1, Lambda_hat_lr_p_t2] = evaluate_Lambda_LR_P(estimated_theta_lr_p, data_cp);
                      
% Calcualte Sigma_Lambda_2_LR_P            
[Sigma_Lambda_2_lr_p_t1, Sigma_Lambda_2_lr_p_t2] = evaluate_Sigma_Lambda_2_LR_P(estimated_theta_lr_p, estimated_gamma_lr_p, data_cp, B2);
          
fprintf('Estimated Theta (lr_p): \n');
disp(estimated_theta_lr_p);

fprintf('Estimated Sigma (lr_p): %.4f\n', estimated_sigma_lr_p);

fprintf('Estimated Theta Tilde (lr_p): \n');
disp(estimated_theta_tilde_lr_p);

fprintf('Estimated GammaTh3 (lr_p): %.4f\n', estimated_gamma_lr_p);
fprintf('Estimated GammaHuang (lr_p): %.4f\n', estimated_gammahuang_lr_p);

fprintf('Lambda Hat lr_p at t1: %.4f\n', Lambda_hat_lr_p_t1);
fprintf('Lambda Hat lr_p at t2: %.4f\n', Lambda_hat_lr_p_t2);

fprintf('Sigma Lambda 2 lr_p at t1: %.4f\n', Sigma_Lambda_2_lr_p_t1);
fprintf('Sigma Lambda 2 lr_p at t2: %.4f\n', Sigma_Lambda_2_lr_p_t2);