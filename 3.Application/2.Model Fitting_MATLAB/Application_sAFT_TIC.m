% Parameters                                             %
alpha = 1.5;
B1 = 1500;                                                                                          
B2 = 1500;                                                                                          
tau_max = 3.5; % Maximum value for tau                                                                                                           
%-------------------------------------

function data_cp = generate_recurrent_data_cp(n, tau_max, alpha)
    % Generate recurrent event data with three covariates Z1, Z2, Z3
    % Each Zk ~ Bernoulli(p), tau ~ Uniform(0, tau_max)

    Z1 = binornd(1, 0.2, [n, 1]);
    Z2 = binornd(1, 0.5, [n, 1]);
    Z3 = binornd(1, 0.8, [n, 1]);
    tau = unifrnd(0, tau_max, [n, 1]);
    data_list = cell(n, 1);

    rho = 0.25;  % Correlation for Gumbel bivariate exponential

    for i = 1:n
        lambda = alpha; % Baseline rate
        % AFT scaling for all covariates
        scaling = exp(0.3*Z1(i) + 0.18*Z2(i) - 0.68*Z3(i));

        gap_times = [];
        k = 1;
        current_time = 0;

        while true
            if k == 1
                gap = exprnd(1 / lambda);
            else
                u = rand;
                v = rand;
                theta = 1 - rho;
                x = -log(u);
                y = -log(v);
                gap = max(x, y / theta) / lambda;
            end

            % Apply AFT-like effect
            gap = gap * scaling;

            if current_time + gap > tau(i)
                break;
            end

            gap_times(end + 1, 1) = gap;
            current_time = current_time + gap;
            k = k + 1;
        end

        % Compute event times
        if ~isempty(gap_times)
            event_times = cumsum(gap_times);
            last_event_time = event_times(end);
        else
            event_times = [];
            last_event_time = 0;
        end

        % Handle censoring
        if last_event_time < tau(i)
            censoring_time = tau(i);
            censoring_gap_time = tau(i) - last_event_time;
        else
            censoring_time = tau(i);
            censoring_gap_time = tau(i);  % No events case
        end

        % Construct data table for this individual
        all_times = [event_times; censoring_time];
        all_gaps = [gap_times; censoring_gap_time];
        all_events = [ones(length(event_times), 1); 0];

        individual_data = table( ...
            i * ones(length(all_times), 1), ...
            all_times, ...
            all_gaps, ...
            all_events, ...
            Z1(i) * ones(length(all_times), 1), ...
            Z2(i) * ones(length(all_times), 1), ...
            Z3(i) * ones(length(all_times), 1), ...
            tau(i) * ones(length(all_times), 1), ...
            'VariableNames', {'id', 'time', 'gap_time', ...
                              'event', 'Z1', 'Z2', 'Z3', 'tau'});

        data_list{i} = individual_data;
    end

    data_cp = vertcat(data_list{:});
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

function [lambda_hat_t1, lambda_hat_t2, lambda_hat_t3] = evaluate_Lambda_G_M(theta_G_M, data_cp)
    % Evaluate Lambda_G_M at t = 180, t = 365, t = 730 for three covariates

    % Time points for evaluation
    t1 = 180;
    t2 = 365;
    t3 = 730;

    % Compute Lambda_G_M at t1 and t2
    lambda_hat_t1 = Lambda_G_M(theta_G_M, t1, data_cp);
    lambda_hat_t2 = Lambda_G_M(theta_G_M, t2, data_cp);
    lambda_hat_t3 = Lambda_G_M(theta_G_M, t3, data_cp);
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

function [Sigma_Lambda_2_G_M_t1, Sigma_Lambda_2_G_M_t2, Sigma_Lambda_2_G_M_t3] = evaluate_Sigma_Lambda_2_G_M(theta_G_M, Gamma_hat_G, data_cp, B)
    % Evaluate Sigma_Lambda_2_G_M at t = 180, t = 365, t = 730 for three covariates

    % Evaluate at t = 180
    t1 = 180;
    first_term_t1 = First_Term_G_M(theta_G_M, t1, data_cp);
    A_hat_t1 = estimate_A_hat_M(theta_G_M, Gamma_hat_G, t1, data_cp, B);
    Sigma_Lambda_2_G_M_t1 = first_term_t1 + (A_hat_t1' * Gamma_hat_G * A_hat_t1);

    % Evaluate at t = 365
    t2 = 365;
    first_term_t2 = First_Term_G_M(theta_G_M, t2, data_cp);
    A_hat_t2 = estimate_A_hat_M(theta_G_M, Gamma_hat_G, t2, data_cp, B);
    Sigma_Lambda_2_G_M_t2 = first_term_t2 + (A_hat_t2' * Gamma_hat_G * A_hat_t2);
    
    % Evaluate at t = 730
    t3 = 730;
    first_term_t3 = First_Term_G_M(theta_G_M, t3, data_cp);
    A_hat_t3 = estimate_A_hat_M(theta_G_M, Gamma_hat_G, t3, data_cp, B);
    Sigma_Lambda_2_G_M_t3 = first_term_t3 + (A_hat_t3' * Gamma_hat_G * A_hat_t3);


end


% Plot Objective Function

function plot_objective_function(data_cp)
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
            obj_values(i, j) = objective_function(theta, data_cp);
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
    % Lambda_LR_M: Estimate cumulative baseline hazard for LR-M model
    % theta_LR_M: column vector (3x1)
    % t: evaluation time
    % data_cp: input data table with covariates, event times, and censoring

    % Get unique subject IDs
    ids = unique(data_cp.id);
    n = numel(ids);

    % Precompute covariates and tau for all subjects, matching your reference code style
    [~, ia] = unique(data_cp.id, 'first');
    tau_vec = data_cp.tau(ia); % n x 1
    Z_mat = [data_cp.Z1(ia), data_cp.Z2(ia), data_cp.Z3(ia)]; % n x 3

    total_sum = 0;

    % Ensure theta_LR_M is a column vector
    theta_LR_M = theta_LR_M(:);

    % Loop over subjects and their observed events
    for idx_i = 1:n
        i = ids(idx_i);
        i_rows = (data_cp.id == i);
        individual_data = data_cp(i_rows, :);
        Z_i = Z_mat(idx_i, :)';      % 3 x 1 column vector
        S_vec = individual_data.time; % all event/censor times for subject i

        for S_ij = S_vec'
            % Numerator: log(t) - log(S_ij) >= theta' * Z_i
            numerator = (log(t) - log(S_ij)) >= (theta_LR_M' * Z_i);

            if numerator
                % Denominator: sum over all subjects
                Z_i_mat = repmat(Z_i', n, 1); % n x 3
                Z_diff = Z_i_mat - Z_mat;     % n x 3
                theta_dot_diff = Z_diff * theta_LR_M; % n x 1
                denom_indicator = (log(tau_vec) - log(S_ij)) >= theta_dot_diff; % n x 1 logical
                weight_denominator = sum(denom_indicator);

                if weight_denominator > 0
                    total_sum = total_sum + (1 / weight_denominator);
                end
            end
        end
    end

    lambda_hat = total_sum;
end

% Function to evaluate Lambda_LR_M at t = 180, t = 365, t = 730
function [lambda_hat_t1, lambda_hat_t2, lambda_hat_t3] = evaluate_Lambda_LR_M(theta_LR_M, data_cp)
    % Generalized evaluation for Lambda_LR_M at three time points
    t1 = 180;
    t2 = 365;
    t3 = 730;

    lambda_hat_t1 = Lambda_LR_M(theta_LR_M, t1, data_cp);
    lambda_hat_t2 = Lambda_LR_M(theta_LR_M, t2, data_cp);
    lambda_hat_t3 = Lambda_LR_M(theta_LR_M, t3, data_cp);
end


% Estimating Sigma_Lambda_2_LR_M
function first_term = First_Term_LR_M(theta_LR_M, t, data_cp)
    % First_Term_LR_M: Efficient, vectorized estimation for three covariates
    % Returns n * sum over (1/denominator^2) for eligible event times

    ids = unique(data_cp.id);
    n = numel(ids);

    % Pre-extract tau and covariates for all subjects (matches previous style)
    [~, ia] = unique(data_cp.id, 'first');
    tau_vec = data_cp.tau(ia);                                 % n x 1
    Z_mat = [data_cp.Z1(ia), data_cp.Z2(ia), data_cp.Z3(ia)];  % n x 3

    total_sum = 0;

    % Ensure theta_LR_M is a column vector
    theta_LR_M = theta_LR_M(:);

    % Loop over subjects and their observed events
    for idx_i = 1:n
        i = ids(idx_i);
        i_rows = (data_cp.id == i);
        individual_data = data_cp(i_rows, :);
        Z_i = Z_mat(idx_i, :)';      % 3 x 1 column vector
        S_vec = individual_data.time; % all event/censor times for subject i

        for S_ij = S_vec'
            % Numerator: log(t) - log(S_ij) >= theta' * Z_i
            numerator = (log(t) - log(S_ij)) >= (theta_LR_M' * Z_i);

            if numerator
                % Denominator: sum over all subjects
                Z_i_mat = repmat(Z_i', n, 1); % n x 3
                Z_diff = Z_i_mat - Z_mat;     % n x 3
                theta_dot_diff = Z_diff * theta_LR_M; % n x 1
                denom_indicator = (log(tau_vec) - log(S_ij)) >= theta_dot_diff; % n x 1 logical
                weight_denominator = sum(denom_indicator);

                if weight_denominator > 0
                    total_sum = total_sum + (1 / (weight_denominator^2));
                end
            end
        end
    end

    first_term = n * total_sum;
end

% Estimate A_hat
function A_hat_B_tilde = estimate_A_hat_M_LR(theta_LR_M, Gamma_hat_LR, t, data_cp, B)
    % Efficient vectorized estimation for three covariates using multivariate bootstrap (LR-M version)

    ids = unique(data_cp.id);
    n = numel(ids);

    % Covariance matrix: ensure correct dimension and scaling
    cov_matrix = (1/n) * Gamma_hat_LR;  % 3x3 covariance matrix

    % Bootstrap samples of theta: each column is a 3x1 vector
    theta_tilde = mvnrnd(theta_LR_M(:)', cov_matrix, B)';  % 3 x B

    % Compute Lambda_LR_M for all bootstrap samples and the original estimate
    lambda_hat_theta_LR = Lambda_LR_M(theta_LR_M(:), t, data_cp); % scalar
    lambda_hat_tilde = zeros(1, B);

    for k = 1:B
        lambda_hat_tilde(k) = Lambda_LR_M(theta_tilde(:, k), t, data_cp);
    end

    % Construct Z and Y for regression
    Z = [ones(B, 1), (theta_LR_M(:)' - theta_tilde')]; % B x 4
    Y = (lambda_hat_tilde' - lambda_hat_theta_LR);     % B x 1

    % OLS regression: (Z'Z)^(-1) Z'Y
    A_hat_B_tilde_full = (Z' * Z) \ (Z' * Y);          % 4 x 1

    % Remove intercept row (first row)
    A_hat_B_tilde = A_hat_B_tilde_full(2:end, :);      % 3 x 1

    % Symmetrize (for vector, this is a no-op, but included for consistency)
    A_hat_B_tilde = (1/2) * (A_hat_B_tilde + A_hat_B_tilde);
end

% Function to evaluate Sigma_Lambda_2_LR_M at t = 180, t = 365, t = 730
function [Sigma_Lambda_2_LR_M_t1, Sigma_Lambda_2_LR_M_t2, Sigma_Lambda_2_LR_M_t3] = evaluate_Sigma_Lambda_2_LR_M(theta_LR_M, Gamma_hat_LR, data_cp, B)
    % Generalized evaluation of Sigma_Lambda_2_LR_M for vector-valued theta (three covariates)
    % Evaluates at t = 180, t = 365, t = 730

    % Evaluate at t = 180
    t1 = 180;
    first_term_t1 = First_Term_LR_M(theta_LR_M, t1, data_cp);
    A_hat_t1 = estimate_A_hat_M_LR(theta_LR_M, Gamma_hat_LR, t1, data_cp, B); % 3x1 vector
    Sigma_Lambda_2_LR_M_t1 = first_term_t1 + (A_hat_t1' * Gamma_hat_LR * A_hat_t1); % scalar

    % Evaluate at t = 365
    t2 = 365;
    first_term_t2 = First_Term_LR_M(theta_LR_M, t2, data_cp);
    A_hat_t2 = estimate_A_hat_M_LR(theta_LR_M, Gamma_hat_LR, t2, data_cp, B); % 3x1 vector
    Sigma_Lambda_2_LR_M_t2 = first_term_t2 + (A_hat_t2' * Gamma_hat_LR * A_hat_t2); % scalar

    % Evaluate at t = 730
    t3 = 730;
    first_term_t3 = First_Term_LR_M(theta_LR_M, t3, data_cp);
    A_hat_t3 = estimate_A_hat_M_LR(theta_LR_M, Gamma_hat_LR, t3, data_cp, B); % 3x1 vector
    Sigma_Lambda_2_LR_M_t3 = first_term_t3 + (A_hat_t3' * Gamma_hat_LR * A_hat_t3); % scalar
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  MAIN FUNCTIONS                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Read the CSV file into a table or matrix
data_cp = readtable('data_cp.csv');  % For table format

% Generate data
%n = 100;
%data_cp = generate_recurrent_data_cp(n, tau_max, alpha);

% Display the first few rows of the data
disp(head(data_cp));

% Assuming 'data_cp' is your table variable and 'id' is the column name
unique_ids = unique(data_cp.id); % Get unique IDs
num_unique_ids = length(unique_ids); % Count the number of unique IDs
n = num_unique_ids;

% Display the number of unique IDs
disp(['Number of unique IDs in the data: ', num2str(num_unique_ids)]);

% Plot Objective Function
plot_objective_function(data_cp);

% Display results Gehan Weight
initial_theta_m = [-0.5; -0.20; 0.01];
disp(objective_function(initial_theta_m, data_cp));

% Estimate for theta_g_m
estimated_theta_g_m = optimize_theta(data_cp);

% Calculate Sigma_G for theta_g_m
estimated_sigma_g_m = calculate_sigma(data_cp, estimated_theta_g_m);

% Optimize theta_tilde_g_m
estimated_theta_tilde_g_m = optimize_theta_tilde(data_cp, estimated_sigma_g_m);
    
            
% Assuming estimated_theta_tilde_g_m is a 3x3 matrix
% and estimated_theta_g_m is a 3x1 column vector

% Initialize the result matrix
result_subtract = zeros(3, 3);

% Loop over each column
for i = 1:3
    result_subtract(:, i) = estimated_theta_tilde_g_m(:, i) - estimated_theta_g_m;
end

% Calculate Gamma for
% theta_g_m
estimated_gamma_g_m = num_unique_ids * (result_subtract)*(result_subtract)';


% Calculate Lambda_hat_g_m
[Lambda_hat_g_m_t1, Lambda_hat_g_m_t2, Lambda_hat_g_m_t3] = evaluate_Lambda_G_M(estimated_theta_g_m, data_cp);
            
% Calcualte Sigma_Lambda_2_G_M
[Sigma_Lambda_2_g_m_t1, Sigma_Lambda_2_g_m_t2, Sigma_Lambda_2_g_m_t3] = evaluate_Sigma_Lambda_2_G_M(estimated_theta_g_m, estimated_gamma_g_m , data_cp, B2);


fprintf('Estimated Theta (g_m): \n');
disp(estimated_theta_g_m);

fprintf('Estimated Sigma (g_m): %.4f\n', estimated_sigma_g_m);

fprintf('Estimated Theta Tilde (g_m): \n');
disp(estimated_theta_tilde_g_m);

fprintf('Estimated Gamma (g_m): %.4f\n', estimated_gamma_g_m);

fprintf('Lambda Hat g_m at t1: %.4f\n', Lambda_hat_g_m_t1);
fprintf('Lambda Hat g_m at t2: %.4f\n', Lambda_hat_g_m_t2);
fprintf('Lambda Hat g_m at t3: %.4f\n', Lambda_hat_g_m_t3);

fprintf('Sigma Lambda 2 g_m at t1: %.4f\n', Sigma_Lambda_2_g_m_t1);
fprintf('Sigma Lambda 2 g_m at t2: %.4f\n', Sigma_Lambda_2_g_m_t2);
fprintf('Sigma Lambda 2 g_m at t3: %.4f\n', Sigma_Lambda_2_g_m_t3);
% Display results Log-Rank Weight

% Calculate theta_lr_m            
score_lr_m = S_LR_M(estimated_theta_g_m, data_cp); 
d_hat_lr_b_m = estimate_D_LR_B(estimated_theta_g_m, estimated_gamma_g_m, data_cp, B1);            
d_hat_lr_b_m_inv = test_finite_d_hat_lr_b_inv(d_hat_lr_b_m);            
estimated_theta_lr_m = estimated_theta_g_m - (d_hat_lr_b_m_inv * score_lr_m);    

% Calculate Sigma_LR for theta_lr_m using the function calculate_sigma_lr_m
estimated_sigma_lr_m = calculate_sigma_lr(data_cp, estimated_theta_lr_m);

% Calculate Gamma for theta_lr_m
d_hat_lr_b_m_inv = test_finite_d_hat_lr_b_inv(d_hat_lr_b_m);
estimated_gamma_lr_m = d_hat_lr_b_m_inv' * estimated_sigma_lr_m * d_hat_lr_b_m_inv;

% Optimize theta_tilde_lr_m
estimated_theta_tilde_lr_m = optimize_theta_tilde_lr_m(data_cp, estimated_sigma_lr_m);

% Initialize the result matrix
result_subtract1 = zeros(3, 3);

% Loop over each column
for i = 1:3
    result_subtract1(:, i) = estimated_theta_tilde_lr_m(:, i) - estimated_theta_lr_m;
end

% Calculate Gamma for
% theta_lr
estimated_gammahuang_lr_m = num_unique_ids * (result_subtract1)*(result_subtract1)';
            
% Calculate Lambda_hat_lr_m            
[Lambda_hat_lr_m_t1, Lambda_hat_lr_m_t2, Lambda_hat_lr_m_t3] = evaluate_Lambda_LR_M(estimated_theta_lr_m, data_cp);
                      
% Calcualte Sigma_Lambda_2_LR_M            
[Sigma_Lambda_2_lr_m_t1, Sigma_Lambda_2_lr_m_t2, Sigma_Lambda_2_lr_m_t3] = evaluate_Sigma_Lambda_2_LR_M(estimated_theta_lr_m, estimated_gamma_lr_m, data_cp, B2);
          
fprintf('Estimated Theta (lr_m): \n');
disp(estimated_theta_lr_m);

fprintf('Estimated Sigma (lr_m): %.4f\n', estimated_sigma_lr_m);

fprintf('Estimated Theta Tilde (lr_m): \n');
disp(estimated_theta_tilde_lr_m);

fprintf('Estimated GammaTh3 (lr_m): %.4f\n', estimated_gamma_lr_m);
fprintf('Estimated GammaHuang (lr_m): %.4f\n', estimated_gammahuang_lr_m);

fprintf('Lambda Hat lr_m at t1: %.4f\n', Lambda_hat_lr_m_t1);
fprintf('Lambda Hat lr_m at t2: %.4f\n', Lambda_hat_lr_m_t2);
fprintf('Lambda Hat lr_m at t3: %.4f\n', Lambda_hat_lr_m_t3);

fprintf('Sigma Lambda 2 lr_m at t1: %.4f\n', Sigma_Lambda_2_lr_m_t1);
fprintf('Sigma Lambda 2 lr_m at t2: %.4f\n', Sigma_Lambda_2_lr_m_t2);
fprintf('Sigma Lambda 2 lr_m at t3: %.4f\n', Sigma_Lambda_2_lr_m_t3);