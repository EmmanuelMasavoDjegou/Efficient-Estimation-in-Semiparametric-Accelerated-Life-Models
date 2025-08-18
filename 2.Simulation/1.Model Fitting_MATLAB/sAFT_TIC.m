%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% AGT-TIC                                                                 %  
% FINITE SAMPLE PROPERTIES                                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set random seed for reproducibility
rng(45);  

% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_values = [50]; % Sample sizes                                                                   
B1 = 10;                                                                                           
B2 = 25;                                                                                           
replications = 100; % Number of replications                                                        
theta_0 = 0.8; % True parameter theta   
% Parameters
alpha = 1.5;                                                                                      
Lambda_0_dist = [1.18, 3.35]; % Baseline cumulative hazard functions at t = 1, 3              
% Store the baseline cumulative hazard functions in a cell array                                                                                  
tau_max = 3.5; % Maximum value for tau                                                              
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function data_cp = generate_recurrent_data_cp(n, tau_max, alpha)
    Z = binornd(1, 0.5, [n, 1]);             % Covariate
    tau = unifrnd(0, tau_max, [n, 1]);       % Censoring times
    data_list = cell(n, 1);                  % Initialize storage

    rho = 0.25;  % Correlation for Gumbel bivariate exponential

    for i = 1:n
        lambda = alpha; % Baseline rate (used for exponential rate)
        scaling = exp(-0.8 * Z(i)); % AFT-like scaling

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

        % Construct data table
        all_times = [event_times; censoring_time];
        all_gaps = [gap_times; censoring_gap_time];
        all_events = [ones(length(event_times), 1); 0];

        individual_data = table( ...
            i * ones(length(all_times), 1), ...
            all_times, ...
            all_gaps, ...
            all_events, ...
            Z(i) * ones(length(all_times), 1), ...
            tau(i) * ones(length(all_times), 1), ...
            'VariableNames', {'id', 'time', 'gap_time', ...
                              'event', 'covariate', 'tau'});

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

% Objective function
function obj_value = objective_function(theta, data_cp)
    unique_ids = unique(data_cp.id);
    n = length(unique_ids);
    total_sum = 0;
    
    for i = unique_ids'
        individual_data = data_cp(data_cp.id == i, :);
        K_i = height(individual_data);
        
        for j = 1:K_i
            S_ij = individual_data.time(j);
            
            for l = unique_ids'
                l_data = data_cp(data_cp.id == l, :);
                tau_l = l_data.tau(1);
                Z_i = individual_data.covariate(1);
                Z_l = l_data.covariate(1);
                
                contribution = log(tau_l) - log(S_ij) - theta * (Z_i - Z_l);
                total_sum = total_sum + max(contribution, 0);
            end
        end
    end
    
    obj_value = total_sum / (n^2);
end

% Optimization to find the parameter theta
function estimated_theta = optimize_theta(data_cp)
    initial_theta = 0.7;
    
    options = optimset('fminsearch');
    options.Display = 'off';
    [estimated_theta, ~] = fminsearch(@(theta) ...
        objective_function(theta, data_cp), initial_theta, options);
end


% Generalized at-risk process Y_l
function indicator = indicator_function(tau_l, S_ij, theta, Z_i, Z_l)
    % This function calculates the indicator function 
    % I(log(tau_l) - log(S_ij) >= theta * (Z_i - Z_l))

    % Compute log values
    log_tau_l = log(tau_l);
    log_S_ij = log(S_ij);
    
    % Compute theta * (Z_i - Z_l)
    theta_Z_diff = theta * (Z_i - Z_l);
    
    % Compute the indicator value
    indicator = (log_tau_l - log_S_ij >= theta_Z_diff);
end

% S0
function S_0 = S_0_func(S_ij, theta, Z_i, data_cp)
    % Calculate \mathbb{S}^{(0)}(\Sij e^{\theta^{\prime} Z_i} \mid \theta)
    n = length(unique(data_cp.id));
    total_sum = 0;
    unique_ids = unique(data_cp.id);
    
    for l = unique_ids'
        l_data = data_cp(data_cp.id == l, :);  % Corrected subsetting
        if ~isempty(l_data)  % Check if l_data is not empty
            tau_l = l_data.tau(1);
            Z_l = l_data.covariate(1);
            Y_l = indicator_function(S_ij, tau_l, theta, Z_i, Z_l);
            total_sum = total_sum + Y_l;
        end
    end
    
    S_0 = (1/n) * total_sum;
end

% S1
function S_1 = S_1_func(S_ij, theta, Z_i, data_cp)
    % Calculate \mathbb{S}^{(1)}(\Sij e^{\theta^{\prime} Z_i} \mid \theta)
    n = length(unique(data_cp.id));
    total_sum = 0;
    unique_ids = unique(data_cp.id);
    
    for l = unique_ids'
        l_data = data_cp(data_cp.id == l, :);  % Corrected subsetting
        if ~isempty(l_data)  % Check if l_data is not empty
            tau_l = l_data.tau(1);
            Z_l = l_data.covariate(1);
            Y_l = indicator_function(S_ij, tau_l, theta, Z_i, Z_l);
            total_sum = total_sum + Z_l * Y_l;
        end
    end
    
    S_1 = (1/n) * total_sum;
end

% S2
function S_2 = S_2_func(S_ij, theta, Z_i, data_cp)
    % Calculate \mathbb{S}^{(2)}(\Sij e^{\theta^{\prime} Z_i} \mid \theta)
    n = length(unique(data_cp.id));
    total_sum = 0;
    unique_ids = unique(data_cp.id);
    
    for l = unique_ids'
        l_data = data_cp(data_cp.id == l, :);  % Corrected subsetting
        if ~isempty(l_data)  % Check if l_data is not empty
            tau_l = l_data.tau(1);
            Z_l = l_data.covariate(1);
            Y_l = indicator_function(S_ij, tau_l, theta, Z_i, Z_l);
            total_sum = total_sum + Z_l^2 * Y_l;
        end
    end
    
    S_2 = (1/n) * total_sum;
end


% Function to compute \varphi(w)
function phi_w = phi_function(S_ij, theta, Z_i, data_cp)
    % Compute S^0, S^1, and S^2
    S_0 = S_0_func(S_ij, theta, Z_i, data_cp);
    S_1 = S_1_func(S_ij, theta, Z_i, data_cp);
    S_2 = S_2_func(S_ij, theta, Z_i, data_cp);
    
    % Calculate phi(w)
    phi_w = S_0 * S_2 - (S_1)^2;
end


% Function to compute \widehat{\Sigma}_G
function Sigma_G = calculate_sigma(data_cp, theta_hat)
    n = length(unique(data_cp.id));
    total_sum = 0;
    unique_ids = unique(data_cp.id);
    
    for i = unique_ids'
        i_data = data_cp(data_cp.id == i, :);  % Subsetting data for 
        % individual i
        K_i = height(i_data);
        for j = 1:K_i
            S_ij = i_data.time(j);  % Assuming time corresponds to S_ij
            Z_i = i_data.covariate(1);
            % Compute \varphi(S_{ij} e^{\theta^{\prime} Z_i})
            phi_w = phi_function(S_ij, theta_hat, Z_i, data_cp);
            total_sum = total_sum + phi_w;
        end
    end
    
    Sigma_G = (1/n) * total_sum;
end

% Step 2

% Estimate Theta_tilde_G

% Objective function_tilde
function obj_value1 = objective_function_tilde(theta, data_cp, sigma)
    unique_ids = unique(data_cp.id);
    n = length(unique_ids);
    total_sum = 0;
    
    for i = unique_ids'
        individual_data = data_cp(data_cp.id == i, :);
        K_i = height(individual_data);
        
        for j = 1:K_i
            S_ij = individual_data.time(j);
            
            for l = unique_ids'
                l_data = data_cp(data_cp.id == l, :);
                tau_l = l_data.tau(1);
                Z_i = individual_data.covariate(1);
                Z_l = l_data.covariate(1);
                
                contribution = (Z_i - Z_l) * indicator_function(tau_l, S_ij, theta, Z_i, Z_l);
                total_sum = total_sum + contribution ;
            end
        end
    end
    
    obj_value1 = abs(total_sum / (n^2) - sqrt((1/n) * sigma));
end

% Optimization to find the parameter theta
function estimated_theta_tilde = optimize_theta_tilde(data_cp, sigma)
    initial_theta_tilde = 0.7;
    
    options = optimset('fminsearch');
    options.Display = 'off';
    [estimated_theta_tilde, ~] = fminsearch(@(theta) ...
        objective_function_tilde(theta, data_cp, sigma), ...
        initial_theta_tilde, options);
end


% Estimating Lambda_G_M

% Lambda_G_M Function
function lambda_hat = Lambda_G_M(theta_G_M, t, data_cp)
    unique_ids = unique(data_cp.id);
    total_sum = 0;

    for i = unique_ids'
        individual_data = data_cp(data_cp.id == i, :);
        K_i = height(individual_data);
        Z_i = individual_data.covariate(1);
        
        for j = 1:K_i
            S_ij = individual_data.time(j);

            numerator = log(t) - log(S_ij) >= theta_G_M * Z_i;
            
            if numerator
                weight_denominator = 0;
                
                for l = unique_ids'
                    l_data = data_cp(data_cp.id == l, :);
                    tau_l = l_data.tau(1);
                    Z_l = l_data.covariate(1);
                    
                    weight_denominator = weight_denominator + (log(tau_l) - log(S_ij) >= theta_G_M * (Z_i - Z_l));
                end

                if weight_denominator > 0
                    total_sum = total_sum + (1 / weight_denominator);
                end
            end
        end
    end

    lambda_hat = total_sum;
end


% Function to evaluate Lambda_G_M at t = 1 and t = 2
function [lambda_hat_t1, lambda_hat_t2] = evaluate_Lambda_G_M(theta_G_M, data_cp)
    % Evaluate at t = 1
    t1 = 1;
    lambda_hat_t1 = Lambda_G_M(theta_G_M, t1, data_cp);

    % Evaluate at t = 3
    t2 = 3;
    lambda_hat_t2 = Lambda_G_M(theta_G_M, t2, data_cp);
end

% Estimating Sigma_Lambda_2_G_M

function first_term = First_Term_G_M(theta_G_M, t, data_cp)
    unique_ids = unique(data_cp.id);
    n = length(unique_ids);
    total_sum = 0;

    for i = unique_ids'
        individual_data = data_cp(data_cp.id == i, :);
        K_i = height(individual_data);
        Z_i = individual_data.covariate(1);
        
        for j = 1:K_i
            S_ij = individual_data.time(j);

            numerator = log(t) - log(S_ij) >= theta_G_M * Z_i;
            
            if numerator
                weight_denominator = 0;
                
                for l = unique_ids'
                    l_data = data_cp(data_cp.id == l, :);
                    tau_l = l_data.tau(1);
                    Z_l = l_data.covariate(1);
                    
                    weight_denominator = weight_denominator + (log(tau_l) - log(S_ij) >= theta_G_M * (Z_i - Z_l));
                end

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
    % Parameters
    n = length(unique(data_cp.id));  % Number of samples
    cov_matrix = (1/n) * Gamma_hat_G;  % Covariance matrix (scalar)

    % Bootstrap samples of theta
    theta_tilde = normrnd(theta_G_M, sqrt(cov_matrix), [1, B]);

    % Compute \bar{S}_{LR}^M for the bootstrap samples and the original estimate
    lambda_hat_theta_G = Lambda_G_M(theta_G_M, t, data_cp);
    lambda_hat_tilde = zeros(1, B);

    for k = 1:B
        lambda_hat_tilde(k) = Lambda_G_M(theta_tilde(k), t, data_cp);
    end

    % Construct \mathfrak{Z} and \mathfrak{Y}
    Z = [ones(B, 1), (theta_G_M - theta_tilde)'];
    Y = (lambda_hat_tilde - lambda_hat_theta_G)';

    % Remove the first row from the result of the matrix multiplication
    A_hat_B_tilde = (Z' * Z) \ (Z' * Y);
    A_hat_B_tilde = A_hat_B_tilde(2:end, :);  % Removing the first row
    A_hat_B_tilde = (1/2) * (A_hat_B_tilde + A_hat_B_tilde);

end


% Function to evaluate Sigma_Lambda_2_G_M at t = 0.5 and t = 1.5
function [Sigma_Lambda_2_G_M_t1, Sigma_Lambda_2_G_M_t2] = evaluate_Sigma_Lambda_2_G_M(theta_G_M, Gamma_hat_G, data_cp, B)
    % Evaluate at t = 1
    t1 = 1;
    first_term_t1 = First_Term_G_M(theta_G_M, t1, data_cp);
    A_hat_t1 = estimate_A_hat_M(theta_G_M, Gamma_hat_G, t1, data_cp, B);
    Sigma_Lambda_2_G_M_t1 = first_term_t1 + (A_hat_t1 * Gamma_hat_G * A_hat_t1);
    
    % Evaluate at t = 3
    t2 = 3;
    first_term_t2 = First_Term_G_M(theta_G_M, t2, data_cp);
    A_hat_t2 = estimate_A_hat_M(theta_G_M, Gamma_hat_G, t2, data_cp, B);
    Sigma_Lambda_2_G_M_t2 = first_term_t2 + (A_hat_t2 * Gamma_hat_G * A_hat_t2);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LOG-RANK WEIGHT                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% AMI MODEL

% ALGORITHM 2

% Accessory Functions

% Score Function S_LR_M
function obj_value2 = S_LR_M(theta, data_cp)
    unique_ids = unique(data_cp.id);
    n = length(unique_ids);
    total_sum = 0;
    
    for i = unique_ids'
        individual_data = data_cp(data_cp.id == i, :);
        K_i = height(individual_data);
        
        for j = 1:K_i
            S_ij = individual_data.time(j);
            Z_i = individual_data.covariate(1);
            
            for l = unique_ids'
                l_data = data_cp(data_cp.id == l, :);
                tau_l = l_data.tau(1);
                Z_l = l_data.covariate(1);
                
                indicator = log(tau_l) - log(S_ij) >= theta * (Z_i - Z_l);
                
                if indicator
                    weight_denominator = 0;
                    for l2 = unique_ids'
                        l2_data = data_cp(data_cp.id == l2, :);
                        tau_l2 = l2_data.tau(1);
                        Z_l2 = l2_data.covariate(1);
                        
                        weight_denominator = weight_denominator + (log(tau_l2) - log(S_ij) >= theta * (Z_i - Z_l2));
                    end
                    
                    if weight_denominator > 0
                        contribution = (Z_i - Z_l) / weight_denominator;
                        total_sum = total_sum + contribution;
                    end
                end
            end
        end
    end
    
    obj_value2 = total_sum / n;
end

% Estimate D_LR_B
function D_LR_B = estimate_D_LR_B(theta_G, Gamma_hat_G, data_cp, B)
    % Parameters
    n = length(unique(data_cp.id));  % Number of samples
    cov_matrix = (1/n) * Gamma_hat_G;  % Covariance matrix (scalar)

    % Bootstrap samples of theta
    theta_tilde = normrnd(theta_G, sqrt(cov_matrix), [1, B]);

    % Compute \bar{S}_{LR}^M for the bootstrap samples and the original estimate
    S_LR_M_theta_G = S_LR_M(theta_G, data_cp);
    S_LR_M_theta_tilde = zeros(1, B);

    for k = 1:B
        S_LR_M_theta_tilde(k) = S_LR_M(theta_tilde(k), data_cp);
    end

    % Construct \mathfrak{Z} and \mathfrak{Y}
    Z = [ones(B, 1), (theta_tilde - theta_G)'];
    Y = (S_LR_M_theta_tilde - S_LR_M_theta_G)';

    % Remove the first row from the result of the matrix multiplication
    D_LR_B_tilde = (Z' * Z) \ (Z' * Y);
    D_LR_B_tilde = D_LR_B_tilde(2:end, :);  % Removing the first row
    D_LR_B = (1/2) * (D_LR_B_tilde + D_LR_B_tilde);

end


function d_hat_lr_b_inv = test_finite_d_hat_lr_b_inv(d_hat_lr_b)
    % Function to test if d_hat_lr_b_inv = (1 / d_hat_lr_b) is finite
    % If not, return NaN
    
    % Calculate the inverse
    d_hat_lr_b_inv = 1 / d_hat_lr_b;
    
    % Check if the result is finite
    if ~isfinite(d_hat_lr_b_inv)
        % If not finite, return NaN
        d_hat_lr_b_inv = NaN;
    end
end


% Function to compute \varphi(w)
function phi_w = phi_function_lr(S_ij, theta, Z_i, data_cp)
    % Compute S^0, S^1, and S^2
    S_0 = S_0_func(S_ij, theta, Z_i, data_cp);
    S_1 = S_1_func(S_ij, theta, Z_i, data_cp);
    S_2 = S_2_func(S_ij, theta, Z_i, data_cp);
    
    % Check if S_0 is greater than 0
    if S_0 > 0
        % Calculate phi(w)
        phi_w = (S_2 / S_0) - ((S_1 / S_0)^2);
    else
        % Handle the case where S_0 <= 0
        warning('S_0 is less than or equal to zero, returning NaN');
        phi_w = NaN;
    end
end


% Function to compute \widehat{\Sigma}_LR
function Sigma_LR = calculate_sigma_lr(data_cp, theta_hat)
    n = length(unique(data_cp.id));
    total_sum = 0;
    unique_ids = unique(data_cp.id);
    
    for i = unique_ids'
        i_data = data_cp(data_cp.id == i, :);  % Subsetting data for individual i
        K_i = height(i_data);
        for j = 1:K_i
            S_ij = i_data.time(j);  % Assuming time corresponds to S_ij
            Z_i = i_data.covariate(1);
            % Compute \varphi(S_{ij} e^{\theta^{\prime} Z_i})
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
    unique_ids = unique(data_cp.id);
    n = length(unique_ids);
    total_sum = 0;
    
    for i = unique_ids'
        individual_data = data_cp(data_cp.id == i, :);
        K_i = height(individual_data);
        
        for j = 1:K_i
            S_ij = individual_data.time(j);
            Z_i = individual_data.covariate(1);
            
            for l = unique_ids'
                l_data = data_cp(data_cp.id == l, :);
                tau_l = l_data.tau(1);
                Z_l = l_data.covariate(1);
                
                indicator = log(tau_l) - log(S_ij) >= theta * (Z_i - Z_l);
                
                if indicator
                    weight_denominator = 0;
                    for l2 = unique_ids'
                        l2_data = data_cp(data_cp.id == l2, :);
                        tau_l2 = l2_data.tau(1);
                        Z_l2 = l2_data.covariate(1);
                        
                        weight_denominator = weight_denominator + (log(tau_l2) - log(S_ij) >= theta * (Z_i - Z_l2));
                    end
                    
                    if weight_denominator > 0
                        contribution = (Z_i - Z_l) / weight_denominator;
                        total_sum = total_sum + contribution;
                    end
                end
            end
        end
    end
    
    obj_value1 = abs(total_sum /(n) - sqrt((1/n) * sigma));
end


% Optimization to find the parameter theta
function estimated_theta_tilde_lr_m = optimize_theta_tilde_lr_m(data_cp, sigma)
    initial_theta_tilde = 0.7;
    
    options = optimset('fminsearch');
    options.Display = 'off';
    [estimated_theta_tilde_lr_m, ~] = fminsearch(@(theta) ...
        objective_function_tilde_lr_m(theta, data_cp, sigma), ...
        initial_theta_tilde, options);
end


% Estimating Lambda_LR_M

% Lambda_LR_M Function
function lambda_hat = Lambda_LR_M(theta_LR_M, t, data_cp)
    unique_ids = unique(data_cp.id);
    n = length(unique_ids);
    total_sum = 0;

    for i = unique_ids'
        individual_data = data_cp(data_cp.id == i, :);
        K_i = height(individual_data);
        Z_i = individual_data.covariate(1);
        
        for j = 1:K_i
            S_ij = individual_data.time(j);

            numerator = log(t) - log(S_ij) >= theta_LR_M * Z_i;
            
            if numerator
                weight_denominator = 0;
                
                for l = unique_ids'
                    l_data = data_cp(data_cp.id == l, :);
                    tau_l = l_data.tau(1);
                    Z_l = l_data.covariate(1);
                    
                    weight_denominator = weight_denominator + (log(tau_l) - log(S_ij) >= theta_LR_M * (Z_i - Z_l));
                end

                if weight_denominator > 0
                    total_sum = total_sum + (1 / weight_denominator);
                end
            end
        end
    end

    lambda_hat = total_sum;
end


% Function to evaluate Lambda_LR_M at t = 0.5 and t = 1.5
function [lambda_hat_t1, lambda_hat_t2] = evaluate_Lambda_LR_M(theta_LR_M, data_cp)
    % Evaluate at t = 1
    t1 = 1;
    lambda_hat_t1 = Lambda_LR_M(theta_LR_M, t1, data_cp);

    % Evaluate at t = 3
    t2 = 3;
    lambda_hat_t2 = Lambda_LR_M(theta_LR_M, t2, data_cp);
end


% Estimating Sigma_Lambda_2_LR_M

function first_term = First_Term_LR_M(theta_LR_M, t, data_cp)
    unique_ids = unique(data_cp.id);
    n = length(unique_ids);
    total_sum = 0;

    for i = unique_ids'
        individual_data = data_cp(data_cp.id == i, :);
        K_i = height(individual_data);
        Z_i = individual_data.covariate(1);
        
        for j = 1:K_i
            S_ij = individual_data.time(j);

            numerator = log(t) - log(S_ij) >= theta_LR_M * Z_i;
            
            if numerator
                weight_denominator = 0;
                
                for l = unique_ids'
                    l_data = data_cp(data_cp.id == l, :);
                    tau_l = l_data.tau(1);
                    Z_l = l_data.covariate(1);
                    
                    weight_denominator = weight_denominator + (log(tau_l) - log(S_ij) >= theta_LR_M * (Z_i - Z_l));
                end

                if weight_denominator > 0
                    total_sum = total_sum + (1 / (weight_denominator)^2);
                end
            end
        end
    end

    first_term = n * total_sum;
end

% Estimate A_hat
function A_hat_B_tilde = estimate_A_hat_M_LR(theta_LR_M, Gamma_hat_LR, t, data_cp, B)
    % Parameters
    n = length(unique(data_cp.id));  % Number of samples
    cov_matrix = (1/n) * Gamma_hat_LR;  % Covariance matrix (scalar)

    % Bootstrap samples of theta
    theta_tilde = normrnd(theta_LR_M, sqrt(cov_matrix), [1, B]);

    % Compute \bar{S}_{LR}^M for the bootstrap samples and the original estimate
    lambda_hat_theta_LR = Lambda_LR_M(theta_LR_M, t, data_cp);
    lambda_hat_tilde = zeros(1, B);

    for k = 1:B
        lambda_hat_tilde(k) = Lambda_LR_M(theta_tilde(k), t, data_cp);
    end

    % Construct \mathfrak{Z} and \mathfrak{Y}
    Z = [ones(B, 1), (theta_LR_M - theta_tilde)'];
    Y = (lambda_hat_tilde - lambda_hat_theta_LR)';

    % Remove the first row from the result of the matrix multiplication
    A_hat_B_tilde = (Z' * Z) \ (Z' * Y);
    A_hat_B_tilde = A_hat_B_tilde(2:end, :);  % Removing the first row
    A_hat_B_tilde = (1/2) * (A_hat_B_tilde + A_hat_B_tilde);

end


% Function to evaluate Sigma_Lambda_2_LR_M at t = 1 and t = 2
function [Sigma_Lambda_2_LR_M_t1, Sigma_Lambda_2_LR_M_t2] = evaluate_Sigma_Lambda_2_LR_M(theta_LR_M, Gamma_hat_LR, data_cp, B)
    % Evaluate at t = 1
    t1 = 1;
    first_term_t1 = First_Term_LR_M(theta_LR_M, t1, data_cp);
    A_hat_t1 = estimate_A_hat_M_LR(theta_LR_M, Gamma_hat_LR, t1, data_cp, B);
    Sigma_Lambda_2_LR_M_t1 = first_term_t1 + (A_hat_t1 * Gamma_hat_LR * A_hat_t1);
    
    % Evaluate at t = 3
    t2 = 3;
    first_term_t2 = First_Term_LR_M(theta_LR_M, t2, data_cp);
    A_hat_t2 = estimate_A_hat_M_LR(theta_LR_M, Gamma_hat_LR, t2, data_cp, B);
    Sigma_Lambda_2_LR_M_t2 = first_term_t2 + (A_hat_t2 * Gamma_hat_LR * A_hat_t2);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  MAIN FUNCTIONS                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Initialize results storage
results_m = table();

% Loop over sample sizes
for n = n_values
        % Store estimates and variances for current sample size and distribution
        estimates_g_m = nan(replications, 1);
        sigmas_g_m = nan(replications, 1);
        estimates_tilde_g_m = nan(replications, 1);
        gammas_g_m = nan(replications, 1);
        Lambdas_hat_g_m_t1 = nan(replications, 1);
        Sigmas_Lambda_2_g_m_t1 = nan(replications, 1);        
        Lambdas_hat_g_m_t2 = nan(replications, 1);
        Sigmas_Lambda_2_g_m_t2 = nan(replications, 1);       
        estimates_lr_m = nan(replications, 1);
        gammas_lr_m = nan(replications, 1);
        gammahuangs_lr_m = nan(replications, 1);
        Lambdas_hat_lr_m_t1 = nan(replications, 1);
        Sigmas_Lambda_2_lr_m_t1 = nan(replications, 1);  
        Lambdas_hat_lr_m_t2 = nan(replications, 1);
        Sigmas_Lambda_2_lr_m_t2 = nan(replications, 1); 
        

        % Perform simulations
        for rep = 1:replications
            % Generate data
            data_cp = generate_recurrent_data_cp(n, tau_max, alpha);

            % Optimize theta_g_m
            estimated_theta_g_m = optimize_theta(data_cp);
    
            % Calculate Sigma_G for theta_g_m using the function calculate_sigma
            estimated_sigma_g_m = calculate_sigma(data_cp, estimated_theta_g_m);

            % Optimize theta_tilde_g_m
            estimated_theta_tilde_g_m = optimize_theta_tilde(data_cp, estimated_sigma_g_m);

            % Calculate Gamma for theta_g_m
            estimated_gamma_g_m = (sqrt(n) * (estimated_theta_tilde_g_m - estimated_theta_g_m))^2;

            % Calculate Lambda_hat_g_m
            [Lambda_hat_g_m_t1, Lambda_hat_g_m_t2] = evaluate_Lambda_G_M(estimated_theta_g_m, data_cp);

            % Calcualte Sigma_Lambda_2_G_M
            [Sigma_Lambda_2_g_m_t1, Sigma_Lambda_2_g_m_t2] = evaluate_Sigma_Lambda_2_G_M(estimated_theta_g_m, estimated_gamma_g_m , data_cp, B2);
          
                       
            % Calculate theta_lr_m
            score_lr_m = S_LR_M(estimated_theta_g_m, data_cp);
            d_hat_lr_b = estimate_D_LR_B(estimated_theta_g_m, estimated_gamma_g_m, data_cp, B1);
            d_hat_lr_b_inv = test_finite_d_hat_lr_b_inv(d_hat_lr_b);
            estimated_theta_lr_m = estimated_theta_g_m - (d_hat_lr_b_inv * score_lr_m);

            % Calculate Sigma_LR for theta_lr_m using the function calculate_sigma_lr
            estimated_sigma_lr_m = calculate_sigma_lr(data_cp, estimated_theta_lr_m);

            % Calculate Gamma for theta_lr_m
            d_hat_lr_b_inv = test_finite_d_hat_lr_b_inv(d_hat_lr_b);
            estimated_gamma_lr_m = d_hat_lr_b_inv * estimated_sigma_lr_m * d_hat_lr_b_inv;

            % Optimize theta_tilde_lr_m
            estimated_theta_tilde_lr_m = optimize_theta_tilde_lr_m(data_cp, estimated_sigma_lr_m);

            % Calculate GammaHuang for theta_lr_m 
            estimated_gammahuang_lr_m = (sqrt(n) * (estimated_theta_tilde_lr_m - estimated_theta_lr_m))^2;

            % Calculate Lambda_hat_lr_m
            [Lambda_hat_lr_m_t1, Lambda_hat_lr_m_t2] = evaluate_Lambda_LR_M(estimated_theta_lr_m, data_cp);

            % Calculate Sigma_Lambda_2_LR_M
            [Sigma_Lambda_2_lr_m_t1, Sigma_Lambda_2_lr_m_t2] = evaluate_Sigma_Lambda_2_LR_M(estimated_theta_lr_m, estimated_gamma_lr_m, data_cp, B2);
          


            % Store the estimates for theta_g_m
            estimates_g_m(rep) = estimated_theta_g_m;
            sigmas_g_m(rep) = estimated_sigma_g_m;
            estimates_tilde_g_m(rep) = estimated_theta_tilde_g_m;
            gammas_g_m(rep) = estimated_gamma_g_m;
            Lambdas_hat_g_m_t1(rep) =  Lambda_hat_g_m_t1;
            Sigmas_Lambda_2_g_m_t1(rep) = Sigma_Lambda_2_g_m_t1;
            Lambdas_hat_g_m_t2(rep) =  Lambda_hat_g_m_t2;
            Sigmas_Lambda_2_g_m_t2(rep) = Sigma_Lambda_2_g_m_t2;

            % Store the estimates for theta_lr_m
            estimates_lr_m(rep) = estimated_theta_lr_m;
            gammas_lr_m(rep) = estimated_gamma_lr_m;
            gammahuangs_lr_m(rep) = estimated_gammahuang_lr_m;
            Lambdas_hat_lr_m_t1(rep) =  Lambda_hat_lr_m_t1;
            Sigmas_Lambda_2_lr_m_t1(rep) = Sigma_Lambda_2_lr_m_t1;
            Lambdas_hat_lr_m_t2(rep) =  Lambda_hat_lr_m_t2;
            Sigmas_Lambda_2_lr_m_t2(rep) = Sigma_Lambda_2_lr_m_t2;

            % Create a summary table for current settings for theta_m
            temp_results_m = table( ...
                repmat(n, replications, 1), ...
                estimates_g_m, sigmas_g_m, estimates_tilde_g_m, gammas_g_m, Lambdas_hat_g_m_t1, Sigmas_Lambda_2_g_m_t1, Lambdas_hat_g_m_t2, Sigmas_Lambda_2_g_m_t2, estimates_lr_m, gammas_lr_m, gammahuangs_lr_m, Lambdas_hat_lr_m_t1, Sigmas_Lambda_2_lr_m_t1, Lambdas_hat_lr_m_t2, Sigmas_Lambda_2_lr_m_t2, 'VariableNames', ...
                {'SampleSize', 'EstimatedThetaG_M', ...
                'EstimatedSigmaG_M', 'EstimatedThetaTildeG_M', 'EstimatedGammaG_M', 'EstimatedLambdaG_M1', 'EstimatedSigmaLambda2G_M1', 'EstimatedLambdaG_M2', 'EstimatedSigmaLambda2G_M2', 'EstimatedThetaLR_M', 'EstimatedGammaLR_M', 'EstimatedGammaHuangLR_M', 'EstimatedLambdaLR_M1', 'EstimatedSigmaLambda2LR_M1', 'EstimatedLambdaLR_M2', 'EstimatedSigmaLambda2LR_M2',});
        
            % Append to results_m
            results_m = [results_m; temp_results_m];
        end
end

% Display results
disp('Results for theta_m:');
disp(results_m);

% Save results to CSV files
writetable(results_m, 'simulation_results_m.csv');

% Initialize tables to store the bias, standard deviation, sqrt(EstimatedGamma), and coverage results
extended_results_m = table();


% Loop over sample sizes
for n = n_values
        % Filter results for current sample size and distribution for theta_m
        current_results_m = results_m(results_m.SampleSize == n,:);

        % Compute for theta_m
        % Compute the sample mean of EstimatedThetaM
        mean_estimated_theta_g_m = mean(current_results_m.EstimatedThetaG_M, 'omitnan');
        mean_estimated_Lambda_g_m_t1 = mean(current_results_m.EstimatedLambdaG_M1, 'omitnan');
        mean_estimated_Lambda_g_m_t2 = mean(current_results_m.EstimatedLambdaG_M2, 'omitnan');
        mean_estimated_theta_lr_m = mean(current_results_m.EstimatedThetaLR_M, 'omitnan');
        mean_estimated_Lambda_lr_m_t1 = mean(current_results_m.EstimatedLambdaLR_M1, 'omitnan');
        mean_estimated_Lambda_lr_m_t2 = mean(current_results_m.EstimatedLambdaLR_M2, 'omitnan');
        

        % Compute the sample standard deviation of EstimatedThetaM
        std_estimated_theta_g_m = std(current_results_m.EstimatedThetaG_M, 'omitnan');
        std_estimated_Lambda_g_m_t1 = std(current_results_m.EstimatedLambdaG_M1, 'omitnan');
        std_estimated_Lambda_g_m_t2 = std(current_results_m.EstimatedLambdaG_M2, 'omitnan');
        std_estimated_theta_lr_m = std(current_results_m.EstimatedThetaLR_M, 'omitnan');
        std_estimated_Lambda_lr_m_t1 = std(current_results_m.EstimatedLambdaLR_M1, 'omitnan');
        std_estimated_Lambda_lr_m_t2 = std(current_results_m.EstimatedLambdaLR_M2, 'omitnan');

        
        % Compute the mean of sqrt(EstimatedGammaM)
        mean_sqrt_gamma_g_m = mean(sqrt(current_results_m.EstimatedGammaG_M), 'omitnan');
        mean_sqrt_gammaThm3_g_m = NaN;
        mean_sqrt_Sigma_Lambda_2_g_m_t1 = mean(sqrt(current_results_m.EstimatedSigmaLambda2G_M1), 'omitnan');
        mean_sqrt_Sigma_Lambda_2_g_m_t2 = mean(sqrt(current_results_m.EstimatedSigmaLambda2G_M2), 'omitnan');
        
        mean_sqrt_gammahuang_lr_m = mean(sqrt(current_results_m.EstimatedGammaHuangLR_M), 'omitnan');
        mean_sqrt_gamma_lr_m = mean(sqrt(current_results_m.EstimatedGammaLR_M), 'omitnan');
        mean_sqrt_Sigma_Lambda_2_lr_m_t1 = mean(sqrt(current_results_m.EstimatedSigmaLambda2LR_M1), 'omitnan');
        mean_sqrt_Sigma_Lambda_2_lr_m_t2 = mean(sqrt(current_results_m.EstimatedSigmaLambda2LR_M2), 'omitnan');
        
        

        % Compute the bias relative to true theta value
        bias_g_m = mean_estimated_theta_g_m - theta_0;
        true_Lambda_0_t1 = Lambda_0_dist(1); % true value at t = 0.5
        bias_Lambda_g_m_t1 = mean_estimated_Lambda_g_m_t1 - true_Lambda_0_t1;
        true_Lambda_0_t2 = Lambda_0_dist(2); % true value at t = 1.5
        bias_Lambda_g_m_t2 = mean_estimated_Lambda_g_m_t2 - true_Lambda_0_t2;  
        bias_lr_m = mean_estimated_theta_lr_m - theta_0;      
        bias_Lambda_lr_m_t1 = mean_estimated_Lambda_lr_m_t1 - true_Lambda_0_t1;   
        bias_Lambda_lr_m_t2 = mean_estimated_Lambda_lr_m_t2 - true_Lambda_0_t2;  



        
        % Wald Method for 95% Coverage for theta_m
        % Confidence interval: EstimatedThetaM ± 1.96 * sqrt(EstimatedGammaM / n)
        ci_wald_lower_g_m = mean_estimated_theta_g_m - 1.96 * mean_sqrt_gamma_g_m/sqrt(n);
        ci_wald_upper_g_m = mean_estimated_theta_g_m + 1.96 * mean_sqrt_gamma_g_m/sqrt(n);

        % Z value for 95% confidence (z_alpha/2 for alpha=0.05 is approximately 1.96)
        z_alpha_2 = 1.96;

        % Calculate confidence intervals using the provided formula
        ci_wald_lower_Lambda_g_m_t1 = mean_estimated_Lambda_g_m_t1 * exp(-z_alpha_2 * mean_sqrt_Sigma_Lambda_2_g_m_t1 / (sqrt(n) * mean_estimated_Lambda_g_m_t1));
        ci_wald_upper_Lambda_g_m_t1 = mean_estimated_Lambda_g_m_t1 * exp(z_alpha_2 * mean_sqrt_Sigma_Lambda_2_g_m_t1 / (sqrt(n) * mean_estimated_Lambda_g_m_t1));

        ci_wald_lower_Lambda_g_m_t2 = mean_estimated_Lambda_g_m_t2 * exp(-z_alpha_2 * mean_sqrt_Sigma_Lambda_2_g_m_t2 / (sqrt(n) * mean_estimated_Lambda_g_m_t2));
        ci_wald_upper_Lambda_g_m_t2 = mean_estimated_Lambda_g_m_t2 * exp(z_alpha_2 * mean_sqrt_Sigma_Lambda_2_g_m_t2 / (sqrt(n) * mean_estimated_Lambda_g_m_t2));


        
        ci_wald_lower_lr_m = mean_estimated_theta_lr_m - 1.96 * mean_sqrt_gamma_lr_m/sqrt(n);
        ci_wald_upper_lr_m = mean_estimated_theta_lr_m + 1.96 * mean_sqrt_gamma_lr_m/sqrt(n);

        
        % Calculate confidence intervals using the provided formula
        ci_wald_lower_Lambda_lr_m_t1 = mean_estimated_Lambda_lr_m_t1 * exp(-z_alpha_2 * mean_sqrt_Sigma_Lambda_2_lr_m_t1 / (sqrt(n) * mean_estimated_Lambda_lr_m_t1));
        ci_wald_upper_Lambda_lr_m_t1 = mean_estimated_Lambda_lr_m_t1 * exp(z_alpha_2 * mean_sqrt_Sigma_Lambda_2_lr_m_t1 / (sqrt(n) * mean_estimated_Lambda_lr_m_t1));

        ci_wald_lower_Lambda_lr_m_t2 = mean_estimated_Lambda_lr_m_t2 * exp(-z_alpha_2 * mean_sqrt_Sigma_Lambda_2_lr_m_t2 / (sqrt(n) * mean_estimated_Lambda_lr_m_t2));
        ci_wald_upper_Lambda_lr_m_t2 = mean_estimated_Lambda_lr_m_t2 * exp(z_alpha_2 * mean_sqrt_Sigma_Lambda_2_lr_m_t2 / (sqrt(n) * mean_estimated_Lambda_lr_m_t2));



        % Determine Wald coverage for theta_m
        coverage_wald_g_m = mean(current_results_m.EstimatedThetaG_M >= ci_wald_lower_g_m & current_results_m.EstimatedThetaG_M <= ci_wald_upper_g_m);
        coverage_wald_Lambda_g_m_t1 = mean(current_results_m.EstimatedLambdaG_M1 >= ci_wald_lower_Lambda_g_m_t1 & current_results_m.EstimatedLambdaG_M1 <= ci_wald_upper_Lambda_g_m_t1);
        coverage_wald_Lambda_g_m_t2 = mean(current_results_m.EstimatedLambdaG_M2 >= ci_wald_lower_Lambda_g_m_t2 & current_results_m.EstimatedLambdaG_M2 <= ci_wald_upper_Lambda_g_m_t2);
        
        
        
        
        % Determine Wald coverage for theta_lr
        coverage_wald_lr_m = mean(current_results_m.EstimatedThetaLR_M >= ci_wald_lower_lr_m & current_results_m.EstimatedThetaLR_M <= ci_wald_upper_lr_m, 'omitnan');
        coverage_wald_Lambda_lr_m_t1 = mean(current_results_m.EstimatedLambdaLR_M1 >= ci_wald_lower_Lambda_lr_m_t1 & current_results_m.EstimatedLambdaLR_M1 <= ci_wald_upper_Lambda_lr_m_t1, 'omitnan');
        coverage_wald_Lambda_lr_m_t2 = mean(current_results_m.EstimatedLambdaLR_M2 >= ci_wald_lower_Lambda_lr_m_t2 & current_results_m.EstimatedLambdaLR_M2 <= ci_wald_upper_Lambda_lr_m_t2, 'omitnan');
        
            
        
        % Percentile Method for 95% Coverage for theta_m
        % Compute percentiles
        ci_percentile_lower_g_m = prctile(current_results_m.EstimatedThetaG_M, 2.5);
        ci_percentile_upper_g_m = prctile(current_results_m.EstimatedThetaG_M, 97.5);

        ci_percentile_lower_Lambda_g_m_t1 = prctile(current_results_m.EstimatedLambdaG_M1, 2.5);
        ci_percentile_upper_Lambda_g_m_t1 = prctile(current_results_m.EstimatedLambdaG_M1, 97.5);

        ci_percentile_lower_Lambda_g_m_t2 = prctile(current_results_m.EstimatedLambdaG_M2, 2.5);
        ci_percentile_upper_Lambda_g_m_t2 = prctile(current_results_m.EstimatedLambdaG_M2, 97.5);


        ci_percentile_lower_lr_m = prctile(current_results_m.EstimatedThetaG_M, 2.5);
        ci_percentile_upper_lr_m = prctile(current_results_m.EstimatedThetaLR_M, 97.5);

        ci_percentile_lower_Lambda_lr_m_t1 = prctile(current_results_m.EstimatedLambdaLR_M1, 2.5);
        ci_percentile_upper_Lambda_lr_m_t1 = prctile(current_results_m.EstimatedLambdaLR_M1, 97.5);

        ci_percentile_lower_Lambda_lr_m_t2 = prctile(current_results_m.EstimatedLambdaLR_M2, 2.5);
        ci_percentile_upper_Lambda_lr_m_t2 = prctile(current_results_m.EstimatedLambdaLR_M2, 97.5);



        % Determine Percentile coverage for theta_m
        coverage_percentile_g_m = mean(current_results_m.EstimatedThetaG_M >= ci_percentile_lower_g_m & current_results_m.EstimatedThetaG_M <= ci_percentile_upper_g_m);
        coverage_percentile_Lambda_g_m_t1 = mean(current_results_m.EstimatedLambdaG_M1 >= ci_percentile_lower_Lambda_g_m_t1 & current_results_m.EstimatedLambdaG_M1 <= ci_percentile_upper_Lambda_g_m_t1);
        coverage_percentile_Lambda_g_m_t2 = mean(current_results_m.EstimatedLambdaG_M2 >= ci_percentile_lower_Lambda_g_m_t2 & current_results_m.EstimatedLambdaG_M2 <= ci_percentile_upper_Lambda_g_m_t2);
       
        coverage_percentile_lr_m = mean(current_results_m.EstimatedThetaLR_M >= ci_percentile_lower_lr_m & current_results_m.EstimatedThetaLR_M <= ci_percentile_upper_lr_m, 'omitnan');
        coverage_percentile_Lambda_lr_m_t1 = mean(current_results_m.EstimatedLambdaLR_M1 >= ci_percentile_lower_Lambda_lr_m_t1 & current_results_m.EstimatedLambdaLR_M1 <= ci_percentile_upper_Lambda_lr_m_t1);
        coverage_percentile_Lambda_lr_m_t2 = mean(current_results_m.EstimatedLambdaLR_M2 >= ci_percentile_lower_Lambda_lr_m_t2 & current_results_m.EstimatedLambdaLR_M2 <= ci_percentile_upper_Lambda_lr_m_t2);
       

        % Store the results for theta_m
        temp_results_m = table( ...
            n, ...
            abs(bias_g_m), ...
            std_estimated_theta_g_m, ...
            mean_sqrt_gammaThm3_g_m,...
            mean_sqrt_gamma_g_m, ...
            coverage_wald_g_m, ...
            coverage_percentile_g_m, ...
            abs(bias_Lambda_g_m_t1),...
            std_estimated_Lambda_g_m_t1,...
            mean_sqrt_Sigma_Lambda_2_g_m_t1,...
            coverage_wald_Lambda_g_m_t1,...
            coverage_percentile_Lambda_g_m_t1,...
            abs(bias_Lambda_g_m_t2),...
            std_estimated_Lambda_g_m_t2,...
            mean_sqrt_Sigma_Lambda_2_g_m_t2,...
            coverage_wald_Lambda_g_m_t2,...
            coverage_percentile_Lambda_g_m_t2,...
            abs(bias_lr_m), ...
            std_estimated_theta_lr_m, ...
            mean_sqrt_gamma_lr_m, ...
            mean_sqrt_gammahuang_lr_m,...
            coverage_wald_lr_m, ...
            coverage_percentile_lr_m, ...
            abs(bias_Lambda_lr_m_t1),...
            std_estimated_Lambda_lr_m_t1,...
            mean_sqrt_Sigma_Lambda_2_lr_m_t1,...
            coverage_wald_Lambda_lr_m_t1,...
            coverage_percentile_Lambda_lr_m_t1,...
            abs(bias_Lambda_lr_m_t2),...
            std_estimated_Lambda_lr_m_t2,...
            mean_sqrt_Sigma_Lambda_2_lr_m_t2,...
            coverage_wald_Lambda_lr_m_t2,...
            coverage_percentile_Lambda_lr_m_t2,...        
            'VariableNames', {'SampleSize', 'AbsoluteBiasG_M', 'StandardDeviationG_M', 'MeanSqrtGammaTh3G_M','MeanSqrtGammaG_M', 'CoverageWaldG_M', 'CoveragePercentileG_M',... 
            'AbsoluteBiasLambdaG_M1', 'StandardDeviationLambdaG_M1', 'MeanSqrtSigmaLambdaG_M1', 'CoverageWaldLambdaG_M1', 'CoveragePercentileLambdaG_M1', 'AbsoluteBiasLambdaG_M2', 'StandardDeviationLambdaG_M2', 'MeanSqrtSigmaLambdaG_M2', 'CoverageWaldLambdaG_M2', 'CoveragePercentileLambdaG_M2', 'AbsoluteBiasLR_M', 'StandardDeviationLR_M', 'MeanSqrtGammaLR_M', ...
            'MeanSqrtGammaHuangLR_M', 'CoverageWaldLR_M', 'CoveragePercentileLR_M', 'AbsoluteBiasLambdaLR_M1', 'StandardDeviationLambdaLR_M1', 'MeanSqrtSigmaLambdaLR_M1', 'CoverageWaldLambdaLR_M1', 'CoveragePercentileLambdaLR_M1', 'AbsoluteBiasLambdaLR_M2', 'StandardDeviationLambdaLR_M2', 'MeanSqrtSigmaLambdaLR_M2', 'CoverageWaldLambdaLR_M2', 'CoveragePercentileLambdaLR_M2'});

        % Append to the extended_results_m table
        extended_results_m = [extended_results_m; temp_results_m];
end

% Reorder rows by Distribution and then by SampleSize
extended_results_m = sortrows(extended_results_m, {'SampleSize'});


% Display the results
disp('Extended Results for theta_m:');
disp(extended_results_m);

% Save the extended results to CSV files
writetable(extended_results_m, 'extended_simulation_results_m.csv');
