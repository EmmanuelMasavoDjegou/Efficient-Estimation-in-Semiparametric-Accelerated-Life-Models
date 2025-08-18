%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sAGT-TIC                                                                 %  
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
% Weibull Parameters
alpha = 1.5;
gamma = 1.5;                                                                                       
Lambda_0_dist = [1.5, 7.79]; % Baseline cumulative hazard functions at t = 1, 3              
% Store the baseline cumulative hazard functions in a cell array                                                                                  
tau_max = 3.5; % Maximum value for tau                                                                                         %  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function data_cp = generate_recurrent_data_cp(n, tau_max, alpha, gamma)
    % Distribution: V ~ Weibull(1.5, 1.5), Z ~ N(0, 1)
    Z = binornd(1, 0.5, [n, 1]);
    
    tau = unifrnd(0, tau_max, [n, 1]);
    
    data_list = cell(n, 1);
    
    for i = 1:n
        current_time = 0;
        event_times = [];
        gap_times = [];
        
        while true
            % Simulate failure times from the Weibull distribution
            U = rand(1, 1);  % Uniform(0,1) sample
            V = (-log(U) / alpha).^(1 / gamma);
 
            gap_time = V * exp(-0.8 * Z(i));
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
            if last_event_time < tau(i)
                censoring_time = tau(i);
                censoring_gap_time = tau(i) - last_event_time;
            else
                censoring_time = tau(i);
                censoring_gap_time = tau(i);
            end
        else
            censoring_time = tau(i);
            censoring_gap_time = tau(i);
        end
                
        individual_data = table( ...
            i * ones(length(event_times) + 1, 1), ...
            [event_times; censoring_time], ...
            [gap_times; censoring_gap_time], ...
            [ones(length(event_times), 1); 0], ...
            Z(i) * ones(length(event_times) + 1, 1), ...
            tau(i) * ones(length(event_times) + 1, 1), ...
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
    % Evaluate at t = 0.5
    t1 = 0.5;
    first_term_t1 = First_Term_G_M(theta_G_M, t1, data_cp);
    A_hat_t1 = estimate_A_hat_M(theta_G_M, Gamma_hat_G, t1, data_cp, B);
    Sigma_Lambda_2_G_M_t1 = first_term_t1 + (A_hat_t1 * Gamma_hat_G * A_hat_t1);
    
    % Evaluate at t = 1.5
    t2 = 1.5;
    first_term_t2 = First_Term_G_M(theta_G_M, t2, data_cp);
    A_hat_t2 = estimate_A_hat_M(theta_G_M, Gamma_hat_G, t2, data_cp, B);
    Sigma_Lambda_2_G_M_t2 = first_term_t2 + (A_hat_t2 * Gamma_hat_G * A_hat_t2);
end


% API MODEL

% ALGORITHM 1

% Accessory Functions

% Step 1

% Estimate Theta_G and Sigma_G

function X_lr_tilde = X_lr_tilde_func(T_lr, tau_l, S_l_kl, theta, Z_l, r, K_l)
    % Function to calculate \widetilde{X}_{lr}(\boldsymbol{\theta})
    
    % Calculate the exponentiated covariate
    exp_theta_Z_l = exp(theta * Z_l);
    
    % Determine the value of \widetilde{X}_{lr} based on r
    if r <= K_l
        % If r <= K_l
        X_lr_tilde = T_lr * exp_theta_Z_l;
    else
        % If r = K_l + 1
        X_lr_tilde = (tau_l - S_l_kl) * exp_theta_Z_l;
    end
end

function X_ij_tilde = X_ij_tilde_func(T_ij, theta, Z_i)
    % Function to calculate \widetilde{X}_{ij}(\boldsymbol{\theta})
    
    % Calculate the exponentiated covariate
    exp_theta_Z_i = exp(theta * Z_i);
    
    % Determine the value of \widetilde{X}_{ij} 
    X_ij_tilde = T_ij * exp_theta_Z_i;
end

function L_P_G = objective_function_p(theta, data_cp)
    % Extract the number of individuals
    n = length(unique(data_cp.id));

    % Initialize the summation variable
    total_sum = 0;

    % Loop over all individuals and events
    for i = 1:n
        % Extract data for individual i
        i_data = data_cp(data_cp.id == i, :);
        K_i = height(i_data); % Number of observed events for individual i
        Z_i = i_data.covariate(1); % Covariate for individual i

        for j = 1:K_i
            T_ij = i_data.gap_time(j);

            for l = 1:n
                % Extract data for individual l
                l_data = data_cp(data_cp.id == l, :);
                K_l = height(l_data); % Number of observed events for individual l
                tau_l = l_data.tau(1); % Censoring time for individual l
                S_l_kl = l_data.time(K_l); % Last observed event time for individual l
                Z_l = l_data.covariate(1); % Covariate for individual l

                for r = 1:(K_l + 1)
                    if r <= K_l
                        T_lr = l_data.gap_time(r); % r-th gap time for individual l
                    else
                        T_lr = (tau_l - S_l_kl); 
                    end

                    % Calculate \widetilde{X}_{lr}(\boldsymbol{\theta}) and \widetilde{X}_{ij}(\boldsymbol{\theta})
                    X_lr_tilde = X_lr_tilde_func(T_lr, tau_l, S_l_kl, theta, Z_l, r, K_l);
                    X_ij_tilde = X_ij_tilde_func(T_ij, theta, Z_i);

                    % Calculate the log difference
                    log_diff = log(X_lr_tilde) - log(X_ij_tilde);

                    % Apply the positive part operation
                    log_diff_pos = max(log_diff, 0);

                    % Add to the total sum
                    total_sum = total_sum + log_diff_pos;
                end
            end
        end
    end

    % Calculate the final objective function value
    L_P_G = (1 / n^2) * total_sum;
end


% Optimization to find the parameter theta
function estimated_theta_p = optimize_theta_p(data_cp)
    initial_theta_p = 0.7;
    
    options = optimset('fminsearch');
    options.Display = 'off';
    [estimated_theta_p, ~] = fminsearch(@(theta) ...
        objective_function_p(theta, data_cp), initial_theta_p, options);
end


% Generalized at-risk process Y_l

function indicator = indicator_tilde_func(T_lr, tau_l, S_l_kl, T_ij, theta, Z_i, Z_l, r, K_l)
    % Calculate \widetilde{X}_{ij}(\boldsymbol{\theta})
    X_ij_tilde = X_ij_tilde_func(T_ij, theta, Z_i);

    % Calculate \widetilde{X}_{lr}(\boldsymbol{\theta})
    X_lr_tilde = X_lr_tilde_func(T_lr, tau_l, S_l_kl, theta, Z_l, r, K_l);
    
    % Compute the indicator value
    indicator = (X_ij_tilde <= X_lr_tilde);
end

function Y_l_P = Y_l_P_func(tau_l, S_l_kl, T_ij, theta, Z_i, Z_l, K_l, l_data)
    % Function to calculate Y_{l}^P(\widetilde{X}_{ij}(\boldsymbol{\theta}) \mid \boldsymbol{\theta})
    
    % Initialize the sum
    sum_indicator = 0;
    
    % Loop over r from 1 to K_l + 1
    for r = 1:(K_l + 1)
        if r <= K_l
            T_lr = l_data.gap_time(r); % r-th gap time for individual l
        else
            T_lr = (tau_l - S_l_kl);
        end
            
        % Calculate the indicator value
        indicator_value = indicator_tilde_func(T_lr, tau_l, S_l_kl, T_ij, theta, Z_i, Z_l, r, K_l);
        sum_indicator = sum_indicator + indicator_value;
    end
    
    % Return the sum
    Y_l_P = sum_indicator;
end


% S0
function S_0 = S_0_func_p(T_ij, theta, Z_i, data_cp)
    % Calculate \mathbb{S}^{(0)}(\Tij e^{\theta^{\prime} Z_i} \mid \theta)
    n = length(unique(data_cp.id));
    total_sum = 0;
    unique_ids = unique(data_cp.id);
    
    for l = unique_ids'
        l_data = data_cp(data_cp.id == l, :);  % Corrected subsetting
        if ~isempty(l_data)  % Check if l_data is not empty
            tau_l = l_data.tau(1);
            Z_l = l_data.covariate(1);
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
    % Calculate \mathbb{S}^{(0)}(\Tij e^{\theta^{\prime} Z_i} \mid \theta)
    n = length(unique(data_cp.id));
    total_sum = 0;
    unique_ids = unique(data_cp.id);
    
    for l = unique_ids'
        l_data = data_cp(data_cp.id == l, :);  % Corrected subsetting
        if ~isempty(l_data)  % Check if l_data is not empty
            tau_l = l_data.tau(1);
            Z_l = l_data.covariate(1);
            K_l = height(l_data);
            S_l_kl = l_data.time(K_l); % Last observed event time for individual l
            Y_l = Y_l_P_func(tau_l, S_l_kl, T_ij, theta, Z_i, Z_l, K_l, l_data);
            total_sum = total_sum +  Z_l * Y_l;
        end
    end
    S_1 = (1/n) * total_sum;
end


% S2
function S_2 = S_2_func_p(T_ij, theta, Z_i, data_cp)
    % Calculate \mathbb{S}^{(0)}(\Tij e^{\theta^{\prime} Z_i} \mid \theta)
    n = length(unique(data_cp.id));
    total_sum = 0;
    unique_ids = unique(data_cp.id);
    
    for l = unique_ids'
        l_data = data_cp(data_cp.id == l, :);  % Corrected subsetting
        if ~isempty(l_data)  % Check if l_data is not empty
            tau_l = l_data.tau(1);
            Z_l = l_data.covariate(1);
            K_l = height(l_data);
            S_l_kl = l_data.time(K_l); % Last observed event time for individual l
            Y_l = Y_l_P_func(tau_l, S_l_kl, T_ij, theta, Z_i, Z_l, K_l, l_data);
            total_sum = total_sum +  Z_l^2 * Y_l;
        end
    end
    
    S_2 = (1/n) * total_sum;
end


% Function to compute \varphi(w)
function phi_w = phi_function_p(T_ij, theta, Z_i, data_cp)
    % Compute S^0, S^1, and S^2
    S_0 = S_0_func_p(T_ij, theta, Z_i, data_cp);
    S_1 = S_1_func_p(T_ij, theta, Z_i, data_cp);
    S_2 = S_2_func_p(T_ij, theta, Z_i, data_cp);
    
    % Calculate phi(w)
    phi_w = S_0 * S_2 - (S_1)^2;
end


% Function to compute \widehat{\Sigma}_G
function Sigma_G = calculate_sigma_p(data_cp, theta_hat)
    n = length(unique(data_cp.id));
    total_sum = 0;
    unique_ids = unique(data_cp.id);
    
    for i = unique_ids'
        i_data = data_cp(data_cp.id == i, :);  % Subsetting data for 
        % individual i
        K_i = height(i_data);
        for j = 1:K_i
            T_ij = i_data.gap_time(j);  % Assuming time corresponds to T_ij
            Z_i = i_data.covariate(1);
            % Compute \varphi(T_{ij} e^{\theta^{\prime} Z_i})
            phi_w = phi_function_p(T_ij, theta_hat, Z_i, data_cp);
            total_sum = total_sum + phi_w;
        end
    end
    
    Sigma_G = (1/n) * total_sum;
end

% Step 2

% Estimate Theta_tilde_G

% Objective function_tilde_p


function S_P_G = objective_function_tilde_p(theta, data_cp,  sigma)
    % Extract the number of individuals
    n = length(unique(data_cp.id));

    % Initialize the summation variable
    total_sum = 0;

    % Loop over all individuals and events
    for i = 1:n
        % Extract data for individual i
        i_data = data_cp(data_cp.id == i, :);
        K_i = height(i_data); % Number of observed events for individual i
        Z_i = i_data.covariate(1); % Covariate for individual i

        for j = 1:K_i
            T_ij = i_data.gap_time(j);

            for l = 1:n
                % Extract data for individual l
                l_data = data_cp(data_cp.id == l, :);
                K_l = height(l_data); % Number of observed events for individual l
                tau_l = l_data.tau(1); % Censoring time for individual l
                S_l_kl = l_data.time(K_l); % Last observed event time for individual l
                Z_l = l_data.covariate(1); % Covariate for individual l
                Y_l = Y_l_P_func(tau_l, S_l_kl, T_ij, theta, Z_i, Z_l, K_l, l_data);
                
                % Calculate the difference
                    Z_diff = Z_i - Z_l;
                % Add the contribution to the total sum
                    total_sum = total_sum + Y_l * Z_diff;
            end
        end
    end
    S_P_G = abs(total_sum / (n^2) - sqrt((1/n) * sigma));
end

% Optimization to find the parameter theta
function estimated_theta_tilde_p = optimize_theta_tilde_p(data_cp, sigma)
    initial_theta_tilde_p = 0.7;
    
    options = optimset('fminsearch');
    options.Display = 'off';
    [estimated_theta_tilde_p, ~] = fminsearch(@(theta) ...
        objective_function_tilde_p(theta, data_cp, sigma), ...
        initial_theta_tilde_p, options);
end


% Estimating Lambda_G_P

% Lambda_G_P Function

function Lambda_0_t = Lambda_G_P(theta, data_cp, t)
    % Extract the number of individuals
    n = length(unique(data_cp.id));

    % Initialize the summation variable
    total_sum = 0;
    
    % Loop over all individuals and events
    for i = 1:n
        % Extract data for individual i
        i_data = data_cp(data_cp.id == i, :);
        K_i = height(i_data); % Number of observed events for individual i
        Z_i = i_data.covariate(1); % Covariate for individual i
        
        for j = 1:K_i
            T_ij = i_data.gap_time(j); % Extract T_ij for individual i at event j

            % Compute Y_l for all individuals
            Y_l_sum = 0;
            for l = 1:n
                % Extract data for individual l
                l_data = data_cp(data_cp.id == l, :);
                K_l = height(l_data); % Number of observed events for individual l
                tau_l = l_data.tau(1); % Censoring time for individual l
                S_l = l_data.time(K_l); % Last observed event time for individual l
                Z_l = l_data.covariate(1); % Covariate for individual l
                
                % Compute Y_l
                Y_l = Y_l_P_func(tau_l, S_l, T_ij, theta, Z_i, Z_l, K_l, l_data);
                
                % Update the weight sum
                Y_l_sum = Y_l_sum + Y_l;
            end

            % Calculate the indicator function
            X_ij = i_data.gap_time(j) * exp(theta * Z_i); % Compute X_ij based on theta and Z_i
            indicator = X_ij <= t;
            
            % Calculate the contribution to the total sum
            if Y_l_sum > 0
                total_sum = total_sum + (indicator / Y_l_sum);
            end
        end
    end
    
    % Compute Lambda_0(t)
    Lambda_0_t = total_sum;
end

% Function to evaluate Lambda_G_P at t = 1 and t = 2
function [lambda_hat_t1, lambda_hat_t2] = evaluate_Lambda_G_P(theta_G_P, data_cp)
    % Evaluate at t = 1
    t1 = 1;
    lambda_hat_t1 = Lambda_G_P(theta_G_P, data_cp, t1);

    % Evaluate at t = 3
    t2 = 3;
    lambda_hat_t2 = Lambda_G_P(theta_G_P, data_cp, t2);
end


% Estimating Sigma_Lambda_2_G_P

function first_term = First_Term_G_P(theta, data_cp, t)
    % Extract the number of individuals
    n = length(unique(data_cp.id));

    % Initialize the summation variable
    total_sum = 0;
    
    % Loop over all individuals and events
    for i = 1:n
        % Extract data for individual i
        i_data = data_cp(data_cp.id == i, :);
        K_i = height(i_data); % Number of observed events for individual i
        Z_i = i_data.covariate(1); % Covariate for individual i
        
        for j = 1:K_i
            T_ij = i_data.gap_time(j); % Extract T_ij for individual i at event j

            % Compute Y_l for all individuals
            Y_l_sum = 0;
            for l = 1:n
                % Extract data for individual l
                l_data = data_cp(data_cp.id == l, :);
                K_l = height(l_data); % Number of observed events for individual l
                tau_l = l_data.tau(1); % Censoring time for individual l
                S_l = l_data.time(K_l); % Last observed event time for individual l
                Z_l = l_data.covariate(1); % Covariate for individual l
                
                % Compute Y_l
                Y_l = Y_l_P_func(tau_l, S_l, T_ij, theta, Z_i, Z_l, K_l, l_data);
                
                % Update the weight sum
                Y_l_sum = Y_l_sum + Y_l;
            end

            % Calculate the indicator function
            X_ij = i_data.gap_time(j) * exp(theta * Z_i); % Compute X_ij based on theta and Z_i
            indicator = X_ij <= t;
            
            % Calculate the contribution to the total sum
            if Y_l_sum > 0
                total_sum = total_sum + (indicator / (Y_l_sum)^2);
            end
        end
    end
    
    % Compute Lambda_0(t)
    first_term = n * total_sum;
end


% Estimate A_hat
function A_hat_B_tilde = estimate_A_hat_P(theta_G_P, Gamma_hat_G, t, data_cp, B)
    % Parameters
    n = length(unique(data_cp.id));  % Number of samples
    cov_matrix = (1/n) * Gamma_hat_G;  % Covariance matrix (scalar)

    % Bootstrap samples of theta
    theta_tilde = normrnd(theta_G_P, sqrt(cov_matrix), [1, B]);

    % Compute \bar{S}_{LR}^M for the bootstrap samples and the original estimate
    lambda_hat_theta_G = Lambda_G_P(theta_G_P, data_cp, t);
    lambda_hat_tilde = zeros(1, B);

    for k = 1:B
        lambda_hat_tilde(k) = Lambda_G_P(theta_tilde(k), data_cp, t);
    end

    % Construct \mathfrak{Z} and \mathfrak{Y}
    Z = [ones(B, 1), (theta_G_P - theta_tilde)'];
    Y = (lambda_hat_tilde - lambda_hat_theta_G)';

    % Remove the first row from the result of the matrix multiplication
    A_hat_B_tilde = (Z' * Z) \ (Z' * Y);
    A_hat_B_tilde = A_hat_B_tilde(2:end, :);  % Removing the first row
    A_hat_B_tilde = (1/2) * (A_hat_B_tilde + A_hat_B_tilde);

end


% Function to evaluate Sigma_Lambda_2_G_M at t = 0.5 and t = 1.5
function [Sigma_Lambda_2_G_P_t1, Sigma_Lambda_2_G_P_t2] = evaluate_Sigma_Lambda_2_G_P(theta_G_P, Gamma_hat_G, data_cp, B)
    % Evaluate at t = 1
    t1 = 1;
    first_term_t1 = First_Term_G_P(theta_G_P, data_cp, t1);
    A_hat_t1 = estimate_A_hat_P(theta_G_P, Gamma_hat_G, t1, data_cp, B);
    Sigma_Lambda_2_G_P_t1 = first_term_t1 + (A_hat_t1 * Gamma_hat_G * A_hat_t1);
    
    % Evaluate at t = 3
    t2 = 3;
    first_term_t2 = First_Term_G_P(theta_G_P, data_cp, t2);
    A_hat_t2 = estimate_A_hat_P(theta_G_P, Gamma_hat_G, t2, data_cp, B);
    Sigma_Lambda_2_G_P_t2 = first_term_t2 + (A_hat_t2 * Gamma_hat_G * A_hat_t2);
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
    % Evaluate at t = 0.5
    t1 = 0.5;
    first_term_t1 = First_Term_LR_M(theta_LR_M, t1, data_cp);
    A_hat_t1 = estimate_A_hat_M_LR(theta_LR_M, Gamma_hat_LR, t1, data_cp, B);
    Sigma_Lambda_2_LR_M_t1 = first_term_t1 + (A_hat_t1 * Gamma_hat_LR * A_hat_t1);
    
    % Evaluate at t = 1.5
    t2 = 1.5;
    first_term_t2 = First_Term_LR_M(theta_LR_M, t2, data_cp);
    A_hat_t2 = estimate_A_hat_M_LR(theta_LR_M, Gamma_hat_LR, t2, data_cp, B);
    Sigma_Lambda_2_LR_M_t2 = first_term_t2 + (A_hat_t2 * Gamma_hat_LR * A_hat_t2);
end




% API MODEL

% ALGORITHM 2

% Accessory Functions

% Score Function S_LR_P
function S_LR_P = S_LR_P(theta, data_cp)
    % Extract the number of individuals
    n = length(unique(data_cp.id));

    % Initialize the summation variable
    total_sum = 0;

    % Loop over all individuals and events
    for i = 1:n
        % Extract data for individual i
        i_data = data_cp(data_cp.id == i, :);
        K_i = height(i_data); % Number of observed events for individual i
        Z_i = i_data.covariate(1); % Covariate for individual i

        for j = 1:K_i
            T_ij = i_data.gap_time(j); % Extract T_ij for individual i at event j

            % Compute Y_l for all individuals
            Y_l_sum = 0;
            Y_l_values = zeros(n, 1);
            for l_inner = 1:n
                l_inner_data = data_cp(data_cp.id == l_inner, :);
                K_l_inner = height(l_inner_data); % Number of observed events for individual l_inner
                tau_l_inner = l_inner_data.tau(1); % Censoring time for individual l_inner
                S_l_kl_inner = l_inner_data.time(K_l_inner); % Last observed event time for individual l_inner
                Z_l_inner = l_inner_data.covariate(1); % Covariate for individual l_inner
                Y_l_inner = Y_l_P_func(tau_l_inner, S_l_kl_inner, T_ij, theta, Z_i, Z_l_inner, K_l_inner, l_inner_data);
                Y_l_values(l_inner) = Y_l_inner;
                Y_l_sum = Y_l_sum + Y_l_inner;
            end

            for l = 1:n
                % Extract data for individual l
                l_data = data_cp(data_cp.id == l, :);
                Z_l = l_data.covariate(1); % Covariate for individual l
                
                % Calculate the weight
                if Y_l_sum > 0
                    Y_l = Y_l_values(l) / Y_l_sum;
                else
                    Y_l = 0; % Handle the case where Y_l_sum is zero to avoid division by zero
                end
                
                % Calculate the difference
                Z_diff = Z_i - Z_l;
                
                % Add the contribution to the total sum
                total_sum = total_sum + Y_l * Z_diff;
            end
        end
    end
    S_LR_P = total_sum / n;
end

% Function to compute \varphi(w)
function phi_w = phi_function_lr_p(T_ij, theta, Z_i, data_cp)
    % Compute S^0, S^1, and S^2
    S_0 = S_0_func_p(T_ij, theta, Z_i, data_cp);
    S_1 = S_1_func_p(T_ij, theta, Z_i, data_cp);
    S_2 = S_2_func_p(T_ij, theta, Z_i, data_cp);
    
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
function Sigma_LR = calculate_sigma_lr_p(data_cp, theta_hat)
    n = length(unique(data_cp.id));
    total_sum = 0;
    unique_ids = unique(data_cp.id);
    
    for i = unique_ids'
        i_data = data_cp(data_cp.id == i, :);  % Subsetting data for individual i
        K_i = height(i_data);
        for j = 1:K_i
            T_ij = i_data.gap_time(j);  % Assuming time corresponds to T_ij
            Z_i = i_data.covariate(1);
            % Compute \varphi(S_{ij} e^{\theta^{\prime} Z_i})
            phi_w = phi_function_lr_p(T_ij, theta_hat, Z_i, data_cp);
            % Only add phi_w if it is not NaN
            if ~isnan(phi_w)
                total_sum = total_sum + phi_w;
            end
        end
    end
    
    Sigma_LR = (1/n) * total_sum;
end


% Objective function_tilde_lr_p
function obj_value1 = objective_function_tilde_lr_p(theta, data_cp, sigma)
    % Extract the number of individuals
    n = length(unique(data_cp.id));

    % Initialize the summation variable
    total_sum = 0;

    % Loop over all individuals and events
    for i = 1:n
        % Extract data for individual i
        i_data = data_cp(data_cp.id == i, :);
        K_i = height(i_data); % Number of observed events for individual i
        Z_i = i_data.covariate(1); % Covariate for individual i

        for j = 1:K_i
            T_ij = i_data.gap_time(j); % Extract T_ij for individual i at event j

            % Compute Y_l for all individuals
            Y_l_sum = 0;
            Y_l_values = zeros(n, 1);
            for l_inner = 1:n
                l_inner_data = data_cp(data_cp.id == l_inner, :);
                K_l_inner = height(l_inner_data); % Number of observed events for individual l_inner
                tau_l_inner = l_inner_data.tau(1); % Censoring time for individual l_inner
                S_l_kl_inner = l_inner_data.time(K_l_inner); % Last observed event time for individual l_inner
                Z_l_inner = l_inner_data.covariate(1); % Covariate for individual l_inner
                Y_l_inner = Y_l_P_func(tau_l_inner, S_l_kl_inner, T_ij, theta, Z_i, Z_l_inner, K_l_inner, l_inner_data);
                Y_l_values(l_inner) = Y_l_inner;
                Y_l_sum = Y_l_sum + Y_l_inner;
            end

            for l = 1:n
                % Extract data for individual l
                l_data = data_cp(data_cp.id == l, :);
                Z_l = l_data.covariate(1); % Covariate for individual l
                
                % Calculate the weight
                if Y_l_sum > 0
                    Y_l = Y_l_values(l) / Y_l_sum;
                else
                    Y_l = 0; % Handle the case where Y_l_sum is zero to avoid division by zero
                end
                
                % Calculate the difference
                Z_diff = Z_i - Z_l;
                
                % Add the contribution to the total sum
                total_sum = total_sum + Y_l * Z_diff;
            end
        end
    end
    
    obj_value1 = abs(total_sum /(n) - sqrt((1/n) * sigma));
end


% Optimization to find the parameter theta
function estimated_theta_tilde_lr_p = optimize_theta_tilde_lr_p(data_cp, sigma)
    initial_theta_tilde = 0.7;
    
    options = optimset('fminsearch');
    options.Display = 'off';
    [estimated_theta_tilde_lr_p, ~] = fminsearch(@(theta) ...
        objective_function_tilde_lr_p(theta, data_cp, sigma), ...
        initial_theta_tilde, options);
end

% Estimating Lambda_LR_P

% Lambda_LR_P Function

function Lambda_0_t = Lambda_LR_P(theta, data_cp, t)
    % Extract the number of individuals
    n = length(unique(data_cp.id));

    % Initialize the summation variable
    total_sum = 0;
    
    % Loop over all individuals and events
    for i = 1:n
        % Extract data for individual i
        i_data = data_cp(data_cp.id == i, :);
        K_i = height(i_data); % Number of observed events for individual i
        Z_i = i_data.covariate(1); % Covariate for individual i
        
        for j = 1:K_i
            T_ij = i_data.gap_time(j); % Extract T_ij for individual i at event j

            % Compute Y_l for all individuals
            Y_l_sum = 0;
            for l = 1:n
                % Extract data for individual l
                l_data = data_cp(data_cp.id == l, :);
                K_l = height(l_data); % Number of observed events for individual l
                tau_l = l_data.tau(1); % Censoring time for individual l
                S_l = l_data.time(K_l); % Last observed event time for individual l
                Z_l = l_data.covariate(1); % Covariate for individual l
                
                % Compute Y_l
                Y_l = Y_l_P_func(tau_l, S_l, T_ij, theta, Z_i, Z_l, K_l, l_data);
                
                % Update the weight sum
                Y_l_sum = Y_l_sum + Y_l;
            end

            % Calculate the indicator function
            X_ij = i_data.gap_time(j) * exp(theta * Z_i); % Compute X_ij based on theta and Z_i
            indicator = X_ij <= t;
            
            % Calculate the contribution to the total sum
            if Y_l_sum > 0
                total_sum = total_sum + (indicator / Y_l_sum);
            end
        end
    end
    
    % Compute Lambda_0(t)
    Lambda_0_t = total_sum;
end

% Function to evaluate Lambda_LR_P at t = 1 and t = 2
function [lambda_hat_t1, lambda_hat_t2] = evaluate_Lambda_LR_P(theta_LR_P, data_cp)
    % Evaluate at t = 1
    t1 = 1;
    lambda_hat_t1 = Lambda_LR_P(theta_LR_P, data_cp, t1);

    % Evaluate at t = 3
    t2 = 3;
    lambda_hat_t2 = Lambda_LR_P(theta_LR_P, data_cp, t2);
end

% Estimating Sigma_Lambda_2_LR_P

function first_term = First_Term_LR_P(theta, data_cp, t)
    % Extract the number of individuals
    n = length(unique(data_cp.id));

    % Initialize the summation variable
    total_sum = 0;
    
    % Loop over all individuals and events
    for i = 1:n
        % Extract data for individual i
        i_data = data_cp(data_cp.id == i, :);
        K_i = height(i_data); % Number of observed events for individual i
        Z_i = i_data.covariate(1); % Covariate for individual i
        
        for j = 1:K_i
            T_ij = i_data.gap_time(j); % Extract T_ij for individual i at event j

            % Compute Y_l for all individuals
            Y_l_sum = 0;
            for l = 1:n
                % Extract data for individual l
                l_data = data_cp(data_cp.id == l, :);
                K_l = height(l_data); % Number of observed events for individual l
                tau_l = l_data.tau(1); % Censoring time for individual l
                S_l = l_data.time(K_l); % Last observed event time for individual l
                Z_l = l_data.covariate(1); % Covariate for individual l
                
                % Compute Y_l
                Y_l = Y_l_P_func(tau_l, S_l, T_ij, theta, Z_i, Z_l, K_l, l_data);
                
                % Update the weight sum
                Y_l_sum = Y_l_sum + Y_l;
            end

            % Calculate the indicator function
            X_ij = i_data.gap_time(j) * exp(theta * Z_i); % Compute X_ij based on theta and Z_i
            indicator = X_ij <= t;
            
            % Calculate the contribution to the total sum
            if Y_l_sum > 0
                total_sum = total_sum + (indicator / (Y_l_sum)^2);
            end
        end
    end
    
    % Compute Lambda_0(t)
    first_term = n * total_sum;
end


% Estimate A_hat
function A_hat_B_tilde = estimate_A_hat_P_LR(theta_LR_P, Gamma_hat_LR, t, data_cp, B)
    % Parameters
    n = length(unique(data_cp.id));  % Number of samples
    cov_matrix = (1/n) * Gamma_hat_LR;  % Covariance matrix (scalar)

    % Bootstrap samples of theta
    theta_tilde = normrnd(theta_LR_P, sqrt(cov_matrix), [1, B]);

    % Compute \bar{S}_{LR}^M for the bootstrap samples and the original estimate
    lambda_hat_theta_LR = Lambda_LR_P(theta_LR_P, data_cp, t);
    lambda_hat_tilde = zeros(1, B);

    for k = 1:B
        lambda_hat_tilde(k) = Lambda_LR_P(theta_tilde(k), data_cp, t);
    end

    % Construct \mathfrak{Z} and \mathfrak{Y}
    Z = [ones(B, 1), (theta_LR_P - theta_tilde)'];
    Y = (lambda_hat_tilde - lambda_hat_theta_LR)';

    % Remove the first row from the result of the matrix multiplication
    A_hat_B_tilde = (Z' * Z) \ (Z' * Y);
    A_hat_B_tilde = A_hat_B_tilde(2:end, :);  % Removing the first row
    A_hat_B_tilde = (1/2) * (A_hat_B_tilde + A_hat_B_tilde);

end


% Function to evaluate Sigma_Lambda_2_LR_M at t = 0.5 and t = 1.5
function [Sigma_Lambda_2_LR_P_t1, Sigma_Lambda_2_LR_P_t2] = evaluate_Sigma_Lambda_2_LR_P(theta_LR_P, Gamma_hat_LR, data_cp, B)
    % Evaluate at t = 1
    t1 = 1;
    first_term_t1 = First_Term_LR_P(theta_LR_P, data_cp, t1);
    A_hat_t1 = estimate_A_hat_P_LR(theta_LR_P, Gamma_hat_LR, t1, data_cp, B);
    Sigma_Lambda_2_LR_P_t1 = first_term_t1 + (A_hat_t1 * Gamma_hat_LR * A_hat_t1);
    
    % Evaluate at t = 3
    t2 = 3;
    first_term_t2 = First_Term_LR_P(theta_LR_P, data_cp, t2);
    A_hat_t2 = estimate_A_hat_P_LR(theta_LR_P, Gamma_hat_LR, t2, data_cp, B);
    Sigma_Lambda_2_LR_P_t2 = first_term_t2 + (A_hat_t2 * Gamma_hat_LR * A_hat_t2);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  MAIN FUNCTIONS                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Initialize results storage
results_p = table();


% Loop over sample sizes
for n = n_values
        % Store estimates and variances for current sample size and distribution
        estimates_g_p = nan(replications, 1);
        sigmas_g_p = nan(replications, 1);
        estimates_tilde_g_p = nan(replications, 1);
        gammas_g_p = nan(replications, 1);
        Lambdas_hat_g_p_t1 = nan(replications, 1);
        Sigmas_Lambda_2_g_p_t1 = nan(replications, 1); 
        Lambdas_hat_g_p_t2 = nan(replications, 1);
        Sigmas_Lambda_2_g_p_t2 = nan(replications, 1); 
        estimates_lr_p = nan(replications, 1);
        gammas_lr_p = nan(replications, 1);
        gammahuangs_lr_p = nan(replications, 1);
        Lambdas_hat_lr_p_t1 = nan(replications, 1);
        Sigmas_Lambda_2_lr_p_t1 = nan(replications, 1); 
        Lambdas_hat_lr_p_t2 = nan(replications, 1);
        Sigmas_Lambda_2_lr_p_t2 = nan(replications, 1); 
     
        % Perform simulations
        for rep = 1:replications
            % Generate data
            data_cp = generate_recurrent_data_cp(n, tau_max, alpha, gamma);
    
            % Repeat for theta_g_p
            estimated_theta_g_p = optimize_theta_p(data_cp);

            % Calculate Sigma_G for theta_g_p
            estimated_sigma_g_p = calculate_sigma_p(data_cp, estimated_theta_g_p);

            % Optimize theta_tilde_g_p
            estimated_theta_tilde_g_p = optimize_theta_tilde_p(data_cp, estimated_sigma_g_p);

            % Calculate Gamma for theta_g_p
            estimated_gamma_g_p = (sqrt(n) * (estimated_theta_tilde_g_p - estimated_theta_g_p))^2;

            % Calculate Lambda_hat_g_p
            [Lambda_hat_g_p_t1, Lambda_hat_g_p_t2] = evaluate_Lambda_G_P(estimated_theta_g_p, data_cp);
            
            % Calcualte Sigma_Lambda_2_G_P
            [Sigma_Lambda_2_g_p_t1, Sigma_Lambda_2_g_p_t2] = evaluate_Sigma_Lambda_2_G_P(estimated_theta_g_p, estimated_gamma_g_p , data_cp, B2);


            % Calculate theta_lr_p
            score_lr_p = S_LR_P(estimated_theta_g_p, data_cp);
            d_hat_lr_b_p = estimate_D_LR_B(estimated_theta_g_p, estimated_gamma_g_p, data_cp, B1);
            d_hat_lr_b_p_inv = test_finite_d_hat_lr_b_inv(d_hat_lr_b_p);
            estimated_theta_lr_p = estimated_theta_g_p - (d_hat_lr_b_p_inv * score_lr_p);    

            % Calculate Sigma_LR for theta_lr_p using the function calculate_sigma_lr_p
            estimated_sigma_lr_p = calculate_sigma_lr_p(data_cp, estimated_theta_lr_p);

            % Calculate Gamma for theta_lr_p
            d_hat_lr_b_p_inv = test_finite_d_hat_lr_b_inv(d_hat_lr_b_p);
            estimated_gamma_lr_p = d_hat_lr_b_p_inv * estimated_sigma_lr_p * d_hat_lr_b_p_inv;

            % Optimize theta_tilde_lr_p
            estimated_theta_tilde_lr_p = optimize_theta_tilde_lr_p(data_cp, estimated_sigma_lr_p);

            % Calculate GammaHuang for theta_lr_p 
            estimated_gammahuang_lr_p = (sqrt(n) * (estimated_theta_tilde_lr_p - estimated_theta_lr_p))^2;

            % Calculate Lambda_hat_lr_p
            [Lambda_hat_lr_p_t1, Lambda_hat_lr_p_t2] = evaluate_Lambda_LR_P(estimated_theta_lr_p, data_cp);
          
            % Calcualte Sigma_Lambda_2_LR_P
            [Sigma_Lambda_2_lr_p_t1, Sigma_Lambda_2_lr_p_t2] = evaluate_Sigma_Lambda_2_LR_P(estimated_theta_lr_p, estimated_gamma_lr_p, data_cp, B2);
          

            % Store the estimates for theta_g_p
            estimates_g_p(rep) = estimated_theta_g_p;
            sigmas_g_p(rep) = estimated_sigma_g_p;
            estimates_tilde_g_p(rep) = estimated_theta_tilde_g_p;
            gammas_g_p(rep) = estimated_gamma_g_p;
            Lambdas_hat_g_p_t1(rep) =  Lambda_hat_g_p_t1;
            Sigmas_Lambda_2_g_p_t1(rep) = Sigma_Lambda_2_g_p_t1;
            Lambdas_hat_g_p_t2(rep) =  Lambda_hat_g_p_t2;
            Sigmas_Lambda_2_g_p_t2(rep) = Sigma_Lambda_2_g_p_t2;

            % Store the estimates for theta_lr_p
            estimates_lr_p(rep) = estimated_theta_lr_p;
            gammas_lr_p(rep) = estimated_gamma_lr_p;
            gammahuangs_lr_p(rep) = estimated_gammahuang_lr_p;
            Lambdas_hat_lr_p_t1(rep) =  Lambda_hat_lr_p_t1;
            Sigmas_Lambda_2_lr_p_t1(rep) = Sigma_Lambda_2_lr_p_t1;
            Lambdas_hat_lr_p_t2(rep) =  Lambda_hat_lr_p_t2;
            Sigmas_Lambda_2_lr_p_t2(rep) = Sigma_Lambda_2_lr_p_t2;
           
        end

        % Create a summary table for current settings for theta_p
        temp_results_p = table( ...
            repmat(n, replications, 1), ...
            estimates_g_p, sigmas_g_p, estimates_tilde_g_p, gammas_g_p, Lambdas_hat_g_p_t1, Sigmas_Lambda_2_g_p_t1, Lambdas_hat_g_p_t2, Sigmas_Lambda_2_g_p_t2, estimates_lr_p, gammas_lr_p, gammahuangs_lr_p, Lambdas_hat_lr_p_t1, Sigmas_Lambda_2_lr_p_t1, Lambdas_hat_lr_p_t2, Sigmas_Lambda_2_lr_p_t2,...
            'VariableNames', ...
            {'SampleSize', 'EstimatedThetaG_P', ...
            'EstimatedSigmaG_P', 'EstimatedThetaTildeG_P', 'EstimatedGammaG_P', 'EstimatedLambdaG_P1', 'EstimatedSigmaLambda2G_P1', 'EstimatedLambdaG_P2', 'EstimatedSigmaLambda2G_P2','EstimatedThetaLR_P', 'EstimatedGammaLR_P', 'EstimatedGammaHuangLR_P', 'EstimatedLambdaLR_P1', 'EstimatedSigmaLambda2LR_P1', 'EstimatedLambdaLR_P2', 'EstimatedSigmaLambda2LR_P2'});
        
        % Append to results_g_p
        results_p = [results_p; temp_results_p];
end

% Display results
disp('Results for theta_p:');
disp(results_p);

% Save results to CSV files
writetable(results_p, 'simulation_results_p.csv');


% Initialize tables to store the bias, standard deviation, sqrt(EstimatedGamma), and coverage results
extended_results_p = table();


% Loop over sample sizes
for n = n_values
    % Loop over distributions
        % Filter results for current sample size
        current_results_p = results_p(results_p.SampleSize == n,:);

        % Compute for theta_p
        % Compute the sample mean of EstimatedThetaP
        mean_estimated_theta_g_p = mean(current_results_p.EstimatedThetaG_P, 'omitnan');
        mean_estimated_Lambda_g_p_t1 = mean(current_results_p.EstimatedLambdaG_P1, 'omitnan');
        mean_estimated_Lambda_g_p_t2 = mean(current_results_p.EstimatedLambdaG_P2, 'omitnan');
        mean_estimated_theta_lr_p = mean(current_results_p.EstimatedThetaLR_P, 'omitnan');
        mean_estimated_Lambda_lr_p_t1 = mean(current_results_p.EstimatedLambdaLR_P1, 'omitnan');
        mean_estimated_Lambda_lr_p_t2 = mean(current_results_p.EstimatedLambdaLR_P2, 'omitnan');
        

        % Compute the sample standard deviation of EstimatedThetaP
        std_estimated_theta_g_p = std(current_results_p.EstimatedThetaG_P, 'omitnan');
        std_estimated_Lambda_g_p_t1 = std(current_results_p.EstimatedLambdaG_P1, 'omitnan');
        std_estimated_Lambda_g_p_t2 = std(current_results_p.EstimatedLambdaG_P2, 'omitnan');
        std_estimated_theta_lr_p = std(current_results_p.EstimatedThetaLR_P, 'omitnan');
        std_estimated_Lambda_lr_p_t1 = std(current_results_p.EstimatedLambdaLR_P1, 'omitnan');
        std_estimated_Lambda_lr_p_t2 = std(current_results_p.EstimatedLambdaLR_P2, 'omitnan');


        % Compute the mean of sqrt(EstimatedGammaP)
        mean_sqrt_gamma_g_p = mean(sqrt(current_results_p.EstimatedGammaG_P), 'omitnan');
        mean_sqrt_gammaThm3_g_p = NaN;
        mean_sqrt_Sigma_Lambda_2_g_p_t1 = mean(sqrt(current_results_p.EstimatedSigmaLambda2G_P1), 'omitnan');
        mean_sqrt_Sigma_Lambda_2_g_p_t2 = mean(sqrt(current_results_p.EstimatedSigmaLambda2G_P2), 'omitnan');
        

        mean_sqrt_gammahuang_lr_p = mean(sqrt(current_results_p.EstimatedGammaHuangLR_P), 'omitnan');
        mean_sqrt_gamma_lr_p = mean(sqrt(current_results_p.EstimatedGammaLR_P), 'omitnan');
        mean_sqrt_Sigma_Lambda_2_lr_p_t1 = mean(sqrt(current_results_p.EstimatedSigmaLambda2LR_P1), 'omitnan');
        mean_sqrt_Sigma_Lambda_2_lr_p_t2 = mean(sqrt(current_results_p.EstimatedSigmaLambda2LR_P2), 'omitnan');
        
        
        % Compute the bias relative to true theta value
        bias_g_p = mean_estimated_theta_g_p - theta_0;
        true_Lambda_0_t1 = Lambda_0_dist(1); % true value at t = 0.5
        bias_Lambda_g_p_t1 = mean_estimated_Lambda_g_p_t1 - true_Lambda_0_t1;
        true_Lambda_0_t2 = Lambda_0_dist(2); % true value at t = 1.5
        bias_Lambda_g_p_t2 = mean_estimated_Lambda_g_p_t2 - true_Lambda_0_t2;  
        bias_lr_p = mean_estimated_theta_lr_p - theta_0;
        bias_Lambda_lr_p_t1 = mean_estimated_Lambda_lr_p_t1 - true_Lambda_0_t1;
        bias_Lambda_lr_p_t2 = mean_estimated_Lambda_lr_p_t2 - true_Lambda_0_t2;  

        % Wald Method for 95% Coverage for theta_p
        % Confidence interval: EstimatedThetaP ± 1.96 * sqrt(EstimatedGammaP / n)
        ci_wald_lower_g_p = mean_estimated_theta_g_p - 1.96 * mean_sqrt_gamma_g_p /sqrt(n);
        ci_wald_upper_g_p = mean_estimated_theta_g_p + 1.96 * mean_sqrt_gamma_g_p/sqrt(n);

        % Z value for 95% confidence (z_alpha/2 for alpha=0.05 is approximately 1.96)
        z_alpha_2 = 1.96;

        % Calculate confidence intervals using the provided formula
        ci_wald_lower_Lambda_g_p_t1 = mean_estimated_Lambda_g_p_t1 * exp(-z_alpha_2 * mean_sqrt_Sigma_Lambda_2_g_p_t1 / (sqrt(n) * mean_estimated_Lambda_g_p_t1));
        ci_wald_upper_Lambda_g_p_t1 = mean_estimated_Lambda_g_p_t1 * exp(z_alpha_2 * mean_sqrt_Sigma_Lambda_2_g_p_t1 / (sqrt(n) * mean_estimated_Lambda_g_p_t1));

        ci_wald_lower_Lambda_g_p_t2 = mean_estimated_Lambda_g_p_t2 * exp(-z_alpha_2 * mean_sqrt_Sigma_Lambda_2_g_p_t2 / (sqrt(n) * mean_estimated_Lambda_g_p_t2));
        ci_wald_upper_Lambda_g_p_t2 = mean_estimated_Lambda_g_p_t2 * exp(z_alpha_2 *  mean_sqrt_Sigma_Lambda_2_g_p_t2 / (sqrt(n) * mean_estimated_Lambda_g_p_t2));

        
        ci_wald_lower_lr_p = mean_estimated_theta_lr_p - 1.96 * mean_sqrt_gamma_lr_p /sqrt(n);
        ci_wald_upper_lr_p = mean_estimated_theta_lr_p + 1.96 * mean_sqrt_gamma_lr_p /sqrt(n);


        % Calculate confidence intervals using the provided formula
        ci_wald_lower_Lambda_lr_p_t1 = mean_estimated_Lambda_lr_p_t1 * exp(-z_alpha_2 * mean_sqrt_Sigma_Lambda_2_lr_p_t1 / (sqrt(n) * mean_estimated_Lambda_lr_p_t1));
        ci_wald_upper_Lambda_lr_p_t1 = mean_estimated_Lambda_lr_p_t1 * exp(z_alpha_2 *  mean_sqrt_Sigma_Lambda_2_lr_p_t1 / (sqrt(n) * mean_estimated_Lambda_lr_p_t1));

        ci_wald_lower_Lambda_lr_p_t2 = mean_estimated_Lambda_lr_p_t2 * exp(-z_alpha_2 * mean_sqrt_Sigma_Lambda_2_lr_p_t2/ (sqrt(n) * mean_estimated_Lambda_lr_p_t2));
        ci_wald_upper_Lambda_lr_p_t2 = mean_estimated_Lambda_lr_p_t2 * exp(z_alpha_2 *  mean_sqrt_Sigma_Lambda_2_lr_p_t2/ (sqrt(n) * mean_estimated_Lambda_lr_p_t2));
        

        % Determine Wald coverage for theta_p
        coverage_wald_g_p = mean(current_results_p.EstimatedThetaG_P >= ci_wald_lower_g_p & current_results_p.EstimatedThetaG_P <= ci_wald_upper_g_p);
        
        coverage_wald_Lambda_g_p_t1 = mean(current_results_p.EstimatedLambdaG_P1 >= ci_wald_lower_Lambda_g_p_t1 & current_results_p.EstimatedLambdaG_P1 <= ci_wald_upper_Lambda_g_p_t1);
        coverage_wald_Lambda_g_p_t2 = mean(current_results_p.EstimatedLambdaG_P2 >= ci_wald_lower_Lambda_g_p_t2 & current_results_p.EstimatedLambdaG_P2 <= ci_wald_upper_Lambda_g_p_t2);
              
        coverage_wald_lr_p = mean(current_results_p.EstimatedThetaLR_P >= ci_wald_lower_lr_p & current_results_p.EstimatedThetaLR_P <= ci_wald_upper_lr_p, 'omitnan');

        coverage_wald_Lambda_lr_p_t1 = mean(current_results_p.EstimatedLambdaLR_P1 >= ci_wald_lower_Lambda_lr_p_t1 & current_results_p.EstimatedLambdaLR_P1 <= ci_wald_upper_Lambda_lr_p_t1);
        coverage_wald_Lambda_lr_p_t2 = mean(current_results_p.EstimatedLambdaLR_P2 >= ci_wald_lower_Lambda_lr_p_t2 & current_results_p.EstimatedLambdaLR_P2 <= ci_wald_upper_Lambda_lr_p_t2);
        

        % Percentile Method for 95% Coverage for theta_p
        % Compute percentiles
        ci_percentile_lower_g_p = prctile(current_results_p.EstimatedThetaG_P, 2.5);
        ci_percentile_upper_g_p = prctile(current_results_p.EstimatedThetaG_P, 97.5);

        ci_percentile_lower_Lambda_g_p_t1 = prctile(current_results_p.EstimatedLambdaG_P1, 2.5);
        ci_percentile_upper_Lambda_g_p_t1 = prctile(current_results_p.EstimatedLambdaG_P1, 97.5);

        ci_percentile_lower_Lambda_g_p_t2 = prctile(current_results_p.EstimatedLambdaG_P2, 2.5);
        ci_percentile_upper_Lambda_g_p_t2 = prctile(current_results_p.EstimatedLambdaG_P2, 97.5);


        ci_percentile_lower_lr_p = prctile(current_results_p.EstimatedThetaLR_P, 2.5);
        ci_percentile_upper_lr_p = prctile(current_results_p.EstimatedThetaLR_P, 97.5);


        ci_percentile_lower_Lambda_lr_p_t1 = prctile(current_results_p.EstimatedLambdaLR_P1, 2.5);
        ci_percentile_upper_Lambda_lr_p_t1 = prctile(current_results_p.EstimatedLambdaLR_P1, 97.5);

        ci_percentile_lower_Lambda_lr_p_t2 = prctile(current_results_p.EstimatedLambdaLR_P2, 2.5);
        ci_percentile_upper_Lambda_lr_p_t2 = prctile(current_results_p.EstimatedLambdaLR_P2, 97.5);




        % Determine Percentile coverage for theta_p
        coverage_percentile_g_p = mean(current_results_p.EstimatedThetaG_P >= ci_percentile_lower_g_p & current_results_p.EstimatedThetaG_P <= ci_percentile_upper_g_p);
        
        % Determine Percentile coverage for Lambda_p
        coverage_percentile_Lambda_g_p_t1 = mean(current_results_p.EstimatedLambdaG_P1 >= ci_percentile_lower_Lambda_g_p_t1 & current_results_p.EstimatedLambdaG_P1 <= ci_percentile_upper_Lambda_g_p_t1);
        coverage_percentile_Lambda_g_p_t2 = mean(current_results_p.EstimatedLambdaG_P2 >= ci_percentile_lower_Lambda_g_p_t2 & current_results_p.EstimatedLambdaG_P2 <= ci_percentile_upper_Lambda_g_p_t2);
       
       % Determine Percentile coverage for Lambda_p
        coverage_percentile_Lambda_lr_p_t1 = mean(current_results_p.EstimatedLambdaLR_P1 >= ci_percentile_lower_Lambda_lr_p_t1 & current_results_p.EstimatedLambdaLR_P1 <= ci_percentile_upper_Lambda_lr_p_t1);
        coverage_percentile_Lambda_lr_p_t2 = mean(current_results_p.EstimatedLambdaLR_P2 >= ci_percentile_lower_Lambda_lr_p_t2 & current_results_p.EstimatedLambdaLR_P2 <= ci_percentile_upper_Lambda_lr_p_t2);
        
        
        % Determine Percentile coverage for theta_p
        coverage_percentile_lr_p = mean(current_results_p.EstimatedThetaLR_P >= ci_percentile_lower_lr_p & current_results_p.EstimatedThetaLR_P <= ci_percentile_upper_lr_p, 'omitnan');

        
        % Store the results for theta_p
        temp_results_p = table( ...
            n, ...
            abs(bias_g_p), ...
            std_estimated_theta_g_p, ...
            mean_sqrt_gammaThm3_g_p, ...
            mean_sqrt_gamma_g_p, ...
            coverage_wald_g_p, ...
            coverage_percentile_g_p, ...
            abs(bias_Lambda_g_p_t1), ...
            std_estimated_Lambda_g_p_t1,...
            mean_sqrt_Sigma_Lambda_2_g_p_t1,...
            coverage_wald_Lambda_g_p_t1,...
            coverage_percentile_Lambda_g_p_t1,...
            abs(bias_Lambda_g_p_t2),...
            std_estimated_Lambda_g_p_t2, ...
            mean_sqrt_Sigma_Lambda_2_g_p_t2,...
            coverage_wald_Lambda_g_p_t2,...
            coverage_percentile_Lambda_g_p_t2,...
            abs(bias_lr_p), ...
            std_estimated_theta_lr_p,...
            mean_sqrt_gamma_lr_p, ...
            mean_sqrt_gammahuang_lr_p, ...
            coverage_wald_lr_p, ...
            coverage_percentile_lr_p, ...
            abs(bias_Lambda_lr_p_t1), ...
            std_estimated_Lambda_lr_p_t1,...
            mean_sqrt_Sigma_Lambda_2_lr_p_t1,...
            coverage_wald_Lambda_lr_p_t1,...
            coverage_percentile_Lambda_lr_p_t1,...
            abs(bias_Lambda_lr_p_t2),...
            std_estimated_Lambda_lr_p_t2, ...
            mean_sqrt_Sigma_Lambda_2_lr_p_t2,...
            coverage_wald_Lambda_lr_p_t2,...
            coverage_percentile_Lambda_lr_p_t2,...
            'VariableNames', {'SampleSize', 'AbsoluteBiasG_P', 'StandardDeviationG_P', 'MeanSqrtGammaTh3G_P', 'MeanSqrtGammaG_P', 'CoverageWaldG_P', 'CoveragePercentileG_P', ...
            'AbsoluteBiasLambdaG_P1', 'StandardDeviationLambdaG_P1', 'MeanSqrtSigmaLambdaG_P1', 'CoverageWaldLambdaG_P1', 'CoveragePercentileLambdaG_P1', 'AbsoluteBiasLambdaG_P2', 'StandardDeviationLambdaG_P2', 'MeanSqrtSigmaLambdaG_P2', 'CoverageWaldLambdaG_P2', 'CoveragePercentileLambdaG_P2', 'AbsoluteBiasLR_P', 'StandardDeviationLR_P', 'MeanSqrtGammaLR_P', ...
            'MeanSqrtGammaHuangLR_P', 'CoverageWaldLR_P', 'CoveragePercentileLR_P', 'AbsoluteBiasLambdaLR_P1', 'StandardDeviationLambdaLR_P1', 'MeanSqrtSigmaLambdaLR_P1', 'CoverageWaldLambdaLR_P1', 'CoveragePercentileLambdaLR_P1', 'AbsoluteBiasLambdaLR_P2', 'StandardDeviationLambdaLR_P2', 'MeanSqrtSigmaLambdaLR_P2', 'CoverageWaldLambdaLR_P2', 'CoveragePercentileLambdaLR_P2'});

        % Append to the extended_results_p table
        extended_results_p = [extended_results_p; temp_results_p];
end

% Reorder rows by Distribution and then by SampleSize
extended_results_p = sortrows(extended_results_p, 'SampleSize');

% Display the results
disp('Extended Results for theta_p:');
disp(extended_results_p);


% Save the extended results to CSV files
writetable(extended_results_p, 'extended_simulation_results_p.csv');
