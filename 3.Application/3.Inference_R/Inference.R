# Function to calculate p-value and confidence interval
calculate_p_value_and_ci <- function(estimate, se, sample_size, confidence_level = 0.95) {
  # Degrees of freedom
  df <- sample_size - 1
  
  # Calculate the t-statistic
  t_statistic <- estimate / se
  
  # Calculate the p-value (two-tailed test)
  p_value <- 2 * pt(-abs(t_statistic), df)
  
  # Calculate the critical t-value for the confidence interval
  alpha <- 1 - confidence_level
  critical_t <- qt(1 - alpha / 2, df)
  
  # Calculate the margin of error
  margin_of_error <- critical_t * se
  
  # Calculate the confidence interval
  ci_lower <- estimate - margin_of_error
  ci_upper <- estimate + margin_of_error
  
  # Return the results
  list(
    p_value = p_value,
    confidence_interval = c(ci_lower, ci_upper)
  )
}

# Example usage
estimate <-  0.1161   # Example estimate
se <- sqrt(19.6678)        # Example standard error
sample_size <- 50  # Example sample size

results <- calculate_p_value_and_ci(estimate, se, sample_size)
print(results)


# Data
labels <- c("Chemo", "Sex", "Dukes")
theta_L <- c(0.36, -0.14, 0.25)
theta_G <- c(0.33, -0.15, 0.28)
ci_L <- matrix(c(-2.53, 3.26, -3.12, 2.83, -2.00, 2.51), ncol = 2, byrow = TRUE)
ci_G <- matrix(c(-2.83, 3.50, -3.44, 3.12, -3.67, 4.23), ncol = 2, byrow = TRUE)

# Data
labels <- c("Chemo", "Sex", "Dukes")
theta_L <- c(0.36, -0.14, 0.25)
theta_G <- c(0.33, -0.15, 0.28)
ci_L <- matrix(c(-2.53, 3.26, -3.12, 2.83, -2.00, 2.51), ncol = 2, byrow = TRUE)
ci_G <- matrix(c(-2.83, 3.50, -3.44, 3.12, -3.67, 4.23), ncol = 2, byrow = TRUE)

# Create data frame
df <- data.frame(
  labels = rep(labels, 2),
  estimate = c(theta_L, theta_G),
  lower = c(ci_L[,1], ci_G[,1]),
  upper = c(ci_L[,2], ci_G[,2]),
  method = rep(c("Log-rank", "Gehan"), each = length(labels))
)

# Plot
library(ggplot2)

ggplot(df, aes(x = estimate, y = labels, color = method)) +
  geom_point(size = 3) +
  geom_errorbarh(aes(xmin = lower, xmax = upper), height = 0.2) +
  labs(x = "Estimate (\u03B8)", title = "Forest Plot of Regression Estimates") +
  theme_minimal() +
  theme(legend.title = element_blank())

# Variance Data
A_L <- c(2.00, 2.11, 1.21)
A_G <- c(2.40, 4.13, 3.74)

# Create data frame for variance comparison
df_variance <- data.frame(
  labels = labels,
  Log_rank = A_L,
  Gehan = A_G
)

# Plot
ggplot(df_variance, aes(x = labels)) +
  geom_bar(aes(y = Log_rank, fill = "Log-rank"), stat = "identity", position = "dodge") +
  geom_bar(aes(y = Gehan, fill = "Gehan"), stat = "identity", position = "dodge") +
  labs(x = "Variable", y = "Variance", title = "Comparison of Variances (A_L vs. A_G)") +
  scale_fill_manual(values = c("Log-rank" = "blue", "Gehan" = "red")) +
  theme_minimal() +
  theme(legend.title = element_blank())

# Time Points and Hazard Data
time <- c(50, 100)
Lambda_L <- c(0.19, 0.25)
Lambda_G <- c(0.19, 0.25)
B_L <- c(0.28, 0.32) # Variance
B_G <- c(0.36, 0.42)

# Create data frame for hazard estimation
df_hazard <- data.frame(
  time = rep(time, 2),
  Lambda = c(Lambda_L, Lambda_G),
  variance = c(B_L, B_G),
  method = rep(c("Log-rank", "Gehan"), each = length(time))
)

# Plot
ggplot(df_hazard, aes(x = time, y = Lambda, color = method)) +
  geom_line() +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = Lambda - sqrt(variance), ymax = Lambda + sqrt(variance)), width = 0.1) +
  labs(x = "Time", y = "Cumulative Baseline Hazard (\u03BB)", title = "Cumulative Baseline Hazard Estimates") +
  theme_minimal() +
  theme(legend.title = element_blank())

