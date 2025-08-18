# Load a dataset from a CSV file
data <- read.csv("simulation_results_p.csv")

# View the first few rows of the dataset
head(data)

library(ggplot2)
library(reshape2)
library(gridExtra)

# Filter data for SampleSize 30 and 50
data_filtered <- subset(data, SampleSize %in% c(30, 50))

# Replace numeric distribution labels with names
data_filtered$Distribution <- factor(data_filtered$Distribution, 
                                     levels = c(1, 2), 
                                     labels = c("Weibull", "Gamma"))

# Reshape data for better comparison with shorter Theta names
data_long <- melt(data_filtered, id.vars = c("SampleSize", "Distribution"), 
                  measure.vars = c("EstimatedThetaG_P", "EstimatedThetaLR_P"), 
                  variable.name = "Theta", value.name = "Theta_Value")

# Replace long Theta names with shorter ones
data_long$Theta <- factor(data_long$Theta, 
                          levels = c("EstimatedThetaG_P", "EstimatedThetaLR_P"), 
                          labels = c("ThetaG", "ThetaLR"))

# Boxplot comparison with green shades, no grid background
boxplot_comp <- ggplot(data_long, aes(x = Theta, y = Theta_Value, fill = Theta)) +
  geom_boxplot(alpha = 0.8, color = "black") +
  labs(title = "", x = "Theta Type", y = "Theta Values") +
  scale_fill_manual(values = c("ThetaG" = "orange", "ThetaLR" = "firebrick")) +
  facet_grid(SampleSize ~ Distribution) +
  theme(
    panel.background = element_blank(),  # Remove the panel background
    plot.background = element_blank(),   # Remove the plot background
    panel.grid.major = element_blank(),  # Remove major grid lines
    panel.grid.minor = element_blank(),  # Remove minor grid lines
    axis.line = element_line(color = "black")  # Keep the axis lines visible
  )

# Histogram comparison with fewer bins and no grid background
hist_comp <- ggplot(data_long, aes(x = Theta_Value, fill = Theta)) +
  geom_histogram(bins = 12, color = "black", alpha = 0.6, position = "identity") +
  labs(title = "", x = "Theta Values", y = "Frequency") +
  scale_fill_manual(values = c("ThetaG" = "orange", "ThetaLR" = "firebrick")) +
  facet_grid(SampleSize ~ Distribution) +
  theme(
    panel.background = element_blank(),  # Remove the panel background
    plot.background = element_blank(),   # Remove the plot background
    panel.grid.major = element_blank(),  # Remove major grid lines
    panel.grid.minor = element_blank(),  # Remove minor grid lines
    axis.line = element_line(color = "black")  # Keep the axis lines visible
  )

hist_comp
boxplot_comp

# Save the plots
png("Theta_Comparison_Weibull_Gamma.png", width = 14, height = 8, units = "in", res = 300)
grid.arrange(boxplot_comp, hist_comp, ncol = 2)
dev.off()

# Load necessary libraries
library(ggplot2)
library(reshape2)
library(dplyr)

# Filter data for SampleSize 30 and 50
data_filtered <- subset(data, SampleSize %in% c(30, 50))

# Rename distributions for better readability
data_filtered$Distribution <- factor(data_filtered$Distribution, levels = c(1, 2), labels = c("Weibull", "Gamma"))

# Reshape data for comparison
data_long <- melt(data_filtered, id.vars = c("SampleSize", "Distribution"), 
                  measure.vars = c("EstimatedGammaHuangLR_P", "EstimatedGammaLR_P"), 
                  variable.name = "A_Type", value.name = "A_Value")

# Define concise labels for Gamma_Type
data_long$A_Type <- factor(data_long$A_Type, 
                               levels = c("EstimatedGammaHuangLR_P", "EstimatedGammaLR_P"), 
                               labels = c("A_LR (Huang)", "A_LR (Reg)"))

# Define colors for each Gamma Type
colors <- c("A_LR (Huang)" = "#228B22", "A_LR (Reg)" = "#7CFC00")

# Boxplot comparison with concise labels and color applied to the boxes
boxplot_comp <- ggplot(data_long, aes(x = A_Type, y = A_Value, fill = A_Type)) +
  geom_boxplot(alpha = 0.8, color = "black") +  # Color the boxes using 'fill'
  labs(title = "", x = "A_Type", y = "A_Values") +
  scale_fill_manual(values = colors) +  # Apply custom colors for boxplot
  facet_grid(SampleSize ~ Distribution) +
  theme(
    panel.background = element_blank(),  # Remove the panel background
    plot.background = element_blank(),   # Remove the plot background
    panel.grid.major = element_blank(),  # Remove major grid lines
    panel.grid.minor = element_blank(),  # Remove minor grid lines
    axis.line = element_line(color = "black")  # Keep the axis lines visible
  )

# Save the boxplot
png("A_LR_Comparison_Boxplot_Reg_Huang.png", width = 10, height = 6, units = "in", res = 300)
print(boxplot_comp)
dev.off()

