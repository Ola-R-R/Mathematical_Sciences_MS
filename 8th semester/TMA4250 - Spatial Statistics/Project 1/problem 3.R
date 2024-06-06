library(ggplot2)
library(geoR)

# Parameters for the GRF
sigma2 <- 2 # Marginal variance
a <- 3 # Spatial scale

# ------Problem 3a------

# Create 30x30 grid
x <- seq(1, 30)
y <- seq(1, 30)
expand.grid(x,y)

# Simulation grid
coords = cbind(rep(x, 30), rep(y, each = 30))

# Covariance matrix
Sigma = sigma2*exp(-as.matrix(dist(coords))/a)
L = t(chol(Sigma))
z = L%*%rnorm(dim(Sigma)[1])

# Plotting
df = data.frame(s1 = coords[,1],
                s2 = coords[,2],
                X = z)  
fig = ggplot(data = df, aes(s1, s2)) + 
  geom_raster(aes(fill = X)) +
  scale_fill_viridis_c(limits = c(-4,4), name = "Response") + 
  coord_fixed() +
  ggtitle("Simulation") +
  labs(x = "x", y = "y")
fig

# ------Problem 3b------

# Convert to geodata
geodata <- as.geodata(data.frame(coords = df[, c("s1", "s2")], data = df$X))

# Compute empirical semivariogram
empirical_variogram <- variog(geodata)

# Theoretical semivariogram for exponential model
theoretical_variogram <- function(h, sigma2, a) {
  sigma2 * (1 - exp(-h / a))
}

# Generate distances
distances <- seq(0, max(empirical_variogram$u), by = 0.1)

# Find values of the theoretical semivariogram
theoretical_values <- theoretical_variogram(distances, sigma2, a)

# Create data frame for theoretical semivariogram
theoretical_semi_df <- data.frame(h = distances, gamma = theoretical_values)

# Create data frame for empirical semivariogram
# $u - a vector with distances
# $v - a vector with estimated variogram values at distances given in 'u'
empirical_semi_df <- data.frame(h = empirical_variogram$u, gamma = empirical_variogram$v)

# Combine data frames
semi_df <- rbind(
  data.frame(h = empirical_semi_df$h, gamma = empirical_semi_df$gamma, Type = "Empirical"),
  data.frame(h = theoretical_semi_df$h, gamma = theoretical_semi_df$gamma, Type = "Theoretical")
)

# Plotting
ggplot(semi_df, aes(x = h, y = gamma, color = Type)) +
  geom_line(data = subset(semi_df, Type == "Empirical")) +
  geom_line(data = subset(semi_df, Type == "Theoretical")) +
  scale_color_manual(values = c("Empirical" = "black", "Theoretical" = "red")) +             
  labs(x = "Distance", y = "Semivariance", color = "Type") +
  ggtitle("Empirical and Theoretical Semivariogram")

# ------Problem 3c------

# Run 3a and 3b three times

# ------Problem 3d------

set.seed(425)

# Function to perform steps for different number of observations
find_semivariogram <- function(n, df, sigma2, a) {
  
  # Sample n random locations
  sampled_indices <- sample(1:nrow(df), n, replace = FALSE)
  sampled_locations <- df[sampled_indices, ]
  sampled_geodata <- as.geodata(data.frame(coords = sampled_locations[, c("s1", "s2")], data = sampled_locations$X))
  
  # Compute empirical variogram with samples
  empirical_variogram_sampled <- variog(sampled_geodata)
  
  # Estimate a and sigma2 using full ML
  full_fit <- likfit(geodata, ini.cov.pars=c(1, 1), cov.model="exponential")
  
  # Estimate parameters using MLE
  fit_sampled <- likfit(sampled_geodata, ini.cov.pars=c(1, 1), cov.model="exponential")
  
  # Estimate parameters
  sigma2_full <- full_fit$sigmasq
  a_full <- full_fit$phi
  
  # Extract estimated parameters
  estimated_sigma2 <- fit_sampled$sigmasq
  estimated_a <- fit_sampled$phi
  
  # Generate distances for theoretical variogram
  max_dist <- max(empirical_variogram_sampled$u)
  distances <- seq(0, max_dist, by = 0.1)
  
  # Calculate theoretical variogram values
  theoretical_values <- theoretical_variogram(distances, sigma2, a)
  

  # Data frames for plotting
  df_theoretical <- data.frame(Distance = distances, Semivariance = theoretical_variogram(distances, sigma2, a), Type = "Theoretical")
  # df_full_empirical <- data.frame(Distance = empirical_variogram$u, Semivariance = empirical_variogram$v, Type = "Full Empirical")
  df_sampled_empirical <- data.frame(Distance = empirical_variogram_sampled$u, Semivariance = empirical_variogram_sampled$v, Type = "Empirical")
  df_full_fit <- data.frame(Distance = distances, Semivariance = theoretical_variogram(distances, full_fit$sigmasq, full_fit$phi), Type = "Full estimate")
  df_sampled_fit <- data.frame(Distance = distances, Semivariance = theoretical_variogram(distances, fit_sampled$sigmasq, fit_sampled$phi), Type = "Observed estimate")
  
  
  # Plotting
  ggplot() +
    geom_line(data = df_theoretical, aes(x = Distance, y = Semivariance, color = Type)) +
    geom_line(data = df_sampled_empirical, aes(x = Distance, y = Semivariance, color = Type)) +
    geom_line(data = df_full_fit, aes(x = Distance, y = Semivariance, color = Type)) +
    geom_line(data = df_sampled_fit, aes(x = Distance, y = Semivariance, color = Type)) +
    scale_color_manual(values = c("Theoretical" = "red", "Empirical" = "green", "Full estimate" = "blue", "Observed estimate" = "orange")) +
    labs(x = "Distance", y = "Semivariance", color = "Type") +
    ggtitle(paste("Semivariogram Analysis for", n, "Observations")) +
    theme_minimal()
}

# Theoretical variogram function
theoretical_variogram <- function(h, sigma2, a) {
  sigma2 * (1 - exp(-h / a))
}

# Plotting
plot_36 <- find_semivariogram(36, df, sigma2, a)
print(plot_36)

# ------Problem 3e------

# Perform analysis for 9 and 100 observations
plot_9 <- find_semivariogram(9, df, sigma2, a)
plot_100 <- find_semivariogram(100, df, sigma2, a)

# Display plots
print(plot_9)
print(plot_100)
