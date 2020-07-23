##############################################################################
#                                                                            #
#           Anomaly Detection Using Bayesian Hierarchical Models             # 
#                            Simulation Study                                #
#                                                                            #
##############################################################################

# Simulation of count data, setting of control limits and analysis

# The models and the simulation code are based on an example in 
# McElreath [2020] (401-414) and examples in Gelman et al. [2014] (43-51).

# Refer to thesis for explanation of data and model.

# Set-up #####################################################################

rm(list = ls())

# Packages assumed to be installed:
# rstan, rethinking, extraDistr, actuar

library(rethinking)
library(extraDistr)
library(actuar)

# Version info
packageVersion("rstan")
packageVersion("rethinking")
packageVersion("extraDistr")
packageVersion("actuar")
sessionInfo()

# Avoid scientific notation in output
options("scipen" = 100, "digits" = 4)
# Avoid line breaks in output
options("width" = 120)

random_seed <- 200718

# Set seed for R session (Stan has separate seed that needs to be passed as 
# parameter in call of function ulam - see below)
set.seed(random_seed)

# Simulate initial data ######################################################

# Set number of observational units
units <- 1000

# First, simulate reference count values (used as a measure of exposure)

# Draw values from a Pareto distribution with suitable parameters
v <- rpareto2(units, min = 0, shape = 5, scale = 1000)

# Result
summary(v)
hist(v, breaks = 30, main = "Values Drawn from Pareto Distribution", 
     xlab = "value", ylab = "frequency", col = "black")
hist(v[v < 2000], breaks = 30, main = "Values Drawn from Pareto Distribution",
     xlab = "value", ylab = "frequency", col = "black")
hist(v[v < 1000], breaks = 30, main = "Values Drawn from Pareto Distribution",
     xlab = "value", ylab = "frequency", col = "black")
hist(v[v < 500], breaks = 30, main = "Values Drawn from Pareto Distribution",
     xlab = "value", ylab = "frequency", col = "black")
hist(v[v < 200], breaks = 10, main = "Values Drawn from Pareto Distribution",
     xlab = "value", ylab = "frequency", col = "black")

# Transform to discrete values for count data
n <- as.integer(ceiling(v))

# n must never be 0 (and will not, since v will never be exactly 0)
length(n)
length(n[n == 0])
n[n == 0] <- 1
length(n[n == 0])

# Sort values to facilitate visualization
n <- sort(n)

# Result
summary(n)
hist(n, breaks = 30, main = "Pareto Values as Integers (Ceiling)", 
     xlab = "value", ylab = "frequency", col = "black")

# Set "true" value of alpha, the parameter determining the distribution 
# of theta in the simulated data
alpha <- 0.05

# Set "true" value of theta for each unit by drawing from a half-normal 
# distribution with parameter alpha
true_theta <- rhnorm(units, alpha)

# Result
summary(true_theta)
hist(true_theta, breaks = 30, main = "True Theta", 
     xlab = "theta", ylab = "frequency", col = "black")

# Construct data frame with
# - unit:          index for unit
# - n:             reference count value
# - true_theta:    true value of theta
# - y:             count value of interest
# - theta_nopool:  no-pooling estimate of theta
d <- data.frame(unit = 1:units, n = n, true_theta = true_theta)
d$y <- rpois(units, lambda = d$n * d$true_theta)
d$theta_nopool <- d$y / d$n

# Result
str(d)
summary(d$n)
summary(d$y)
summary(d$true_theta)
summary(d$theta_nopool)
hist(d$theta_nopool, breaks = 30, main = "No-Pooling Estimate of Theta", 
     xlab = "theta", ylab = "frequency", col = "black")

# Stan model #################################################################

# Convert data to list of vectors
dat <- list(y = d$y, n = d$n, unit = d$unit)
str(dat)

# Model as described in thesis

# Four chains with 4000 iterations each, of which half are used for warm-up,
# giving 8000 samples for each of the parameters

m <- ulam(
  alist(
    y ~ dpois(lambda),
    lambda <- n * theta[unit],
    theta[unit] ~ dhalfnorm(0, alpha),
    alpha ~ dhalfnorm(0, 0.376)
  ), 
  data = dat, 
  chains = 4, 
  iter = 4000,
  seed = random_seed)

# If warnings, check chains with:
# traceplot(m)
# trankplot(m)
# dev.off()

# Generated Stan code for model
stancode(m)

# Check max.print option, change if needed
getOption("max.print")
# options(max.print = 10000)
# getOption("max.print")

# Summary and diagnostic statistics 
# (n_eff should be ~10000 for theta, Rhat should be 1)
precis(m, depth = 1)
precis(m, depth = 2)

# Note that alpha was set to a "true" value of 0.05, and that this true 
# value has been recovered precisely (standard deviation of ~0) by the model, 
# despite the prior dhalfnorm(0, 0.376), which has a mean of 0.3.

# Extract samples drawn from posterior distribution
post <- extract.samples(m)

str(post)
dim(post$theta)
nrow(post$theta)
ncol(post$theta)

# Mean of theta samples drawn from posterior distribution 
post_theta_means <- apply(post$theta, 2, mean)

# Result
str(post_theta_means)
summary(post_theta_means)
var(post_theta_means)
sd(post_theta_means)
hist(post_theta_means, breaks = 20, 
     main = "Mean of Theta Samples Drawn From Posterior", 
     xlab = "theta", ylab = "frequency", col = "black")

# Standard deviation of theta samples drawn from posterior distribution 
post_theta_sds <- apply(post$theta, 2, sd)

# Result
str(post_theta_sds)
summary(post_theta_sds)
var(post_theta_sds)
sd(post_theta_sds)
hist(post_theta_sds, breaks = 20, 
     main = "Standard Deviation of Theta Samples Drawn From Posterior", 
     xlab = "standard deviation of samples", ylab = "frequency", col = "black")

# Note that the theta values cannot be recovered precisely by the model.

# Relationship between means and standard deviations

plot(post_theta_means, post_theta_sds,
     main = "Relationship Between Mean and Standard Deviation", 
     xlab = "mean of theta samples", 
     ylab = "standard deviation of theta samples")

plot(log(post_theta_means), log(post_theta_sds),
     main = "Relationship Between Logs of Mean and Standard Deviation", 
     xlab = "log of mean of theta samples", 
     ylab = "log of standard deviation of theta samples")

# Note that higher standard deviations tend to be associated with higher means,
# although there is substantial variation.

# We will use the mean of the theta samples drawn from the posterior 
# distribution as the partial pooling estimate for theta.

# Adding to data frame 
# - theta_partpool:  partial-pooling estimate of theta
d$theta_partpool <- post_theta_means

# Now there are three values for theta in the data frame:
# the true value and the no-pooling and partial-pooling estimates

str(d)

# Compare

summary(d$true_theta)
summary(d$theta_nopool)
summary(d$theta_partpool)

sd(d$true_theta)
sd(d$theta_nopool)
sd(d$theta_partpool)

# The following histograms show the frequencies for the range from 0 to 0.2:

par(mfrow = c(3, 1))

hist(d$true_theta, breaks = seq(0, 1, 0.01), 
     main = "True Theta", 
     xlab = "theta", ylab = "frequency", 
     xlim = c(0, 0.2), ylim = c(0, 300),
     col = "black")

hist(d$theta_nopool, breaks = seq(0, 1, 0.01), 
     main = "No-Pooling Estimate of Theta", 
     xlab = "theta", ylab = "frequency", 
     xlim = c(0, 0.2), ylim = c(0, 300),
     col = "black")

hist(d$theta_partpool, breaks = seq(0, 1, 0.01), 
     main = "Partial-Pooling Estimate of Theta", 
     xlab = "theta", ylab = "frequency", 
     xlim = c(0, 0.2), ylim = c(0, 300),
     col = "black")

par(mfrow = c(1, 1))

# Analyze absolute errors ####################################################

# Since in this case the true value of theta is known, it is possible to 
# determine and compare the absolute error, that is, by how much the estimates 
# miss the true value. 

nopool_error <- abs(d$theta_nopool - d$true_theta)
partpool_error <- abs(d$theta_partpool - d$true_theta)

summary(nopool_error)
summary(partpool_error)

plot(1:units, nopool_error, 
     main = "Absolute Error of Theta Estimates",
     xlab="unit", ylab = "absolute error",
     col = rgb(1, 0, 0, 0.5), pch = 16)
points(1:units, partpool_error, 
       col = rgb(0, 1, 0, 0.5), pch = 16)
legend("topright", inset = 0.05, 
       legend = c("no pooling", "partial pooling"), 
       col = c(rgb(1, 0, 0, 0.5), rgb(0, 1, 0, 0.5)), 
       pch = 16, box.lty = 0, cex = 0.8)

# Means for various ranges of units (units sorted by reference count value)

# Units 1-100
mean(nopool_error[1:100])
mean(partpool_error[1:100])

# Units 1-200
mean(nopool_error[1:200])
mean(partpool_error[1:200])

# Units 1-500
mean(nopool_error[1:500])
mean(partpool_error[1:500])

# All units (1-1000)
mean(nopool_error)
mean(partpool_error)

# Units 501-1000
mean(nopool_error[501:1000])
mean(partpool_error[501:1000])

# Units 801-1000
mean(nopool_error[801:1000])
mean(partpool_error[801:1000])

# Show how the mean of the absolute error for the first N units
# changes for increasing values of N

mean_errors_up_to_n <- function(n, errors) {
  mean(errors[1:n])
}

mean(nopool_error[1:51])
mean(nopool_error[1:1000])
apply(as.matrix(c(51:53, 998:1000)), 1, 
      mean_errors_up_to_n, errors = nopool_error)

mean(partpool_error[1:51])
mean(partpool_error[1:1000])
apply(as.matrix(c(51:53, 998:1000)), 1, 
      mean_errors_up_to_n, errors = partpool_error)

m_np_errs <- apply(as.matrix(1:1000), 1, 
                   mean_errors_up_to_n, errors = nopool_error)
m_pp_errs <- apply(as.matrix(1:1000), 1, 
                   mean_errors_up_to_n, errors = partpool_error)

m_np_errs[1000]
mean(nopool_error)

m_pp_errs[1000]
mean(partpool_error)

plot(m_np_errs, ylim = c(0, 0.1),
     main = "Mean Absolute Error of Theta Estimates up to N Units",
     xlab="N", ylab = "mean absolute error",
     col = "firebrick2")
points(m_pp_errs, col = "green3")
legend("topright", inset = 0.05, 
       legend = c("no pooling", "partial pooling"), 
       col = c("firebrick2", "green3"), 
       pch = 1, box.lty = 0, cex = 0.8)

# Analyze estimated mean of count values of interest #########################

y_hat_true_theta <- d$true_theta * d$n
y_hat_nopool <- d$theta_nopool * d$n
y_hat_partpool <- d$theta_partpool * d$n

summary(y_hat_true_theta)
summary(y_hat_nopool)
summary(y_hat_partpool)

sd(y_hat_true_theta)
sd(y_hat_nopool)
sd(y_hat_partpool)

par(mfrow = c(3, 1))

hist(y_hat_true_theta[y_hat_true_theta <= 50], breaks = seq(0, 50, 1), 
     main = "Expected Mean of Counts, Given True Theta", 
     xlab = "expected mean", ylab = "frequency", 
     xlim = c(0, 50), ylim = c(0, 350),
     col = "black")

hist(y_hat_nopool[y_hat_nopool <= 50], breaks = seq(0, 50, 1), 
     main = "Expected Mean of Counts, Given No-Pooling Estimate of Theta", 
     xlab = "expected mean", ylab = "frequency", 
     xlim = c(0, 50), ylim = c(0, 350),
     col = "black")

hist(y_hat_partpool[y_hat_partpool <= 50], breaks = seq(0, 50, 1), 
     main = "Expected Mean of Counts, Given Partial-Pooling Estimate of Theta", 
     xlab = "expected mean", ylab = "frequency", 
     xlim = c(0, 50), ylim = c(0, 350),
     col = "black")

par(mfrow = c(1, 1))

# In order to get a clearer picture for the values near 0, only expected means
# up 50 are displayed.

# We now examine the estimates of the expected means for the units sorted by 
# the reference count value.

# All units (1-1000)
plot(1:units, y_hat_nopool, 
     main = "Estimated Means",
     xlab="unit", ylab = "y_hat",
     col = rgb(1, 0, 0, 0.5), pch = 16)
points(1:units, y_hat_partpool, 
       col = rgb(0, 1, 0, 0.5), pch = 16)
legend("topleft", inset = 0.05, 
       legend = c("no pooling", "partial pooling"), 
       col = c(rgb(1, 0, 0, 0.5), rgb(0, 1, 0, 0.5)), 
       pch = 16, box.lty = 0, cex = 0.8)

# The following subgroups are examined:

# - units 1-200
# - units 401-600
# - units 801-1000

# For each subgroup, a second plot also indicates the expected means 
# given the true values of theta.

# Units 1-200 

par(mfrow = c(2, 1))

plot(1:200, y_hat_nopool[1:200], 
     main = "Estimated Means",
     xlab="unit", ylab = "y_hat",
     col = rgb(1, 0, 0, 0.7), pch = 16)
points(1:200, y_hat_partpool[1:200], 
       col = rgb(0, 1, 0, 0.7), pch = 16)
legend("topleft", inset = 0.05, 
       legend = c("no pooling", "partial pooling"), 
       col = c(rgb(1, 0, 0, 0.5), rgb(0, 1, 0, 0.5)), 
       pch = 16, box.lty = 0, cex = 0.8)

mean(y_hat_nopool == d$y)

plot(1:200, y_hat_nopool[1:200], 
     main = "Estimated and True Means",
     xlab="unit", ylab = "y_hat",
     col = rgb(1, 0, 0, 0.7), pch = 16)
points(1:200, y_hat_partpool[1:200], 
       col = rgb(0, 1, 0, 0.7), pch = 16)
points(1:200, y_hat_true_theta[1:200], pch = 1)
legend("topleft", inset = 0.05, 
       legend = c("no pooling", "partial pooling", "true"), 
       col = c(rgb(1, 0, 0, 0.5), rgb(0, 1, 0, 0.5), "black"), 
       pch = c(16, 16, 1), box.lty = 0, cex = 0.8)

par(mfrow = c(1, 1))

# Units 401-600

par(mfrow = c(2, 1))

plot(401:600, y_hat_nopool[401:600], 
     main = "Estimated Means",
     xlab="unit", ylab = "y_hat",
     col = rgb(1, 0, 0, 0.7), pch = 16)
points(401:600, y_hat_partpool[401:600], 
       col = rgb(0, 1, 0, 0.7), pch = 16)
legend("topleft", inset = 0.05, 
       legend = c("no pooling", "partial pooling"), 
       col = c(rgb(1, 0, 0, 0.5), rgb(0, 1, 0, 0.5)), 
       pch = 16, box.lty = 0, cex = 0.8)

plot(401:600, y_hat_nopool[401:600], 
     main = "Estimated and True Means",
     xlab="unit", ylab = "y_hat",
     col = rgb(1, 0, 0, 0.7), pch = 16)
points(401:600, y_hat_partpool[401:600], 
       col = rgb(0, 1, 0, 0.7), pch = 16)
points(401:600, y_hat_true_theta[401:600], pch = 1)
legend("topleft", inset = 0.05, 
       legend = c("no pooling", "partial pooling", "true"), 
       col = c(rgb(1, 0, 0, 0.5), rgb(0, 1, 0, 0.5), "black"), 
       pch = c(16, 16, 1), box.lty = 0, cex = 0.8)

par(mfrow = c(1, 1))

# Units 801-1000

par(mfrow = c(2, 1))

plot(801:1000, y_hat_nopool[801:1000], 
     main = "Estimated Means",
     xlab="unit", ylab = "y_hat",
     col = rgb(1, 0, 0, 0.7), pch = 16)
points(801:1000, y_hat_partpool[801:1000], 
       col = rgb(0, 1, 0, 0.7), pch = 16)
legend("topleft", inset = 0.05, 
       legend = c("no pooling", "partial pooling"), 
       col = c(rgb(1, 0, 0, 0.5), rgb(0, 1, 0, 0.5)), 
       pch = 16, box.lty = 0, cex = 0.8)

plot(801:1000, y_hat_nopool[801:1000], 
     main = "Estimated and True Means",
     xlab="unit", ylab = "y_hat",
     col = rgb(1, 0, 0, 0.7), pch = 16)
points(801:1000, y_hat_partpool[801:1000], 
       col = rgb(0, 1, 0, 0.7), pch = 16)
points(801:1000, y_hat_true_theta[801:1000], pch = 1)
legend("topleft", inset = 0.05, 
       legend = c("no pooling", "partial pooling", "true"), 
       col = c(rgb(1, 0, 0, 0.5), rgb(0, 1, 0, 0.5), "black"), 
       pch = c(16, 16, 1), box.lty = 0, cex = 0.8)

par(mfrow = c(1, 1))

# The overlap of the points is exaggerated due to the compressed scale.
# If we only plot expected means up to 20 the plot changes as follows:

par(mfrow = c(2, 1))

plot(801:1000, y_hat_nopool[801:1000], 
     main = "Estimated Means up to 20",
     xlab="unit", ylab = "y_hat", ylim = c(0, 20),
     col = rgb(1, 0, 0, 0.7), pch = 16)
points(801:1000, y_hat_partpool[801:1000], 
       col = rgb(0, 1, 0, 0.7), pch = 16)
legend("topleft",  
       legend = c("no pooling", "partial pooling"), 
       col = c(rgb(1, 0, 0, 0.5), rgb(0, 1, 0, 0.5)), 
       pch = 16, box.lty = 1, bg = "white", cex = 0.8)

plot(801:1000, y_hat_nopool[801:1000], 
     main = "Estimated and True Means up to 20",
     xlab="unit", ylab = "y_hat", ylim = c(0, 20),
     col = rgb(1, 0, 0, 0.7), pch = 16)
points(801:1000, y_hat_partpool[801:1000], 
       col = rgb(0, 1, 0, 0.7), pch = 16)
points(801:1000, y_hat_true_theta[801:1000], pch = 1)
legend("topleft", 
       legend = c("no pooling", "partial pooling", "true"), 
       col = c(rgb(1, 0, 0, 0.5), rgb(0, 1, 0, 0.5), "black"), 
       pch = c(16, 16, 1), box.lty = 1, bg = "white", cex = 0.8)

par(mfrow = c(1, 1))

# Generate new data for test of control limits ###############################

# Finally, we will determine control limits for new reference count values
# and examine the performance of these control limits with simulated new count
# values of interest.

# Assuming new data from subsequent period of about half the length of the 
# original period, and, in proportion, about half of n as reference counts

# Adding to data frame
# - n_new:  new reference count values
# - y_new:  new count values of interest

d$n_new <- as.integer(ceiling(d$n / 2)) 
d$y_new <- rpois(units, lambda = d$n_new * d$true_theta)

# Result
str(d)
summary(d$n_new)
summary(d$y_new)

# Set upper control limits (UCL) #############################################

# Formula for UCL: estimated mean + 3 * estimated standard deviation
# (In this case, assuming a Poisson distribution, both the estimated mean and 
# the estimated variance are theta * n_new)

# Round the result to the nearest integer
# Add 0.5, so that count values are either above or below the UCL (as in 
# Wheeler & Chambers [1992])

# Adding to data frame
# - ucl_true_theta: UCL based on n_new and true value of theta
# - ucl_nopool:     UCL based on n_new and no-pooling estimate of theta
# - ucl_partpool:   UCL based on n_new and partial-pooling estimate of theta

d$ucl_true_theta <- round(
  d$true_theta * d$n_new + 3 * sqrt(d$true_theta * d$n_new)
) + 0.5
d$ucl_nopool <- round(
  d$theta_nopool * d$n_new + 3 * sqrt(d$theta_nopool * d$n_new)
) + 0.5
d$ucl_partpool <- round(
  d$theta_partpool * d$n_new + 3 * sqrt(d$theta_partpool * d$n_new)
) + 0.5

# Result
str(d)
summary(d$ucl_true_theta)
summary(d$ucl_nopool)
summary(d$ucl_partpool)

# check calculation for minimum value of true_theta * n_new 
min(d$true_theta * d$n_new)
sqrt(min(d$true_theta * d$n_new))
3 * sqrt(min(d$true_theta * d$n_new))
min(d$true_theta * d$n_new) + 
  3 * sqrt(min(d$true_theta * d$n_new))
round(min(d$true_theta * d$n_new) + 
        3 * sqrt(min(d$true_theta * d$n_new)))
round(min(d$true_theta * d$n_new) + 
        3 * sqrt(min(d$true_theta * d$n_new))) + 0.5

# check calculation for minimum value of theta_partpool * n_new 
min(d$theta_partpool * d$n_new)
sqrt(min(d$theta_partpool * d$n_new))
3 * sqrt(min(d$theta_partpool * d$n_new))
min(d$theta_partpool * d$n_new) + 
  3 * sqrt(min(d$theta_partpool * d$n_new))
round(min(d$theta_partpool * d$n_new) + 
        3 * sqrt(min(d$theta_partpool * d$n_new)))
round(min(d$theta_partpool * d$n_new) + 
        3 * sqrt(min(d$theta_partpool * d$n_new))) + 0.5

# Analyze effectiveness of control limits with new data ######################

# All counts were generated according to the specified probability model and
# are therefore to be considered within the normal range (negatives, meaning
# no anomalies).

# Even the UCLs based on the true value of theta are likely to lead to some
# false positives when 1000 counts are checked

# The UCLs based on estimates of theta are expected to lead to more false 
# positives, with the UCL based on the partial-pooling estimate of theta
# being likely to perform better especially for low counts

sum(d$y_new - d$ucl_true_theta > 0)
sum(d$y_new - d$ucl_nopool > 0)
sum(d$y_new - d$ucl_partpool > 0)

# This result confirms what has been expected.

par(mfrow = c(3, 1))

plot(d$y_new - d$ucl_true_theta,
     main = "Data Relative to UCL Based on True Theta",
     xlab="unit", ylab = "y_new - ucl_true_theta")
abline(h = 0, lty = 1, lwd = 2)

plot(d$y_new - d$ucl_nopool, col = "firebrick2",
     main = "Data Relative to UCL Based on No-Pooling Estimate",
     xlab="unit", ylab = "y_new - ucl_nopool")
abline(h = 0, lty = 1, lwd = 2)

plot(d$y_new - d$ucl_partpool, col = "green3",
     main = "Data Relative to UCL Based on Partial-Pooling Estimate",
     xlab="unit", ylab = "y_new - ucl_partpool")
abline(h = 0, lty = 1, lwd = 2)

par(mfrow = c(1, 1))

d[d$y_new - d$ucl_true_theta > 0,]
d[d$y_new - d$ucl_nopool > 0,]
d[d$y_new - d$ucl_partpool > 0,]

nrow(d[d$y_new - d$ucl_nopool > 0,])
nrow(d[d$y_new - d$ucl_nopool > 0 & d$y == 0,]) 
nrow(d[d$y_new - d$ucl_nopool > 0 & d$y > 0,])

nrow(d[d$y_new - d$ucl_nopool > 0 & d$y == 0,]) / 
  nrow(d[d$y_new - d$ucl_nopool > 0,])

nrow(d[d$y_new - d$ucl_nopool > 0 & d$y > 0,]) /  
  nrow(d[d$y_new - d$ucl_nopool > 0,])

d[d$y_new - d$ucl_nopool > 0 & d$y > 0 & ! d$y_new - d$ucl_partpool > 0,]
d[d$y_new - d$ucl_partpool > 0 & ! d$y_new - d$ucl_nopool > 0,]

nrow(d[d$y_new - d$ucl_nopool > 0 & d$y > 0 & ! d$y_new - d$ucl_partpool > 0,])
nrow(d[d$y_new - d$ucl_partpool > 0 & ! d$y_new - d$ucl_nopool > 0,])

# Analyze behavior with increasing number of anomalies #######################

par(mfrow = c(2, 1))

# Based on y_new_expected = ceiling(n_new * true_theta)

# Effectively we use a minimum count of one, since the true parameter theta 
# in this simulation is never exactly 0, and n_new is ceiling(n / 2) and 
# therefore also never 0 (since n is never 0).

# Factor y_new_expected will be multiplied by, from 1 to 8, in steps of 0.01
# (The higher the factor, to more likely the result will exceed the UCL)
factor <- seq.int(1, 8, 0.01)

# Create vectors
true_theta_anomaly <- integer(length = length(factor))
nopool_anomaly <- integer(length = length(factor))
partpool_anomaly <- integer(length = length(factor))
str(true_theta_anomaly)

# A) y_new_expected >= 1 (all)

d_test <- d
d_test$y_new_expected <- as.integer(ceiling(d_test$n_new * d_test$true_theta))
d_test <- d_test[d_test$y_new_expected >= 1,]

nrow(d_test)
str(d_test)

summary(d_test$y_new_expected)
summary(d_test$y_new)

summary(d_test$ucl_true_theta)
summary(d_test$ucl_nopool)
summary(d_test$ucl_partpool)

for (i in 1:length(factor)) {
  f <- factor[i]
  true_theta_anomaly[i] <- sum(f * d_test$y_new_expected 
                               - d_test$ucl_true_theta > 0)
  nopool_anomaly[i] <- sum(f * d_test$y_new_expected 
                           - d_test$ucl_nopool > 0)
  partpool_anomaly[i] <- sum(f * d_test$y_new_expected 
                             - d_test$ucl_partpool > 0)
}

plot(factor, true_theta_anomaly, 
     main = "Counts Exceeding UCL With Increasing Multiples of Counts (1)",
     xlab="factor", ylab = "counts exceeding UCL",
     type = "n")
lines(factor, true_theta_anomaly, col = "black", lwd = 3)
lines(factor, nopool_anomaly, col = "firebrick2", lwd = 3)
lines(factor, partpool_anomaly, col = "green3", lwd = 3)
legend("bottomright", inset = 0.05, 
       legend = c("no pooling", "partial pooling", "true"), 
       col = c("firebrick2", "green3", "black"), 
       lwd = 3, box.lty = 0, cex = 0.8)

sum(1 * d_test$y_new_expected - d_test$ucl_nopool > 0)
sum(1 * d_test$y_new_expected - d_test$ucl_nopool > 0 
    & d_test$ucl_nopool == 0.5)
sum(1 * d_test$y_new_expected - d_test$ucl_nopool > 0 
    & d_test$ucl_nopool == 0.5) / 
  sum(1 * d_test$y_new_expected - d_test$ucl_nopool > 0)

sum(1 * d_test$y_new_expected - d_test$ucl_true_theta > 0)
sum(1 * d_test$y_new_expected - d_test$ucl_true_theta > 0 
    & d_test$ucl_true_theta == 0.5)
sum(1 * d_test$y_new_expected - d_test$ucl_true_theta > 0 
    & d_test$ucl_true_theta == 0.5) / 
  sum(1 * d_test$y_new_expected - d_test$ucl_true_theta > 0)

sum(1 * d_test$y_new_expected - d_test$ucl_partpool > 0)

sum(2 * d_test$y_new_expected - d_test$ucl_nopool > 0)
sum(2 * d_test$y_new_expected - d_test$ucl_partpool > 0)

sum(3 * d_test$y_new_expected - d_test$ucl_nopool > 0)
sum(3 * d_test$y_new_expected - d_test$ucl_partpool > 0)

# B) y_new_expected >= 2

d_test <- d
d_test$y_new_expected <- as.integer(ceiling(d_test$n_new * d_test$true_theta))
d_test <- d_test[d_test$y_new_expected >= 2,]

nrow(d_test)
str(d_test)

summary(d_test$y_new_expected)
summary(d_test$y_new)

summary(d_test$ucl_true_theta)
summary(d_test$ucl_nopool)
summary(d_test$ucl_partpool)

for (i in 1:length(factor)) {
  f <- factor[i]
  true_theta_anomaly[i] <- sum(f * d_test$y_new_expected 
                               - d_test$ucl_true_theta > 0)
  nopool_anomaly[i] <- sum(f * d_test$y_new_expected 
                           - d_test$ucl_nopool > 0)
  partpool_anomaly[i] <- sum(f * d_test$y_new_expected 
                             - d_test$ucl_partpool > 0)
}

plot(factor, true_theta_anomaly, 
     main = "Counts Exceeding UCL With Increasing Multiples of Counts (2)",
     xlab="factor", ylab = "counts exceeding UCL",
     type = "n")
lines(factor, true_theta_anomaly, col = "black", lwd = 3)
lines(factor, nopool_anomaly, col = "firebrick2", lwd = 3)
lines(factor, partpool_anomaly, col = "green3", lwd = 3)
legend("bottomright", inset = 0.05, 
       legend = c("no pooling", "partial pooling", "true"), 
       col = c("firebrick2", "green3", "black"), 
       lwd = 3, box.lty = 0, cex = 0.8)

sum(1 * d_test$y_new_expected - d_test$ucl_nopool > 0)
sum(1 * d_test$y_new_expected - d_test$ucl_nopool > 0 
    & d_test$ucl_nopool == 0.5)
sum(1 * d_test$y_new_expected - d_test$ucl_nopool > 0 
    & d_test$ucl_nopool == 0.5) / 
  sum(1 * d_test$y_new_expected - d_test$ucl_nopool > 0)

sum(1 * d_test$y_new_expected - d_test$ucl_true_theta > 0)

sum(1 * d_test$y_new_expected - d_test$ucl_partpool > 0)

sum(2 * d_test$y_new_expected - d_test$ucl_nopool > 0)
sum(2 * d_test$y_new_expected - d_test$ucl_partpool > 0)

sum(3 * d_test$y_new_expected - d_test$ucl_nopool > 0)
sum(3 * d_test$y_new_expected - d_test$ucl_partpool > 0)

par(mfrow = c(1, 1))
