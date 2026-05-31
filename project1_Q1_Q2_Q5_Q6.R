## Project 1 - Bayesian Weibull regression on the Veteran lung cancer data

library(survival)   # contains the 'veteran' dataset
library(rjags)      # interface to JAGS
library(coda)       # MCMC diagnostics
library(ggplot2)    # GGplot visualization
library(tidyr)
library(readr)

set.seed(20260518)

data(cancer, package = "survival")

## Questions 1 and 2

# Use only patients with observed event (status == 1), as instructed
vet <- veteran[veteran$status == 1, ]

# trt: 1 = standard, 2 = test chemotherapy
# We recode as 0/1 so that beta_trt is the contrast (test vs standard)
trt01 <- vet$trt - 1

# celltype has 4 levels: squamous, smallcell, adenocarcinoma, large
# We use 'squamous' as the reference category and create 3 dummy variables
vet$celltype <- relevel(factor(vet$celltype), ref = "squamous")
X_cell <- model.matrix(~ celltype, data = vet)[, -1]
# columns: celltypesmallcell, celltypeadeno, celltypelarge

# JAGS data list 
jags_data <- list(
  N         = nrow(vet),
  time      = vet$time,
  trt       = trt01,
  karno     = vet$karno - mean(vet$karno),
  smallcell = X_cell[, "celltypesmallcell"],
  adeno     = X_cell[, "celltypeadeno"],
  large     = X_cell[, "celltypelarge"]
)

# --- Model (BUGS code) --------------------------------------------------------
# Likelihood: T_i ~ Weibull(k, lambda_i)
#   - k       : shape parameter (>0); k>1 -> increasing hazard, k<1 -> decreasing
#   - lambda_i: scale-related parameter linked to covariates via log(lambda_i) = -eta_i
#
# Linear predictor:
#   eta_i = b0 + b_trt*trt + b_karno*karno + b_sc*smallcell + b_ad*adeno + b_lg*large
#
# Priors (uninformative / vague):
#   - All regression coefficients: N(0, 1e4)  (precision 1e-4 in JAGS notation)
#   - Shape k: Gamma(0.01, 0.01), a standard vague prior on a positive parameter

model_string <- "
model {
  for (i in 1:N) {
    time[i] ~ dweib(k, lambda[i])
    log(lambda[i]) <- -eta[i]
    eta[i] <- beta0
            + beta_trt       * trt[i]
            + beta_karno     * karno[i]
            + beta_smallcell * smallcell[i]
            + beta_adeno     * adeno[i]
            + beta_large     * large[i]
  }

  # Vague priors on regression coefficients
  beta0          ~ dnorm(0, 0.0001)
  beta_trt       ~ dnorm(0, 0.0001)
  beta_karno     ~ dnorm(0, 0.0001)
  beta_smallcell ~ dnorm(0, 0.0001)
  beta_adeno     ~ dnorm(0, 0.0001)
  beta_large     ~ dnorm(0, 0.0001)

  # Vague prior on the Weibull shape (k > 0)
  k ~ dgamma(0.01, 0.01)
}
"

writeLines(model_string, "weibull_model.txt")
file.show("weibull_model.txt")

# --- MCMC settings ------------------------------------------------------------
# Three independent chains with overdispersed starting values, so that the
# Gelman-Rubin diagnostic is meaningful.
inits <- list(
  list(beta0 = -5,  beta_trt = -1, beta_karno = -0.1,
       beta_smallcell =  1, beta_adeno =  1, beta_large =  1, k = 0.5),
  list(beta0 =  0,  beta_trt =  0, beta_karno =  0,
       beta_smallcell =  0, beta_adeno =  0, beta_large =  0, k = 1.0),
  list(beta0 =  5,  beta_trt =  1, beta_karno =  0.1,
       beta_smallcell = -1, beta_adeno = -1, beta_large = -1, k = 2.0)
)

n.chains  <- 3
n.adapt   <- 2000     # JAGS adaptation phase
n.burn    <- 5000     # burn-in to discard
n.iter    <- 50000    # iterations kept (per chain)
n.thin    <- 10        # increase if autocorrelation is high

params <- c("beta0", "beta_trt", "beta_karno",
            "beta_smallcell", "beta_adeno", "beta_large", "k")

# --- Run JAGS -----------------------------------------------------------------
jmod <- jags.model("weibull_model.txt",
                   data    = jags_data,
                   inits   = inits,
                   n.chains = n.chains,
                   n.adapt  = n.adapt)

# Burn-in
update(jmod, n.iter = n.burn)

# Sampling
samples <- coda.samples(jmod,
                        variable.names = params,
                        n.iter = n.iter, #per chain
                        thin   = n.thin)

# --- Convergence diagnostics --------------------------------------------------

# Numerical summary
summary(samples)

# 1. Trace and density plots
#fig.width=12
#fig.height=10
#plot(samples)

# 2. Gelman-Rubin (potential scale reduction factor)
#    Rule of thumb: point estimate and upper CI < 1.1 (ideally < 1.05) -> converged
gelman.diag(samples, multivariate = FALSE)
gelman.plot(samples)

# 3. Autocorrelation
autocorr.plot(samples)
autocorr.diag(samples)   # numerical autocorrelations at lags 0, 1, 5, 10, 50

# 4. Effective sample size
effectiveSize(samples)

# 5. Geweke diagnostic (z-scores; |z| < 2 indicates no evidence of non-stationarity)
geweke.diag(samples)

# 6. Heidelberger-Welch stationarity / halfwidth tests (optional, complementary)
heidel.diag(samples)

# If high autocorrelation / low ESS:
#   - increase n.iter and/or n.thin (e.g. thin = 5 or 10),
#   - or re-parameter / centre covariates (karno could be centred to help mixing).

## --------------------------------------------------------------------------------------
## QUESTIONS 5 and 6
## --------------------------------------------------------------------------------------

## Question 5
samples.mat <- as.matrix(samples)

## Extract parameters from the matrix
b0   <- samples.mat[, "beta0"]
b_trt <- samples.mat[, "beta_trt"]
b_sc <- samples.mat[, "beta_smallcell"]
b_ad <- samples.mat[, "beta_adeno"]
b_lg <- samples.mat[, "beta_large"]
k    <- samples.mat[, "k"]

## Calculate eta for each profile
eta_squamous <- b0
eta_small    <- b0 + b_sc
eta_adeno    <- b0 + b_ad
eta_large    <- b0 + b_lg

## Compute posterior samples for median survival times
median_squamous <- (log(2) * exp(eta_squamous))^(1 / k)
median_small    <- (log(2) * exp(eta_small))^(1 / k)
median_adeno    <- (log(2) * exp(eta_adeno))^(1 / k)
median_large    <- (log(2) * exp(eta_large))^(1 / k)

## Combine into a data frame
medians_df <- data.frame(
  Squamous  = median_squamous,
  SmallCell = median_small,
  Adeno     = median_adeno,
  Large     = median_large
)

## Posterior Summaries Table (Point and Interval Estimates)
median_summary <- t(apply(medians_df, 2, function(x) {
  c(mean = mean(x), sd = sd(x), median = median(x),
    q2.5 = unname(quantile(x, 0.025)), q97.5 = unname(quantile(x, 0.975)))
}))

print(round(median_summary, 3))

## Visualization via Caterpillar Plot
plot_data <- as.data.frame(median_summary)
plot_data$CellType <- rownames(plot_data)

ggplot(plot_data, aes(x = CellType, y = median)) +
  geom_point(size = 4, color = "darkblue") +
  geom_errorbar(aes(ymin = q2.5, ymax = q97.5), width = 0.2, size = 1, color = "red") +
  coord_flip() +
  labs(title = "Posterior Median Survival Times (with 95% Credible Intervals)",
       subtitle = "Profile: Treatment 1, Average Karnofsky Score",
       x = "Tumor Cell Type", y = "Median Survival (Days)") + theme_minimal()

## Question 6
t_val <- 60

## Profile A: Trt 1, Avg Karno, Squamous
lambda_A <- exp(-b0)
SA_60   <- exp(-lambda_A * (t_val^k))

## Profile B: Trt 2, Avg Karno, Squamous
lambda_B <- exp(-(b0 + b_trt))
SB_60   <- exp(-lambda_B * (t_val^k))

# Combine and compute summaries
S60_df <- data.frame(Profile_A = SA_60, Profile_B = SB_60)
S60_summary <- t(apply(S60_df, 2, function(x) {
  c(mean = mean(x), sd = sd(x), median = median(x),
    q2.5 = unname(quantile(x, 0.025)), q97.5 = unname(quantile(x, 0.975)))
}))

print(round(S60_summary, 3))

## Posterior Density Plots
S60_long <- gather(S60_df, key = "Profile", value = "Probability")

ggplot(S60_long, aes(x = Probability, fill = Profile)) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("Profile_A" = "red", "Profile_B" = "green"),
                    labels = c("Profile A", "Profile B")) +
  labs(title = "Posterior Density of Survival Probability at t = 60 Days",
       x = "Survival Probability S(60)", y = "Density", fill = "Patient Profile") + theme_minimal()


## --------------------------------------------------------------------------------------
## QUESTIONS 7 and 8
## --------------------------------------------------------------------------------------

model_string2 <- "
model {
  for (i in 1:N) {
    time[i] ~ dweib(k, lambda[i])
    log(lambda[i]) <- -eta[i]
    eta[i] <- beta0
            + beta_trt       * trt[i]
            + beta_karno     * karno[i]
            + beta_smallcell * smallcell[i]
            + beta_adeno     * adeno[i]
            + beta_large     * large[i]
  }

  beta0          ~ dnorm(0, 0.0001)
  beta_trt       ~ dnorm(0, 0.0001)
  beta_karno     ~ dnorm(0, 0.0001)
  beta_smallcell ~ dnorm(0, 0.0001)
  beta_adeno     ~ dnorm(0, 0.0001)
  beta_large     ~ dnorm(0, 0.0001)
  k              ~ dgamma(0.01, 0.01)

  lambda_A <- exp(-beta0)
  lambda_B <- exp(-(beta0 + beta_trt))
  SA_60 <- exp(-lambda_A * pow(60, k))
  SB_60 <- exp(-lambda_B * pow(60, k))
  
  #Q7 derived variable
  prob_A_better_B <- step(SA_60 - SB_60)

  #Q8 derived variables
  median_squamous <- pow(log(2) * exp(beta0), 1/k)
  median_small    <- pow(log(2) * exp(beta0 + beta_smallcell), 1/k)
  median_adeno    <- pow(log(2) * exp(beta0 + beta_adeno), 1/k)
  median_large    <- pow(log(2) * exp(beta0 + beta_large), 1/k)
  prob_med_sq_100 <- step(median_squamous - 100)
  prob_med_sc_100 <- step(median_small - 100)
  prob_med_ad_100 <- step(median_adeno - 100)
  prob_med_lg_100 <- step(median_large - 100)
}
"

writeLines(model_string2, "weibull_model2.txt")
file.show("weibull_model2.txt")

params2 <- c("beta0", "beta_trt", "beta_karno",
            "beta_smallcell", "beta_adeno", "beta_large", "k",
            "prob_A_better_B", 
            "prob_med_sq_100", "prob_med_sc_100", "prob_med_ad_100", "prob_med_lg_100")

jmod2 <- jags.model("weibull_model2.txt",
                   data    = jags_data,
                   inits   = inits,
                   n.chains = n.chains,
                   n.adapt  = n.adapt)

update(jmod2, n.iter = n.burn)

samples <- coda.samples(jmod2,
                        variable.names = params2,
                        n.iter = n.iter,
                        thin   = n.thin)

summary_stats <- summary(samples)$statistics
summary_stats

# Q7 result:
cat("Question 7: P(SA(60) > SB(60) | data) =", 
    round(summary_stats["prob_A_better_B", "Mean"], 4))

# Q8 results:
prob_squamous_100 <- round(summary_stats["prob_med_sq_100", "Mean"], 4)
prob_small_100    <- round(summary_stats["prob_med_sc_100", "Mean"], 4)
prob_adeno_100    <- round(summary_stats["prob_med_ad_100", "Mean"], 4)
prob_large_100    <- round(summary_stats["prob_med_lg_100", "Mean"], 4)

prob_100_df <- data.frame(
  Cell_Type = c("Squamous", "Small Cell", "Adeno", "Large"),
  Prob_Exceeds_100 = c(prob_squamous_100, prob_small_100, prob_adeno_100, prob_large_100)
)
prob_100_df
