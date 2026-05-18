# Project 1 - Bayesian Weibull regression on the Veteran lung cancer data
# Questions 1 and 2


library(survival)   # contains the 'veteran' dataset
library(rjags)      # interface to JAGS
library(coda)       # MCMC diagnostics

set.seed(20260518)

data(cancer, package = "survival")

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
                        n.iter = n.iter,
                        thin   = n.thin)

# --- Convergence diagnostics --------------------------------------------------

# Numerical summary
summary(samples)

# 1. Trace and density plots
#+ fig.width=12, fig.height=10
plot(samples)

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
#   - or re-parameterise / centre covariates (karno could be centred to help mixing).
