# --- Q4: Hazard ratios --------------------------------------------------------

# Convert MCMC samples to a matrix
samples.mat <- as.matrix(samples)
head(samples.mat)

unique(vet$celltype)
table(vet$celltype)
class(vet$trt)

# Compute posterior samples of hazard ratios
# ref: trt -> standard / celltype -> squamous
# HR = exp(- beta * k)
HR <- data.frame(
  HR_trt_test = exp(-samples.mat[, "beta_trt"] * samples.mat[, "k"]),
  HR_karno = exp(-samples.mat[, "beta_karno"] * samples.mat[, "k"]),
  HR_smallcell = exp(-samples.mat[, "beta_smallcell"] * samples.mat[, "k"]),
  HR_adeno = exp(-samples.mat[, "beta_adeno"] * samples.mat[, "k"]),
  HR_large = exp(-samples.mat[, "beta_large"] * samples.mat[, "k"])
)

# Posterior summaries
HR_summary <- t(apply(HR, 2, function(x) {
  c(
    mean = mean(x),
    sd = sd(x),
    median = median(x),
    q2.5 = unname(quantile(x, 0.025)),
    q97.5 = unname(quantile(x, 0.975))
  )
}))

round(HR_summary, 3)
