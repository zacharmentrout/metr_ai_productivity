# Betancourt Bayesian Modeling Reference

Consolidated notes from Michael Betancourt's materials, organized by workflow steps.

---

## The Workflow

1. **Conceptual Analysis** - Define domain, quantities, questions
2. **Observational Model Development** - Build the generative model
3. **Prior Model Development** - Choose priors from domain expertise
4. **Prior Predictive Check** - Verify model produces plausible data
5. **Fit the Model** - Run HMC/MCMC
6. **Computational Diagnostics** - Check Rhat, ESS, divergences
7. **Posterior Retrodictive Check** - Compare predictions to data
8. **Model Critique and Iteration** - Refine based on misfits

---

## 1. Conceptual Analysis

**Goal**: Translate domain expertise into explicit model structure before writing any code.

Key questions:
- What is the estimand (quantity we want to learn)?
- What is the data generating process?
- What variables are observed vs latent?
- What are the causal relationships?

**Generative thinking**: Models are stories about how data arise. Think narratively:
- What causes what?
- What would we observe if we could re-run the experiment?
- What aspects of the process matter at our measurement resolution?

---

## 2. Observational Model Development

### Core Structure

The joint model separates into likelihood and prior:
```
π(y, θ) = π(y | θ) π(θ)
```

### Regression Models (Taylor Approximation)

For modeling how covariates influence outcomes:

```
y ~ Normal(μ, σ)
μ = α + β₁(x - x₀) + β₂(x - x₀)² + ...
```

**Key insight**: Taylor regression treats linear models as *local approximations*. Parameters have interpretations:
- α = f(x₀) — function value at baseline
- β₁ = df/dx(x₀) — first derivative at baseline
- β₂ = ½ d²f/dx²(x₀) — second derivative (scaled)

**Baseline choice (x₀)**: Pick a meaningful point in the covariate space where you want to anchor interpretation. Does NOT need to be the empirical mean.

### Hierarchical Models

When data has group structure (e.g., multiple observations per developer):

```stan
// Population level
mu ~ normal(0, scale);
tau ~ exponential(1);

// Group level (non-centered parameterization)
alpha_raw ~ std_normal();
alpha = mu + tau * alpha_raw;

// Observation level
y ~ normal(alpha[group_id] + X * beta, sigma);
```

**Partial pooling**: Groups with few observations shrink toward population mean. Groups with many observations retain their empirical estimates.

**Non-centered parameterization**: Critical for HMC efficiency. Use `alpha_raw ~ std_normal()` then transform, rather than `alpha ~ normal(mu, tau)` directly.

---

## 3. Prior Model Development

### Philosophy

> "A useful prior model does not have to incorporate all of our domain expertise but rather just enough domain expertise to ensure well-behaved inferences."

Priors compensate for what the likelihood cannot inform. Don't encode redundant information.

### Containment Prior Method

**Step 1**: Define extremity thresholds from domain expertise
- At what value does the parameter become "ridiculous"?
- Thresholds need only order-of-magnitude accuracy

**Step 2**: Choose tail probability (typically 1% beyond each threshold)

**Step 3**: Compute prior SD

For **Normal** priors (unconstrained parameters):
```
SD = threshold / 2.32    (for 1% tail probability)
```

For **Half-Normal** priors (positive parameters like σ):
```
SD = threshold / 2.57    (for 1% tail probability)
```

### Prior Recommendations by Parameter Type

**Intercept (α)**:
- Reason about plausible outcome values at baseline
- Normal(0, SD) where SD reflects outcome scale

**Regression slopes (β)**:
- Think about how outcome changes across covariate range
- Heuristic: SD ≈ (expected outcome variation) / (covariate range)

**Residual SD (σ)**:
- Half-Normal(0, SD) where SD = (max plausible residual) / 2.57

**Hierarchical scale (τ)**:
- Exponential(1) or Half-Normal work well
- Exponential(1) puts ~63% mass below 1

### Common Mistakes

1. **Naive product priors**: Don't multiply independent marginal constraints together — use hierarchical or correlated structures instead

2. **Counterfeiting domain expertise**: Don't design priors to fix pathological likelihood behavior

3. **Translation-invariant ("uniform") priors**: These concentrate mass at infinity, not at finite values

4. **Assuming marginal independence suffices**: When constraints share dependencies, use correlated priors

---

## 4. Prior Predictive Checks

**Purpose**: Verify the prior produces data consistent with domain expertise *before* seeing real data.

### Stan Template: Prior Predictive Check (Recommended)

Use `generated quantities` block only — no likelihood, no inference. Just sample from priors and push forward.

```stan
// prior_predictive.stan
data {
  int<lower=0> M;       // Number of covariates
  int<lower=0> N;       // Number of observations
  vector[M] x0;         // Covariate baselines
  matrix[N, M] X;       // Covariate design matrix
}

transformed data {
  matrix[N, M] deltaX;
  for (n in 1:N) {
    deltaX[n,] = X[n] - x0';
  }
}

generated quantities {
  // Sample from priors directly using _rng functions
  real alpha = normal_rng(alpha_mean, alpha_sd);
  vector[M] beta;
  for (m in 1:M) {
    beta[m] = normal_rng(beta_mean[m], beta_sd[m]);
  }
  real<lower=0> sigma = fabs(normal_rng(0, sigma_sd));

  // Push forward to observation scale
  vector[N] mu = alpha + deltaX * beta;

  // Optional: generate full fake observations
  vector[N] y_sim;
  for (n in 1:N) {
    y_sim[n] = normal_rng(mu[n], sigma);
  }
}
```

### Alternative: Using Model Block

If you want to use Stan's sampling for the priors (useful for complex priors):

```stan
// prior_predictive_model.stan
data {
  int<lower=0> N;
  vector[N] x;
  real x0;
}

transformed data {
  vector[N] deltaX = x - x0;
}

parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
}

model {
  // Prior model only — no likelihood!
  alpha ~ normal(alpha_mean, alpha_sd);
  beta ~ normal(beta_mean, beta_sd);
  sigma ~ normal(0, sigma_sd);  // half-normal via constraint
}

generated quantities {
  vector[N] mu = alpha + beta * deltaX;
  vector[N] y_sim;
  for (n in 1:N) {
    y_sim[n] = normal_rng(mu[n], sigma);
  }
}
```

### R Code for Visualizing Prior Predictive

```r
fit_prior <- stan(file = "prior_predictive.stan",
                  data = stan_data,
                  algorithm = "Fixed_param",  # For generated quantities only
                  seed = 1234)

samples <- extract(fit_prior)

# Marginal check: distribution of mu across all observations
util$plot_line_hist(as.vector(samples$mu),
                    bin_min, bin_max, bin_delta,
                    xlab = "Prior predictive mu")

# Conditional check: mu vs each covariate
par(mfrow = c(1, M))
for (m in 1:M) {
  # Plot quantile bands of mu against covariate m
  # Overlay baseline with vertical line
  abline(v = x0[m], lwd = 2, col = "grey", lty = 3)
}
```

### Procedure (Manual in R)

```r
# 1. Draw from priors
alpha_sim <- rnorm(N_sim, alpha_mean, alpha_sd)
beta_sim <- rnorm(N_sim, beta_mean, beta_sd)
sigma_sim <- abs(rnorm(N_sim, 0, sigma_sd))

# 2. Generate fake data for each prior draw
y_sim <- matrix(NA, N_obs, N_sim)
for (i in 1:N_sim) {
  mu <- alpha_sim[i] + beta_sim[i] * x
  y_sim[, i] <- rnorm(N_obs, mu, sigma_sim[i])
}

# 3. Check: Do simulated datasets look plausible?
# - Are values in reasonable range?
# - Do summary statistics match expectations?
```

### What to Check

- **Marginal distribution**: Histogram of simulated outcomes
- **Conditional behavior**: How does outcome vary with covariates?
- **Summary statistics**: Mean, SD, quantiles of simulated data

### Iteration ("Doog Loop")

1. Construct initial prior
2. Perform prior predictive checks
3. Compare to domain expertise
4. If inconsistent, update prior
5. Repeat until satisfied

---

## 5. Fitting the Model

### Stan Model Template

```stan
data {
  int<lower=1> N;
  vector[N] y;
  vector[N] x;
  real x0;  // baseline for Taylor expansion
}

parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
}

model {
  // Priors (from domain expertise calibration)
  alpha ~ normal(0, alpha_sd);
  beta ~ normal(0, beta_sd);
  sigma ~ normal(0, sigma_sd);  // half-normal via constraint

  // Likelihood
  y ~ normal(alpha + beta * (x - x0), sigma);
}

generated quantities {
  vector[N] y_rep;
  vector[N] log_lik;

  for (n in 1:N) {
    real mu_n = alpha + beta * (x[n] - x0);
    y_rep[n] = normal_rng(mu_n, sigma);
    log_lik[n] = normal_lpdf(y[n] | mu_n, sigma);
  }
}
```

### R Fitting Code

```r
stan_data <- list(
  N = nrow(dat),
  y = dat$y,
  x = dat$x,
  x0 = median(dat$x)  # or domain-meaningful baseline
)

fit <- stan(
  file = "model.stan",
  data = stan_data,
  seed = 1234,
  refresh = 0
)
```

---

## 6. Computational Diagnostics

**Run these BEFORE interpreting results. If diagnostics fail, stop and fix.**

### Key Diagnostics

```r
util$check_all_diagnostics(fit)
```

| Diagnostic | Threshold | Meaning |
|------------|-----------|---------|
| Rhat | < 1.01 | Chains have mixed |
| ESS/N | > 0.1 | Sufficient effective samples |
| Divergences | 0 | No numerical issues |
| Tree depth | Rarely saturated | Sampler not struggling |
| E-FMI | > 0.2 | Energy diagnostic ok |

### If Diagnostics Fail

- **Divergences**: Reparameterize (non-centered), tighten priors, or simplify model
- **Low ESS**: Run longer, reparameterize
- **High Rhat**: Chains haven't converged — run longer or fix identifiability

---

## 7. Posterior Retrodictive Checks

**Purpose**: Compare model predictions to observed data to identify systematic misfits.

### Two-Pronged Approach

**A. Marginal check**: Does the model capture the overall distribution of outcomes?

```r
# Compare histogram of observed vs posterior predictive
util$plot_hist_quantiles(samples, 'y_rep',
                         bin_min, bin_max, bin_width,
                         baseline_values = y_obs)
```

**B. Conditional check**: Does the model capture how outcomes vary with covariates?

```r
# Plot quantile bands along covariate grid
# Overlay observed data points
util$plot_conditional_mean_quantiles(samples, pred_names,
                                     covariate, baseline_values = y_obs)
```

### Interpreting Misfits

| Marginal Check | Conditional Check | Diagnosis |
|----------------|-------------------|-----------|
| Poor | Poor | Taylor approximation inadequate (need higher order) |
| Poor | Good | Probabilistic variation model inadequate |
| Good | Good | Model is adequate |

### Shape-Based Diagnostics

- Positive deviations at extremes, negative in middle → need quadratic term
- Heavy tails → need heavy-tailed likelihood (Student-t)
- Systematic bias in subgroups → missing covariate or interaction

---

## 8. Model Critique and Iteration

### Progressive Complexity Strategy

Start simple, add complexity only when justified:

1. **Model 1**: Simplest reasonable model
2. Run full workflow (priors → fit → diagnostics → retrodiction)
3. Identify misfits from retrodictive checks
4. **Model 2**: Add one feature to address misfit
5. Compare: Does added complexity improve fit?
6. Repeat until adequate

### What to Add

- **Curvature**: Add quadratic terms if conditional checks show curvature
- **Heterogeneity**: Add hierarchical structure if groups differ
- **Heavy tails**: Switch to Student-t if marginal check shows heavy tails
- **Interactions**: If effect varies by subgroup

### When to Stop

- Retrodictive checks show no systematic misfits
- Additional complexity doesn't improve fit
- Model answers the scientific question adequately

---

## Key Code Patterns

### Extracting Samples

```r
samples <- util$extract_expectand_vals(fit)
diagnostics <- util$extract_hmc_diagnostics(fit)
```

### Histogram Visualization

```r
util$plot_line_hist(data, bin_min, bin_max, bin_delta,
                    xlab = "Variable name")
```

### Quantile Bands

```r
# For prior/posterior predictive
util$plot_expectand_pushforward(samples, n_bins,
                                flim = c(lower, upper),
                                display_name = "Description")
```

### Diagnostic Checks

```r
util$check_all_hmc_diagnostics(diagnostics)
util$check_all_expectand_diagnostics(samples)
```

---

## Quick Reference: Prior Calibration

| Parameter Type | Distribution | SD Formula |
|----------------|--------------|------------|
| Unconstrained (α, β) | Normal(0, SD) | threshold / 2.32 |
| Positive (σ, τ) | Half-Normal(0, SD) | threshold / 2.57 |
| Positive (τ) | Exponential(rate) | rate = 1/expected_value |
| Probability | Beta(a, b) | Solve for 1% tails |
| Ordered | positive_ordered + Half-Normal | As above |

---

## Checklist for Each Model

- [ ] Conceptual analysis documented
- [ ] Priors calibrated with domain expertise (thresholds stated)
- [ ] Prior predictive check passed
- [ ] Model compiles and runs
- [ ] All MCMC diagnostics pass (Rhat, ESS, divergences)
- [ ] Marginal retrodictive check passed
- [ ] Conditional retrodictive check passed
- [ ] Results interpreted in context
