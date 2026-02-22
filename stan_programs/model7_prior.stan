// Model 7 Prior Predictive: Hierarchical Factor Model
//
// Latent structure:
//   mu_j ~ Normal(0, sigma_mu)          [developer mean]
//   delta_n ~ Normal(0, sigma_delta)    [task deviation from developer mean]
//   eta_n = mu_j + delta_n              [total latent variable]
//
// Outcomes:
//   forecast ~ NegBinomial2(exp(alpha_f + eta), phi)
//   exposure ~ OrderedLogistic(-lambda_e * delta, cuts_e)
//   resources ~ OrderedLogistic(lambda_r * delta, cuts_r)
//   y ~ Lognormal(alpha_t + beta_eta * eta + beta_trt * ai + beta_eta_trt * eta * ai, sigma_t)

functions {
  int ordinal_shifted_logistic_rng(vector c, real gamma) {
    int K = num_elements(c) + 1;
    vector[K - 1] Pi = inv_logit(c - gamma);
    vector[K] p = append_row(Pi, [1]') - append_row([0]', Pi);
    return categorical_rng(p);
  }

  vector derived_cut_points(vector p) {
    int K = num_elements(p);
    vector[K - 1] c;
    real cum_sum = 0;
    for (k in 1:(K - 1)) {
      cum_sum += p[k];
      c[k] = logit(cum_sum);
    }
    return c;
  }
}

data {
  int<lower=0> N;                             // Number of observations
  int<lower=0> J;                             // Number of developers
  array[N] int<lower=1, upper=J> dev_idx;     // Developer index for each obs
  array[N] int<lower=0, upper=1> ai_access;

  // Dirichlet hyperparameters
  vector<lower=0>[5] rho_e;
  real<lower=0> tau_e;
  vector<lower=0>[3] rho_r;
  real<lower=0> tau_r;
}

transformed data {
  vector[5] alpha_dir_e = rho_e / tau_e + rep_vector(1, 5);
  vector[3] alpha_dir_r = rho_r / tau_r + rep_vector(1, 3);
}

generated quantities {
  //==========================================================================
  // DISPERSION AND LOCATION PARAMETERS
  //==========================================================================

  real<lower=0> sigma_mu = abs(normal_rng(0, 0.78));      // Developer mean spread
  real<lower=0> sigma_delta = abs(normal_rng(0, 0.39));   // Task deviation spread
  real<lower=0> sigma_t = abs(normal_rng(0, 0.25));       // Completion noise
  real<lower=0> kappa = abs(normal_rng(0, 0.10));         // Forecast overdispersion
  real phi = 1.0 / kappa;

  real alpha_f = normal_rng(log(17), 0.30);               // Log expected forecast baseline
  real alpha_t = normal_rng(log(90), 0.40);               // Log median completion baseline

  //==========================================================================
  // COEFFICIENTS
  //==========================================================================

  real beta_eta = normal_rng(1, 0.3);                     // eta -> completion
  real beta_trt = normal_rng(0, 0.7);                     // Treatment effect (at eta = 0)
  real beta_eta_trt = normal_rng(0, 0.15);                // eta x treatment interaction

  real lambda_e = normal_rng(1, 0.43);                    // delta -> exposure
  real lambda_r = normal_rng(1, 0.43);                    // delta -> resources

  //==========================================================================
  // CUTPOINTS
  //==========================================================================

  simplex[5] baseline_p_e = dirichlet_rng(alpha_dir_e);
  ordered[4] cut_points_e = derived_cut_points(baseline_p_e);

  simplex[3] baseline_p_r = dirichlet_rng(alpha_dir_r);
  ordered[2] cut_points_r = derived_cut_points(baseline_p_r);

  //==========================================================================
  // SIMULATE LATENT VARIABLES (non-centered)
  //==========================================================================

  vector[J] mu;
  for (j in 1:J) {
    mu[j] = sigma_mu * normal_rng(0, 1);
  }

  vector[N] delta;
  vector[N] eta;
  for (n in 1:N) {
    delta[n] = sigma_delta * normal_rng(0, 1);
    eta[n] = mu[dev_idx[n]] + delta[n];
  }

  //==========================================================================
  // SIMULATE OUTCOMES
  //==========================================================================

  array[N] int exposure_sim;
  array[N] int resources_sim;
  array[N] int forecast_shifted_sim;
  vector[N] forecast_sim;
  vector[N] y_sim;
  vector[N] log_y_sim;

  for (n in 1:N) {
    // Exposure: -lambda_e * delta
    real gamma_e = -lambda_e * delta[n];
    exposure_sim[n] = ordinal_shifted_logistic_rng(cut_points_e, gamma_e);

    // Resources: lambda_r * delta
    real gamma_r = lambda_r * delta[n];
    resources_sim[n] = ordinal_shifted_logistic_rng(cut_points_r, gamma_r);

    // Forecast: eta coefficient = 1 for identification
    real mu_f = exp(alpha_f + eta[n]);
    forecast_shifted_sim[n] = neg_binomial_2_rng(mu_f, phi);
    forecast_sim[n] = 5.0 * (forecast_shifted_sim[n] + 1);

    // Completion time: eta + heterogeneous treatment
    real mu_t = alpha_t + beta_eta * eta[n] + beta_trt * ai_access[n] + beta_eta_trt * eta[n] * ai_access[n];
    y_sim[n] = lognormal_rng(mu_t, sigma_t);
    log_y_sim[n] = log(y_sim[n]);
  }

  // Treatment effect at eta = 0 (percentage lift)
  real pct_lift = (exp(beta_trt) - 1) * 100;

  // Treatment effect at different eta values (percentage lift)
  real pct_lift_low_eta = (exp(beta_trt + beta_eta_trt * (-1)) - 1) * 100;
  real pct_lift_high_eta = (exp(beta_trt + beta_eta_trt * (1)) - 1) * 100;

  // Summary statistics
  real mean_exposure = mean(to_vector(exposure_sim));
  real mean_resources = mean(to_vector(resources_sim));
  real mean_eta = mean(eta);
  real sd_eta = sd(eta);
  real mean_delta = mean(delta);
  real sd_delta = sd(delta);
}
