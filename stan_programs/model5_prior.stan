// Model 5 Prior Predictive: Developer Comfort + Task Burden
//
// Latent structure:
//   comfort_j ~ Normal(0, sigma_c)          [developer-level, J=16]
//   task_burden_n ~ Normal(0, sigma_b)      [observation-level, N]
//
// Outcomes:
//   exposure ~ OrderedLogistic(lambda_e * comfort, cut_points_e)
//   resources ~ OrderedLogistic(lambda_r_c * comfort + lambda_r_b * burden, cut_points_r)
//   forecast ~ NegBinomial2(exp(alpha_f - comfort + burden), phi)
//   y ~ Lognormal(alpha_t + beta_t_c * comfort + beta_burden * burden + beta_trt * ai, sigma_t)

functions {
  // Betancourt's ordinal functions
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
  // DECIDED: rho_e = (1/5, ..., 1/5), tau_e = 0.1 for exposure (K=5)
  // DECIDED: rho_r = (1/3, 1/3, 1/3), tau_r = 0.2 for resources (K=3)
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
  // PARAMETERS FROM M4 (DECIDED - keep same priors)
  //==========================================================================

  real<lower=0> sigma_b = abs(normal_rng(0, 0.39));   // Task burden spread
  real<lower=0> sigma_t = abs(normal_rng(0, 0.25));   // Completion noise
  real<lower=0> kappa = abs(normal_rng(0, 0.10));     // Forecast overdispersion
  real phi = 1.0 / kappa;

  real alpha_f = normal_rng(log(17), 0.30);           // Log expected (forecast/5 - 1)
  real alpha_t = normal_rng(log(90), 0.40);           // Log median completion

  real beta_burden = normal_rng(1, 0.32);             // Burden -> completion
  real beta_trt = normal_rng(0, 0.7);                 // Treatment effect
  real lambda_r_b = normal_rng(1, 0.43);              // Burden -> resources

  //==========================================================================
  // NEW PARAMETERS FOR MODEL 5
  //==========================================================================

  // Developer comfort spread (identified by coefficient=-1 in forecast)
  // 99% prior mass below 2 (would be shocked by 5x+ multiplicative differences)
  real<lower=0> sigma_c = abs(normal_rng(0, 0.78));   // HalfNormal(0, 2/2.57)

  // Comfort -> exposure (positive: more comfort -> higher familiarity)
  real lambda_e = normal_rng(1, 0.43);

  // Comfort -> resources (negative: more comfort -> lower needs)
  real lambda_r_c = normal_rng(-1, 0.43);

  // Comfort -> log completion (expect negative: more comfort -> faster)
  // Centered at -1 to match forecast identification; SD=0.6 allows some overconfidence
  real beta_t_c = normal_rng(-1, 0.6);

  // NOTE: comfort coefficient fixed to -1 in forecast for identification (higher comfort -> lower forecast)

  //==========================================================================
  // CUTPOINTS (DECIDED)
  //==========================================================================

  // Exposure cutpoints (K=5)
  simplex[5] baseline_p_e = dirichlet_rng(alpha_dir_e);
  ordered[4] cut_points_e = derived_cut_points(baseline_p_e);

  // Resources cutpoints (K=3)
  simplex[3] baseline_p_r = dirichlet_rng(alpha_dir_r);
  ordered[2] cut_points_r = derived_cut_points(baseline_p_r);

  //==========================================================================
  // SIMULATE LATENT VARIABLES (non-centered)
  //==========================================================================

  vector[J] comfort;
  for (j in 1:J) {
    comfort[j] = sigma_c * normal_rng(0, 1);
  }

  vector[N] task_burden;
  for (n in 1:N) {
    task_burden[n] = sigma_b * normal_rng(0, 1);
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
    int j = dev_idx[n];

    // Exposure: comfort only
    real gamma_e = lambda_e * comfort[j];
    exposure_sim[n] = ordinal_shifted_logistic_rng(cut_points_e, gamma_e);

    // Resources: comfort + burden
    real gamma_r = lambda_r_c * comfort[j] + lambda_r_b * task_burden[n];
    resources_sim[n] = ordinal_shifted_logistic_rng(cut_points_r, gamma_r);

    // Forecast: -comfort + burden (comfort=-1, burden=+1 for identification)
    real mu_f = exp(alpha_f - comfort[j] + task_burden[n]);
    forecast_shifted_sim[n] = neg_binomial_2_rng(mu_f, phi);
    forecast_sim[n] = 5.0 * (forecast_shifted_sim[n] + 1);

    // Completion time: comfort + burden + treatment (coefficients free)
    real mu_t = alpha_t + beta_t_c * comfort[j] + beta_burden * task_burden[n]
                + beta_trt * ai_access[n];
    y_sim[n] = lognormal_rng(mu_t, sigma_t);
    log_y_sim[n] = log(y_sim[n]);
  }

  // Treatment effect as percentage lift
  real pct_lift = (exp(beta_trt) - 1) * 100;

  // Summary statistics for simulated outcomes
  real mean_exposure = mean(to_vector(exposure_sim));
  real mean_resources = mean(to_vector(resources_sim));
}
