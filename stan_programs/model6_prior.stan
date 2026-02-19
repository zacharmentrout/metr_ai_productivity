// Model 6 Prior Predictive: Heterogeneous Treatment Effect (Gap Interaction)
//
// Extends Model 5 by allowing the treatment effect to vary with gap.
//
// Latent structure:
//   comfort_j ~ Normal(0, sigma_c)          [developer-level, J]
//   task_burden_n ~ Normal(0, sigma_b)      [observation-level, N]
//   gap_n = task_burden_n - comfort_j       [key derived quantity]
//
// Outcomes:
//   exposure ~ OrderedLogistic(lambda_e * comfort, cut_points_e)
//   resources ~ OrderedLogistic(lambda_r * gap, cut_points_r)
//   forecast ~ NegBinomial2(exp(alpha_f + gap), phi)
//   y ~ Lognormal(alpha_t + beta_gap * gap + beta_trt * ai + beta_gap_trt * gap * ai, sigma_t)
//
// Treatment effect = beta_trt + beta_gap_trt * gap
//   beta_trt: effect at gap = 0 (average task for average developer)
//   beta_gap_trt: how effect changes per unit gap
//     > 0: AI helps less for harder tasks
//     < 0: AI helps more for harder tasks

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

  real<lower=0> sigma_c = abs(normal_rng(0, 0.78));   // Developer comfort spread
  real<lower=0> sigma_b = abs(normal_rng(0, 0.39));   // Task burden spread
  real<lower=0> sigma_t = abs(normal_rng(0, 0.25));   // Completion noise
  real<lower=0> kappa = abs(normal_rng(0, 0.10));     // Forecast overdispersion
  real phi = 1.0 / kappa;

  real alpha_f = normal_rng(log(17), 0.30);           // Log expected (forecast/5 - 1)
  real alpha_t = normal_rng(log(90), 0.40);           // Log median completion

  //==========================================================================
  // COEFFICIENTS
  //==========================================================================

  real beta_gap = normal_rng(1, 0.3);                 // Gap -> completion
  real beta_trt = normal_rng(0, 0.7);                 // Treatment effect (at gap = 0)
  real beta_gap_trt = normal_rng(0, 0.15);            // Gap x treatment interaction

  real lambda_e = normal_rng(1, 0.43);                // Comfort -> exposure
  real lambda_r = normal_rng(1, 0.43);                // Gap -> resources

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
  vector[N] gap_sim;  // Store gap for analysis

  for (n in 1:N) {
    int j = dev_idx[n];
    real gap = task_burden[n] - comfort[j];
    gap_sim[n] = gap;

    // Exposure: comfort only
    real gamma_e = lambda_e * comfort[j];
    exposure_sim[n] = ordinal_shifted_logistic_rng(cut_points_e, gamma_e);

    // Resources: gap
    real gamma_r = lambda_r * gap;
    resources_sim[n] = ordinal_shifted_logistic_rng(cut_points_r, gamma_r);

    // Forecast: gap coefficient = 1 for identification
    real mu_f = exp(alpha_f + gap);
    forecast_shifted_sim[n] = neg_binomial_2_rng(mu_f, phi);
    forecast_sim[n] = 5.0 * (forecast_shifted_sim[n] + 1);

    // Completion time: gap + heterogeneous treatment
    real mu_t = alpha_t + beta_gap * gap + beta_trt * ai_access[n] + beta_gap_trt * gap * ai_access[n];
    y_sim[n] = lognormal_rng(mu_t, sigma_t);
    log_y_sim[n] = log(y_sim[n]);
  }

  // Treatment effect at gap = 0 (percentage lift)
  real pct_lift = (exp(beta_trt) - 1) * 100;

  // Treatment effect at different gap values (percentage lift)
  real pct_lift_low_gap = (exp(beta_trt + beta_gap_trt * (-1)) - 1) * 100;   // Easy task (gap = -1)
  real pct_lift_high_gap = (exp(beta_trt + beta_gap_trt * (1)) - 1) * 100;   // Hard task (gap = +1)

  // Summary statistics
  real mean_exposure = mean(to_vector(exposure_sim));
  real mean_resources = mean(to_vector(resources_sim));
  real mean_gap = mean(gap_sim);
  real sd_gap = sd(gap_sim);
}
