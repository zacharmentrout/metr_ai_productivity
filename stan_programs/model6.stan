// Model 6: Heterogeneous Treatment Effect (Gap Interaction)
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
  real ordinal_shifted_logistic_lpmf(int y, vector c, real gamma) {
    int K = num_elements(c) + 1;
    vector[K - 1] Pi = inv_logit(c - gamma);
    vector[K] p = append_row(Pi, [1]') - append_row([0]', Pi);
    return categorical_lpmf(y | p);
  }

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

  vector<lower=0>[N] y;                       // Completion time
  vector<lower=0>[N] forecast;                // Forecast time
  array[N] int<lower=0, upper=1> ai_access;   // Treatment indicator

  // Exposure (ordinal 1-5, with missingness)
  int<lower=0> N_obs_exposure;
  array[N_obs_exposure] int<lower=1, upper=N> exposure_idx;
  array[N_obs_exposure] int<lower=1, upper=5> exposure;

  // Resources (ordinal 1-3, with missingness)
  int<lower=0> N_obs_resources;
  array[N_obs_resources] int<lower=1, upper=N> resources_idx;
  array[N_obs_resources] int<lower=1, upper=3> resources;

  // Dirichlet hyperparameters
  vector<lower=0>[5] rho_e;
  real<lower=0> tau_e;
  vector<lower=0>[3] rho_r;
  real<lower=0> tau_r;
}

transformed data {
  vector[N] log_y = log(y);

  // Shift forecasts: forecast = 5 * (Z + 1), so Z = forecast/5 - 1
  array[N] int<lower=0> forecast_shifted;
  for (n in 1:N) {
    forecast_shifted[n] = to_int(round(forecast[n] / 5.0)) - 1;
    if (forecast_shifted[n] < 0) forecast_shifted[n] = 0;
  }

  vector[N] ai_access_vec;
  for (n in 1:N) {
    ai_access_vec[n] = ai_access[n];
  }

  // Dirichlet concentrations
  vector[5] alpha_dir_e = rho_e / tau_e + rep_vector(1, 5);
  vector[3] alpha_dir_r = rho_r / tau_r + rep_vector(1, 3);
}

parameters {
  // Dispersion parameters
  real<lower=0> sigma_c;    // Developer comfort spread
  real<lower=0> sigma_b;    // Task burden spread
  real<lower=0> sigma_t;    // Completion time noise
  real<lower=0> kappa;      // Forecast overdispersion

  // Location parameters
  real alpha_f;             // Log expected forecast baseline
  real alpha_t;             // Log median completion baseline

  // Coefficients
  real beta_gap;            // Gap -> completion
  real beta_trt;            // Treatment effect (at gap = 0)
  real beta_gap_trt;        // Gap x treatment interaction
  real lambda_e;            // Comfort -> exposure
  real lambda_r;            // Gap -> resources

  // Baseline ordinal probabilities
  simplex[5] baseline_p_e;  // Exposure (K=5)
  simplex[3] baseline_p_r;  // Resources (K=3)

  // Latent variables (non-centered)
  vector[J] comfort_raw;
  vector[N] task_burden_raw;
}

transformed parameters {
  real<lower=0> phi = 1.0 / kappa;

  // Derived cut points
  ordered[4] cut_points_e = derived_cut_points(baseline_p_e);
  ordered[2] cut_points_r = derived_cut_points(baseline_p_r);

  // Non-centered parameterization
  vector[J] comfort = sigma_c * comfort_raw;
  vector[N] task_burden = sigma_b * task_burden_raw;
}

model {
  // Priors on dispersion
  sigma_c ~ normal(0, 0.78);
  sigma_b ~ normal(0, 0.39);
  sigma_t ~ normal(0, 0.25);
  kappa ~ normal(0, 0.10);

  // Priors on location
  alpha_f ~ normal(log(17), 0.30);
  alpha_t ~ normal(log(90), 0.40);

  // Priors on coefficients
  beta_gap ~ normal(1, 0.3);
  beta_trt ~ normal(0, 0.7);
  beta_gap_trt ~ normal(0, 0.15);  // Interaction centered at zero
  lambda_e ~ normal(1, 0.43);
  lambda_r ~ normal(1, 0.43);

  // Priors on baseline probabilities
  baseline_p_e ~ dirichlet(alpha_dir_e);
  baseline_p_r ~ dirichlet(alpha_dir_r);

  // Latent variables (non-centered)
  comfort_raw ~ std_normal();
  task_burden_raw ~ std_normal();

  // Likelihood for forecasts (gap coefficient = 1 for identification)
  {
    vector[N] mu_f;
    for (n in 1:N) {
      real gap = task_burden[n] - comfort[dev_idx[n]];
      mu_f[n] = exp(alpha_f + gap);
    }
    forecast_shifted ~ neg_binomial_2(mu_f, phi);
  }

  // Likelihood for exposure (ordinal, observed cases only)
  for (i in 1:N_obs_exposure) {
    int n = exposure_idx[i];
    real gamma_e = lambda_e * comfort[dev_idx[n]];
    exposure[i] ~ ordinal_shifted_logistic(cut_points_e, gamma_e);
  }

  // Likelihood for resources (ordinal, observed cases only)
  for (i in 1:N_obs_resources) {
    int n = resources_idx[i];
    real gap = task_burden[n] - comfort[dev_idx[n]];
    real gamma_r = lambda_r * gap;
    resources[i] ~ ordinal_shifted_logistic(cut_points_r, gamma_r);
  }

  // Likelihood for completion times (with heterogeneous treatment effect)
  {
    vector[N] mu_t;
    for (n in 1:N) {
      real gap = task_burden[n] - comfort[dev_idx[n]];
      mu_t[n] = alpha_t + beta_gap * gap + beta_trt * ai_access[n] + beta_gap_trt * gap * ai_access[n];
    }
    log_y ~ normal(mu_t, sigma_t);
  }
}

generated quantities {
  // Posterior predictive for completion times
  vector[N] y_pred;
  vector[N] log_y_pred;

  for (n in 1:N) {
    real gap = task_burden[n] - comfort[dev_idx[n]];
    real mu_t_n = alpha_t + beta_gap * gap + beta_trt * ai_access[n] + beta_gap_trt * gap * ai_access[n];
    y_pred[n] = lognormal_rng(mu_t_n, sigma_t);
    log_y_pred[n] = log(y_pred[n]);
  }

  // Posterior predictive for forecasts
  array[N] int forecast_shifted_pred;
  vector[N] forecast_pred;

  for (n in 1:N) {
    real gap = task_burden[n] - comfort[dev_idx[n]];
    real mu_f_n = exp(alpha_f + gap);
    forecast_shifted_pred[n] = neg_binomial_2_rng(mu_f_n, phi);
    forecast_pred[n] = 5.0 * (forecast_shifted_pred[n] + 1);
  }

  // Posterior predictive for exposure (all observations)
  array[N] int exposure_pred;
  for (n in 1:N) {
    real gamma_e = lambda_e * comfort[dev_idx[n]];
    exposure_pred[n] = ordinal_shifted_logistic_rng(cut_points_e, gamma_e);
  }

  // Posterior predictive for resources (all observations)
  array[N] int resources_pred;
  for (n in 1:N) {
    real gap = task_burden[n] - comfort[dev_idx[n]];
    real gamma_r = lambda_r * gap;
    resources_pred[n] = ordinal_shifted_logistic_rng(cut_points_r, gamma_r);
  }

  // Treatment effect at gap = 0 (percentage lift)
  real pct_lift = (exp(beta_trt) - 1) * 100;

  // Treatment effect at different gap values (percentage lift)
  real pct_lift_low_gap = (exp(beta_trt + beta_gap_trt * (-1)) - 1) * 100;   // Easy task (gap = -1)
  real pct_lift_high_gap = (exp(beta_trt + beta_gap_trt * (1)) - 1) * 100;   // Hard task (gap = +1)
}
