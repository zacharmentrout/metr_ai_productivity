// Model 4: Task Burden -> Forecasts + Resources + Completion Time
//
// Generative structure:
//   task_burden[n] ~ Normal(0, sigma_b)
//   forecast[n] = 5 * (Z[n] + 1), where Z[n] ~ NegBinomial2(mu, phi)
//   resources[n] ~ OrderedLogistic(lambda_r * task_burden[n], cut_points)  [when observed]
//   y[n] ~ Lognormal(alpha_t + beta_trt * ai_access[n] + beta_burden * task_burden[n], sigma_t)
//
// Cut points derived from Dirichlet:
//   p ~ Dirichlet(alpha), alpha = rho / tau + 1
//   cut_points = derived_cut_points(p)

functions {
  // Ordinal probability mass function assuming a
  // latent shifted logistic density function.
  real ordinal_shifted_logistic_lpmf(int y, vector c, real gamma) {
    int K = num_elements(c) + 1;
    vector[K - 1] Pi = inv_logit(c - gamma);
    vector[K] p = append_row(Pi, [1]') - append_row([0]', Pi);
    return categorical_lpmf(y | p);
  }

  // Ordinal pseudo-random number generator assuming
  // a latent shifted logistic density function.
  int ordinal_shifted_logistic_rng(vector c, real gamma) {
    int K = num_elements(c) + 1;
    vector[K - 1] Pi = inv_logit(c - gamma);
    vector[K] p = append_row(Pi, [1]') - append_row([0]', Pi);
    return categorical_rng(p);
  }

  // Derive cut points from baseline probabilities
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
  int<lower=0> N;
  vector<lower=0>[N] y;
  vector<lower=0>[N] forecast;
  array[N] int<lower=0, upper=1> ai_access;

  // Resources (ordinal 1-3, with missingness)
  int<lower=0> N_obs_resources;
  array[N_obs_resources] int<lower=1, upper=N> resources_idx;
  array[N_obs_resources] int<lower=1, upper=3> resources;

  // Dirichlet hyperparameters
  vector<lower=0>[3] rho;
  real<lower=0> tau;
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

  // Dirichlet concentration
  vector[3] alpha_dir = rho / tau + rep_vector(1, 3);
}

parameters {
  // Dispersion parameters
  real<lower=0> sigma_b;
  real<lower=0> sigma_t;
  real<lower=0> kappa;

  // Location parameters
  real alpha_f;
  real alpha_t;

  // Coefficients
  real beta_burden;
  real beta_trt;
  real lambda_r;  // Effect of task burden on resources (sign learned from data)

  // Baseline ordinal probabilities
  simplex[3] baseline_p;

  // Latent task burden (non-centered)
  vector[N] task_burden_raw;
}

transformed parameters {
  real<lower=0> phi = 1.0 / kappa;
  ordered[2] cut_points = derived_cut_points(baseline_p);

  // Non-centered parameterization
  vector[N] task_burden = sigma_b * task_burden_raw;
}

model {
  // Priors on dispersion
  sigma_b ~ normal(0, 0.39);
  sigma_t ~ normal(0, 0.25);
  kappa ~ normal(0, 0.10);

  // Priors on location
  alpha_f ~ normal(log(17), 0.30);
  alpha_t ~ normal(log(90), 0.40);

  // Priors on coefficients
  beta_burden ~ normal(1, 0.32);
  beta_trt ~ normal(0, 0.7);
  lambda_r ~ normal(1, 0.43);

  // Prior on baseline probabilities
  baseline_p ~ dirichlet(alpha_dir);

  // Latent task burden (non-centered)
  task_burden_raw ~ std_normal();

  // Likelihood for forecasts (Shifted Negative Binomial)
  {
    vector[N] mu_f = exp(alpha_f + task_burden);
    forecast_shifted ~ neg_binomial_2(mu_f, phi);
  }

  // Likelihood for resources (ordinal, observed cases only)
  for (i in 1:N_obs_resources) {
    int n = resources_idx[i];
    real gamma_r = lambda_r * task_burden[n];
    resources[i] ~ ordinal_shifted_logistic(cut_points, gamma_r);
  }

  // Likelihood for completion times
  log_y ~ normal(alpha_t + beta_trt * ai_access_vec + beta_burden * task_burden, sigma_t);
}

generated quantities {
  // Posterior predictive for completion times
  vector[N] mu_t = alpha_t + beta_trt * ai_access_vec + beta_burden * task_burden;
  vector[N] y_pred;
  vector[N] log_y_pred;

  for (n in 1:N) {
    y_pred[n] = lognormal_rng(mu_t[n], sigma_t);
    log_y_pred[n] = log(y_pred[n]);
  }

  // Posterior predictive for forecasts
  array[N] int forecast_shifted_pred;
  vector[N] forecast_pred;

  for (n in 1:N) {
    real mu_f_n = exp(alpha_f + task_burden[n]);
    forecast_shifted_pred[n] = neg_binomial_2_rng(mu_f_n, phi);
    forecast_pred[n] = 5.0 * (forecast_shifted_pred[n] + 1);
  }

  // Posterior predictive for resources (all observations)
  array[N] int resources_pred;
  for (n in 1:N) {
    real gamma_r = lambda_r * task_burden[n];
    resources_pred[n] = ordinal_shifted_logistic_rng(cut_points, gamma_r);
  }

  // Treatment effect as percentage lift
  real pct_lift = (exp(beta_trt) - 1) * 100;
}
