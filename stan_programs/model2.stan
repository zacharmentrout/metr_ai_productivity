// Model 2: Latent Task Burden (Non-Centered Parameterization)
//
// Generative structure:
//   task_burden[n] ~ Normal(0, sigma_b)
//   forecast[n] ~ Lognormal(alpha_f + task_burden[n], sigma_f)
//   y[n] ~ Lognormal(alpha_t + beta_trt * ai_access[n] + beta_burden * task_burden[n], sigma_t)
//
// Prior model:
//   sigma_b ~ Half-Normal(0, 0.39)
//   sigma_f ~ Half-Normal(0, 0.39)
//   sigma_t ~ Half-Normal(0, 0.25)
//   alpha_f ~ Normal(log(90), 0.30)
//   alpha_t ~ Normal(log(90), 0.40)
//   beta_burden ~ Normal(1, 0.32)
//   beta_trt ~ Normal(0, 0.7)
//
// Implementation note: Uses non-centered parameterization for task_burden
// to avoid funnel geometry. task_burden = sigma_b * task_burden_raw where
// task_burden_raw ~ Normal(0, 1).

data {
  int<lower=0> N;
  vector<lower=0>[N] y;                      // Observed completion time (minutes)
  vector<lower=0>[N] forecast;               // Observed forecast time (minutes)
  array[N] int<lower=0, upper=1> ai_access;  // 1 = AI allowed, 0 = restricted
}

transformed data {
  vector[N] log_y = log(y);
  vector[N] log_forecast = log(forecast);

  vector[N] ai_access_vec;
  for (n in 1:N) {
    ai_access_vec[n] = ai_access[n];
  }
}

parameters {
  // Dispersion parameters
  real<lower=0> sigma_b;
  real<lower=0> sigma_f;
  real<lower=0> sigma_t;

  // Location parameters
  real alpha_f;
  real alpha_t;

  // Coefficients
  real beta_burden;
  real beta_trt;

  // Latent task burden (raw, for non-centered parameterization)
  vector[N] task_burden_raw;
}

transformed parameters {
  // Non-centered parameterization: task_burden = sigma_b * task_burden_raw
  vector[N] task_burden = sigma_b * task_burden_raw;
}

model {
  // Priors on dispersion
  sigma_b ~ normal(0, 0.39);
  sigma_f ~ normal(0, 0.39);
  sigma_t ~ normal(0, 0.25);

  // Priors on location
  alpha_f ~ normal(log(90), 0.30);
  alpha_t ~ normal(log(90), 0.40);

  // Priors on coefficients
  beta_burden ~ normal(1, 0.32);
  beta_trt ~ normal(0, 0.7);

  // Latent task burden (non-centered: task_burden_raw ~ std_normal)
  task_burden_raw ~ std_normal();

  // Likelihood for forecasts
  log_forecast ~ normal(alpha_f + task_burden, sigma_f);

  // Likelihood for completion times
  log_y ~ normal(alpha_t + beta_trt * ai_access_vec + beta_burden * task_burden, sigma_t);
}

generated quantities {
  // Posterior predictive for completion times
  vector[N] mu = alpha_t + beta_trt * ai_access_vec + beta_burden * task_burden;
  vector[N] y_pred;
  vector[N] log_y_pred;

  for (n in 1:N) {
    y_pred[n] = lognormal_rng(mu[n], sigma_t);
    log_y_pred[n] = log(y_pred[n]);
  }

  // Posterior predictive for forecasts
  vector[N] forecast_pred;
  vector[N] log_forecast_pred;

  for (n in 1:N) {
    forecast_pred[n] = lognormal_rng(alpha_f + task_burden[n], sigma_f);
    log_forecast_pred[n] = log(forecast_pred[n]);
  }

  // Treatment effect as percentage lift
  real pct_lift = (exp(beta_trt) - 1) * 100;
}
