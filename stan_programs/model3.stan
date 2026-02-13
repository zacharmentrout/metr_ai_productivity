// Model 3: Latent Task Burden with Shifted Negative Binomial Forecasts
//
// Generative structure:
//   task_burden[n] ~ Normal(0, sigma_b)
//   forecast[n] = 5 * (Z[n] + 1), where Z[n] ~ NegBinomial2(mu, phi)
//   y[n] ~ Lognormal(alpha_t + beta_trt * ai_access[n] + beta_burden * task_burden[n], sigma_t)
//
// The shift ensures minimum forecast is 5 minutes (when Z = 0).
//
// Parameterization: kappa = 1/phi, so Var = mu + mu^2 * kappa
//   - kappa = 0: Poisson (no overdispersion)
//   - kappa > 0: overdispersion (larger kappa = more variance)
//
// Prior model:
//   sigma_b ~ Half-Normal(0, 0.39)
//   sigma_t ~ Half-Normal(0, 0.25)
//   kappa ~ Half-Normal(0, 0.10)        [overdispersion; 99% < 0.25]
//   alpha_f ~ Normal(log(17), 0.30)     [log expected (forecast/5 - 1) at avg burden]
//   alpha_t ~ Normal(log(90), 0.40)
//   beta_burden ~ Normal(1, 0.32)
//   beta_trt ~ Normal(0, 0.7)

data {
  int<lower=0> N;
  vector<lower=0>[N] y;                      // Observed completion time (minutes)
  vector<lower=0>[N] forecast;               // Observed forecast time (minutes)
  array[N] int<lower=0, upper=1> ai_access;  // 1 = AI allowed, 0 = restricted
}

transformed data {
  vector[N] log_y = log(y);

  // Shift forecasts: forecast = 5 * (Z + 1), so Z = forecast/5 - 1
  array[N] int<lower=0> forecast_shifted;
  for (n in 1:N) {
    forecast_shifted[n] = to_int(round(forecast[n] / 5.0)) - 1;
    // Ensure non-negative
    if (forecast_shifted[n] < 0) forecast_shifted[n] = 0;
  }

  vector[N] ai_access_vec;
  for (n in 1:N) {
    ai_access_vec[n] = ai_access[n];
  }
}

parameters {
  // Dispersion parameters
  real<lower=0> sigma_b;
  real<lower=0> sigma_t;
  real<lower=0> kappa;  // Overdispersion: 1/phi

  // Location parameters
  real alpha_f;
  real alpha_t;

  // Coefficients
  real beta_burden;
  real beta_trt;

  // Latent task burden
  vector[N] task_burden;
}

transformed parameters {
  real<lower=0> phi = 1.0 / kappa;  // NegBinom dispersion parameter
}

model {
  // Priors on dispersion
  sigma_b ~ normal(0, 0.39);
  sigma_t ~ normal(0, 0.25);
  kappa ~ normal(0, 0.10);

  // Priors on location
  alpha_f ~ normal(log(17), 0.30);  // E[forecast] = 5*(exp(alpha_f)+1) = 90 at avg burden
  alpha_t ~ normal(log(90), 0.40);

  // Priors on coefficients
  beta_burden ~ normal(1, 0.32);
  beta_trt ~ normal(0, 0.7);

  // Latent task burden
  task_burden ~ normal(0, sigma_b);

  // Likelihood for forecasts (Shifted Negative Binomial)
  // forecast_shifted = forecast/5 - 1, so forecast = 5*(forecast_shifted + 1)
  {
    vector[N] mu_f = exp(alpha_f + task_burden);
    forecast_shifted ~ neg_binomial_2(mu_f, phi);
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

  // Posterior predictive for forecasts (shifted, on original and log scale)
  array[N] int forecast_shifted_pred;
  vector[N] forecast_pred;
  vector[N] log_forecast_pred;

  for (n in 1:N) {
    real mu_f_n = exp(alpha_f + task_burden[n]);
    forecast_shifted_pred[n] = neg_binomial_2_rng(mu_f_n, phi);
    forecast_pred[n] = 5.0 * (forecast_shifted_pred[n] + 1);
    log_forecast_pred[n] = log(forecast_pred[n]);
  }

  // Treatment effect as percentage lift
  real pct_lift = (exp(beta_trt) - 1) * 100;
}
