// Model 2: Latent Task Burden (Marginalized)
//
// Original generative structure:
//   task_burden[n] ~ Normal(0, sigma_b)
//   forecast[n] ~ Lognormal(alpha_f + task_burden[n], sigma_f)
//   y[n] ~ Lognormal(alpha_t + beta_trt * ai_access[n] + beta_burden * task_burden[n], sigma_t)
//
// Marginalized form (integrating out task_burden):
//   [log_forecast; log_y] ~ BivariateNormal(mu, Sigma)
//   where:
//     mu = [alpha_f; alpha_t + beta_trt * ai_access]
//     Sigma[1,1] = sigma_b^2 + sigma_f^2
//     Sigma[2,2] = (beta_burden * sigma_b)^2 + sigma_t^2
//     Sigma[1,2] = Sigma[2,1] = beta_burden * sigma_b^2
//
// Prior model:
//   sigma_b ~ Half-Normal(0, 0.39)
//   sigma_f ~ Half-Normal(0, 0.39)
//   sigma_t ~ Half-Normal(0, 0.25)
//   alpha_f ~ Normal(log(90), 0.30)
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

  // Marginalized bivariate normal likelihood
  // Covariance matrix components (constant across observations)
  real var_f = square(sigma_b) + square(sigma_f);
  real var_t = square(beta_burden * sigma_b) + square(sigma_t);
  real cov_ft = beta_burden * square(sigma_b);

  // Correlation and conditional variance for efficient computation
  real rho = cov_ft / sqrt(var_f * var_t);
  real sd_f = sqrt(var_f);
  real sd_t = sqrt(var_t);
  real sd_t_cond = sd_t * sqrt(1 - square(rho));

  for (n in 1:N) {
    // Means
    real mu_f = alpha_f;
    real mu_t = alpha_t + beta_trt * ai_access_vec[n];

    // Bivariate normal using conditional factorization:
    // p(log_forecast, log_y) = p(log_forecast) * p(log_y | log_forecast)
    real z_f = (log_forecast[n] - mu_f) / sd_f;
    real mu_t_cond = mu_t + rho * sd_t * z_f;

    log_forecast[n] ~ normal(mu_f, sd_f);
    log_y[n] ~ normal(mu_t_cond, sd_t_cond);
  }
}

generated quantities {
  // Variance components for interpretation
  real var_f = square(sigma_b) + square(sigma_f);
  real var_t = square(beta_burden * sigma_b) + square(sigma_t);
  real cov_ft = beta_burden * square(sigma_b);
  real rho = cov_ft / sqrt(var_f * var_t);

  // Posterior predictive (need to sample task_burden first, then outcomes)
  vector[N] task_burden;
  vector[N] y_pred;
  vector[N] log_y_pred;
  vector[N] forecast_pred;
  vector[N] log_forecast_pred;

  for (n in 1:N) {
    task_burden[n] = normal_rng(0, sigma_b);
    forecast_pred[n] = lognormal_rng(alpha_f + task_burden[n], sigma_f);
    log_forecast_pred[n] = log(forecast_pred[n]);

    real mu_t = alpha_t + beta_trt * ai_access_vec[n] + beta_burden * task_burden[n];
    y_pred[n] = lognormal_rng(mu_t, sigma_t);
    log_y_pred[n] = log(y_pred[n]);
  }

  // Treatment effect as percentage lift
  real pct_lift = (exp(beta_trt) - 1) * 100;
}
