// Model 1: Prior Predictive Check
//
// Observational model:
//   y ~ Lognormal(mu, sigma)
//   mu = alpha + beta_trt * ai_access + beta_forecast * log(forecast / x0)
//
// Prior model:
//   alpha ~ Normal(log(90), 0.30)         [median 45-180 min at baseline]
//   beta_trt ~ Normal(0, 0.7)             [treatment effect on log scale]
//   beta_forecast ~ Normal(1, 0.32)       [elasticity, centered at 1]
//   sigma ~ Normal(0, 0.8 / 2.57)                   

data {
  int<lower=0> N;                         // Number of observations
  vector<lower=0>[N] forecast;            // Forecast time (minutes)
  array[N] int<lower=0, upper=1> ai_access;  // 1 = AI allowed, 0 = restricted
  real<lower=0> x0;                       // Baseline forecast (90 minutes)
}

transformed data {
  vector[N] log_forecast_ratio = log(forecast / x0);

  vector[N] ai_access_vec;
  for (n in 1:N) {
    ai_access_vec[n] = ai_access[n];
  }
}

generated quantities {
  // Sample from priors
  real alpha = normal_rng(log(90), 0.30);
  real beta_trt = normal_rng(0, 0.7);
  real beta_forecast = normal_rng(1, 0.32);
  real<lower=0> sigma = abs(normal_rng(0, 0.8 / 2.57));

  // Log median completion time (mu)
  vector[N] mu = alpha + beta_trt * ai_access_vec + beta_forecast * log_forecast_ratio;
  vector[N] mu_no_trt = alpha + beta_forecast * log_forecast_ratio;
  real mu_baseline = alpha;
  real exp_mu_baseline = exp(alpha);

  // Simulated observations
  vector[N] y_sim;
  vector[N] y_sim_no_trt;
  real y_sim_baseline;
  vector[N] log_y_sim;

  for (n in 1:N) {
    y_sim[n] = lognormal_rng(mu[n], sigma);
    y_sim_no_trt[n] = lognormal_rng(mu_no_trt[n], sigma);
    log_y_sim[n] = log(y_sim[n]);
  }

  y_sim_baseline = lognormal_rng(mu_baseline, sigma);
}
