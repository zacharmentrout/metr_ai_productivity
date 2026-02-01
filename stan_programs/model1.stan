// Model 1: Bayesian Regression
//
// Observational model:
//   y ~ Lognormal(mu, sigma)
//   mu = alpha + beta_trt * ai_access + beta_forecast * log(forecast / x0)
//
// Prior model:
//   alpha ~ Normal(log(90), 0.30)         [median 45-180 min at baseline]
//   beta_trt ~ Normal(0, 0.7)             [treatment effect on log scale]
//   beta_forecast ~ Normal(1, 0.32)       [elasticity, centered at 1]
//   sigma ~ Half-Normal(0, 0.8 / 2.57)

data {
  int<lower=0> N;
  vector<lower=0>[N] y;                      // Observed completion time (minutes)
  vector<lower=0>[N] forecast;               // Forecast time (minutes)
  array[N] int<lower=0, upper=1> ai_access;  // 1 = AI allowed, 0 = restricted
  real<lower=0> x0;                          // Baseline forecast (90 minutes)
}

transformed data {
  vector[N] log_y = log(y);
  vector[N] log_forecast_ratio = log(forecast / x0);

  vector[N] ai_access_vec;
  for (n in 1:N) {
    ai_access_vec[n] = ai_access[n];
  }
}

parameters {
  real alpha;
  real beta_trt;
  real beta_forecast;
  real<lower=0> sigma;
}

model {
  // Priors
  alpha ~ normal(log(90), 0.30);
  beta_trt ~ normal(0, 0.7);
  beta_forecast ~ normal(1, 0.32);
  sigma ~ normal(0, 0.8 / 2.57);

  // Likelihood
  log_y ~ normal(alpha + beta_trt * ai_access_vec + beta_forecast * log_forecast_ratio, sigma);
}

generated quantities {
  // Posterior predictive
  vector[N] mu = alpha + beta_trt * ai_access_vec + beta_forecast * log_forecast_ratio;
  vector[N] y_rep;
  vector[N] log_y_rep;

  for (n in 1:N) {
    y_rep[n] = lognormal_rng(mu[n], sigma);
    log_y_rep[n] = log(y_rep[n]);
  }

  // Treatment effect as percentage lift
  real pct_lift = (exp(beta_trt) - 1) * 100;
}
