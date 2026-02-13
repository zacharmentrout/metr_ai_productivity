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
  int<lower=1> N;
  int<lower=1> N_developers;
  array[N] int<lower=1, upper=N_developers> dev_nums;
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
  real alpha; // intercept: log median at baseline forecast and no AI
  real beta_trt; // treatment coefficient
  real beta_forecast; // forecast coefficient
  real<lower=0> sigma; // completion time variability
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
  vector[N] y_pred;
  vector[N] log_y_pred;

  array[N_developers] real mean_outcome_dev_pred = rep_array(0, N_developers);
  array[N_developers] real C = rep_array(0, N_developers);

  for (n in 1:N) {
    y_pred[n] = lognormal_rng(mu[n], sigma);
    log_y_pred[n] = log(y_pred[n]);
    
    real delta = 0;
    int c = dev_nums[n];

    C[c] += 1;
    delta = y_pred[n] - mean_outcome_dev_pred[c];
    mean_outcome_dev_pred[c] += delta / C[c];
  }

  // Treatment effect as percentage lift
  real pct_lift = (exp(beta_trt) - 1) * 100;
}
