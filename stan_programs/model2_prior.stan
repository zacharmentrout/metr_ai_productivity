// Model 2: Prior Predictive Check (Latent Task Burden)
//
// Generative structure:
//   task_burden[n] ~ Normal(0, sigma_b)
//   forecast[n] ~ Lognormal(alpha_f + task_burden[n], sigma_f)
//   y[n] ~ Lognormal(alpha_t + beta_trt * ai_access[n] + beta_task_burden * task_burden[n], sigma_t)
//
// Prior model:
//   sigma_b ~ Half-Normal(0, 0.39)           [task burden spread; 99% < 1]
//   sigma_f ~ Half-Normal(0, 0.39)           [forecast noise; 99% < 1]
//   sigma_t ~ Half-Normal(0, 0.25)           [completion noise; 99% < 0.64]
//   alpha_f ~ Normal(log(90), 0.30)          [log median forecast at avg burden]
//   alpha_t ~ Normal(log(90), 0.40)          [log median completion at avg burden, no AI]
//   beta_task_burden ~ Normal(1, 0.32)       [burden effect on completion; 0.25-1.75]
//   beta_trt ~ Normal(0, 0.7)                [treatment effect on log scale]

data {
  int<lower=0> N;
  array[N] int<lower=0, upper=1> ai_access;
}

transformed data {
  vector[N] ai_access_vec;
  for (n in 1:N) {
    ai_access_vec[n] = ai_access[n];
  }
}

generated quantities {
  // Sample from priors
  real<lower=0> sigma_b = abs(normal_rng(0, 0.39));
  real<lower=0> sigma_f = abs(normal_rng(0, 0.39));
  real<lower=0> sigma_t = abs(normal_rng(0, 0.25));

  real alpha_f = normal_rng(log(90), 0.30);
  real alpha_t = normal_rng(log(90), 0.40);
  real beta_task_burden = normal_rng(1, 0.32);
  real beta_trt = normal_rng(0, 0.7);

  // Latent task burden
  vector[N] task_burden;
  for (n in 1:N) {
    task_burden[n] = normal_rng(0, sigma_b);
  }

  // Simulated forecasts
  vector[N] forecast_sim;
  vector[N] log_forecast_sim;
  for (n in 1:N) {
    forecast_sim[n] = lognormal_rng(alpha_f + task_burden[n], sigma_f);
    log_forecast_sim[n] = log(forecast_sim[n]);
  }

  // Log median completion time
  vector[N] mu = alpha_t + beta_trt * ai_access_vec + beta_task_burden * task_burden;
  vector[N] mu_no_trt = alpha_t + beta_task_burden * task_burden;

  // Simulated completion times
  vector[N] y_sim;
  vector[N] y_sim_no_trt;
  vector[N] log_y_sim;
  for (n in 1:N) {
    y_sim[n] = lognormal_rng(mu[n], sigma_t);
    y_sim_no_trt[n] = lognormal_rng(mu_no_trt[n], sigma_t);
    log_y_sim[n] = log(y_sim[n]);
  }

  // Baseline (average burden, no treatment)
  real y_sim_baseline = lognormal_rng(alpha_t, sigma_t);

  // Treatment effect as percentage lift
  real pct_lift = (exp(beta_trt) - 1) * 100;
}
