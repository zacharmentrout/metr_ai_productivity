// Model 4 Prior Predictive: Task Burden -> Forecasts + Resources + Completion Time
//
// Uses Betancourt's ordinal modeling functions.

functions {
  // Ordinal probability mass function assuming a
  // latent shifted logistic density function.
  //
  // Positive gamma shifts baseline ordinal
  // probabilities towards larger values.
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
  // and latent logistic density function.
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
  array[N] int<lower=0, upper=1> ai_access;

  // Dirichlet hyperparameters for cutpoints
  vector<lower=0>[3] rho;  // (1/3, 1/3, 1/3)
  real<lower=0> tau;       // 0.2
}

transformed data {
  vector[3] alpha_dir = rho / tau + rep_vector(1, 3);
}

generated quantities {
  // Sample priors
  real<lower=0> sigma_b = abs(normal_rng(0, 0.39));
  real<lower=0> sigma_t = abs(normal_rng(0, 0.25));
  real<lower=0> kappa = abs(normal_rng(0, 0.10));
  real phi = 1.0 / kappa;

  real alpha_f = normal_rng(log(17), 0.30);
  real alpha_t = normal_rng(log(90), 0.40);

  real beta_burden = normal_rng(1, 0.32);
  real beta_trt = normal_rng(0, 0.7);
  real lambda_r = normal_rng(1, 0.43);

  // Sample baseline probabilities from Dirichlet
  simplex[3] baseline_p = dirichlet_rng(alpha_dir);

  // Derive cutpoints using Betancourt's function
  ordered[2] cut_points = derived_cut_points(baseline_p);

  // Simulate data
  vector[N] task_burden;
  array[N] int forecast_shifted_sim;
  vector[N] forecast_sim;
  array[N] int resources_sim;
  vector[N] y_sim;
  vector[N] log_y_sim;

  for (n in 1:N) {
    // Latent task burden
    task_burden[n] = normal_rng(0, sigma_b);

    // Forecast (shifted negative binomial)
    real mu_f = exp(alpha_f + task_burden[n]);
    forecast_shifted_sim[n] = neg_binomial_2_rng(mu_f, phi);
    forecast_sim[n] = 5.0 * (forecast_shifted_sim[n] + 1);

    // Resources (ordinal shifted logistic)
    // Positive gamma shifts toward larger categories
    real gamma_r = lambda_r * task_burden[n];
    resources_sim[n] = ordinal_shifted_logistic_rng(cut_points, gamma_r);

    // Completion time
    real mu_t = alpha_t + beta_trt * ai_access[n] + beta_burden * task_burden[n];
    y_sim[n] = lognormal_rng(mu_t, sigma_t);
    log_y_sim[n] = log(y_sim[n]);
  }

  // Treatment effect as percentage lift
  real pct_lift = (exp(beta_trt) - 1) * 100;

  // Summary: resource distribution across simulated data
  int n_resources_1 = 0;
  int n_resources_2 = 0;
  int n_resources_3 = 0;
  for (n in 1:N) {
    if (resources_sim[n] == 1) n_resources_1 += 1;
    else if (resources_sim[n] == 2) n_resources_2 += 1;
    else n_resources_3 += 1;
  }
  real prop_resources_1 = n_resources_1 * 1.0 / N;
  real prop_resources_2 = n_resources_2 * 1.0 / N;
  real prop_resources_3 = n_resources_3 * 1.0 / N;
}
