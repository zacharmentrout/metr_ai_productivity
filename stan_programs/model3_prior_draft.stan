functions {
  
  // Log probability density function over cut points
  // induced by a Dirichlet probability density function
  // over baseline probabilities and a latent logistic
  // density function.
  real induced_dirichlet_lpdf(vector c, vector alpha) {
    int K = num_elements(c) + 1;
    vector[K - 1] Pi = inv_logit(c);
    vector[K] p = append_row(Pi, [1]') - append_row([0]', Pi);

    // Log Jacobian correction
    real logJ = 0;
    for (k in 1:(K - 1)) {
      if (c[k] >= 0)
        logJ += -c[k] - 2 * log(1 + exp(-c[k]));
      else
        logJ += +c[k] - 2 * log(1 + exp(+c[k]));
    }

    return dirichlet_lpdf(p | alpha) + logJ;
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
  
  // Ordinal pseudo-random number generator assuming
  // a latent standard logistic density function.
  vector shifted_derived_ordinal_probs(vector c, real gamma) {
    int K = num_elements(c) + 1;
    vector[K - 1] Pi = inv_logit(c - gamma);
    vector[K] p = append_row(Pi, [1]') - append_row([0]', Pi);
    return p;
  }

}

data {
  int<lower=1> N;
  int<lower=1> N_developers;
  vector<lower=0>[N] forecast;               // Forecast time (minutes)
  array[N] int<lower=0, upper=1> ai_access;  // 1 = AI allowed, 0 = restricted
  real<lower=0> x0;                          // Baseline forecast (90 minutes)
  
  int<lower=1> K_exposure; // number of exposure categories 
  int<lower=1> K_resources;
  array[N] int<lower=1, upper=N_developers> dev_idxs; // developer ids

  simplex[K_exposure] rho_exposure; // ordinal prob locations
  real<lower=0> tau_exposure; // ordinal prob concentration
  
simplex[K_exposure] rho_resources; // ordinal prob locations
  real<lower=0> tau_resources; // ordinal prob concentration
}


generated quantities {
  // exposure / comfort parameters
  ordered[K_exposure - 1] cp_exposure; // Interior cut points for exposure
  array[N] int<lower=1, upper=K_exposure> exposure;
  vector[K_exposure] ordinal_probs_exposure;
  real<lower=0> tau_comfort; // linear elo coef population scale
  real mu_comfort; // linear elo coef population mean
  vector[N_developers] comfort; // developer comfort in context
  vector[N_developers-1] eta_comfort;
  vector[N_developers-1] comfort_free;
  
  // resources parameters
  ordered[K_resources - 1] cp_resources;
  array[N]  int<lower=1, upper=K_resources> resources;
  vector[K_resources] ordinal_probs_resources;
  real<lower=0> tau_difficulty;
  real mu_difficulty;
  vector[N] difficulty;
  real<lower=0> sigma_forecast;
  
  


  ordinal_probs_exposure = dirichlet_rng(rho_exposure/tau_exposure + rep_vector(1, K_exposure));
  cp_exposure = derived_cut_points(ordinal_probs_exposure);
  
  ordinal_probs_resources = dirichlet_rng(rho_resources / tau_resources + rep_vector(1, K_resources));
  cp_resources = dirived_cut_pointes(ordinal_probs_resources);
  
  for (n in 1:(N_developers - 1)) {
    eta_comfort[n] = normal_rng(0,1);
    comfort_free = mu_comfort + eta_comfort[n] * tau_comfort;
  }
  
  comfort = append_row([0]', comfort_free);
  
  sigma_difficulty = exponential(1);

  for (n in 1:N) {
    difficulty[n] = normal_rng(0, sigma_difficulty);
    exposure[n] = ordered_logistic_rng(comfort[dev_idxs[n]], cp_exposure);
    resources[n] = ordered_logistic_rng(difficulty[n] - comfort[dev_idxs[n]], cp_resources);
    forecast[n] = lognormal_rng(alpha_forecast + beta_comfort_forecast * comfort[dev_idxs[n] + beta_difficulty_forecast * difficulty[n]], sigma_forecast);
    mu[n] = alpha_time 
            + beta_difficulty_time * difficulty[n] 
            + beta_comfort_time * comfort[dev_idxs[n]] 
            + beta_trt * ai_access[n]
            + beta_forecast * log(forecast[n] / 90);
  }
  
  
  
    
}

