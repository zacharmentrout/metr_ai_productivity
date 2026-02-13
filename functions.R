
build_adj_matrix <- function(N_items, N_comparisons, idx1, idx2) {
  if (min(idx1) < 1 | max(idx1) > N_items) {
    stop("Out of bounds idx1 values.")
  }
  if (min(idx2) < 1 | max(idx2) > N_items) {
    stop("Out of bounds idx2 values.")
  }
  
  adj <- matrix(0, nrow=N_items, ncol=N_items)
  
  # Compute directed edges
  for (n in 1:N_comparisons) {
    adj[idx1[n], idx2[n]] <- adj[idx1[n], idx2[n]] + 1
  }
  
  # Compute symmetric, undirected edges
  for (p1 in 1:(N_items - 1)) {
    for (p2 in (p1 + 1):N_items) {
      N1 <- adj[p1, p2]
      N2 <- adj[p2, p1]
      adj[p1, p2] <- N1 + N2
      adj[p2, p1] <- N1 + N2
    }
  }
  
  (adj)
}

elo1 <- function(x) {
  1 / (1 + 10^(-x/400))
}

# test function 
elo2 <- function(x) {
  1 / (1+ exp(-x/173.7178))
}



derived_cut_points <- function(p) {
  K <- length(p)
  c <- rep(NA, length.out=K-1)
  cum_sum <- 0
  for (k in 1:(K-1)) {
    cum_sum <- cum_sum + p[k]
    c[k] <- logit(cum_sum)
  }
  return(c)
}

rdirichlet <- function(n, rho, tau) {
  alpha <- rho / tau + 1
  rBeta2009::rdirichlet(n, alpha)
}


logit <- function(p) {
  if (any(p <= 0 | p >= 1)) stop("Probabilities must be strictly between 0 and 1.")
  log(p / (1 - p))
}

my_q <- function(x) {
  quantile(x, c(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99))
}

ordinal_probs_to_cutpoints <- function(probs) {
  # Ensure the probabilities are cumulative
  cum_probs <- cumsum(probs)
  
  # Exclude the last value since the cumulative probability for the final category should be 1
  if (!all.equal(tail(cum_probs, 1), 1)) stop("Probabilities must sum to 1.")
  
  # Compute cutpoints using the logit transformation
  cutpoints <- qlogis(cum_probs[-length(cum_probs)])
  
  return(cutpoints)
}


rordered_logistic <- function(n, cutpoints, shift) {
  # Ensure cutpoints are sorted
  cutpoints <- sort(cutpoints)
  
  # Sample uniform values and map to categorical outcomes
  #u <- runif(n)
  
  # Compute cumulative probabilities based on the logistic CDF
  probs <- plogis(cutpoints - shift)
  
  # Compute valid category probabilities
  probs <- c(probs[1], diff(probs), 1 - tail(probs, 1))
  
  # Sample from the categorical distribution
  categories <- sample(seq_along(probs), size = n, replace = TRUE, prob = probs)
  
  return(categories)
  #probs
}

probs_ordered_logistic <- function(n, cutpoints, shift) {
  # Ensure cutpoints are sorted
  cutpoints <- sort(cutpoints)
  
  # Sample uniform values and map to categorical outcomes
  u <- runif(n)
  
  # Compute cumulative probabilities based on the logistic CDF
  probs <- plogis(cutpoints - shift)
  
  # Compute valid category probabilities
  probs <- c(probs[1], diff(probs), 1 - tail(probs, 1))
  
  # Sample from the categorical distribution
  categories <- sample(seq_along(probs), size = n, replace = TRUE, prob = probs)
  
  return(probs)
}


plot_cut_point_overlay <- function(expectand_vals_list, prefix,
                                   flim, fname, ylim, main=NULL) {
  name <- paste0(prefix, 1, ']')
  util$plot_expectand_pushforward(expectand_vals_list[[name]],
                                  45, flim=flim, display_name=fname,
                                  col=util$c_dark, border="#DDDDDDDD",
                                  ylim=ylim, main=main)
  name <- paste0(prefix, 2, ']')
  util$plot_expectand_pushforward(expectand_vals_list[[name]],
                                  45, flim=flim,
                                  col=util$c_mid_highlight,
                                  border="#DDDDDDDD",
                                  add=TRUE)
}


plot_cut_point_post_vs_prior_overlay <- function(expectand_vals_list, prefix, prefix_prior,
                                   flim, fname, ylim, main=NULL) {
  # posterior
  name <- paste0(prefix, 1, ']')
  util$plot_expectand_pushforward(expectand_vals_list[[name]],
                                  45, flim=flim, display_name=fname,
                                  col=util$c_dark, border="#DDDDDDDD",
                                  ylim=ylim, main=main)
  
  name <- paste0(prefix, 2, ']')
  util$plot_expectand_pushforward(expectand_vals_list[[name]],
                                  45, flim=flim,
                                  col=util$c_mid_highlight,
                                  border="#DDDDDDDD",
                                  add=TRUE)
  
  # prior
  name <- paste0(prefix_prior, 1, ']')
  util$plot_expectand_pushforward(expectand_vals_list[[name]],
                                  45, flim=flim, display_name=fname,
                                  col=util$c_dark_teal, border="#DDDDDDDD",
                                  ylim=ylim,
                                  add=TRUE)
  
  name <- paste0(prefix_prior, 2, ']')
  util$plot_expectand_pushforward(expectand_vals_list[[name]],
                                  45, flim=flim,
                                  col=util$c_mid_teal,
                                  border="#DDDDDDDD",
                                  add=TRUE)  
  
}

plot_ordinal_post_vs_prior_overlay <- function(expectand_vals_list, prefix, prefix_prior,
                                                 flim, fname, ylim, main=NULL) {
  # prior
  name <- paste0(prefix_prior, 1, ']')
  util$plot_expectand_pushforward(expectand_vals_list[[name]],
                                  45, flim=flim, display_name=fname,
                                  col=util$c_dark_teal, border="#DDDDDDDD",
                                  ylim=ylim,
                                  main=main)
  
  name <- paste0(prefix_prior, 2, ']')
  util$plot_expectand_pushforward(expectand_vals_list[[name]],
                                  45, flim=flim,
                                  col=util$c_mid_teal,
                                  border="#DDDDDDDD",
                                  add=TRUE)  
  
  name <- paste0(prefix_prior, 3, ']')
  util$plot_expectand_pushforward(expectand_vals_list[[name]],
                                  45, flim=flim,
                                  col=util$c_light_teal,
                                  border="#DDDDDDDD",
                                  add=TRUE)
  
  # posterior
  name <- paste0(prefix, 1, ']')
  util$plot_expectand_pushforward(expectand_vals_list[[name]],
                                  45, flim=flim, display_name=fname,
                                  col=util$c_dark, border="#DDDDDDDD",
                                  ylim=ylim,
                                  add=TRUE)
  
  name <- paste0(prefix, 2, ']')
  util$plot_expectand_pushforward(expectand_vals_list[[name]],
                                  45, flim=flim,
                                  col=util$c_mid_highlight,
                                  border="#DDDDDDDD",
                                  add=TRUE)
  
  name <- paste0(prefix, 3, ']')
  util$plot_expectand_pushforward(expectand_vals_list[[name]],
                                  45, flim=flim,
                                  col=util$c_light,
                                  border="#DDDDDDDD",
                                  add=TRUE)
  

  
}

