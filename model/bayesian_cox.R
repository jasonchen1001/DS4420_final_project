# Bayesian Cox Proportional Hazards Model with MCMC
# Manual implementation - no rstan or survival package for inference
# DS4420 Final Project - Breast Cancer Prognosis

# Load required packages (only for data manipulation, not MCMC)
library(dplyr)
library(ggplot2)

# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

load_survival_data <- function(csv_path = "../data/brca_multimodal.csv",
                                n_features = 100,
                                seed = 42) {
  # Load data
  df <- read.csv(csv_path)

  # Extract survival variables
  time <- df$OS.time
  status <- df$OS  # 1 = event (short survival), 0 = censored (long survival)

  # Drop non-feature columns
  drop_cols <- c("sample", "OS", "OS.time", "sampleID_x", "sampleID_y", "sampleID",
                 "ER.Status", "PR.Status", "HER2.Final.Status",
                 "age_at_initial_pathologic_diagnosis")
  X <- df %>%
    select(-any_of(drop_cols)) %>%
    select(where(is.numeric)) %>%
    mutate(across(where(is.numeric), ~ifelse(is.na(.), mean(., na.rm = TRUE), .)))

  # Feature selection: select top features by variance
  feature_vars <- apply(X, 2, var)
  top_features <- names(sort(feature_vars, decreasing = TRUE))[1:min(n_features, ncol(X))]

  X_selected <- X[, top_features]

  # Standardize features (important for MCMC convergence)
  X_standardized <- scale(X_selected)
  X_standardized[is.na(X_standardized)] <- 0

  # Convert to matrix
  X_matrix <- as.matrix(X_standardized)

  # Set seed for reproducibility
  set.seed(seed)

  list(X = X_matrix, time = time, status = status,
       feature_names = colnames(X_matrix), n = nrow(X_matrix), p = ncol(X_matrix))
}

# =============================================================================
# Cox Partial Likelihood Functions
# =============================================================================

# Compute log partial likelihood for Cox model
log_partial_likelihood <- function(beta, X, time, status) {
  n <- nrow(X)

  # Compute linear predictor
  eta <- X %*% beta

  # Get unique event times (sorted)
  event_times <- sort(unique(time[status == 1]))
  n_events <- length(event_times)

  # Initialize log likelihood
  log_lik <- 0

  for (t in event_times) {
    # Individuals at risk at time t
    at_risk <- which(time >= t)

    # Individuals with event at time t
    events <- which(time == t & status == 1)

    if (length(events) == 0) next

    # Breslow approximation for tied events
    # Sum of exp(eta) over risk set
    risk_sum <- sum(exp(eta[at_risk]))

    if (risk_sum <= 0) risk_sum <- 1e-10

    # Log likelihood contribution
    log_lik <- log_lik + sum(eta[events]) - length(events) * log(risk_sum)
  }

  return(log_lik)
}

# Numerical gradient of log partial likelihood
gradient_log_likelihood <- function(beta, X, time, status, eps = 1e-8) {
  p <- length(beta)
  grad <- numeric(p)

  # Compute baseline log likelihood
  log_lik_base <- log_partial_likelihood(beta, X, time, status)

  for (j in 1:p) {
    beta_perturb <- beta
    beta_perturb[j] <- beta_perturb[j] + eps
    log_lik_perturb <- log_partial_likelihood(beta_perturb, X, time, status)
    grad[j] <- (log_lik_perturb - log_lik_base) / eps
  }

  return(grad)
}

# =============================================================================
# Prior Functions
# =============================================================================

# Log prior for coefficients (Normal(0, sigma^2))
log_prior <- function(beta, sigma = 10) {
  sum(dnorm(beta, mean = 0, sd = sigma, log = TRUE))
}

# =============================================================================
# Metropolis-Hastings MCMC
# =============================================================================

# Single Metropolis-Hastings step for one parameter
mh_step <- function(beta, X, time, status, idx, step_size, sigma_prior = 10) {
  # Propose new value
  beta_proposed <- beta
  beta_proposed[idx] <- rnorm(1, beta[idx], step_size[idx])

  # Compute log posterior (log likelihood + log prior)
  log_lik_current <- log_partial_likelihood(beta, X, time, status)
  log_lik_proposed <- log_partial_likelihood(beta_proposed, X, time, status)

  log_prior_current <- log_prior(beta, sigma_prior)
  log_prior_proposed <- log_prior(beta_proposed, sigma_prior)

  log_post_current <- log_lik_current + log_prior_current
  log_post_proposed <- log_lik_proposed + log_prior_proposed

  # Acceptance probability
  log_alpha <- min(0, log_post_proposed - log_post_current)

  # Accept or reject
  if (log(runif(1)) < log_alpha) {
    return(list(beta = beta_proposed, accepted = TRUE))
  } else {
    return(list(beta = beta, accepted = FALSE))
  }
}

# Adaptive MCMC - adjust step sizes to achieve target acceptance
adaptive_mcmc <- function(X, time, status,
                         n_iter = 10000,
                         burnin = 1000,
                         sigma_prior = 10,
                         target_accept = 0.234,
                         adapt_period = 1000,
                         seed = 42) {

  set.seed(seed)

  n <- nrow(X)
  p <- ncol(X)

  # Initialize beta (all zeros)
  beta <- rep(0, p)

  # Initialize step sizes
  step_sizes <- rep(0.1, p)

  # Store samples
  samples <- matrix(NA, nrow = n_iter, ncol = p)
  colnames(samples) <- colnames(X)

  # Track acceptance rates
  accept_count <- rep(0, p)

  cat("Running MCMC...\n")
  for (iter in 1:n_iter) {
    # Update each parameter sequentially
    for (j in 1:p) {
      result <- mh_step(beta, X, time, status, j, step_sizes, sigma_prior)
      beta <- result$beta
      if (result$accepted) {
        accept_count[j] <- accept_count[j] + 1
      }
    }

    # Store sample
    samples[iter, ] <- beta

    # Adaptive tuning (during adapt_period)
    if (iter < adapt_period) {
      for (j in 1:p) {
        accept_rate <- accept_count[j] / iter
        if (accept_rate < target_accept) {
          step_sizes[j] <- step_sizes[j] * 0.9
        } else if (accept_rate > target_accept) {
          step_sizes[j] <- step_sizes[j] * 1.1
        }
      }
    }

    # Progress report
    if (iter %% 1000 == 0) {
      overall_accept <- sum(accept_count) / (iter * p)
      cat(sprintf("Iteration %d/%d, Acceptance rate: %.3f\n",
                  iter, n_iter, overall_accept))
    }
  }

  cat(sprintf("MCMC complete. Final acceptance rate: %.3f\n",
              sum(accept_count) / (n_iter * p)))

  return(list(
    samples = samples[(burnin+1):n_iter, , drop = FALSE],
    acceptance_rate = accept_count / n_iter,
    step_sizes = step_sizes,
    burnin = burnin
  ))
}

# Run multiple chains for convergence diagnostics
run_multiple_chains <- function(X, time, status,
                                 n_chains = 3,
                                 n_iter = 10000,
                                 burnin = 1000,
                                 seeds = c(42, 123, 456)) {

  chains <- vector("list", n_chains)

  cat(paste0("Running ", n_chains, " chains...\n"))
  for (i in 1:n_chains) {
    cat(sprintf("Chain %d/%d\n", i, n_chains))
    chains[[i]] <- adaptive_mcmc(X, time, status,
                                  n_iter = n_iter,
                                  burnin = burnin,
                                  seed = seeds[i])
  }

  return(chains)
}

# =============================================================================
# Convergence Diagnostics
# =============================================================================

# Gelman-Rubin R-hat statistic
gelman_rubin <- function(chains) {
  n_chains <- length(chains)
  p <- ncol(chains[[1]]$samples)
  
  rhat <- numeric(p)
  
  for (j in 1:p) {
    chain_samples <- sapply(chains, function(chain) chain$samples[, j])
    
    n_iter <- nrow(chain_samples)
    
    # within-chain variance
    W <- mean(apply(chain_samples, 2, var))
    
    # between-chain variance
    chain_means <- colMeans(chain_samples)
    B <- var(chain_means) * n_iter
    
    # posterior variance estimate
    V <- ((n_iter - 1) * W + B) / n_iter
    
    rhat[j] <- sqrt(V / W)
  }
  
  return(rhat)
}

# Effective Sample Size
effective_sample_size <- function(chains) {
  # Combine chains
  n_chains <- length(chains)
  p <- ncol(chains[[1]]$samples)

  ess <- numeric(p)

  for (j in 1:p) {
    combined <- unlist(sapply(chains, function(chain) chain$samples[, j]))

    # Autocorrelation function
    acf_vals <- acf(combined, plot = FALSE, lag.max = min(1000, length(combined) - 1))$acf

    # Sum of autocorrelations
    sum_acf <- sum(acf_vals)
    if (sum_acf < 1) sum_acf <- 1

    ess[j] <- length(combined) / sum_acf
  }

  return(ess)
}

# =============================================================================
# Posterior Summary
# =============================================================================

summarize_posterior <- function(samples, ci_width = 0.95) {
  summary_df <- data.frame(
    feature = colnames(samples),
    mean = colMeans(samples),
    median = apply(samples, 2, median),
    sd = apply(samples, 2, sd),
    q025 = apply(samples, 2, quantile, probs = 0.025),
    q975 = apply(samples, 2, quantile, probs = 0.975)
  )

  # Hazard ratios
  summary_df$hr_mean <- exp(summary_df$mean)
  summary_df$hr_q025 <- exp(summary_df$q025)
  summary_df$hr_q975 <- exp(summary_df$q975)

  # Posterior probability of significance (compute manually to avoid dplyr masking)
  n_samples <- nrow(samples)
  prob_positive <- colMeans(samples > 0)
  prob_negative <- colMeans(samples < 0)
  summary_df$prob_positive <- prob_positive
  summary_df$prob_negative <- prob_negative

  # Significance flag (CI doesn't contain 0 for beta, or 1 for HR)
  summary_df$significant <- (summary_df$q025 > 0) | (summary_df$q975 < 0)

  return(summary_df)
}

# =============================================================================
# Visualization
# =============================================================================

# Trace plot for a single parameter
plot_trace <- function(chains, feature_idx, save_path = "../figures/cox_trace_plots.png") {
  n_chains <- length(chains)
  feature_name <- colnames(chains[[1]]$samples)[feature_idx]

  df <- data.frame()
  for (i in 1:n_chains) {
    chain_df <- data.frame(
      iteration = 1:nrow(chains[[i]]$samples),
      value = chains[[i]]$samples[, feature_idx],
      chain = as.factor(i)
    )
    df <- rbind(df, chain_df)
  }

  p <- ggplot(df, aes(x = iteration, y = value, color = chain)) +
    geom_line(alpha = 0.7, linewidth = 0.3) +
    labs(
      title = paste("Trace Plot:", feature_name),
      x = "Iteration",
      y = expression(beta),
      color = "Chain"
    ) +
    theme_minimal()

  ggsave(save_path, p, width = 10, height = 4, dpi = 150)
  cat(paste0("Trace plot saved to: ", save_path, "\n"))
}

# Forest plot of hazard ratios
plot_forest <- function(summary_df, top_n = 20, save_path = "../figures/cox_forest_plot.png") {
  # Sort by absolute mean beta
  df <- summary_df %>%
    arrange(desc(abs(mean))) %>%
    head(top_n) %>%
    mutate(feature = reorder(feature, hr_mean))

  p <- ggplot(df, aes(x = hr_mean, y = feature)) +
    geom_point(size = 2) +
    geom_errorbarh(aes(xmin = hr_q025, xmax = hr_q975), height = 0.2, linewidth = 0.5) +
    geom_vline(xintercept = 1, linetype = "dashed", color = "red") +
    scale_x_log10() +
    labs(
      title = "Hazard Ratios with 95% Credible Intervals",
      subtitle = paste0("Top ", top_n, " features by absolute coefficient magnitude"),
      x = "Hazard Ratio (HR)",
      y = "Feature"
    ) +
    theme_minimal() +
    theme(
      axis.text.y = element_text(size = 8)
    )

  ggsave(save_path, p, width = 8, height = max(6, top_n * 0.25), dpi = 150)
  cat(paste0("Forest plot saved to: ", save_path, "\n"))
}

# =============================================================================
# Main Analysis Function
# =============================================================================

run_analysis <- function(csv_path = "../data/brca_multimodal.csv",
                         n_features = 100,
                         n_iter = 10000,
                         burnin = 1000,
                         n_chains = 3) {
  
  dir.create("../results", showWarnings = FALSE)
  dir.create("../figures", showWarnings = FALSE)
  
  cat("=====================================\n")
  cat("Bayesian Cox Proportional Hazards Model\n")
  cat("Manual MCMC Implementation\n")
  cat("=====================================\n\n")

  # Load data
  cat("Loading data...\n")
  data <- load_survival_data(csv_path, n_features)
  cat(sprintf("n = %d samples, p = %d features\n\n", data$n, data$p))

  # Run multiple chains
  chains <- run_multiple_chains(data$X, data$time, data$status,
                                n_chains = n_chains,
                                n_iter = n_iter,
                                burnin = burnin)

  # Convergence diagnostics
  cat("\n=====================================\n")
  cat("Convergence Diagnostics\n")
  cat("=====================================\n")

  rhat <- gelman_rubin(chains)
  ess <- effective_sample_size(chains)

  conv_df <- data.frame(
    feature = colnames(chains[[1]]$samples),
    rhat = rhat,
    ess = ess
  )

  converged <- sum(rhat < 1.1 & ess > 100)
  cat(sprintf("Parameters with R-hat < 1.1 and ESS > 100: %d/%d\n", converged, length(rhat)))
  cat(sprintf("Mean R-hat: %.3f (target < 1.1)\n", mean(rhat)))
  cat(sprintf("Mean ESS: %.0f\n", mean(ess)))

  # Combine chains for final analysis
  combined_samples <- do.call(rbind, lapply(chains, function(c) c$samples))

  # Posterior summary
  cat("\n=====================================\n")
  cat("Posterior Summary\n")
  cat("=====================================\n")

  summary_df <- summarize_posterior(combined_samples)

  significant_biomarkers <- summary_df %>%
    filter(significant) %>%
    arrange(desc(abs(mean)))

  cat(sprintf("\nSignificant biomarkers (95%% CI excludes 0): %d\n",
              nrow(significant_biomarkers)))
  print(head(significant_biomarkers, 10))

  # Visualizations
  cat("\n=====================================\n")
  cat("Generating Visualizations\n")
  cat("=====================================\n")

  # Trace plot for top feature
  top_feature_idx <- which(colnames(chains[[1]]$samples) ==
                           significant_biomarkers$feature[1])
  plot_trace(chains, top_feature_idx)

  # Forest plot
  plot_forest(summary_df, top_n = 20)

  # Save results
  results_path <- "/Users/chenyanzhen/Documents/DS4420_final_project-2/results/cox_coefficients.csv"
  summary_df %>%
    arrange(desc(abs(mean))) %>%
    write.csv(results_path, row.names = FALSE)

  biomarkers_path <- "/Users/chenyanzhen/Documents/DS4420_final_project-2/results/biomarkers.csv"
  significant_biomarkers %>%
    write.csv(biomarkers_path, row.names = FALSE)

  conv_path <- "/Users/chenyanzhen/Documents/DS4420_final_project-2/results/cox_convergence.csv"
  conv_df %>%
    write.csv(conv_path, row.names = FALSE)

  cat("\n=====================================\n")
  cat("Results Saved\n")
  cat("=====================================\n")
  cat(paste0("- Coefficient summary: ", results_path, "\n"))
  cat(paste0("- Significant biomarkers: ", biomarkers_path, "\n"))
  cat(paste0("- Convergence diagnostics: ", conv_path, "\n"))

  return(list(
    summary = summary_df,
    biomarkers = significant_biomarkers,
    convergence = conv_df,
    chains = chains
  ))
}

# =============================================================================
# Main Execution
# =============================================================================

# Run the analysis
results <- run_analysis(csv_path = "/Users/chenyanzhen/Documents/DS4420_final_project-2/data/brca_multimodal.csv",
                        n_features = 100,
                        n_iter = 5000,
                        burnin = 1000,
                        n_chains = 3)

