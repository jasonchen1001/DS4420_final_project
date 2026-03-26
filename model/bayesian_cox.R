library(dplyr)
library(ggplot2)
library(survival)

# Project paths
PROJECT_ROOT <- "/Users/chenyanzhen/Documents/DS4420_final_project-2"
DATA_DIR <- file.path(PROJECT_ROOT, "data")
RESULTS_DIR <- file.path(PROJECT_ROOT, "results")
FIGURES_DIR <- file.path(PROJECT_ROOT, "figures")

dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(FIGURES_DIR, showWarnings = FALSE, recursive = TRUE)

# Helper: identify omics blocks
identify_feature_blocks <- function(feature_names) {
  clinical_cols <- c(
    "ER.Status",
    "PR.Status",
    "HER2.Final.Status",
    "age_at_initial_pathologic_diagnosis"
  )

  clinical <- intersect(feature_names, clinical_cols)

  protein <- feature_names[
    grepl("^[A-Z0-9]+$", feature_names) &
      !(feature_names %in% clinical)
  ]

  gene <- setdiff(feature_names, c(protein, clinical))

  list(
    gene = gene,
    protein = protein,
    clinical = clinical
  )
}

# Load preprocessed survival data from Python pipeline
load_survival_data <- function(
    train_csv = file.path(DATA_DIR, "cox_train.csv"),
    test_csv = file.path(DATA_DIR, "cox_test.csv")) {

  train_df <- read.csv(train_csv, check.names = FALSE)
  test_df  <- read.csv(test_csv, check.names = FALSE)

  # survival variables
  train_time <- train_df$OS.time
  train_status <- train_df$OS

  test_time <- test_df$OS.time
  test_status <- test_df$OS

  # feature matrices
  X_train <- train_df %>%
    select(-OS, -OS.time) %>%
    as.matrix()

  X_test <- test_df %>%
    select(-OS, -OS.time) %>%
    as.matrix()

  storage.mode(X_train) <- "double"
  storage.mode(X_test) <- "double"

  # Safety cleanup: guard against NA/Inf from mixed-type CSV columns
  bad_train <- !is.finite(X_train)
  bad_test <- !is.finite(X_test)
  n_bad <- sum(bad_train) + sum(bad_test)
  if (n_bad > 0) {
    X_train[bad_train] <- 0
    X_test[bad_test] <- 0
    cat(sprintf("Warning: replaced %d non-finite feature values with 0\n", n_bad))
  }

  feature_names <- colnames(X_train)
  blocks <- identify_feature_blocks(feature_names)

  cat("Loaded processed survival data\n")
  cat(sprintf("Train: n = %d, p = %d\n", nrow(X_train), ncol(X_train)))
  cat(sprintf("Test : n = %d, p = %d\n", nrow(X_test), ncol(X_test)))
  cat(sprintf("Gene: %d | Protein: %d | Clinical: %d\n",
              length(blocks$gene), length(blocks$protein), length(blocks$clinical)))

  list(
    X_train = X_train,
    time_train = train_time,
    status_train = train_status,
    X_test = X_test,
    time_test = test_time,
    status_test = test_status,
    feature_names = feature_names,
    blocks = blocks,
    n_train = nrow(X_train),
    n_test = nrow(X_test),
    p = ncol(X_train)
  )
}

# Cox partial log-likelihood (Breslow for ties)
log_partial_likelihood <- function(beta, X, time, status) {
  eta <- as.vector(X %*% beta)
  
  event_times <- sort(unique(time[status == 1]))
  log_lik <- 0
  
  for (t in event_times) {
    at_risk <- which(time >= t)
    events <- which(time == t & status == 1)
    
    if (length(events) == 0) next
    
    eta_risk <- eta[at_risk]
    
    max_eta <- max(eta_risk)
    risk_sum <- sum(exp(eta_risk - max_eta))
    
    if (!is.finite(risk_sum) || risk_sum <= 0) {
      return(-Inf)
    }
    
    log_risk <- log(risk_sum) + max_eta
    
    log_lik <- log_lik + sum(eta[events]) - length(events) * log_risk
  }
  
  log_lik
}

# Block-specific prior
log_prior_block <- function(beta, feature_names,
                            sigma_gene = 1.0,
                            sigma_protein = 1.0,
                            sigma_clinical = 2.0) {

  blocks <- identify_feature_blocks(feature_names)

  lp <- 0

  if (length(blocks$gene) > 0) {
    idx <- match(blocks$gene, feature_names)
    lp <- lp + sum(dnorm(beta[idx], mean = 0, sd = sigma_gene, log = TRUE))
  }

  if (length(blocks$protein) > 0) {
    idx <- match(blocks$protein, feature_names)
    lp <- lp + sum(dnorm(beta[idx], mean = 0, sd = sigma_protein, log = TRUE))
  }

  if (length(blocks$clinical) > 0) {
    idx <- match(blocks$clinical, feature_names)
    lp <- lp + sum(dnorm(beta[idx], mean = 0, sd = sigma_clinical, log = TRUE))
  }

  lp
}

log_posterior <- function(beta, X, time, status, feature_names,
                          sigma_gene = 1.0,
                          sigma_protein = 1.0,
                          sigma_clinical = 2.0) {
  log_partial_likelihood(beta, X, time, status) +
    log_prior_block(beta, feature_names,
                    sigma_gene = sigma_gene,
                    sigma_protein = sigma_protein,
                    sigma_clinical = sigma_clinical)
}

# Single MH step (coordinate-wise)
mh_step <- function(beta, current_log_post,
                    X, time, status, feature_names,
                    idx, step_sizes,
                    sigma_gene = 1.0,
                    sigma_protein = 1.0,
                    sigma_clinical = 1.0) {
  
  beta_prop <- beta
  beta_prop[idx] <- rnorm(1, mean = beta[idx], sd = step_sizes[idx])
  
  if (any(abs(beta_prop) > 20)) {
    return(list(beta = beta, log_post = current_log_post, accepted = FALSE))
  }
  
  prop_log_post <- log_posterior(
    beta_prop, X, time, status, feature_names,
    sigma_gene = sigma_gene,
    sigma_protein = sigma_protein,
    sigma_clinical = sigma_clinical
  )
  
  if (!is.finite(prop_log_post)) {
    return(list(beta = beta, log_post = current_log_post, accepted = FALSE))
  }
  
  log_alpha <- min(0, prop_log_post - current_log_post)
  
  if (!is.finite(log_alpha)) {
    return(list(beta = beta, log_post = current_log_post, accepted = FALSE))
  }
  
  if (log(runif(1)) < log_alpha) {
    list(beta = beta_prop, log_post = prop_log_post, accepted = TRUE)
  } else {
    list(beta = beta, log_post = current_log_post, accepted = FALSE)
  }
}

# Adaptive MCMC
adaptive_mcmc <- function(X, time, status, feature_names,
                          n_iter = 5000,
                          burnin = 1000,
                          sigma_gene = 1.0,
                          sigma_protein = 1.0,
                          sigma_clinical = 2.0,
                          init_step_size = 0.001,
                          target_accept = 0.234,
                          adapt_period = 1000,
                          seed = 42) {

  set.seed(seed)

  p <- ncol(X)
  beta <- rep(0, p)
  step_sizes <- rep(init_step_size, p)

  samples <- matrix(NA_real_, nrow = n_iter, ncol = p)
  colnames(samples) <- feature_names

  accept_count <- rep(0, p)

  current_log_post <- log_posterior(
    beta, X, time, status, feature_names,
    sigma_gene = sigma_gene,
    sigma_protein = sigma_protein,
    sigma_clinical = sigma_clinical
  )

  cat("Running MCMC...\n")

  for (iter in 1:n_iter) {
    for (j in 1:p) {
      step_result <- mh_step(
        beta = beta,
        current_log_post = current_log_post,
        X = X,
        time = time,
        status = status,
        feature_names = feature_names,
        idx = j,
        step_sizes = step_sizes,
        sigma_gene = sigma_gene,
        sigma_protein = sigma_protein,
        sigma_clinical = sigma_clinical
      )

      beta <- step_result$beta
      current_log_post <- step_result$log_post

      if (step_result$accepted) {
        accept_count[j] <- accept_count[j] + 1
      }
    }

    samples[iter, ] <- beta

    # adapt step sizes during early iterations
    if (iter < adapt_period) {
      for (j in 1:p) {
        acc_rate <- accept_count[j] / iter
        if (acc_rate < target_accept) {
          step_sizes[j] <- step_sizes[j] * 0.9
        } else if (acc_rate > target_accept) {
          step_sizes[j] <- step_sizes[j] * 1.1
        }
      }
    }

    if (iter %% 100 == 0) {
      overall_accept <- sum(accept_count) / (iter * p)
      cat(sprintf("Iteration %d/%d | acceptance = %.6f\n",
                  iter, n_iter, overall_accept))
    }
  }

  final_accept <- sum(accept_count) / (n_iter * p)
  cat(sprintf("MCMC complete. Final acceptance rate: %.6f\n", final_accept))

  list(
    samples = samples[(burnin + 1):n_iter, , drop = FALSE],
    acceptance_rate = accept_count / n_iter,
    final_acceptance = final_accept,
    step_sizes = step_sizes,
    burnin = burnin
  )
}

# Run multiple chains
run_multiple_chains <- function(X, time, status, feature_names,
                                n_chains = 3,
                                n_iter = 5000,
                                burnin = 1000,
                                seeds = c(42, 123, 456),
                                sigma_gene = 1.0,
                                sigma_protein = 1.0,
                                sigma_clinical = 2.0,
                                init_step_size = 0.001) {

  chains <- vector("list", n_chains)

  cat(sprintf("Running %d chains...\n", n_chains))

  if (length(seeds) < n_chains) {
    set.seed(999)
    seeds <- c(seeds, sample.int(1e6, n_chains - length(seeds)))
  }

  for (i in 1:n_chains) {
    cat(sprintf("Chain %d/%d\n", i, n_chains))
    chains[[i]] <- adaptive_mcmc(
      X = X,
      time = time,
      status = status,
      feature_names = feature_names,
      n_iter = n_iter,
      burnin = burnin,
      seed = seeds[i],
      sigma_gene = sigma_gene,
      sigma_protein = sigma_protein,
      sigma_clinical = sigma_clinical,
      init_step_size = init_step_size
    )

    ar <- chains[[i]]$acceptance_rate
    cat(sprintf(
      "Chain %d parameter-level acceptance: min=%.6f | median=%.6f | max=%.6f\n",
      i, min(ar), median(ar), max(ar)
    ))
  }

  chains
}

# Gelman-Rubin R-hat
gelman_rubin <- function(chains) {
  n_chains <- length(chains)
  p <- ncol(chains[[1]]$samples)
  rhat <- numeric(p)

  for (j in 1:p) {
    chain_samples <- sapply(chains, function(ch) ch$samples[, j])
    n_iter <- nrow(chain_samples)

    W <- mean(apply(chain_samples, 2, var))
    chain_means <- colMeans(chain_samples)
    B <- var(chain_means) * n_iter
    V <- ((n_iter - 1) * W + B) / n_iter

    rhat[j] <- sqrt(V / W)
  }

  rhat
}

# Effective Sample Size
effective_sample_size <- function(chains) {
  p <- ncol(chains[[1]]$samples)
  ess <- numeric(p)

  for (j in 1:p) {
    combined <- unlist(lapply(chains, function(ch) ch$samples[, j]))

    acf_vals <- acf(combined,
                    plot = FALSE,
                    lag.max = min(500, length(combined) - 1))$acf

    sum_acf <- sum(acf_vals)
    if (!is.finite(sum_acf) || sum_acf < 1) sum_acf <- 1

    ess[j] <- length(combined) / sum_acf
  }

  ess
}

# Posterior summary
summarize_posterior <- function(samples) {
  summary_df <- data.frame(
    feature = colnames(samples),
    mean = colMeans(samples),
    median = apply(samples, 2, median),
    sd = apply(samples, 2, sd),
    q025 = apply(samples, 2, quantile, probs = 0.025),
    q975 = apply(samples, 2, quantile, probs = 0.975)
  )

  summary_df$hr_mean <- exp(summary_df$mean)
  summary_df$hr_q025 <- exp(summary_df$q025)
  summary_df$hr_q975 <- exp(summary_df$q975)

  summary_df$prob_positive <- colMeans(samples > 0)
  summary_df$prob_negative <- colMeans(samples < 0)

  summary_df$significant <- (summary_df$q025 > 0) | (summary_df$q975 < 0)

  summary_df
}

# C-index on test set
compute_test_cindex <- function(beta, X_test, time_test, status_test) {
  risk_score <- as.vector(X_test %*% beta)

  cobj <- survival::concordance(
    survival::Surv(time_test, status_test) ~ risk_score,
    reverse = TRUE
  )

  list(
    c_index = unname(cobj$concordance),
    se = unname(sqrt(cobj$var))
  )
}

# Plots
plot_trace <- function(chains, feature_idx,
                       save_path = file.path(FIGURES_DIR, "cox_trace_plot.png")) {

  feature_name <- colnames(chains[[1]]$samples)[feature_idx]

  plot_df <- bind_rows(lapply(seq_along(chains), function(i) {
    data.frame(
      iteration = 1:nrow(chains[[i]]$samples),
      value = chains[[i]]$samples[, feature_idx],
      chain = factor(i)
    )
  }))

  p <- ggplot(plot_df, aes(x = iteration, y = value, color = chain)) +
    geom_line(alpha = 0.7, linewidth = 0.3) +
    labs(
      title = paste("Trace Plot:", feature_name),
      x = "Iteration",
      y = expression(beta),
      color = "Chain"
    ) +
    theme_minimal()

  ggsave(save_path, p, width = 10, height = 4, dpi = 150)
  cat("Trace plot saved to:", save_path, "\n")
}

plot_forest <- function(summary_df, top_n = 20,
                        save_path = file.path(FIGURES_DIR, "cox_forest_plot.png")) {

  df <- summary_df %>%
    arrange(desc(abs(mean))) %>%
    slice_head(n = top_n) %>%
    mutate(feature = reorder(feature, hr_mean))

  p <- ggplot(df, aes(x = hr_mean, y = feature)) +
    geom_point(size = 2) +
    geom_errorbarh(aes(xmin = hr_q025, xmax = hr_q975), height = 0.2, linewidth = 0.5) +
    geom_vline(xintercept = 1, linetype = "dashed", color = "red") +
    scale_x_log10() +
    labs(
      title = "Hazard Ratios with 95% Credible Intervals",
      subtitle = paste("Top", top_n, "features by |posterior mean|"),
      x = "Hazard Ratio (log scale)",
      y = "Feature"
    ) +
    theme_minimal() +
    theme(axis.text.y = element_text(size = 8))

  ggsave(save_path, p, width = 8, height = max(6, top_n * 0.25), dpi = 150)
  cat("Forest plot saved to:", save_path, "\n")
}

# Full analysis
run_analysis <- function(
    train_csv = file.path(DATA_DIR, "cox_train.csv"),
    test_csv = file.path(DATA_DIR, "cox_test.csv"),
    n_iter = 5000,
    burnin = 1000,
    n_chains = 3,
    sigma_gene = 1.0,
    sigma_protein = 1.0,
    sigma_clinical = 2.0,
    init_step_size = 0.001) {

  cat("Loading processed data...\n")
  data <- load_survival_data(train_csv, test_csv)

  cat("\nRunning Bayesian Cox model...\n")
  chains <- run_multiple_chains(
    X = data$X_train,
    time = data$time_train,
    status = data$status_train,
    feature_names = data$feature_names,
    n_chains = n_chains,
    n_iter = n_iter,
    burnin = burnin,
    sigma_gene = sigma_gene,
    sigma_protein = sigma_protein,
    sigma_clinical = sigma_clinical,
    init_step_size = init_step_size
  )

  cat("\nConvergence diagnostics...\n")
  rhat <- gelman_rubin(chains)
  ess <- effective_sample_size(chains)

  conv_df <- data.frame(
    feature = data$feature_names,
    rhat = rhat,
    ess = ess
  )

  converged <- sum(rhat < 1.1 & ess > 100)
  cat(sprintf("Parameters with R-hat < 1.1 and ESS > 100: %d/%d\n",
              converged, length(rhat)))
  cat(sprintf("Mean R-hat: %.3f\n", mean(rhat)))
  cat(sprintf("Mean ESS: %.1f\n", mean(ess)))

  combined_samples <- do.call(rbind, lapply(chains, function(ch) ch$samples))
  summary_df <- summarize_posterior(combined_samples)

  significant_biomarkers <- summary_df %>%
    filter(significant) %>%
    arrange(desc(abs(mean)))

  cat(sprintf("\nSignificant biomarkers (95%% CI excludes 0): %d\n",
              nrow(significant_biomarkers)))

  if (nrow(significant_biomarkers) > 0) {
    print(head(significant_biomarkers, 10))
  }

  # posterior mean beta for prediction
  beta_mean <- colMeans(combined_samples)

  test_perf <- compute_test_cindex(
    beta = beta_mean,
    X_test = data$X_test,
    time_test = data$time_test,
    status_test = data$status_test
  )

  cat(sprintf("\nTest C-index: %.3f (SE = %.3f)\n",
              test_perf$c_index, test_perf$se))

  # plots
  if (nrow(significant_biomarkers) > 0) {
    top_feature_name <- significant_biomarkers$feature[1]
    top_feature_idx <- which(colnames(chains[[1]]$samples) == top_feature_name)
    plot_trace(chains, top_feature_idx)
  }

  plot_forest(summary_df, top_n = 20)

  # save outputs
  write.csv(
    summary_df %>% arrange(desc(abs(mean))),
    file.path(RESULTS_DIR, "cox_coefficients.csv"),
    row.names = FALSE
  )

  write.csv(
    significant_biomarkers,
    file.path(RESULTS_DIR, "biomarkers.csv"),
    row.names = FALSE
  )

  write.csv(
    conv_df,
    file.path(RESULTS_DIR, "cox_convergence.csv"),
    row.names = FALSE
  )

  write.csv(
    data.frame(
      test_c_index = test_perf$c_index,
      test_c_index_se = test_perf$se
    ),
    file.path(RESULTS_DIR, "cox_test_performance.csv"),
    row.names = FALSE
  )

  cat("\nResults saved to:\n")
  cat(file.path(RESULTS_DIR, "cox_coefficients.csv"), "\n")
  cat(file.path(RESULTS_DIR, "biomarkers.csv"), "\n")
  cat(file.path(RESULTS_DIR, "cox_convergence.csv"), "\n")
  cat(file.path(RESULTS_DIR, "cox_test_performance.csv"), "\n")

  list(
    summary = summary_df,
    biomarkers = significant_biomarkers,
    convergence = conv_df,
    test_performance = test_perf,
    chains = chains,
    data = data
  )
}

# Run
results <- run_analysis(
  train_csv = file.path(DATA_DIR, "cox_train.csv"),
  test_csv = file.path(DATA_DIR, "cox_test.csv"),
  n_iter = 2000,
  burnin = 500,
  n_chains = 2,
  sigma_gene = 1.0,
  sigma_protein = 1.0,
  sigma_clinical = 2.0,
  init_step_size = 0.001
)