# Cox Survival Analysis Pipeline (Streamlined)
# Flow: KM -> Univariate -> P-filter -> Penalized Cox -> Bayesian Cox -> Validation -> Stability

library(dplyr)
library(ggplot2)
library(survival)
library(survminer)
library(glmnet)
library(plotmo)

PROJECT_ROOT <- "/Users/chenyanzhen/Documents/DS4420_final_project-2"
DATA_DIR     <- file.path(PROJECT_ROOT, "data")
RESULTS_DIR  <- file.path(PROJECT_ROOT, "results", "cox")
FIGURES_DIR  <- file.path(PROJECT_ROOT, "figures", "cox")
dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(FIGURES_DIR, showWarnings = FALSE, recursive = TRUE)


# ---- Load data ----
load_data <- function() {
  train <- read.csv(file.path(DATA_DIR, "train.csv"), check.names = FALSE)
  test  <- read.csv(file.path(DATA_DIR, "test.csv"),  check.names = FALSE)
  feat  <- read.csv(file.path(DATA_DIR, "features.csv"))
  
  features       <- feat$feature
  clinical_feats <- feat$feature[feat$is_clinical == TRUE]
  
  X_train <- as.matrix(train[, features]); storage.mode(X_train) <- "double"
  X_test  <- as.matrix(test[, features]);  storage.mode(X_test)  <- "double"
  X_train[!is.finite(X_train)] <- 0
  X_test[!is.finite(X_test)]   <- 0
  
  train$time_years <- train$OS.time / 365.25
  test$time_years  <- test$OS.time / 365.25
  
  cat(sprintf("Train: %d samples, %d features, %d events\n",
              nrow(X_train), ncol(X_train), sum(train$OS)))
  cat(sprintf("Test:  %d samples, %d features, %d events\n",
              nrow(X_test), ncol(X_test), sum(test$OS)))
  
  list(train=train, test=test, X_train=X_train, X_test=X_test,
       features=features, clinical_feats=clinical_feats)
}


# ---- Step 1: Kaplan-Meier ----
run_km <- function(data) {
  cat("\n[Step 1] Kaplan-Meier\n")
  train <- data$train
  assign("train", train, envir = .GlobalEnv)
  on.exit(rm("train", envir = .GlobalEnv), add = TRUE)
  
  sfit <- survfit(Surv(time_years, OS) ~ 1, data = train)
  cat("Survival at 1, 3, 5 years:\n")
  print(summary(sfit, times = c(1, 3, 5)))
  
  ggsurv <- ggsurvplot(sfit, data=train, conf.int=TRUE, risk.table=TRUE,
                       xlab="Time (years)", legend="none", surv.median.line="hv")
  pdf(file.path(FIGURES_DIR, "km_overall.pdf"), width=8, height=6)
  print(ggsurv)
  dev.off()
  cat("Saved: km_overall.pdf\n")
}


# ---- Step 2: Univariate Cox ----
run_univariate <- function(data) {
  cat("\n[Step 2] Univariate Cox screening\n")
  train <- data$train
  
  # Age: PH assumption check
  fit_age <- coxph(Surv(time_years, OS) ~ age_at_initial_pathologic_diagnosis, data=train)
  cat("Age Cox:\n"); print(summary(fit_age))
  zph <- cox.zph(fit_age, transform="log")
  cat("PH check:\n"); print(zph)
  
  # Screen all features
  results <- list()
  for (feat in data$features) {
    tryCatch({
      tmp <- data.frame(T=train$time_years, E=train$OS, x=data$X_train[,feat])
      s <- summary(coxph(Surv(T, E) ~ x, data=tmp))
      results[[feat]] <- data.frame(feature=feat, coef=s$coefficients[,"coef"],
                                    HR=s$coefficients[,"exp(coef)"], p_value=s$coefficients[,"Pr(>|z|)"])
    }, error=function(e) NULL)
  }
  uni_df <- bind_rows(results) %>% arrange(p_value)
  cat(sprintf("Tested: %d | p<0.05: %d | p<0.1: %d\n",
              nrow(uni_df), sum(uni_df$p_value<0.05), sum(uni_df$p_value<0.1)))
  uni_df
}


# ---- Step 3: P-value filter ----
pvalue_filter <- function(uni_df, data, alpha=0.1) {
  cat(sprintf("\n[Step 3] P-value filter (alpha=%g)\n", alpha))
  selected <- union(uni_df$feature[uni_df$p_value < alpha], data$clinical_feats)
  cat(sprintf("Selected: %d features\n", length(selected)))
  selected
}


# ---- Step 4: Penalized Cox ----
run_penalized_cox <- function(data, candidate_feats) {
  cat("\n[Step 4] Penalized Cox models\n")
  x <- data$X_train[, candidate_feats]
  y <- Surv(data$train$time_years, data$train$OS)
  pf <- rep(1, length(candidate_feats))
  pf[candidate_feats %in% data$clinical_feats] <- 0
  
  # Lasso
  set.seed(123)
  cvfit_lasso <- cv.glmnet(x, y, family="cox", alpha=1, nfolds=5, penalty.factor=pf)
  lambda_opt <- cvfit_lasso$lambda.1se
  coefs <- as.vector(coef(cvfit_lasso, s="lambda.1se"))
  names(coefs) <- candidate_feats
  selected <- names(coefs[abs(coefs) > 1e-6])
  cat(sprintf("Lasso (lambda.1se=%.4f): %d features\n", lambda_opt, length(selected)))
  
  if (length(selected) < 2) {
    cat("lambda.1se too strict, using lambda.min\n")
    lambda_opt <- cvfit_lasso$lambda.min
    coefs <- as.vector(coef(cvfit_lasso, s="lambda.min"))
    names(coefs) <- candidate_feats
    selected <- names(coefs[abs(coefs) > 1e-6])
    cat(sprintf("Lasso (lambda.min=%.4f): %d features\n", lambda_opt, length(selected)))
  }
  
  # Elastic Net
  alpha_seq <- seq(0.1, 1, length=10)
  fitEN <- list()
  set.seed(123)
  for (i in seq_along(alpha_seq))
    fitEN[[i]] <- cv.glmnet(x, y, family="cox", alpha=alpha_seq[i], nfolds=5, penalty.factor=pf)
  best_idx <- which.min(sapply(fitEN, function(xx) xx$cvm[xx$lambda == xx$lambda.min]))
  coefs_en <- as.vector(coef(fitEN[[best_idx]], s="lambda.1se"))
  names(coefs_en) <- candidate_feats
  selected_en <- names(coefs_en[abs(coefs_en) > 1e-6])
  cat(sprintf("Elastic Net (alpha=%.1f): %d features\n", alpha_seq[best_idx], length(selected_en)))
  
  # Adaptive Lasso
  set.seed(123)
  fit_ridge <- cv.glmnet(x, y, family="cox", alpha=0, nfolds=5)
  weights <- abs(1 / as.vector(coef(fit_ridge, s="lambda.min")))
  weights[candidate_feats %in% data$clinical_feats] <- 0
  cvfit_al <- cv.glmnet(x, y, family="cox", nfolds=5, penalty.factor=weights)
  coefs_al <- as.vector(coef(cvfit_al, s="lambda.1se"))
  names(coefs_al) <- candidate_feats
  selected_al <- names(coefs_al[abs(coefs_al) > 1e-6])
  cat(sprintf("Adaptive Lasso: %d features\n", length(selected_al)))
  
  overlap <- Reduce(intersect, list(selected, selected_en, selected_al))
  cat(sprintf("Intersection (all 3): %s\n", paste(overlap, collapse=", ")))
  
  # Lasso coefficient path plot
  mod <- cvfit_lasso$glmnet.fit
  pdf(file.path(FIGURES_DIR, "lasso_path.pdf"), width=8, height=6)
  plot_glmnet(mod, label=TRUE, s=lambda_opt,
              xlab=expression(log~~lambda), ylab=expression(beta))
  title("Lasso Cox\n\n")
  dev.off()
  cat("Saved: lasso_path.pdf\n")
  
  list(lasso=list(cvfit=cvfit_lasso, selected=selected, coefs=coefs),
       en_selected=selected_en, alasso_selected=selected_al,
       candidate_feats=candidate_feats, pf=pf, lambda_opt=lambda_opt)
}


# ---- Step 5: Bayesian Cox (Laplace prior, MCMC) ----
log_partial_likelihood <- function(beta, X, time, status) {
  eta <- as.vector(X %*% beta)
  ll <- 0
  for (t_k in sort(unique(time[status==1]))) {
    at_risk <- which(time >= t_k)
    events  <- which(time == t_k & status == 1)
    if (length(events)==0) next
    eta_r <- eta[at_risk]; mx <- max(eta_r)
    lr <- log(sum(exp(eta_r - mx))) + mx
    if (!is.finite(lr)) return(-Inf)
    ll <- ll + sum(eta[events]) - length(events)*lr
  }
  ll
}

log_posterior <- function(beta, X, time, status, feat_names, clin_feats,
                          lambda=1.0, sigma_clin=5.0) {
  lp <- 0
  for (j in seq_along(beta)) {
    if (feat_names[j] %in% clin_feats)
      lp <- lp + dnorm(beta[j], 0, sigma_clin, log=TRUE)
    else
      lp <- lp + log(lambda/2) - lambda*abs(beta[j])
  }
  log_partial_likelihood(beta, X, time, status) + lp
}

run_bayesian_cox <- function(data, selected_feats,
                             n_iter=10000, burnin=5000, n_chains=3) {
  cat(sprintf("\n[Step 5] Bayesian Cox MCMC (%d features, %d iter, %d chains)\n",
              length(selected_feats), n_iter, n_chains))
  
  X <- data$X_train[, match(selected_feats, data$features), drop=FALSE]
  time <- data$train$time_years; status <- data$train$OS
  p <- length(selected_feats)
  chains <- vector("list", n_chains)
  
  for (ch in 1:n_chains) {
    cat(sprintf("  Chain %d/%d\n", ch, n_chains))
    set.seed(c(42,123,456)[ch])
    beta <- rep(0, p); step <- rep(0.02, p); acc <- rep(0, p)
    samples <- matrix(NA_real_, n_iter, p); colnames(samples) <- selected_feats
    cur_lp <- log_posterior(beta, X, time, status, selected_feats, data$clinical_feats)
    
    for (iter in 1:n_iter) {
      for (j in 1:p) {
        bp <- beta; bp[j] <- rnorm(1, beta[j], step[j])
        if (abs(bp[j]) > 10) next
        pp <- log_posterior(bp, X, time, status, selected_feats, data$clinical_feats)
        if (is.finite(pp) && log(runif(1)) < (pp - cur_lp)) {
          beta <- bp; cur_lp <- pp; acc[j] <- acc[j]+1
        }
      }
      samples[iter,] <- beta
      if (iter <= burnin && iter%%50==0)
        for (j in 1:p) {
          r <- acc[j]/iter
          if (r < 0.39) step[j] <- step[j]*0.9
          if (r > 0.49) step[j] <- step[j]*1.1
        }
      if (iter%%2000==0)
        cat(sprintf("    iter %d/%d accept=%.3f\n", iter, n_iter, sum(acc)/(iter*p)))
    }
    cat(sprintf("    Done. Accept=%.3f\n", sum(acc)/(n_iter*p)))
    chains[[ch]] <- samples[(burnin+1):n_iter, , drop=FALSE]
  }
  
  # Convergence
  rhat <- sapply(1:p, function(j) {
    cd <- sapply(chains, function(ch) ch[,j]); n <- nrow(cd)
    W <- mean(apply(cd,2,var)); B <- var(colMeans(cd))*n
    ifelse(W>0, sqrt(((n-1)*W+B)/(n*W)), NA)
  })
  cat(sprintf("Converged (R-hat<1.1): %d/%d\n", sum(rhat<1.1, na.rm=TRUE), p))
  
  # Posterior summary
  combined <- do.call(rbind, chains)
  post_df <- data.frame(
    feature  = selected_feats,
    post_mean = colMeans(combined),
    post_sd  = apply(combined, 2, sd),
    ci_025   = apply(combined, 2, quantile, 0.025),
    ci_975   = apply(combined, 2, quantile, 0.975),
    HR       = exp(colMeans(combined)),
    HR_ci025 = exp(apply(combined, 2, quantile, 0.025)),
    HR_ci975 = exp(apply(combined, 2, quantile, 0.975)),
    prob_pos = colMeans(combined > 0),
    signif   = (apply(combined, 2, quantile, 0.025)>0) | (apply(combined, 2, quantile, 0.975)<0)
  )
  cat(sprintf("Significant biomarkers (95%% CI excludes 0): %d\n", sum(post_df$signif)))
  print(post_df[post_df$signif, c("feature","HR","HR_ci025","HR_ci975","prob_pos")])
  
  # Forest plot
  fdf <- post_df %>% mutate(feature=reorder(feature, HR))
  pp <- ggplot(fdf, aes(x=HR, y=feature)) +
    geom_point(aes(color=signif), size=3) +
    geom_errorbarh(aes(xmin=HR_ci025, xmax=HR_ci975), height=0.3) +
    geom_vline(xintercept=1, linetype="dashed", color="red") +
    scale_color_manual(values=c("TRUE"="#E24B4A","FALSE"="#888780")) +
    scale_x_log10() +
    labs(title="Bayesian Cox: Hazard Ratios (95% CI)", x="HR (log scale)", y=NULL) +
    theme_minimal() + theme(legend.position="none")
  ggsave(file.path(FIGURES_DIR, "forest_plot.pdf"), pp, width=8, height=max(4, p*0.4))
  cat("Saved: forest_plot.pdf\n")
  
  list(post_df=post_df, combined=combined, chains=chains)
}


# ---- Step 6: Bootstrap stability ----
run_stability <- function(data, penalized, n_boot=100) {
  cat(sprintf("\n[Step 6] Bootstrap stability (%d rounds)\n", n_boot))
  cand <- penalized$candidate_feats; pf <- penalized$pf
  lambda_use <- penalized$lambda_opt
  counts <- setNames(rep(0, length(cand)), cand)
  
  set.seed(123)
  for (i in 1:n_boot) {
    idx <- sample(nrow(data$X_train), replace=TRUE)
    tryCatch({
      fit <- glmnet(data$X_train[idx, cand], Surv(data$train$time_years[idx], data$train$OS[idx]),
                    family="cox", alpha=1, penalty.factor=pf)
      co <- as.vector(coef(fit, s=lambda_use)); names(co) <- cand
      counts[names(co[abs(co)>1e-6])] <- counts[names(co[abs(co)>1e-6])] + 1
    }, error=function(e) NULL)
    if (i%%25==0) cat(sprintf("  %d/%d\n", i, n_boot))
  }
  
  freq <- sort(counts/n_boot, decreasing=TRUE)
  stable <- freq[freq>=0.5]
  cat(sprintf("Stable features (>=50%%): %d\n", length(stable)))
  print(stable)
  
  # Stability plot
  top_n <- min(25, sum(freq>0))
  pd <- data.frame(feature=factor(names(freq)[1:top_n], levels=rev(names(freq)[1:top_n])),
                   freq=freq[1:top_n])
  pp <- ggplot(pd, aes(x=freq, y=feature)) +
    geom_col(aes(fill=freq>=0.5)) +
    scale_fill_manual(values=c("TRUE"="#1D9E75","FALSE"="#B4B2A9"), guide="none") +
    geom_vline(xintercept=0.5, linetype="dashed", color="red") +
    labs(x="Selection frequency", y=NULL,
         title=sprintf("Feature Stability (%d stable)", length(stable))) +
    theme_minimal()
  ggsave(file.path(FIGURES_DIR, "feature_stability.pdf"), pp,
         width=9, height=max(5, top_n*0.3))
  cat("Saved: feature_stability.pdf\n")
  
  list(stable=stable, freq=freq)
}


# ---- Validation (inside run_all_plots) ----
run_validation_plots <- function(data, penalized, bayes, bayes_feats) {
  # Lasso predictions
  lambda_pred <- penalized$lambda_opt
  pred_test <- predict(penalized$lasso$cvfit,
                       newx=data$X_test[, penalized$candidate_feats], s=lambda_pred, type="link")
  
  # KM risk groups
  rg <- ifelse(pred_test > median(pred_test), "High", "Low")
  sfit_h <- survfit(Surv(data$test$time_years[rg=="High"], data$test$OS[rg=="High"]) ~ 1)
  sfit_l <- survfit(Surv(data$test$time_years[rg=="Low"],  data$test$OS[rg=="Low"]) ~ 1)
  .km_t <<- data$test$time_years; .km_s <<- data$test$OS; .km_g <<- rg
  pval <- 1 - pchisq(survdiff(Surv(.km_t, .km_s) ~ .km_g)$chisq, 1)
  rm(.km_t, .km_s, .km_g, envir=.GlobalEnv)
  
  pdf(file.path(FIGURES_DIR, "km_risk_groups.pdf"), width=8, height=6)
  plot(sfit_h, col="#E24B4A", lwd=2, mark.time=TRUE,
       xlab="Time (years)", ylab="Survival probability",
       main=sprintf("Risk Stratification (Test, p=%.4f)", pval))
  lines(sfit_l, col="#378ADD", lwd=2, mark.time=TRUE)
  legend("bottomleft", c("High risk","Low risk"), col=c("#E24B4A","#378ADD"), lwd=2)
  dev.off()
  cat(sprintf("Log-rank p=%.4f | Saved: km_risk_groups.pdf\n", pval))
  
  # C-index
  c_lasso <- glmnet::Cindex(pred=pred_test[,1], y=Surv(data$test$OS.time, data$test$OS))
  beta_b <- bayes$post_df$post_mean
  risk_b <- as.vector(data$X_test[, match(bayes_feats, data$features)] %*% beta_b)
  c_bayes <- concordance(Surv(data$test$time_years, data$test$OS) ~ risk_b, reverse=TRUE)$concordance
  cat(sprintf("Lasso C-index: %.4f | Bayesian C-index: %.4f\n", c_lasso, c_bayes))
  
  perf <- data.frame(metric=c("Lasso_Cindex","Bayesian_Cindex","LogRank_p"),
                     value=c(c_lasso, c_bayes, pval))
  write.csv(perf, file.path(RESULTS_DIR, "test_performance.csv"), row.names=FALSE)
  perf
}


# ---- Main: compute & save ----
main <- function() {
  data <- load_data()
  run_km(data)
  uni_df     <- run_univariate(data)
  candidates <- pvalue_filter(uni_df, data, alpha=0.1)
  penalized  <- run_penalized_cox(data, candidates)
  
  # Pick features for Bayesian
  if (length(penalized$lasso$selected) >= 2) {
    bfeats <- penalized$lasso$selected
  } else if (length(penalized$en_selected) >= 2) {
    bfeats <- penalized$en_selected
  } else {
    bfeats <- penalized$alasso_selected
  }
  cat(sprintf("Bayesian Cox using %d features\n", length(bfeats)))
  
  bayes     <- run_bayesian_cox(data, bfeats, n_iter=10000, burnin=5000, n_chains=3)
  stability <- run_stability(data, penalized, n_boot=100)
  
  # Save all results
  results <- list(data=data, uni_df=uni_df, candidates=candidates,
                  penalized=penalized, bayes_feats=bfeats,
                  bayes=bayes, stability=stability)
  saveRDS(results, file.path(RESULTS_DIR, "cox_results.rds"))
  write.csv(bayes$post_df, file.path(RESULTS_DIR, "bayesian_cox_summary.csv"), row.names=FALSE)
  cat(sprintf("\nResults saved to %s/\n", RESULTS_DIR))
  
  # Plots
  run_all_plots(results)
}

# ---- Re-plot without re-training ----
run_all_plots <- function(results=NULL) {
  if (is.null(results)) {
    cat("Loading saved results...\n")
    results <- readRDS(file.path(RESULTS_DIR, "cox_results.rds"))
  }
  perf <- run_validation_plots(results$data, results$penalized,
                               results$bayes, results$bayes_feats)
  
  cat("\n========== SUMMARY ==========\n")
  cat(sprintf("Lasso: %d | EN: %d | AdaLasso: %d\n",
              length(results$penalized$lasso$selected),
              length(results$penalized$en_selected),
              length(results$penalized$alasso_selected)))
  cat(sprintf("Bayesian significant: %d | Bootstrap stable: %d\n",
              sum(results$bayes$post_df$signif), length(results$stability$stable)))
  print(perf)
}

# Run
main()
# To re-plot only: run_all_plots()