library(dplyr)
library(ggplot2)
library(survival)
library(survminer)
library(glmnet)

PROJECT_ROOT <- "/Users/chenyanzhen/Documents/DS4420_final_project-2"
DATA_DIR     <- file.path(PROJECT_ROOT, "data")
RESULTS_DIR  <- file.path(PROJECT_ROOT, "results", "cox")
FIGURES_DIR  <- file.path(PROJECT_ROOT, "figures", "cox")
dir.create(RESULTS_DIR, showWarnings=FALSE, recursive=TRUE)
dir.create(FIGURES_DIR, showWarnings=FALSE, recursive=TRUE)


load_data <- function() {
  train <- read.csv(file.path(DATA_DIR, "cox_train.csv"), check.names=FALSE)
  test  <- read.csv(file.path(DATA_DIR, "cox_test.csv"), check.names=FALSE)
  feat  <- read.csv(file.path(DATA_DIR, "features.csv"))
  
  features <- feat$feature
  clin_feats <- feat$feature[feat$is_clinical == TRUE]
  
  X_train <- as.matrix(train[, features])
  X_test  <- as.matrix(test[, features])
  storage.mode(X_train) <- "double"
  storage.mode(X_test) <- "double"
  X_train[!is.finite(X_train)] <- 0
  X_test[!is.finite(X_test)] <- 0
  
  train$time_years <- train$OS.time / 365.25
  test$time_years <- test$OS.time / 365.25
  
  list(train=train, test=test, X_train=X_train, X_test=X_test,
       features=features, clin_feats=clin_feats)
}


# kaplan meier baseline
run_km <- function(data) {
  train <- data$train
  assign("train", train, envir=.GlobalEnv)
  on.exit(rm("train", envir=.GlobalEnv), add=TRUE)
  survfit(Surv(time_years, OS) ~ 1, data=train)
}


# screen each feature one by one
run_univariate <- function(data) {
  train <- data$train
  res <- list()
  for (f in data$features) {
    tryCatch({
      tmp <- data.frame(T=train$time_years, E=train$OS, x=data$X_train[,f])
      s <- summary(coxph(Surv(T, E) ~ x, data=tmp))
      res[[f]] <- data.frame(feature=f, coef=s$coefficients[,"coef"],
                              HR=s$coefficients[,"exp(coef)"],
                              p_value=s$coefficients[,"Pr(>|z|)"])
    }, error=function(e) NULL)
  }
  bind_rows(res) %>% arrange(p_value)
}


# keep features with p < alpha
pvalue_filter <- function(uni, data, alpha=0.1) {
  union(uni$feature[uni$p_value < alpha], data$clin_feats)
}


# lasso, elastic net, adaptive lasso
run_penalized <- function(data, cands) {
  x <- data$X_train[, cands]
  y <- Surv(data$train$time_years, data$train$OS)
  pf <- rep(1, length(cands))
  pf[cands %in% data$clin_feats] <- 0  # dont penalize clinical
  
  # lasso
  set.seed(123)
  cv_lasso <- cv.glmnet(x, y, family="cox", alpha=1, nfolds=5, penalty.factor=pf)
  lam <- cv_lasso$lambda.1se
  co <- as.vector(coef(cv_lasso, s="lambda.1se"))
  names(co) <- cands
  lasso_sel <- names(co[abs(co) > 1e-6])
  
  if (length(lasso_sel) < 2) {
    lam <- cv_lasso$lambda.min
    co <- as.vector(coef(cv_lasso, s="lambda.min"))
    names(co) <- cands
    lasso_sel <- names(co[abs(co) > 1e-6])
  }
  
  # elastic net - try different alphas
  alphas <- seq(0.1, 1, length=10)
  en_fits <- list()
  set.seed(123)
  for (i in seq_along(alphas))
    en_fits[[i]] <- cv.glmnet(x, y, family="cox", alpha=alphas[i],
                               nfolds=5, penalty.factor=pf)
  best <- which.min(sapply(en_fits, function(f) f$cvm[f$lambda == f$lambda.min]))
  co_en <- as.vector(coef(en_fits[[best]], s="lambda.1se"))
  names(co_en) <- cands
  en_sel <- names(co_en[abs(co_en) > 1e-6])
  
  # adaptive lasso
  set.seed(123)
  ridge <- cv.glmnet(x, y, family="cox", alpha=0, nfolds=5)
  wts <- abs(1 / as.vector(coef(ridge, s="lambda.min")))
  wts[cands %in% data$clin_feats] <- 0
  cv_al <- cv.glmnet(x, y, family="cox", nfolds=5, penalty.factor=wts)
  co_al <- as.vector(coef(cv_al, s="lambda.1se"))
  names(co_al) <- cands
  al_sel <- names(co_al[abs(co_al) > 1e-6])
  
  list(lasso=list(cvfit=cv_lasso, sel=lasso_sel, coefs=co),
       en_sel=en_sel, al_sel=al_sel,
       cands=cands, pf=pf, lam=lam)
}


## bayesian cox - hand written MCMC

log_partial_likelihood <- function(beta, X, time, status) {
  eta <- as.vector(X %*% beta)
  ll <- 0
  for (tk in sort(unique(time[status==1]))) {
    risk <- which(time >= tk)
    dead <- which(time == tk & status == 1)
    if (length(dead)==0) next
    eta_r <- eta[risk]
    mx <- max(eta_r)
    lr <- log(sum(exp(eta_r - mx))) + mx
    if (!is.finite(lr)) return(-Inf)
    ll <- ll + sum(eta[dead]) - length(dead)*lr
  }
  ll
}

log_posterior <- function(beta, X, time, status, fnames, clin,
                          lam=1.0, sig_clin=5.0) {
  lp <- 0
  for (j in seq_along(beta)) {
    if (fnames[j] %in% clin)
      lp <- lp + dnorm(beta[j], 0, sig_clin, log=TRUE)     # normal prior for age
    else
      lp <- lp + log(lam/2) - lam*abs(beta[j])              # laplace prior for genes
  }
  log_partial_likelihood(beta, X, time, status) + lp
}

run_bayes <- function(data, feats, n_iter=10000, burnin=5000, n_chains=3) {
  X <- data$X_train[, match(feats, data$features), drop=FALSE]
  time <- data$train$time_years
  status <- data$train$OS
  p <- length(feats)
  chains <- vector("list", n_chains)
  
      for (ch in 1:n_chains) {
    cat(sprintf("chain %d/%d running...\n", ch, n_chains))
    set.seed(c(42,123,456)[ch])
    beta <- rep(0, p)
    step <- rep(0.02, p)
    acc <- rep(0, p)
    samps <- matrix(NA_real_, n_iter, p)
    colnames(samps) <- feats
    cur_lp <- log_posterior(beta, X, time, status, feats, data$clin_feats)
    
    for (it in 1:n_iter) {
      for (j in 1:p) {
        bp <- beta
        bp[j] <- rnorm(1, beta[j], step[j])
        if (abs(bp[j]) > 10) next
        pp <- log_posterior(bp, X, time, status, feats, data$clin_feats)
        if (is.finite(pp) && log(runif(1)) < (pp - cur_lp)) {
          beta <- bp
          cur_lp <- pp
          acc[j] <- acc[j] + 1
        }
      }
      samps[it, ] <- beta
      
      # adapt step size during burnin
      if (it <= burnin && it %% 50 == 0) {
        for (j in 1:p) {
          r <- acc[j]/it
          if (r < 0.39) step[j] <- step[j]*0.9
          if (r > 0.49) step[j] <- step[j]*1.1
        }
      }
    }
    chains[[ch]] <- samps[(burnin+1):n_iter, , drop=FALSE]
  }
  
  # summarize posterior
  all_samps <- do.call(rbind, chains)
  post <- data.frame(
    feature   = feats,
    post_mean = colMeans(all_samps),
    post_sd   = apply(all_samps, 2, sd),
    ci_025    = apply(all_samps, 2, quantile, 0.025),
    ci_975    = apply(all_samps, 2, quantile, 0.975),
    HR        = exp(colMeans(all_samps)),
    HR_ci025  = exp(apply(all_samps, 2, quantile, 0.025)),
    HR_ci975  = exp(apply(all_samps, 2, quantile, 0.975)),
    prob_pos  = colMeans(all_samps > 0),
    signif    = (apply(all_samps, 2, quantile, 0.025) > 0) |
                (apply(all_samps, 2, quantile, 0.975) < 0)
  )
  
  # forest plot
  fdf <- post %>% mutate(feature = reorder(feature, HR))
  g <- ggplot(fdf, aes(x=HR, y=feature)) +
    geom_point(aes(color=signif), size=3) +
    geom_errorbarh(aes(xmin=HR_ci025, xmax=HR_ci975), height=0.3) +
    geom_vline(xintercept=1, linetype="dashed", color="red") +
    scale_color_manual(values=c("TRUE"="#E24B4A", "FALSE"="#888780")) +
    scale_x_log10() +
    labs(title="Bayesian Cox: Hazard Ratios (95% CI)",
         x="HR (log scale)", y=NULL) +
    theme_minimal() + theme(legend.position="none")
  ggsave(file.path(FIGURES_DIR, "forest_plot.pdf"), g,
         width=8, height=max(4, p*0.4))
  
  list(post=post, all_samps=all_samps, chains=chains)
}


# bootstrap to check if features are stable
run_bootstrap <- function(data, pen, n_boot=100) {
  cands <- pen$cands
  pf <- pen$pf
  lam <- pen$lam
  counts <- setNames(rep(0, length(cands)), cands)
  
  set.seed(123)
  for (i in 1:n_boot) {
    idx <- sample(nrow(data$X_train), replace=TRUE)
    tryCatch({
      fit <- glmnet(data$X_train[idx, cands],
                    Surv(data$train$time_years[idx], data$train$OS[idx]),
                    family="cox", alpha=1, penalty.factor=pf)
      co <- as.vector(coef(fit, s=lam))
      names(co) <- cands
      got <- names(co[abs(co) > 1e-6])
      counts[got] <- counts[got] + 1
    }, error=function(e) NULL)
  }
  
  freq <- sort(counts/n_boot, decreasing=TRUE)
  list(stable=freq[freq >= 0.5], freq=freq)
}


# evaluate on test set
run_validation <- function(data, pen, bayes, bfeats) {
  pred <- predict(pen$lasso$cvfit, newx=data$X_test[, pen$cands],
                  s=pen$lam, type="link")
  
  # c-index
  c_lasso <- glmnet::Cindex(pred=pred[,1], y=Surv(data$test$OS.time, data$test$OS))
  b_beta <- bayes$post$post_mean
  b_risk <- as.vector(data$X_test[, match(bfeats, data$features)] %*% b_beta)
  c_bayes <- concordance(Surv(data$test$time_years, data$test$OS) ~ b_risk,
                          reverse=TRUE)$concordance
  
  # split into high/low risk for KM plot
  rg <- ifelse(pred > median(pred), "High", "Low")
  .km_t <<- data$test$time_years
  .km_s <<- data$test$OS
  .km_g <<- rg
  pval <- 1 - pchisq(survdiff(Surv(.km_t, .km_s) ~ .km_g)$chisq, 1)
  rm(.km_t, .km_s, .km_g, envir=.GlobalEnv)
  
  sfit_h <- survfit(Surv(data$test$time_years[rg=="High"],
                         data$test$OS[rg=="High"]) ~ 1)
  sfit_l <- survfit(Surv(data$test$time_years[rg=="Low"],
                         data$test$OS[rg=="Low"]) ~ 1)
  
  pdf(file.path(FIGURES_DIR, "km_risk_groups.pdf"), width=8, height=6)
  plot(sfit_h, col="#E24B4A", lwd=2, mark.time=TRUE,
       xlab="Time (years)", ylab="Survival probability",
       main=sprintf("Risk Stratification (Test, p=%.4f)", pval))
  lines(sfit_l, col="#378ADD", lwd=2, mark.time=TRUE)
  legend("bottomleft", c("High risk","Low risk"),
         col=c("#E24B4A","#378ADD"), lwd=2)
  dev.off()
  
  perf <- data.frame(metric=c("Lasso_Cindex","Bayesian_Cindex","LogRank_p"),
                     value=c(c_lasso, c_bayes, pval))
  write.csv(perf, file.path(RESULTS_DIR, "test_performance.csv"), row.names=FALSE)
  perf
}


# run everything
main <- function() {
  data <- load_data()
  run_km(data)
  uni <- run_univariate(data)
  cands <- pvalue_filter(uni, data, alpha=0.1)
  pen <- run_penalized(data, cands)
  
  # use lasso features for bayesian, fallback to EN or adaptive lasso
  if (length(pen$lasso$sel) >= 2) {
    bfeats <- pen$lasso$sel
  } else if (length(pen$en_sel) >= 2) {
    bfeats <- pen$en_sel
  } else {
    bfeats <- pen$al_sel
  }
  
  bayes <- run_bayes(data, bfeats)
  stab  <- run_bootstrap(data, pen)
  perf  <- run_validation(data, pen, bayes, bfeats)
  
  # save
  results <- list(data=data, uni=uni, cands=cands, pen=pen,
                  bfeats=bfeats, bayes=bayes, stab=stab)
  saveRDS(results, file.path(RESULTS_DIR, "cox_results.rds"))
  write.csv(bayes$post, file.path(RESULTS_DIR, "bayesian_cox_summary.csv"),
            row.names=FALSE)
  cat("done\n")
}

main()