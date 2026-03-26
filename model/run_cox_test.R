# Test script for Bayesian Cox model
# Load and execute with reduced iterations for testing

source("bayesian_cox.R")

# Run analysis with reduced iterations for testing
results <- run_analysis(csv_path = "../data/brca_multimodal.csv",
                     n_features = 50,
                     n_iter = 2000,
                     burnin = 500,
                     n_chains = 2)

cat("\n=====================================\n")
cat("Test Run Complete\n")
cat("=====================================\n")
