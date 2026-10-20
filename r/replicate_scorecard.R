# R replication of the WoE logistic scorecard.
#
# Requires R with only base packages (glm, read.csv, pROC is optional).
# Run AFTER the Python pipeline has produced the inputs, i.e.:
#
#     python -m src.pipeline pd
#     Rscript r/replicate_scorecard.R
#
# See r/README.md. In the build environment used for this repository R was not
# installed, so this script was NOT executed and no R numbers are quoted in the
# docs.

results_dir <- file.path("reports", "results")
woe_path    <- file.path(results_dir, "pd_woe_oot.csv")
coef_path   <- file.path(results_dir, "pd_logit_coefficients.csv")

if (!file.exists(woe_path)) {
  stop(sprintf("Missing %s - run `python -m src.pipeline pd` first.", woe_path))
}

dat <- read.csv(woe_path, check.names = FALSE)
if (!("default_flag" %in% names(dat))) {
  stop("pd_woe_oot.csv must contain a `default_flag` column.")
}

feat_cols <- setdiff(names(dat), "default_flag")
form <- as.formula(paste("default_flag ~", paste(sprintf("`%s`", feat_cols), collapse = " + ")))

fit <- glm(form, data = dat, family = binomial(link = "logit"))
summary(fit)

# --- out-of-time AUC (rank-based, so no extra packages required) -------------
auc_rank <- function(y, p) {
  r <- rank(p)
  n1 <- sum(y == 1); n0 <- sum(y == 0)
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}
p_hat <- predict(fit, type = "response")
r_auc <- auc_rank(dat$default_flag, p_hat)
cat(sprintf("R out-of-time AUC = %.4f\n", r_auc))

# --- coefficient comparison vs the Python scorecard --------------------------
if (file.exists(coef_path)) {
  py <- read.csv(coef_path, check.names = FALSE)
  py_coef <- setNames(py$coefficient, py$variable)
  r_coef  <- coef(fit)
  r_coef["(Intercept)"] <- r_coef["(Intercept)"]
  common <- intersect(names(py_coef), c(names(r_coef), "(Intercept)"))
  # Map R intercept name to the Python "const" name.
  r_names <- names(r_coef)
  r_names[r_names == "(Intercept)"] <- "const"
  names(r_coef) <- r_names
  common <- intersect(names(py_coef), names(r_coef))
  cmp <- data.frame(
    variable  = common,
    python    = as.numeric(py_coef[common]),
    r         = as.numeric(r_coef[common])
  )
  cmp$abs_diff <- abs(cmp$python - cmp$r)
  print(cmp)
  write.csv(cmp, file.path(results_dir, "r_scorecard_comparison.csv"), row.names = FALSE)
  cat(sprintf("Max |Python - R| coefficient difference = %.6g\n", max(cmp$abs_diff)))
} else {
  cat(sprintf("Python coefficients not found at %s - skipping the comparison.\n", coef_path))
}
