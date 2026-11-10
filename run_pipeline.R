# =============================================================================
#  run_pipeline.R  -  Runs the complete analysis in order.
#  Usually started from RUN.R. If interrupted, run it again:
#  finished steps are skipped and the analysis resumes where it stopped.
#
# =============================================================================
t0 <- Sys.time()
adim <- function(dosya, kontrol = NULL) {
  if (!is.null(kontrol) && file.exists(kontrol)) {
    cat(">>", dosya, "skipped (already done)\n"); return(invisible())
  }
  cat("\n==========", dosya, "started:", format(Sys.time(), "%H:%M"), "==========\n")
  log_yaz(paste("STARTED", dosya))
  withCallingHandlers(
    source(dosya, local = new.env(), echo = FALSE),
    warning = function(w) log_yaz(paste("WARNING", dosya, ":", conditionMessage(w))),
    error   = function(e) {
      log_yaz(paste("ERROR", dosya, ":", conditionMessage(e)))
      log_yaz(paste("  cagri:", paste(deparse(conditionCall(e)), collapse = " ")))
    })
  log_yaz(paste("FINISHED", dosya))
}
source("config.R")
# Start/end/error records of every step: <output>/run_log.txt
log_yaz <- function(...) cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), ..., "\n",
                             file = yol("run_log.txt"), append = TRUE)
adim("01_prepare_data.R", kontrol = yol("data", "folds.rds"))
adim("02_classical_models.R")   # resumes internally
adim("03_deep_models.R")    # resumes internally (longest step)
adim("04_gradcam_quantitative.R")
adim("05_statistics_tables.R")
adim("06_additional_analyses.R")      # additional analyses (background swap, probes, PP2020)
if (!file.exists(yol("tables", "T2_main_results.csv")))
  stop("Step 05 produced no tables. See ", yol("run_log.txt"), ".")
cat("\nTotal time:", format(round(difftime(Sys.time(), t0, units = "hours"), 2)), "\n")
cat("Done. Tables and figures are in", yol("tables"), "and", yol("figures"), "\n")
