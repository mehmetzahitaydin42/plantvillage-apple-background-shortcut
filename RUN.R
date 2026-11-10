# =============================================================================
#  RUN.R  -  Runs everything from one file
#
#  Usage (RStudio):
#    1) Open this repository folder as the working directory
#       (Session > Set Working Directory > To Source File Location).
#    2) Choose MODE below: first "test", then "full".
#    3) Click "Source" (Ctrl+Shift+S).
#
#  What it does:
#    - Installs missing R packages and a Python/TensorFlow environment (first run only)
#    - Downloads the PlantVillage apple images (~90 MB, 6,342 files)
#    - Runs the complete analysis in the selected mode (resumes if interrupted)
#    - Step C of 06 additionally needs the Plant Pathology 2020 data (see README)
# =============================================================================

MODE <- "test"                      # "test" (5-10 min, code check)  or  "full" (paper)
ROOT <- normalizePath(getwd())      # repository folder

# -----------------------------------------------------------------------------
stopifnot("config.R not found: set the working directory to the repository folder" = file.exists("config.R"))
cat("Working directory:", ROOT, "\n")
options(timeout = max(600, getOption("timeout")))

# ---- 1) Packages ---------------------------------------------------------------
needed <- c("magick", "ranger", "e1071", "xgboost", "dplyr", "tidyr", "readr",
            "ggplot2", "patchwork", "keras3", "reticulate")
missing_pkgs <- needed[!vapply(needed, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_pkgs)) {
  cat("Installing missing packages:", paste(missing_pkgs, collapse = ", "), "\n")
  install.packages(missing_pkgs, repos = "https://cloud.r-project.org")
}

# Python 3.11 + "r-keras" virtual environment for TensorFlow (installed once).
# If the environment already contains tensorflow and keras, nothing is changed.
KERAS_FLAG <- file.path(ROOT, ".keras_setup_done")
venv_dir <- file.path(reticulate::virtualenv_root(), "r-keras")
sp <- c(file.path(venv_dir, "Lib", "site-packages"),                       # Windows
        Sys.glob(file.path(venv_dir, "lib", "python3*", "site-packages")))  # Linux/macOS
tf_ok <- any(dir.exists(file.path(sp, "tensorflow")) & dir.exists(file.path(sp, "keras")))
if (tf_ok && !file.exists(KERAS_FLAG)) writeLines(format(Sys.time()), KERAS_FLAG)
if (!file.exists(KERAS_FLAG)) {
  if (.Platform$OS.type == "windows") {
    # Official python.org installer, per-user and silent: no administrator rights or git needed.
    py_exe <- file.path(Sys.getenv("LOCALAPPDATA"), "Programs", "Python", "Python311", "python.exe")
    if (!file.exists(py_exe)) {
      cat("Downloading and installing Python 3.11 (~25 MB)...\n")
      inst <- file.path(tempdir(), "python-3.11.9-amd64.exe")
      download.file("https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe", inst, mode = "wb")
      system2(inst, c("/quiet", "InstallAllUsers=0", "PrependPath=0", "Include_test=0",
                      "Include_launcher=0", "Shortcuts=0", "AssociateFiles=0"), wait = TRUE)
      if (!file.exists(py_exe)) stop("Python could not be installed. Install Python 3.11.9 from python.org and retry.")
    }
    cat("Installing the keras / TensorFlow environment (~1 GB, 5-15 min)...\n")
    if (!reticulate::virtualenv_exists("r-keras"))
      reticulate::virtualenv_create("r-keras", python = py_exe, packages = NULL)
    reticulate::py_install(c("numpy<2", "scipy", "pandas", "Pillow", "h5py", "tensorflow", "keras==3.*"),
                           envname = "r-keras", pip = TRUE)
  } else {
    keras3::install_keras(backend = "tensorflow", restart_session = FALSE)
  }
  writeLines(format(Sys.time()), KERAS_FLAG)
  cat("keras / TensorFlow setup complete.\n")
}
reticulate::use_virtualenv("r-keras", required = TRUE)

# ---- 2) Data ---------------------------------------------------------------------
DATA_ROOT <- file.path(ROOT, "PlantVillage-Dataset")
paths  <- readLines(file.path(ROOT, "plantvillage_apple_file_list.csv"), encoding = "UTF-8")[-1]
target <- file.path(DATA_ROOT, paths)
absent <- !file.exists(target) | file.size(target) %in% c(0, NA)
if (any(absent)) {
  cat(sprintf("Downloading PlantVillage apple images: %d / %d files missing\n", sum(absent), length(target)))
  base <- "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/"
  url <- paste0(base, vapply(paths, function(p)
    paste(vapply(strsplit(p, "/")[[1]], URLencode, character(1), reserved = TRUE), collapse = "/"), character(1)))
  for (d in unique(dirname(target[absent]))) dir.create(d, recursive = TRUE, showWarnings = FALSE)
  idx <- which(absent); chunks <- split(idx, ceiling(seq_along(idx) / 50))
  for (k in seq_along(chunks)) {
    ix <- chunks[[k]]
    try(suppressWarnings(download.file(url[ix], target[ix], method = "libcurl", mode = "wb", quiet = TRUE)), silent = TRUE)
    if (k %% 10 == 0 || k == length(chunks)) cat(sprintf("  %d / %d\n", min(k * 50, length(idx)), length(idx)))
  }
  left <- which(!file.exists(target) | file.size(target) %in% c(0, NA))
  for (i in left) try(download.file(url[i], target[i], mode = "wb", quiet = TRUE), silent = TRUE)
  left <- which(!file.exists(target) | file.size(target) %in% c(0, NA))
  if (length(left)) stop(length(left), " files could not be downloaded. Check the connection and run RUN.R again.")
  cat("Data complete.\n")
} else cat("Data already present (", length(target), " files).\n")

# ---- 3) Analysis -------------------------------------------------------------------
Sys.setenv(PV_KOK = file.path(DATA_ROOT, "raw"), PV_HIZLI = if (MODE == "test") "TRUE" else "FALSE",
           PV_PP2020 = file.path(ROOT, "PlantPathology2020"), TF_CPP_MIN_LOG_LEVEL = "2")
cat("\n>>> MODE:", MODE, if (MODE == "test") "(output: results_quick_test/)" else
      "(output: results/ - long run; if interrupted, run again)", "\n")
source("run_pipeline.R", local = new.env())
cat(if (MODE == "test") "\n*** TEST PASSED. Set MODE <- \"full\" and source again. ***\n" else
      "\n*** FULL ANALYSIS FINISHED. See results/tables and results/figures. ***\n")
