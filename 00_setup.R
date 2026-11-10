# =============================================================================
#  00_setup.R  -  Installs the required packages ONCE (RUN.R does this automatically)
# =============================================================================
paketler <- c("magick", "ranger", "e1071", "xgboost", "dplyr", "tidyr",
              "readr", "ggplot2", "patchwork", "keras3")
eksik <- paketler[!vapply(paketler, requireNamespace, logical(1), quietly = TRUE)]
if (length(eksik)) install.packages(eksik)

# keras3 needs a Python/TensorFlow environment. Install it once:
#   (takes a few minutes, downloads ~1 GB)
if (!reticulate::virtualenv_exists("r-keras")) {
  keras3::install_keras(backend = "tensorflow")   # CPU version is sufficient
}

# Check
library(keras3)
cat("TensorFlow version:", as.character(tensorflow::tf_version()), "\n")
cat("Setup complete.\n")
