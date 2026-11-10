# =============================================================================
#  02_classical_models.R
#  40 colour + texture features -> Random Forest, SVM (RBF), XGBoost
#  4 variants x TEKRAR_KLASIK x K_KAT. Test-fold predictions of every fold are saved.
# =============================================================================
source("config.R")
suppressPackageStartupMessages({ library(ranger); library(e1071); library(dplyr) })
XGB_VAR <- requireNamespace("xgboost", quietly = TRUE)
if (!XGB_VAR) message("xgboost is not installed; XGBoost will be skipped. install.packages('xgboost')")

liste  <- readRDS(yol("data", "image_list.rds"))
KATLAR <- readRDS(yol("data", "folds.rds"))
y <- liste$sinif

# ---- 1) Feature extraction (40 features per image) ---------------------------
#  RGB mean+SD (6) + HSV mean+SD (6) + H/S/V histograms with 8 bins each (24)
#  + GLCM contrast, homogeneity, energy, correlation (4)  = 40
GLCM_SEVIYE <- 16
glcm_ozellik <- function(gri) {             # gri: 0-255 grey matrix
  q <- pmin(floor(gri / 256 * GLCM_SEVIYE), GLCM_SEVIYE - 1)
  h <- nrow(q); w <- ncol(q); L <- GLCM_SEVIYE
  kaydir <- list(c(0, 1), c(-1, 1), c(-1, 0), c(-1, -1))   # 0, 45, 90, 135 degrees; d = 1
  P <- matrix(0, L, L)
  for (o in kaydir) {
    r1 <- max(1, 1 - o[1]):min(h, h - o[1]); c1 <- max(1, 1 - o[2]):min(w, w - o[2])
    a <- q[r1, c1]; b <- q[r1 + o[1], c1 + o[2]]
    t <- matrix(tabulate(a * L + b + 1, L * L), L, L, byrow = TRUE)
    P <- P + t + t(t)                                        # symmetric
  }
  P <- P / sum(P)
  i <- row(P) - 1; j <- col(P) - 1
  mi <- sum(i * P); mj <- sum(j * P)
  si <- sqrt(sum((i - mi)^2 * P)); sj <- sqrt(sum((j - mj)^2 * P))
  c(kontrast = sum((i - j)^2 * P),
    homojenlik = sum(P / (1 + abs(i - j))),
    enerji = sum(P^2),
    korelasyon = if (si * sj > 0) sum((i - mi) * (j - mj) * P) / (si * sj) else 0)
}
ozellik <- function(img, sadece_bolge = FALSE) {   # img: b x b x 3, 0-255
  rgb <- matrix(img, ncol = 3)
  # In the leaf/background variants the black region is excluded from the colour features;
  # otherwise the fraction of black pixels would leak leaf size.
  if (sadece_bolge) { b <- rowSums(rgb) > 0; if (sum(b) > 10) rgb <- rgb[b, , drop = FALSE] }
  hsv <- t(grDevices::rgb2hsv(t(rgb), maxColorValue = 255))
  hst <- function(v) tabulate(pmin(floor(v * 8), 7) + 1, 8) / length(v)
  gri <- matrix(0.299 * img[, , 1] + 0.587 * img[, , 2] + 0.114 * img[, , 3], nrow(img))
  c(colMeans(rgb), apply(rgb, 2, sd), colMeans(hsv), apply(hsv, 2, sd),
    hst(hsv[, 1]), hst(hsv[, 2]), hst(hsv[, 3]), glcm_ozellik(gri))
}

F_LISTE <- list()
for (v in VARYANTLAR) {
  f_dosya <- yol("data", paste0("F_", v, ".rds"))
  if (file.exists(f_dosya)) { F_LISTE[[v]] <- readRDS(f_dosya); next }
  cat("Extracting features:", v, "\n")
  X <- readRDS(yol("data", paste0("X_", v, "_", GORUNTU_BOYUT, ".rds")))
  F <- t(vapply(seq_len(dim(X)[1]), function(i)
    ozellik(X[i, , , ], sadece_bolge = v %in% c("yaprak", "arkaplan")), numeric(40)))
  colnames(F) <- c(paste0(c("R", "G", "B"), "_ort"), paste0(c("R", "G", "B"), "_ss"),
                   paste0(c("H", "S", "V"), "_ort"), paste0(c("H", "S", "V"), "_ss"),
                   paste0("H_hist", 1:8), paste0("S_hist", 1:8), paste0("V_hist", 1:8),
                   "GLCM_kontrast", "GLCM_homojenlik", "GLCM_enerji", "GLCM_korelasyon")
  saveRDS(F, f_dosya); F_LISTE[[v]] <- F; rm(X); gc()
}

# ---- 2) Models -----------------------------------------------------------------
sinif_agirlik <- function(yy) { tb <- table(factor(yy, levels = seq_along(SINIFLAR)))
  w <- sum(tb) / (length(tb) * pmax(tb, 1)); as.numeric(w) }

model_rf <- function(Ftr, ytr, Fte) {
  d <- data.frame(Ftr, y = factor(ytr, levels = seq_along(SINIFLAR)))
  m <- ranger(y ~ ., data = d, num.trees = 300, class.weights = sinif_agirlik(ytr),
              probability = TRUE, seed = TOHUM, num.threads = 1)
  p <- predict(m, data.frame(Fte))$predictions
  list(tahmin = max.col(p, ties.method = "first"), olasilik = p,
       onem = NULL)
}

model_svm <- function(Ftr, ytr, Fte) {
  # Cost C chosen by inner 3-fold CV (gamma = 1/number of features); scaling fitted on the training set
  mu <- colMeans(Ftr); s <- apply(Ftr, 2, sd); s[s == 0] <- 1
  Ztr <- scale(Ftr, mu, s); Zte <- scale(Fte, mu, s)
  w <- sinif_agirlik(ytr); names(w) <- seq_along(SINIFLAR)
  yf <- factor(ytr, levels = seq_along(SINIFLAR))
  ic <- sample(rep_len(1:3, length(ytr)))
  C_aday <- c(1, 10, 100)
  basari <- sapply(C_aday, function(C) mean(sapply(1:3, function(k) {
    m <- svm(Ztr[ic != k, ], yf[ic != k], kernel = "radial", cost = C, class.weights = w)
    mean(predict(m, Ztr[ic == k, ]) == yf[ic == k]) })))
  m <- svm(Ztr, yf, kernel = "radial", cost = C_aday[which.max(basari)],
           class.weights = w, probability = TRUE)
  p <- attr(predict(m, Zte, probability = TRUE), "probabilities")
  p <- p[, as.character(seq_along(SINIFLAR)), drop = FALSE]
  list(tahmin = max.col(p, ties.method = "first"), olasilik = p)
}

model_xgb <- function(Ftr, ytr, Fte) {
  w <- sinif_agirlik(ytr)[ytr]
  dtr <- xgboost::xgb.DMatrix(Ftr, label = ytr - 1, weight = w)
  prm <- list(objective = "multi:softprob", num_class = length(SINIFLAR),
              eta = 0.1, max_depth = 6, subsample = 0.8, colsample_bytree = 0.8,
              nthread = 1)
  m <- xgboost::xgb.train(params = prm, data = dtr, nrounds = 300, verbose = 0)
  p <- predict(m, xgboost::xgb.DMatrix(Fte))
  if (is.null(dim(p))) p <- matrix(p, ncol = length(SINIFLAR), byrow = TRUE)
  list(tahmin = max.col(p, ties.method = "first"), olasilik = p)
}

MODELLER <- list(RF = model_rf, SVM = model_svm)
if (XGB_VAR) MODELLER$XGBoost <- model_xgb

# ---- 3) Tekrarli CV loop (repeated CV; resumes where it stopped) -------------
for (v in VARYANTLAR) for (mn in names(MODELLER)) for (r in seq_len(TEKRAR_KLASIK)) {
  dosya <- yol("predictions", sprintf("%s__%s__r%02d.rds", mn, v, r))
  if (file.exists(dosya)) next
  t0 <- Sys.time()
  F <- F_LISTE[[v]]; kat <- KATLAR[, r]
  sonuc <- data.frame(id = seq_along(y), gercek = y, tahmin = NA_integer_, kat = kat)
  for (k in seq_len(K_KAT)) {
    set.seed(TOHUM + 100 * r + k)
    te <- which(kat == k); tr <- which(kat != k)
    out <- MODELLER[[mn]](F[tr, ], y[tr], F[te, ])
    sonuc$tahmin[te] <- out$tahmin
  }
  sonuc$model <- mn; sonuc$varyant <- v; sonuc$tekrar <- r
  saveRDS(sonuc, dosya)
  cat(sprintf("%-8s %-9s repetition %2d  accuracy %.3f  (%.0f s)\n", mn, v, r,
              mean(sonuc$tahmin == sonuc$gercek), as.numeric(difftime(Sys.time(), t0, units = "secs"))))
}

# ---- 4) RF feature importance (original variant, all data) -------------------
d <- data.frame(F_LISTE$orijinal, y = factor(y))
rf_onem <- ranger(y ~ ., data = d, num.trees = 300, importance = "permutation",
                  class.weights = sinif_agirlik(y), seed = TOHUM, num.threads = 1)
onem <- sort(rf_onem$variable.importance, decreasing = TRUE)
write_out(data.frame(ozellik = names(onem), onem = round(onem, 5)),
                 yol("tables", "RF_feature_importance.csv"))

# ---- 5) RF inference time (including feature extraction) - complexity table ---
X <- readRDS(yol("data", paste0("X_orijinal_", GORUNTU_BOYUT, ".rds")))
n_olc <- min(200, dim(X)[1])
t_oz <- system.time(for (i in seq_len(n_olc)) ozellik(X[i, , , ]))[["elapsed"]] / n_olc
m <- ranger(y ~ ., data = d, num.trees = 300, seed = TOHUM, num.threads = 1)
t_rf <- system.time(for (i in seq_len(n_olc)) predict(m, d[i, -41, drop = FALSE]))[["elapsed"]] / n_olc
tmp <- tempfile(fileext = ".rds"); saveRDS(m, tmp)
write_out(data.frame(
  metrik = c("feature_extraction_ms_per_image", "rf_prediction_ms_per_image", "total_ms_per_image",
             "rf_model_boyutu_MB"),
  deger = round(c(1000 * t_oz, 1000 * t_rf, 1000 * (t_oz + t_rf), file.size(tmp) / 2^20), 2)),
  yol("tables", "RF_complexity.csv"))
cat("02 done.\n")
