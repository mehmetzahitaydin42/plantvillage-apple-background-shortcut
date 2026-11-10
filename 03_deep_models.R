# =============================================================================
#  03_deep_models.R
#  (a) Lightweight CNN (architecture of the paper, 257,860 parameters)
#  (b) MobileNetV3-Small, ImageNet weights, frozen backbone (transfer learning)
#  Main experiment: 2 models x 4 variants x TEKRAR_DERIN x 5 folds
#  Ablation       : lightweight CNN + data augmentation / without class weights (original, all folds)
#  Resolution     : lightweight CNN at 128x128 (original)
#  Every fold is saved when it finishes; if interrupted, the script resumes where it stopped.
# =============================================================================
source("config.R")
suppressPackageStartupMessages({ library(keras3); library(dplyr) })
np <- reticulate::import("numpy", convert = FALSE)

liste  <- readRDS(yol("data", "image_list.rds"))
KATLAR <- readRDS(yol("data", "folds.rds"))
y <- liste$sinif - 1L                     # 0..3 for keras
n_sinif <- length(SINIFLAR)

# ImageNet weights are downloaded. Without internet, set PV_IMAGENET=FALSE for random
# initialisation (results are then NOT valid; for code testing only).
IMAGENET <- as.logical(Sys.getenv("PV_IMAGENET", unset = "TRUE"))

# ---- Models -------------------------------------------------------------------
ozel_cnn <- function(boyut, artirma = FALSE) {
  reg <- regularizer_l2(1e-4)
  girdi <- keras_input(shape = c(boyut, boyut, 3))
  x <- girdi
  if (artirma) x <- x |> layer_random_flip("horizontal_and_vertical") |>
      layer_random_rotation(0.1) |> layer_random_zoom(0.1) |>
      layer_random_brightness(0.1, value_range = c(0, 255))
  x <- x |> layer_rescaling(1 / 255)
  filtreler <- c(32, 64, 128, 128)
  for (i in seq_along(filtreler)) {
    x <- x |> layer_conv_2d(filtreler[i], 3, padding = "same", activation = "relu",
                            kernel_regularizer = reg,
                            name = if (i == 4) "son_konv" else NULL) |>
      layer_max_pooling_2d(2)
  }
  cikti <- x |> layer_global_average_pooling_2d() |> layer_dropout(0.4) |>
    layer_dense(128, activation = "relu", kernel_regularizer = reg) |>
    layer_dropout(0.4) |> layer_dense(n_sinif, activation = "softmax")
  m <- keras_model(girdi, cikti)
  m |> compile(optimizer = optimizer_adam(OGR_HIZI),
               loss = "sparse_categorical_crossentropy", metrics = "accuracy")
  m
}

mobilenet <- function(boyut) {
  govde <- application_mobilenet_v3_small(
    include_top = FALSE, weights = if (IMAGENET) "imagenet" else NULL,
    input_shape = c(boyut, boyut, 3), include_preprocessing = TRUE)  # input 0-255
  govde$trainable <- FALSE
  girdi <- keras_input(shape = c(boyut, boyut, 3))
  cikti <- girdi |> govde(training = FALSE) |> layer_global_average_pooling_2d() |>
    layer_dropout(0.3) |> layer_dense(n_sinif, activation = "softmax")
  m <- keras_model(girdi, cikti)
  m |> compile(optimizer = optimizer_adam(OGR_HIZI),
               loss = "sparse_categorical_crossentropy", metrics = "accuracy")
  m
}

# ---- Training of one fold -------------------------------------------------------
kat_egit <- function(X, tr, te, model_fn, agirlikli = TRUE, tohum) {
  set.seed(tohum); set_random_seed(tohum)
  # 15% of the training portion as a stratified validation subset
  dg <- unlist(lapply(split(tr, y[tr]), function(ix)
    ix[sample.int(length(ix), round(DOGRULAMA_ORAN * length(ix)))]))
  eg <- setdiff(tr, dg)
  tb <- table(factor(y[eg], levels = 0:(n_sinif - 1)))
  w  <- as.numeric(sum(tb) / (n_sinif * pmax(tb, 1)))
  sw <- if (agirlikli) w[y[eg] + 1] else rep(1, length(eg))
  m <- model_fn()
  t0 <- Sys.time()
  gecmis <- m |> fit(
    np$array(X[eg, , , , drop = FALSE], dtype = "float32"), y[eg],
    sample_weight = np$array(sw, dtype = "float32"),
    validation_data = list(np$array(X[dg, , , , drop = FALSE], dtype = "float32"), y[dg]),
    epochs = EPOCH_MAKS, batch_size = BATCH, verbose = 0,
    callbacks = list(callback_early_stopping(monitor = "val_accuracy", mode = "max",
                                             patience = SABIR, restore_best_weights = TRUE)))
  sure <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  p <- m |> predict(np$array(X[te, , , , drop = FALSE], dtype = "float32"),
                    batch_size = 64, verbose = 0)
  list(model = m, tahmin = max.col(p, ties.method = "first"), olasilik = p,
       sure = sure, epoch = length(gecmis$metrics$loss),
       en_iyi_epoch = which.max(gecmis$metrics$val_accuracy), gecmis = gecmis$metrics)
}

# ---- Deney listesi (experiment list) ------------------------------------------
deneyler <- list()
for (v in VARYANTLAR) {
  deneyler[[length(deneyler) + 1]] <- list(ad = "OzelCNN", varyant = v, boyut = GORUNTU_BOYUT,
    fn = function() ozel_cnn(GORUNTU_BOYUT), agirlikli = TRUE)
  deneyler[[length(deneyler) + 1]] <- list(ad = "MobileNetV3", varyant = v, boyut = GORUNTU_BOYUT,
    fn = function() mobilenet(GORUNTU_BOYUT), agirlikli = TRUE)
}
deneyler[[length(deneyler) + 1]] <- list(ad = "OzelCNN_artirma", varyant = "orijinal",
  boyut = GORUNTU_BOYUT, fn = function() ozel_cnn(GORUNTU_BOYUT, artirma = TRUE), agirlikli = TRUE)
deneyler[[length(deneyler) + 1]] <- list(ad = "OzelCNN_agirliksiz", varyant = "orijinal",
  boyut = GORUNTU_BOYUT, fn = function() ozel_cnn(GORUNTU_BOYUT), agirlikli = FALSE)
for (b in setdiff(COZUNURLUKLER, GORUNTU_BOYUT))
  deneyler[[length(deneyler) + 1]] <- local({ bb <- b; list(ad = paste0("OzelCNN_", bb),
    varyant = "orijinal", boyut = bb, fn = function() ozel_cnn(bb), agirlikli = TRUE) })

# ---- Main loop --------------------------------------------------------------------
dir.create(yol("predictions", "per_fold"), showWarnings = FALSE)
X_onbellek <- NULL; X_ad <- ""
egitim_kayit <- list()
for (d in deneyler) for (r in seq_len(TEKRAR_DERIN)) {
  hedef <- yol("predictions", sprintf("%s__%s__r%02d.rds", d$ad, d$varyant, r))
  if (file.exists(hedef)) next
  x_dosya <- yol("data", sprintf("X_%s_%d.rds", d$varyant, d$boyut))
  if (X_ad != x_dosya) { rm(X_onbellek); gc(); X_onbellek <- readRDS(x_dosya); X_ad <- x_dosya }
  kat <- KATLAR[, r]
  for (k in seq_len(K_KAT)) {
    kat_dosya <- yol("predictions", "per_fold", sprintf("%s__%s__r%02d__k%d.rds", d$ad, d$varyant, r, k))
    if (file.exists(kat_dosya)) next
    te <- which(kat == k); tr <- which(kat != k)
    out <- kat_egit(X_onbellek, tr, te, d$fn, d$agirlikli, tohum = TOHUM + 100 * r + k)
    saveRDS(list(te = te, tahmin = out$tahmin, olasilik = out$olasilik, sure = out$sure,
                 epoch = out$epoch, en_iyi_epoch = out$en_iyi_epoch, gecmis = out$gecmis),
            kat_dosya)
    # keep the models of repetition 1 for Grad-CAM and complexity
    if (r == 1 && d$ad %in% c("OzelCNN", "MobileNetV3") && d$varyant %in% c("orijinal", "yaprak"))
      save_model(out$model, yol("models", sprintf("%s__%s__k%d.keras", d$ad, d$varyant, k)),
                 overwrite = TRUE)
    cat(sprintf("%-18s %-9s r%02d k%d  accuracy %.3f  epoch %2d (best %2d)  %.0f s\n",
                d$ad, d$varyant, r, k, mean(out$tahmin == y[te] + 1), out$epoch,
                out$en_iyi_epoch, out$sure))
    rm(out); clear_session(); gc()
  }
  # all 5 folds finished -> merge into one file (same format as the classical models)
  parcalar <- lapply(seq_len(K_KAT), function(k) readRDS(
    yol("predictions", "per_fold", sprintf("%s__%s__r%02d__k%d.rds", d$ad, d$varyant, r, k))))
  sonuc <- data.frame(id = seq_along(y), gercek = y + 1L, tahmin = NA_integer_, kat = kat)
  for (p in parcalar) sonuc$tahmin[p$te] <- p$tahmin
  sonuc$model <- d$ad; sonuc$varyant <- d$varyant; sonuc$tekrar <- r
  saveRDS(sonuc, hedef)
  egitim_kayit[[length(egitim_kayit) + 1]] <- data.frame(
    model = d$ad, varyant = d$varyant, tekrar = r, kat = seq_len(K_KAT),
    sure_sn = sapply(parcalar, `[[`, "sure"), epoch = sapply(parcalar, `[[`, "epoch"),
    en_iyi_epoch = sapply(parcalar, `[[`, "en_iyi_epoch"))
}
if (length(egitim_kayit)) {
  ek <- do.call(rbind, egitim_kayit)
  f <- yol("predictions", "training_times_raw.csv")       # internal log, appended across runs
  if (file.exists(f)) ek <- rbind(readr::read_csv(f, show_col_types = FALSE), ek)
  readr::write_csv(ek, f)
  write_out(ek, yol("tables", "training_times.csv"))      # English copy
}

# ---- Complexity measurements ------------------------------------------------------
X1 <- np$array(readRDS(yol("data", sprintf("X_orijinal_%d.rds", GORUNTU_BOYUT)))[1:64, , , ,
                drop = FALSE], dtype = "float32")
olc <- function(ad, m) {
  wf <- tempfile(fileext = ".weights.h5"); save_model_weights(m, wf)
  kf <- tempfile(fileext = ".keras"); save_model(m, kf)
  x1 <- reticulate::py_eval("lambda a: a[0:1]")(X1)
  for (i in 1:5) m(x1, training = FALSE)                     # warm-up
  t_tek <- system.time(for (i in 1:50) m(x1, training = FALSE))[["elapsed"]] / 50
  t_toplu <- system.time(for (i in 1:5) m(X1, training = FALSE))[["elapsed"]] / (5 * 64)
  data.frame(model = ad,
             egitilebilir_parametre = sum(sapply(m$trainable_weights, function(w) length(as.array(w)))),
             toplam_parametre = m$count_params(),
             agirlik_dosyasi_MB = round(file.size(wf) / 2^20, 2),
             keras_dosyasi_MB = round(file.size(kf) / 2^20, 2),
             tek_goruntu_ms = round(1000 * t_tek, 1), toplu_ms_goruntu = round(1000 * t_toplu, 2))
}
# Analytical FLOPs of the lightweight CNN (multiply + add = 2 x MAC), 3x3 'same' convolutions
flop_ozel <- function(b) { f <- c(3, 32, 64, 128, 128); s <- b / 2^(0:3); tot <- 0
  for (i in 1:4) tot <- tot + 2 * s[i]^2 * 9 * f[i] * f[i + 1]
  tot + 2 * (128 * 128 + 128 * n_sinif) }
kar <- rbind(olc("OzelCNN", ozel_cnn(GORUNTU_BOYUT)), olc("MobileNetV3", mobilenet(GORUNTU_BOYUT)))
kar$MFLOP <- c(round(flop_ozel(GORUNTU_BOYUT) / 1e6, 1), NA)
print(kar)
write_out(kar, yol("tables", "CNN_complexity.csv"))
cat("03 done.\n")
