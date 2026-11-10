# =============================================================================
#  06_additional_analyses.R  -  Additional analyses (run after 01-05)
#
#  A) Minimal background probes: mean background colour only (3 numbers) and
#     four corner patches only (12 numbers), Random Forest, same 10x5 CV folds.
#  B) Counterfactual background swap: a leaf of class A is pasted onto the background
#     of class B. Does the prediction follow the leaf or the background?
#     Control: swap within the same class (are there compositing artefacts?).
#  C) External field test: models trained on PlantVillage are tested on Plant Pathology 2020
#     (Thapa et al. 2020) field images (scab, rust, healthy).
#     Skipped if the data folder is not present.
# =============================================================================
source("config.R")
suppressPackageStartupMessages({ library(dplyr); library(ranger); library(magick) })
dir.create(yol("predictions_extra"), showWarnings = FALSE)

liste  <- readRDS(yol("data", "image_list.rds"))
KATLAR <- readRDS(yol("data", "folds.rds"))
MASKE  <- readRDS(yol("data", "masks.rds"))
X0     <- readRDS(yol("data", sprintf("X_orijinal_%d.rds", GORUNTU_BOYUT)))
ysin   <- liste$sinif
n      <- length(ysin)
B      <- GORUNTU_BOYUT
NIR    <- max(table(ysin)) / n

# Load the feature functions and models of 02 (cached features are read)
k02 <- readLines("02_classical_models.R")
eval(parse(text = k02[1:(grep("^# ---- 3\\) Tekrarli CV", k02) - 1)]))

genislet <- function(m) {        # dilate a binary mask by 1 pixel (4-neighbourhood)
  h <- nrow(m); w <- ncol(m); g <- m
  g[-1, ] <- g[-1, ] | m[-h, ]; g[-h, ] <- g[-h, ] | m[-1, ]
  g[, -1] <- g[, -1] | m[, -w]; g[, -w] <- g[, -w] | m[, -1]
  g
}

# =============================================================================
# A) MINIMAL BACKGROUND PROBES
# =============================================================================
cat("\n--- A) Minimal background probes ---\n")
F_bg <- t(vapply(seq_len(n), function(i) {
  bg <- !genislet(MASKE[i, , ])
  if (sum(bg) < 20) bg <- !MASKE[i, , ]
  vapply(1:3, function(c) mean(X0[i, , , c][bg]), numeric(1))
}, numeric(3)))
colnames(F_bg) <- c("bg_R", "bg_G", "bg_B")
ks <- 8                                       # corner patch size (pixels, within 96x96)
koseler <- list(1:ks, (B - ks + 1):B)
F_kose <- t(vapply(seq_len(n), function(i) {
  unlist(lapply(koseler, function(r) lapply(koseler, function(c)
    vapply(1:3, function(ch) mean(X0[i, r, c, ch]), numeric(1)))))
}, numeric(12)))
kose_yaprak <- mean(vapply(seq_len(n), function(i)
  mean(c(MASKE[i, koseler[[1]], koseler[[1]]], MASKE[i, koseler[[1]], koseler[[2]]],
         MASKE[i, koseler[[2]], koseler[[1]]], MASKE[i, koseler[[2]], koseler[[2]]])), numeric(1)))

basit <- list(arkaplan_ortRGB = F_bg, kose_yamalari = F_kose)
for (nm in names(basit)) for (r in seq_len(TEKRAR_KLASIK)) {
  dosya <- yol("predictions_extra", sprintf("RF__%s__r%02d.rds", nm, r))
  if (file.exists(dosya)) next
  kat <- KATLAR[, r]; tah <- integer(n)
  for (k in seq_len(K_KAT)) {
    te <- which(kat == k); tr <- which(kat != k)
    tah[te] <- model_rf(basit[[nm]][tr, , drop = FALSE], ysin[tr], basit[[nm]][te, , drop = FALSE])$tahmin
  }
  saveRDS(data.frame(id = seq_len(n), gercek = ysin, tahmin = tah, kat = kat,
                     model = "RF", varyant = nm, tekrar = r), dosya)
}
ta <- bind_rows(lapply(list.files(yol("predictions_extra"), "^RF__(arkaplan_ortRGB|kose_yamalari)__r", full.names = TRUE), readRDS)) %>%
  group_by(varyant, tekrar, kat) %>%
  summarise(m = list(metrikler(gercek, tahmin)), .groups = "drop") %>%
  mutate(dogruluk = sapply(m, `[[`, "dogruluk"), makro_f1 = sapply(m, `[[`, "makro_f1")) %>%
  group_by(varyant) %>%
  summarise(ozellik_sayisi = ifelse(first(varyant) == "arkaplan_ortRGB", 3, 12), bolme = n(),
            dogruluk = sprintf("%.1f \u00b1 %.1f", 100 * mean(dogruluk), 100 * sd(dogruluk)),
            makro_f1 = sprintf("%.3f \u00b1 %.3f", mean(makro_f1), sd(makro_f1)), .groups = "drop")
ta$kose_yamalarinda_yaprak_orani <- ifelse(ta$varyant == "kose_yamalari", round(kose_yaprak, 4), NA)
print(ta)
write_out(ta, yol("tables", "T7_simple_probes.csv"))

# =============================================================================
# B) COUNTERFACTUAL BACKGROUND SWAP
# =============================================================================
cat("\n--- B) Counterfactual background swap ---\n")
doldur <- function(img, delik) {   # fill hole pixels from neighbours (diffusion)
  h <- dim(img)[1]; w <- dim(img)[2]
  for (ch in 1:3) {
    a <- img[, , ch]; a[delik] <- NA; it <- 0
    while (anyNA(a) && it < 200) {
      it <- it + 1
      p <- rbind(NA, a[-h, ]); s <- rbind(a[-1, ], NA)
      l <- cbind(NA, a[, -w]); r <- cbind(a[, -1], NA)
      say <- (!is.na(p)) + (!is.na(s)) + (!is.na(l)) + (!is.na(r))
      top <- ifelse(is.na(p), 0, p) + ifelse(is.na(s), 0, s) + ifelse(is.na(l), 0, l) + ifelse(is.na(r), 0, r)
      yeni <- is.na(a) & say > 0
      a[yeni] <- top[yeni] / say[yeni]
    }
    a[is.na(a)] <- mean(a, na.rm = TRUE)
    # Smoothing: relax the hole towards the neighbour mean (harmonic fill, no streaks)
    for (g in 1:150) {
      p <- rbind(a[1, ], a[-h, ]); s <- rbind(a[-1, ], a[h, ])
      l <- cbind(a[, 1], a[, -w]); r <- cbind(a[, -1], a[, w])
      a[delik] <- ((p + s + l + r) / 4)[delik]
    }
    img[, , ch] <- a
  }
  img
}
birlestir <- function(i, j) {      # leaf of i + background of j
  # 1) remove the leaf of j (dilated by 1 pixel) and fill it from the background of j only,
  #    giving a clean background plate; 2) paste the leaf of i on top.
  mi <- MASKE[i, , ]; bgj <- !genislet(MASKE[j, , ])
  plaka <- doldur(array(as.numeric(X0[j, , , ]), c(B, B, 3)), !bgj)
  img <- plaka
  for (ch in 1:3) img[, , ch] <- ifelse(mi, X0[i, , , ch], plaka[, , ch])
  img
}

set.seed(TOHUM + 7)
kat1 <- KATLAR[, 1]
ciftler <- bind_rows(lapply(seq_len(K_KAT), function(k) {
  te <- which(kat1 == k)
  bind_rows(lapply(te, function(i) {
    farkli <- te[ysin[te] != ysin[i]]; ayni <- setdiff(te[ysin[te] == ysin[i]], i)
    data.frame(kat = k, yaprak = i, arkaplan = c(farkli[sample.int(length(farkli), 1)],
                                                  ayni[sample.int(length(ayni), 1)]),
               tur = c("farkli_sinif", "ayni_sinif"))
  }))
}))
ciftler$yaprak_sinif <- ysin[ciftler$yaprak]; ciftler$arkaplan_sinif <- ysin[ciftler$arkaplan]

bf <- yol("data", "X_swap.rds")
if (file.exists(bf)) XC <- readRDS(bf) else {
  cat("Building composite images (", nrow(ciftler), ")...\n")
  XC <- array(0L, c(nrow(ciftler), B, B, 3))
  for (q in seq_len(nrow(ciftler))) {
    XC[q, , , ] <- as.integer(round(birlestir(ciftler$yaprak[q], ciftler$arkaplan[q])))
    if (q %% 1000 == 0) cat("  ", q, "\n")
  }
  saveRDS(XC, bf, compress = FALSE)
}
FC <- t(vapply(seq_len(nrow(ciftler)), function(q) ozellik(XC[q, , , ]), numeric(40)))
colnames(FC) <- colnames(F_LISTE$orijinal)

tahmin_C <- data.frame()
for (k in seq_len(K_KAT)) {          # each fold: model trained on training folds, composites built from the test fold
  tr <- which(kat1 != k); qk <- which(ciftler$kat == k)
  for (mn in names(MODELLER)) {
    set.seed(TOHUM + 100 + k)
    out <- MODELLER[[mn]](F_LISTE$orijinal[tr, ], ysin[tr], FC[qk, , drop = FALSE])
    tahmin_C <- rbind(tahmin_C, data.frame(q = qk, model = mn, tahmin = out$tahmin))
  }
}
keras_var <- requireNamespace("keras3", quietly = TRUE)
if (keras_var) {
  suppressPackageStartupMessages(library(keras3))
  np <- reticulate::import("numpy", convert = FALSE)
  for (mn in c("OzelCNN", "MobileNetV3")) for (k in seq_len(K_KAT)) {
    mf <- yol("models", sprintf("%s__orijinal__k%d.keras", mn, k))
    if (!file.exists(mf)) next
    m <- load_model(mf); qk <- which(ciftler$kat == k)
    p <- m |> predict(np$array(XC[qk, , , , drop = FALSE], dtype = "float32"), batch_size = 64, verbose = 0)
    tahmin_C <- rbind(tahmin_C, data.frame(q = qk, model = mn, tahmin = max.col(p, ties.method = "first")))
    rm(m); clear_session(); gc()
  }
}
tahmin_C <- cbind(tahmin_C, ciftler[tahmin_C$q, c("tur", "yaprak_sinif", "arkaplan_sinif")])
saveRDS(tahmin_C, yol("predictions_extra", "background_swap.rds"))
tb <- tahmin_C %>% group_by(model, tur) %>%
  summarise(n = n(), yaprak_sinifi = mean(tahmin == yaprak_sinif),
            arkaplan_sinifi = mean(tahmin == arkaplan_sinif & tur == "farkli_sinif"),
            diger = mean(tahmin != yaprak_sinif & (tahmin != arkaplan_sinif | tur == "ayni_sinif")),
            .groups = "drop") %>%
  mutate(across(c(yaprak_sinifi, arkaplan_sinifi, diger), ~ round(100 * .x, 1)))
print(tb, n = 30)
write_out(tb, yol("tables", "T8_background_swap.csv"))
# By class pair (different class): share following the background
tb2 <- tahmin_C %>% filter(tur == "farkli_sinif") %>%
  group_by(model, yaprak = SINIF_ETIKET[yaprak_sinif], arkaplan = SINIF_ETIKET[arkaplan_sinif]) %>%
  summarise(n = n(), yaprak_sinifi = round(100 * mean(tahmin == yaprak_sinif), 1),
            arkaplan_sinifi = round(100 * mean(tahmin == arkaplan_sinif), 1), .groups = "drop")
write_out(tb2, yol("tables", "T8b_swap_class_pairs.csv"))

# Example figure
set.seed(3)
orn <- unlist(lapply(1:4, function(s) { w <- which(ciftler$tur == "farkli_sinif" & ciftler$yaprak_sinif == s)
  w[sample.int(length(w), 1)] }))
tah_orn <- tahmin_C %>% filter(q %in% orn)
png(yol("figures", "S_background_swap.png"), width = 2000, height = 1500, res = 170)
par(mfrow = c(3, 4), mar = c(0.3, 0.3, 2.6, 0.3))
for (satir in 1:3) for (q in orn) {
  i <- ciftler$yaprak[q]; j <- ciftler$arkaplan[q]
  img <- switch(satir, X0[i, , , ], X0[j, , , ], XC[q, , , ])
  plot(as.raster(img / 255))
  ttl <- switch(satir, paste("Leaf source:", SINIF_ETIKET[ysin[i]]),
                paste("Background source:", SINIF_ETIKET[ysin[j]]), {
    tt <- tah_orn %>% filter(q == !!q)
    paste0("Composite\n", paste(sprintf("%s: %s", tt$model,
           SINIF_ETIKET[tt$tahmin]), collapse = "; "))
  })
  title(ttl, cex.main = if (satir == 3) 0.45 else 0.75)
}
dev.off()

# =============================================================================
# C) EXTERNAL FIELD TEST: PLANT PATHOLOGY 2020
PP_KOK <- Sys.getenv("PV_PP2020", unset = "PlantPathology2020")
PP_KOK <- Sys.getenv("PV_PP2020", unset = file.path(dirname(normalizePath(PV_KOK, mustWork = FALSE)), "..", "PlantPathology2020"))
PP_KOK <- normalizePath(PP_KOK, mustWork = FALSE)
if (!file.exists(file.path(PP_KOK, "train.csv"))) {
  cat("\n--- C) Plant Pathology 2020 not found (", PP_KOK, ") -> skipped ---\n")
} else {
  cat("\n--- C) External field test: Plant Pathology 2020 ---\n")
  pp <- read.csv(file.path(PP_KOK, "train.csv"))
  pp <- pp[pp$multiple_diseases == 0, ]
  pp$sinif <- ifelse(pp$scab == 1, 1L, ifelse(pp$rust == 1, 3L, ifelse(pp$healthy == 1, 4L, NA)))
  pp <- pp[!is.na(pp$sinif), ]
  if (HIZLI_TEST) pp <- pp[unlist(lapply(split(seq_len(nrow(pp)), pp$sinif), head, 30)), ]
  pf <- yol("data", sprintf("X_pp2020_%d.rds", B))
  if (file.exists(pf) && dim(readRDS(pf))[1] == nrow(pp)) XP <- readRDS(pf) else {
    cat("Reading PP2020 images (", nrow(pp), ")...\n")
    XP <- array(0L, c(nrow(pp), B, B, 3))
    for (q in seq_len(nrow(pp))) {
      f <- file.path(PP_KOK, "images", paste0(pp$image_id[q], ".jpg"))
      im <- image_read(f); inf <- image_info(im); s <- min(inf$width, inf$height)
      im <- image_crop(im, sprintf("%dx%d+%d+%d", s, s, (inf$width - s) %/% 2, (inf$height - s) %/% 2)) |>
        image_resize(sprintf("%dx%d!", B, B))
      XP[q, , , ] <- as.integer(image_data(im, "rgb"))
      if (q %% 250 == 0) cat("  ", q, "\n")
    }
    saveRDS(XP, pf, compress = FALSE)
  }
  FP <- t(vapply(seq_len(nrow(pp)), function(q) ozellik(XP[q, , , ]), numeric(40)))
  colnames(FP) <- colnames(F_LISTE$orijinal)
  yp <- pp$sinif
  tum <- seq_len(n)
  sonucP <- data.frame()
  for (mn in names(MODELLER)) {
    set.seed(TOHUM + 500)
    out <- MODELLER[[mn]](F_LISTE$orijinal, ysin, FP)
    sonucP <- rbind(sonucP, data.frame(model = mn, tohum = 1, id = seq_along(yp), gercek = yp, tahmin = out$tahmin))
  }
  if (keras_var) {
    k03 <- readLines("03_deep_models.R")
    eval(parse(text = k03[1:(grep("^# ---- Deney listesi", k03) - 1)]))
    X_pv <- X0
    for (s in 1:3) {
      for (mn in c("OzelCNN", "MobileNetV3")) {
        fn <- if (mn == "OzelCNN") function() ozel_cnn(B) else function() mobilenet(B)
        out <- kat_egit(X_pv, tum, 1:2, fn, TRUE, tohum = TOHUM + 900 + s)
        p <- out$model |> predict(np$array(XP, dtype = "float32"), batch_size = 64, verbose = 0)
        sonucP <- rbind(sonucP, data.frame(model = mn, tohum = s, id = seq_along(yp), gercek = yp,
                                           tahmin = max.col(p, ties.method = "first")))
        cat(sprintf("%-12s seed %d  PP2020 accuracy %.3f\n", mn, s, mean(sonucP$tahmin[sonucP$model == mn & sonucP$tohum == s] == yp)))
        rm(out); clear_session(); gc()
      }
    }
  }
  saveRDS(sonucP, yol("predictions_extra", "pp2020.rds"))
  NIR_pp <- max(table(yp)) / length(yp)
  mevcut <- sort(unique(yp))
  tp <- sonucP %>% group_by(model, tohum) %>%
    summarise(dogruluk = mean(tahmin == gercek),
              makro_f1_3sinif = metrikler(gercek, tahmin)$f1[mevcut] |> mean(),
              scab_duyarlilik = mean(tahmin[gercek == 1] == 1),
              rust_duyarlilik = mean(tahmin[gercek == 3] == 3),
              healthy_duyarlilik = mean(tahmin[gercek == 4] == 4),
              blackrot_tahmin_orani = mean(tahmin == 2), .groups = "drop") %>%
    group_by(model) %>%
    summarise(across(-tohum, ~ round(100 * mean(.x), 1)), tohum_sayisi = n()) %>%
    mutate(n_goruntu = length(yp), NIR = round(100 * NIR_pp, 1))
  print(tp)
  write_out(tp, yol("tables", "T9_external_test_pp2020.csv"))
  dag <- as.data.frame(table(sinif = SINIF_ETIKET[yp])); write_out(dag, yol("tables", "T9b_pp2020_class_distribution.csv"))
  # confusion (first seed of each model)
  km <- sonucP %>% filter(tohum == 1) %>% count(model, gercek = SINIF_ETIKET[gercek], tahmin = SINIF_ETIKET[tahmin])
  write_out(km, yol("tables", "T9c_pp2020_confusion.csv"))
}
cat("\n06 done.\n")
