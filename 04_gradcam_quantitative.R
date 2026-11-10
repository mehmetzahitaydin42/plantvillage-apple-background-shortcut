# =============================================================================
#  04_gradcam_quantitative.R
#  Evaluates Grad-CAM QUANTITATIVELY (repetition 1, all test images of the 5 folds):
#   - Leaf attention ratio (EGO): sum(CAM * mask) / sum(CAM)
#     Expected value for uniform attention = leaf area fraction. EGO > area means
#     that the model attends to the leaf more than chance.
#   - Pointing game: does the maximum of the CAM lie on the leaf?
#  The model trained on original images is compared with the model trained on leaf-only images.
# =============================================================================
source("config.R")
suppressPackageStartupMessages({ library(keras3); library(tensorflow); library(dplyr); library(ggplot2) })
np <- reticulate::import("numpy", convert = FALSE)

liste  <- readRDS(yol("data", "image_list.rds"))
KATLAR <- readRDS(yol("data", "folds.rds"))
MASKE  <- readRDS(yol("data", "masks.rds"))
B <- GORUNTU_BOYUT

gradcam <- function(m, x_r) {                 # x_r: R array n x B x B x 3 (0-255)
  gm <- keras_model(m$inputs, list(m$get_layer("son_konv")$output, m$output))
  x <- tf$constant(np$array(x_r, dtype = "float32"))
  with(tf$GradientTape() %as% tape, {
    out <- gm(x, training = FALSE); konv <- out[[1]]; p <- out[[2]]
    sinif <- tf$argmax(p, axis = 1L)
    skor <- tf$gather(p, sinif, axis = 1L, batch_dims = 1L)
  })
  g <- tape$gradient(skor, konv)
  w <- tf$reduce_mean(g, axis = c(1L, 2L), keepdims = TRUE)
  cam <- tf$nn$relu(tf$reduce_sum(konv * w, axis = -1L, keepdims = TRUE))
  cam <- tf$image$resize(cam, as.integer(c(B, B)), method = "bilinear")
  list(cam = as.array(cam)[, , , 1, drop = TRUE], tahmin = as.integer(as.array(sinif)) + 1L)
}

sonuc <- list(); ornek_cam <- list()
ornek_id <- unlist(lapply(seq_along(SINIFLAR), function(s) head(which(liste$sinif == s), 2)))
for (v in c("orijinal", "yaprak")) {
  X <- readRDS(yol("data", sprintf("X_%s_%d.rds", v, B)))
  for (k in seq_len(K_KAT)) {
    mf <- yol("models", sprintf("OzelCNN__%s__k%d.keras", v, k))
    if (!file.exists(mf)) { message("Model not found: ", mf, " (run 03 first)"); next }
    m <- load_model(mf)
    te <- which(KATLAR[, 1] == k)
    for (parca in split(te, ceiling(seq_along(te) / 64))) {
      gc_out <- gradcam(m, X[parca, , , , drop = FALSE])
      cam <- gc_out$cam; if (length(parca) == 1) cam <- array(cam, c(1, B, B))
      for (j in seq_along(parca)) {
        i <- parca[j]; c_ <- cam[j, , ]; mk <- MASKE[i, , ]
        sonuc[[length(sonuc) + 1]] <- data.frame(
          egitim_varyanti = v, id = i, sinif = liste$sinif[i],
          dogru = gc_out$tahmin[j] == liste$sinif[i],
          ego = if (sum(c_) > 0) sum(c_ * mk) / sum(c_) else NA,
          alan = mean(mk),
          isaret = mk[which.max(c_)])
        if (i %in% ornek_id) ornek_cam[[paste(v, i)]] <- c_
      }
    }
    rm(m); clear_session(); gc()
  }
}
sonuc <- do.call(rbind, sonuc)
saveRDS(sonuc, yol("predictions", "gradcam_quantitative.rds"))

ozet <- sonuc %>% filter(!is.na(ego)) %>%
  group_by(egitim_varyanti, sinif = SINIF_ETIKET[sinif]) %>%
  summarise(n = n(), EGO_ort = mean(ego), EGO_ss = sd(ego), yaprak_alani = mean(alan),
            fazla_odak = mean(ego - alan), isaret_oyunu = mean(isaret), .groups = "drop") %>%
  bind_rows(sonuc %>% filter(!is.na(ego)) %>% group_by(egitim_varyanti) %>%
              summarise(sinif = "Tumu", n = n(), EGO_ort = mean(ego), EGO_ss = sd(ego),
                        yaprak_alani = mean(alan), fazla_odak = mean(ego - alan),
                        isaret_oyunu = mean(isaret), .groups = "drop")) %>%
  mutate(across(where(is.numeric) & !n, ~ round(.x, 3)))
print(ozet, n = 50)
write_out(ozet, yol("tables", "T_gradcam_quantitative.csv"))

# Paired test: original-trained vs leaf-trained model on the same image
gen <- sonuc %>% select(egitim_varyanti, id, ego) %>%
  tidyr::pivot_wider(names_from = egitim_varyanti, values_from = ego) %>% na.omit()
if (nrow(gen) > 5) {
  wt <- wilcox.test(gen$orijinal, gen$yaprak, paired = TRUE)
  writeLines(sprintf("Leaf attention ratio: original-trained median %.3f, leaf-trained median %.3f, Wilcoxon p = %.2e (n = %d)",
                     median(gen$orijinal), median(gen$yaprak), wt$p.value, nrow(gen)),
             yol("tables", "gradcam_test.txt"))
}

# Figure: example images and Grad-CAM maps of both models
X0 <- readRDS(yol("data", sprintf("X_orijinal_%d.rds", B)))
renk <- grDevices::colorRampPalette(c("#000080", "#0000FF", "#00FFFF", "#FFFF00", "#FF0000"))(256)
bindir <- function(img, c_) {
  c_ <- if (max(c_) > 0) c_ / max(c_) else c_
  h <- grDevices::col2rgb(renk[pmax(1, ceiling(c_ * 255))]) / 255
  h <- aperm(array(t(h), c(B, B, 3)), c(1, 2, 3))
  0.55 * img / 255 + 0.45 * h
}
png(yol("figures", "S_gradcam_examples.png"), width = 2400, height = 950, res = 170)
par(mfrow = c(3, length(ornek_id)), mar = c(0.3, 0.3, 1.6, 0.3))
for (satir in c("goruntu", "orijinal", "yaprak")) for (i in ornek_id) {
  if (satir == "goruntu") { plot(as.raster(X0[i, , , ] / 255)); title(SINIF_ETIKET[liste$sinif[i]], cex.main = 0.75) }
  else {
    c_ <- ornek_cam[[paste(satir, i)]]
    if (is.null(c_)) { plot.new(); next }
    plot(as.raster(bindir(X0[i, , , ], c_)))
    title(if (satir == "orijinal") "Trained on original images" else "Trained on leaf-only images", cex.main = 0.65)
  }
}
dev.off()
cat("04 done.\n")
