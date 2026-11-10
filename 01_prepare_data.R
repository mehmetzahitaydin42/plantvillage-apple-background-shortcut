# =============================================================================
#  01_prepare_data.R
#  - Lists the images and matches original <-> segmented files
#  - Removes near-duplicate images with dHash
#  - Derives leaf masks from the segmented images (filling interior holes)
#  - Saves image arrays for the four variants (original / leaf / background / silhouette)
#  - Saves the repeated stratified CV folds shared by all models
# =============================================================================
source("config.R")
suppressPackageStartupMessages({ library(magick); library(dplyr) })
set.seed(TOHUM)

# ---- 1) File list and matching ---------------------------------------------
liste <- do.call(rbind, lapply(seq_along(SINIFLAR), function(i) {
  s <- SINIFLAR[i]
  org <- list.files(file.path(PV_KOK, "color", s), full.names = TRUE)
  anahtar <- tools::file_path_sans_ext(basename(org))
  seg_dosya <- file.path(PV_KOK, "segmented", s, paste0(anahtar, "_final_masked.jpg"))
  data.frame(sinif = i, org = org, seg = seg_dosya, stringsAsFactors = FALSE)
}))
cat("Total images:", nrow(liste), "\n")
eksik_seg <- !file.exists(liste$seg)
if (any(eksik_seg)) {
  cat("Images without a segmented counterpart:", sum(eksik_seg), "-> removed\n")
  liste <- liste[!eksik_seg, ]
}
if (HIZLI_TEST) liste <- liste %>% group_by(sinif) %>%
  slice_sample(n = HIZLI_N_SINIF) %>% ungroup() %>% as.data.frame()

# ---- 2) Near-duplicate removal with dHash -----------------------------------
dhash <- function(dosya) {
  g <- image_read(dosya) |> image_convert(colorspace = "gray") |>
    image_resize("9x8!")
  m <- as.integer(image_data(g, "gray"))[, , 1]   # size: 8 rows x 9 columns
  as.vector(m[, -1] > m[, -9])                   # 64 bits
}
cat("Computing dHash...\n")
H <- t(vapply(liste$org, dhash, logical(64)))
# Hamming distances (64 bits, fast via binary matrix products)
Hn <- H * 1
ham <- outer(rowSums(Hn), rowSums(Hn), "+") - 2 * tcrossprod(Hn)
diag(ham) <- 99
kopya <- rep(FALSE, nrow(liste))
for (i in seq_len(nrow(liste))) {
  if (kopya[i]) next
  es <- which(ham[i, ] <= DHASH_ESIK & seq_len(nrow(liste)) > i)
  kopya[es] <- TRUE                     # the first image of each cluster is kept
}
capraz <- sum(vapply(which(kopya), function(j)
  any(liste$sinif[ham[j, ] <= DHASH_ESIK] != liste$sinif[j]), logical(1)))
cat(sprintf("Removed as near-duplicates: %d (%.2f%%); matching a different class: %d\n",
            sum(kopya), 100 * mean(kopya), capraz))
liste <- liste[!kopya, ]
rownames(liste) <- NULL

# ---- 3) Image reading, masks and variants ------------------------------------
yaprak_maskesi <- function(seg_img) {
  # Dark pixels (background candidates) -> black, others white
  a <- as.integer(image_data(seg_img, "rgb"))
  koyu <- apply(a, c(1, 2), mean) < MASKE_ESIK
  b <- image_read(array(as.numeric(!koyu), dim = c(dim(koyu), 1)))
  # Flood-fill the dark region connected to the image corners -> true background.
  # (Dark lesions inside the leaf are not connected to the border, so they stay in the leaf.)
  h <- nrow(koyu); w <- ncol(koyu)
  koseler <- list(c(0, 0), c(w - 1, 0), c(0, h - 1), c(w - 1, h - 1))
  kose_koyu <- c(koyu[1, 1], koyu[1, w], koyu[h, 1], koyu[h, w])
  b <- image_convert(b, type = "TrueColor")
  for (k in which(kose_koyu))
    b <- image_fill(b, "red", point = sprintf("+%d+%d", koseler[[k]][1], koseler[[k]][2]), fuzz = 0)
  bb <- as.integer(image_data(b, "rgb"))
  arka <- bb[, , 1] > 200 & bb[, , 2] < 50 & bb[, , 3] < 50
  !arka                                       # TRUE = leaf
}

boyutlandir <- function(arr01, boyut) {
  img <- image_read(arr01) |> image_resize(sprintf("%dx%d!", boyut, boyut))
  as.integer(image_data(img, "rgb"))           # y x x x 3, 0-255
}

boyutlar <- sort(unique(c(GORUNTU_BOYUT, COZUNURLUKLER)))
n <- nrow(liste)
X <- list()
for (b in boyutlar) X[[paste0("orijinal_", b)]] <- array(0L, c(n, b, b, 3))
X[[paste0("yaprak_", GORUNTU_BOYUT)]]   <- array(0L, c(n, GORUNTU_BOYUT, GORUNTU_BOYUT, 3))
X[[paste0("arkaplan_", GORUNTU_BOYUT)]] <- array(0L, c(n, GORUNTU_BOYUT, GORUNTU_BOYUT, 3))
X[[paste0("siluet_", GORUNTU_BOYUT)]]   <- array(0L, c(n, GORUNTU_BOYUT, GORUNTU_BOYUT, 3))
MASKE <- array(FALSE, c(n, GORUNTU_BOYUT, GORUNTU_BOYUT))
yaprak_orani <- numeric(n)

cat("Processing images (", n, ")...\n")
pb <- if (interactive()) txtProgressBar(0, n, style = 3) else NULL
for (i in seq_len(n)) {
  org <- image_read(liste$org[i]); seg <- image_read(liste$seg[i])
  if (any(image_info(org)[, c("width", "height")] != image_info(seg)[, c("width", "height")]))
    seg <- image_resize(seg, sprintf("%dx%d!", image_info(org)$width, image_info(org)$height))
  m  <- yaprak_maskesi(seg)
  # Dilate the mask by ~4 pixels for the background variant so that no leaf margin leaks
  mg <- image_read(array(as.numeric(m), dim = c(dim(m), 1))) |>
    image_morphology("Dilate", "Disk:4")
  mg <- as.integer(image_data(mg, "gray"))[, , 1] > 127
  o  <- as.integer(image_data(org, "rgb")) / 255
  ap <- o * array(!mg, dim(o))                        # leaf black, background kept
  yp <- o * array(m, dim(o))                          # original colours, background black
  for (b in boyutlar) X[[paste0("orijinal_", b)]][i, , , ] <- boyutlandir(o, b)
  X[[paste0("yaprak_", GORUNTU_BOYUT)]][i, , , ]   <- boyutlandir(yp, GORUNTU_BOYUT)
  X[[paste0("arkaplan_", GORUNTU_BOYUT)]][i, , , ] <- boyutlandir(ap, GORUNTU_BOYUT)
  mk <- image_read(array(as.numeric(m), dim = c(dim(m), 1))) |>
    image_resize(sprintf("%dx%d!", GORUNTU_BOYUT, GORUNTU_BOYUT))
  MASKE[i, , ] <- as.integer(image_data(mk, "gray"))[, , 1] > 127
  X[[paste0("siluet_", GORUNTU_BOYUT)]][i, , , ] <- 255L * as.integer(MASKE[i, , ])
  yaprak_orani[i] <- mean(m)
  if (!is.null(pb)) setTxtProgressBar(pb, i) else if (i %% 500 == 0) cat(i, "/", n, "\n")
}
if (!is.null(pb)) close(pb)
liste$yaprak_orani <- yaprak_orani
cat(sprintf("\nMean leaf area: %.1f%% (min %.1f, max %.1f)\n",
            100 * mean(yaprak_orani), 100 * min(yaprak_orani), 100 * max(yaprak_orani)))

# ---- 4) Repeated stratified CV folds -----------------------------------------
tekrar_maks <- max(TEKRAR_KLASIK, TEKRAR_DERIN)
KATLAR <- sapply(seq_len(tekrar_maks), function(r) {
  set.seed(TOHUM + r)
  k <- integer(n)
  for (s in unique(liste$sinif)) {
    idx <- which(liste$sinif == s)
    k[idx] <- sample(rep_len(seq_len(K_KAT), length(idx)))
  }
  k
})
# KATLAR: n x repetition matrix; KATLAR[i, r] = test fold of image i in repetition r

# ---- 5) Save -------------------------------------------------------------------
for (nm in names(X)) saveRDS(X[[nm]], yol("data", paste0("X_", nm, ".rds")), compress = FALSE)
saveRDS(MASKE, yol("data", "masks.rds"))
saveRDS(liste, yol("data", "image_list.rds"))
saveRDS(KATLAR, yol("data", "folds.rds"))

dagilim <- liste %>% count(sinif) %>%
  mutate(sinif = SINIF_ETIKET[sinif], oran = round(100 * n / sum(n), 1))
print(dagilim)
write_out(dagilim, yol("tables", "T1_class_distribution.csv"))
writeLines(c(sprintf("initial_images=%d", n + sum(kopya)),
             sprintf("near_duplicates_removed=%d", sum(kopya)),
             sprintf("dhash_esik=%d", DHASH_ESIK),
             sprintf("mean_leaf_area=%.4f", mean(yaprak_orani))),
           yol("tables", "data_summary.txt"))

# Check figure: four variants + mask for a few examples
png(yol("figures", "S0_variant_examples.png"), width = 2000, height = 1100, res = 150)
par(mfrow = c(4, 5), mar = c(0.5, 0.5, 2, 0.5))
ornek <- sapply(unique(liste$sinif), function(s) which(liste$sinif == s)[1])
for (i in ornek) {
  for (v in c("orijinal", "yaprak", "arkaplan", "siluet")) {
    plot(as.raster(X[[paste0(v, "_", GORUNTU_BOYUT)]][i, , , ] / 255))
    title(paste(SINIF_ETIKET[liste$sinif[i]], "-", c(orijinal = "original", yaprak = "leaf only", arkaplan = "background only", siluet = "silhouette only")[v]), cex.main = 0.8)
  }
  plot(as.raster(X[[paste0("orijinal_", GORUNTU_BOYUT)]][i, , , ] / 255 *
                 array(MASKE[i, , ], c(GORUNTU_BOYUT, GORUNTU_BOYUT, 3)) * 0.5 + 0.5 *
                 array(MASKE[i, , ], c(GORUNTU_BOYUT, GORUNTU_BOYUT, 3))))
  title("mask check", cex.main = 0.8)
}
dev.off()
cat("01 done. Please check", yol("figures", "S0_variant_examples.png"), "\n")
