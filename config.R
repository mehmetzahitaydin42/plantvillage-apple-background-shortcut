# =============================================================================
#  config.R  -  Shared settings for all scripts
#  Only the paths and numbers in this file need to be changed.
# =============================================================================

# --- 1) Path to the PlantVillage 'raw' folder --------------------------------
# This folder must contain the "color" and "segmented" sub-folders:
#   PV_KOK/color/Apple___Apple_scab/...JPG
#   PV_KOK/segmented/Apple___Apple_scab/..._final_masked.jpg
# Download: https://github.com/spMohanty/PlantVillage-Dataset  (raw/ folder); RUN.R downloads the apple images automatically.
PV_KOK <- Sys.getenv("PV_KOK", unset = "PlantVillage-Dataset/raw")

# --- 2) Output folder --------------------------------------------------------
CIKTI <- Sys.getenv("PV_CIKTI", unset = "results")

# --- 3) Quick test mode ------------------------------------------------------
# TRUE  : few images per class, few repetitions and epochs -> checks in 5-10 minutes
#         that the code runs. Results are NOT meaningful.
# FALSE : full analysis reported in the paper.
HIZLI_TEST <- as.logical(Sys.getenv("PV_HIZLI", unset = "FALSE"))

# --- 4) Experimental design ---------------------------------------------------
SINIFLAR <- c("Apple___Apple_scab", "Apple___Black_rot",
              "Apple___Cedar_apple_rust", "Apple___healthy")
SINIF_ETIKET <- c("Apple scab", "Black rot", "Cedar apple rust", "Healthy")

VARYANTLAR <- c("orijinal", "yaprak", "arkaplan", "siluet")
#   orijinal (original)  : PlantVillage colour image (leaf + background)
#   yaprak   (leaf only) : original leaf pixels, background black (mask from PlantVillage 'segmented')
#   arkaplan (background): leaf (+4-pixel margin) set to black, ONLY the background remains
#   siluet   (silhouette): binary leaf mask only (shape/size control; no colour or texture)

K_KAT          <- 5      # number of cross-validation folds
TEKRAR_KLASIK  <- 10     # classical models: 10 x 5 repeated CV
TEKRAR_DERIN   <- 3      # CNN / MobileNet: 3 x 5 (CPU time). Increase to 5-10 if time allows.
GORUNTU_BOYUT  <- 96     # main input resolution
COZUNURLUKLER  <- c(96, 128)   # resolution sensitivity analysis (lightweight CNN, original images only)
DHASH_ESIK     <- 4      # dHash Hamming threshold: <= this value counts as near-duplicate
MASKE_ESIK     <- 25     # pixel is leaf if mean(R,G,B) > this value in the segmented image
TOST_MARJ      <- 0.02   # equivalence margin: +/-2 accuracy points
TOHUM          <- 2026

# CNN training settings
EPOCH_MAKS     <- 40
SABIR          <- 8      # early-stopping patience (val_accuracy)
BATCH          <- 32
OGR_HIZI       <- 1e-3
DOGRULAMA_ORAN <- 0.15

if (HIZLI_TEST) {
  HIZLI_N_SINIF <- 60    # images per class
  TEKRAR_KLASIK <- 2
  TEKRAR_DERIN  <- 1
  EPOCH_MAKS    <- 3
  SABIR         <- 2
  COZUNURLUKLER <- c(96)
  CIKTI         <- paste0(CIKTI, "_quick_test")
}

dir.create(CIKTI, showWarnings = FALSE, recursive = TRUE)
for (d in c("data", "predictions", "models", "tables", "figures"))
  dir.create(file.path(CIKTI, d), showWarnings = FALSE)

yol <- function(...) file.path(CIKTI, ...)

# Shared helper: accuracy and macro F1
metrikler <- function(gercek, tahmin, siniflar = seq_along(SINIFLAR)) {
  gercek <- factor(gercek, levels = siniflar); tahmin <- factor(tahmin, levels = siniflar)
  cm <- table(gercek, tahmin)
  tp <- diag(cm); prec <- tp / pmax(colSums(cm), 1); rec <- tp / pmax(rowSums(cm), 1)
  f1 <- ifelse(prec + rec > 0, 2 * prec * rec / (prec + rec), 0)
  list(dogruluk = sum(tp) / sum(cm), makro_f1 = mean(f1),
       kesinlik = prec, duyarlilik = rec, f1 = f1)
}

# Shared helper: write tables with English column names and values
# (internal identifiers in the code are Turkish abbreviations; see README, "Naming").
COL_EN <- c(Varyant = "variant", Model = "model", Tekrar_x_Kat = "splits", `Dogruluk (%)` = "accuracy_pct",
  `%95 GA (duzeltilmis)` = "CI95_corrected_pct", `Makro F1` = "macro_F1", metrik = "metric", fark = "difference",
  GA95_alt = "CI95_lower", GA95_ust = "CI95_upper", p_duzeltilmis_t = "p_corrected_t", p_Holm = "p_holm",
  varyant = "variant", tekrar = "repetition", kat = "fold", dogruluk = "accuracy", makro_f1 = "macro_F1",
  dogru = "correct", p_binom = "p_binomial", sinif = "class", oran = "proportion_pct", kesinlik = "precision",
  duyarlilik = "recall", egitim_varyanti = "training_variant", EGO_ort = "leaf_attention_mean",
  EGO_ss = "leaf_attention_sd", yaprak_alani = "leaf_area", fazla_odak = "excess_attention",
  isaret_oyunu = "pointing_game", ozellik = "feature", onem = "importance", deger = "value",
  egitilebilir_parametre = "trainable_params", toplam_parametre = "total_params",
  agirlik_dosyasi_MB = "weights_file_MB", keras_dosyasi_MB = "keras_file_MB", tek_goruntu_ms = "single_image_ms",
  toplu_ms_goruntu = "batched_ms_per_image", sure_sn = "time_s", en_iyi_epoch = "best_epoch",
  ozellik_sayisi = "n_features", bolme = "splits", kose_yamalarinda_yaprak_orani = "leaf_fraction_in_corner_patches",
  tur = "condition", yaprak_sinifi = "follows_leaf_pct", arkaplan_sinifi = "follows_background_pct",
  diger = "other_pct", yaprak = "leaf_class", arkaplan = "background_class", makro_f1_3sinif = "macro_F1_3class",
  scab_duyarlilik = "scab_recall", rust_duyarlilik = "rust_recall", healthy_duyarlilik = "healthy_recall",
  blackrot_tahmin_orani = "predicted_black_rot_pct", tohum_sayisi = "n_seeds", n_goruntu = "n_images",
  gercek = "true_class", tahmin = "predicted_class", Freq = "n")
VAL_EN <- c(OzelCNN_agirliksiz = "LightweightCNN_no_class_weights", OzelCNN_artirma = "LightweightCNN_augmentation",
  OzelCNN_128 = "LightweightCNN_128px", OzelCNN = "LightweightCNN", arkaplan_ortRGB = "background_mean_RGB",
  kose_yamalari = "corner_patches", orijinal = "original", yaprak = "leaf_only", arkaplan = "background_only",
  siluet = "silhouette_only", makro_f1 = "macro_F1", dogruluk = "accuracy", ayni_sinif = "same_class",
  farkli_sinif = "different_class", Tumu = "All", GLCM_kontrast = "GLCM_contrast",
  GLCM_homojenlik = "GLCM_homogeneity", GLCM_enerji = "GLCM_energy", GLCM_korelasyon = "GLCM_correlation",
  "_ort" = "_mean", "_ss" = "_sd")
write_out <- function(df, path) {
  df <- as.data.frame(df, stringsAsFactors = FALSE)
  for (nm in names(df)) if (is.character(df[[nm]]) || is.factor(df[[nm]])) {
    v <- as.character(df[[nm]])
    for (k in names(VAL_EN)) v <- gsub(k, VAL_EN[[k]], v, fixed = TRUE)
    df[[nm]] <- v
  }
  names(df) <- ifelse(names(df) %in% names(COL_EN), COL_EN[names(df)], names(df))
  readr::write_csv(df, path)
}
