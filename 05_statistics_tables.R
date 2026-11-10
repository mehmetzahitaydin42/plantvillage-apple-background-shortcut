# =============================================================================
#  05_statistics_tables.R
#  Reads all prediction files and produces the tables and figures of the paper.
#  Statistics:
#   - Nadeau & Bengio (2003) corrected repeated k-fold t-test
#     (training sets of CV folds overlap, so the plain t-test is over-optimistic)
#   - Holm adjustment (multiple comparisons)
#   - TOST equivalence test (margin +/-TOST_MARJ), 95% CI of the difference
#   - McNemar test (pooled out-of-fold predictions of repetition 1)
# =============================================================================
source("config.R")
suppressPackageStartupMessages({ library(dplyr); library(tidyr); library(ggplot2) })

dosyalar <- list.files(yol("predictions"), pattern = "__r\\d+\\.rds$", full.names = TRUE)
tum <- bind_rows(lapply(dosyalar, readRDS))
n_toplam <- n_distinct(tum$id)
NIR <- max(table(readRDS(yol("data", "image_list.rds"))$sinif)) / n_toplam  # no-information rate

# ---- Fold-level metrics -----------------------------------------------------------
kat_m <- tum %>% group_by(model, varyant, tekrar, kat) %>%
  summarise(m = list(metrikler(gercek, tahmin)), .groups = "drop") %>%
  mutate(dogruluk = sapply(m, `[[`, "dogruluk"), makro_f1 = sapply(m, `[[`, "makro_f1")) %>%
  select(-m)
write_out(kat_m, yol("tables", "fold_level_metrics.csv"))

K <- K_KAT; ORAN <- 1 / (K - 1)                  # n_test / n_train
nb_se <- function(d, J) sqrt((1 / J + ORAN) * var(d))

# ---- Table: main results per model x variant ----------------------------------------
ana <- kat_m %>% group_by(model, varyant) %>%
  summarise(J = n(), tekrar = n_distinct(tekrar),
            dog_ort = mean(dogruluk), dog_ss = sd(dogruluk),
            dog_GA_alt = dog_ort - qt(0.975, J - 1) * nb_se(dogruluk, J),
            dog_GA_ust = dog_ort + qt(0.975, J - 1) * nb_se(dogruluk, J),
            f1_ort = mean(makro_f1), f1_ss = sd(makro_f1), .groups = "drop") %>%
  mutate(varyant = factor(varyant, VARYANTLAR)) %>% arrange(varyant, desc(dog_ort))
yaz <- ana %>% mutate(
  `Dogruluk (%)` = sprintf("%.1f \u00b1 %.1f", 100 * dog_ort, 100 * dog_ss),
  `%95 GA (duzeltilmis)` = sprintf("[%.1f, %.1f]", 100 * dog_GA_alt, 100 * pmin(dog_GA_ust, 1)),
  `Makro F1` = sprintf("%.3f \u00b1 %.3f", f1_ort, f1_ss)) %>%
  select(Varyant = varyant, Model = model, `Tekrar_x_Kat` = J, `Dogruluk (%)`,
         `%95 GA (duzeltilmis)`, `Makro F1`)
print(yaz, n = 60)
write_out(yaz, yol("tables", "T2_main_results.csv"))

# ---- Paired comparison function -------------------------------------------------------
karsilastir <- function(a_model, a_var, b_model, b_var, metrik = "dogruluk") {
  a <- kat_m %>% filter(model == a_model, varyant == a_var)
  b <- kat_m %>% filter(model == b_model, varyant == b_var)
  ab <- inner_join(a, b, by = c("tekrar", "kat"), suffix = c("_a", "_b"))
  if (nrow(ab) < 3) return(NULL)
  d <- ab[[paste0(metrik, "_a")]] - ab[[paste0(metrik, "_b")]]
  J <- length(d); se <- nb_se(d, J); df <- J - 1
  t <- mean(d) / se
  p_tost <- if (se > 0) max(pt((mean(d) + TOST_MARJ) / se, df, lower.tail = FALSE),
                            pt((mean(d) - TOST_MARJ) / se, df)) else NA
  # McNemar: repetition 1, image level
  ta <- tum %>% filter(model == a_model, varyant == a_var, tekrar == 1)
  tb <- tum %>% filter(model == b_model, varyant == b_var, tekrar == 1)
  mc <- NA
  if (nrow(ta) && nrow(tb)) {
    j <- inner_join(ta, tb, by = "id")
    tab <- table(factor(j$tahmin.x == j$gercek.x, c(TRUE, FALSE)),
                 factor(j$tahmin.y == j$gercek.y, c(TRUE, FALSE)))
    mc <- if (tab[1, 2] + tab[2, 1] > 0) mcnemar.test(tab)$p.value else 1
  }
  data.frame(A = paste(a_model, a_var, sep = " / "), B = paste(b_model, b_var, sep = " / "),
             metrik = metrik, J = J, fark = mean(d),
             GA95_alt = mean(d) - qt(0.975, df) * se, GA95_ust = mean(d) + qt(0.975, df) * se,
             t = t, p_duzeltilmis_t = 2 * pt(-abs(t), df),
             p_TOST = p_tost, p_McNemar = mc)
}
bicim <- function(df) df %>% mutate(across(c(fark, GA95_alt, GA95_ust), ~ round(100 * .x, 2)),
                                    across(c(t), ~ round(.x, 2)),
                                    across(starts_with("p_"), ~ signif(.x, 3)))

# ---- Table: comparison between models (original variant) --------------------------
modeller <- intersect(c("OzelCNN", "MobileNetV3", "RF", "SVM", "XGBoost"), unique(kat_m$model))
ciftler <- combn(modeller, 2, simplify = FALSE)
mk <- bind_rows(lapply(c("dogruluk", "makro_f1"), function(mt)
  bind_rows(lapply(ciftler, function(cf) karsilastir(cf[1], "orijinal", cf[2], "orijinal", mt))))) %>%
  group_by(metrik) %>% mutate(p_Holm = p.adjust(p_duzeltilmis_t, "holm")) %>% ungroup()
print(bicim(mk))
write_out(bicim(mk), yol("tables", "T3_model_comparison.csv"))

# ---- Table: variant effect (shortcut learning) -----------------------------------------
vk <- bind_rows(lapply(intersect(modeller, unique(kat_m$model)), function(m)
  bind_rows(lapply(setdiff(VARYANTLAR, "orijinal"), function(v)
    karsilastir(m, "orijinal", m, v))))) %>%
  mutate(p_Holm = p.adjust(p_duzeltilmis_t, "holm"))
print(bicim(vk))
write_out(bicim(vk), yol("tables", "T4_variant_effect.csv"))

# Versus the no-information rate: are background / silhouette models better than chance? (binomial, repetition 1)
sans <- tum %>% filter(tekrar == 1, varyant %in% c("arkaplan", "siluet")) %>%
  group_by(model, varyant) %>%
  summarise(dogru = sum(tahmin == gercek), n = n(), .groups = "drop") %>%
  rowwise() %>% mutate(dogruluk = dogru / n, NIR = NIR,
                       p_binom = binom.test(dogru, n, NIR, alternative = "greater")$p.value) %>%
  ungroup()
print(sans)
write_out(sans, yol("tables", "T4b_chance_level.csv"))

# ---- Table: ablation and resolution ---------------------------------------------------
abl_m <- intersect(c("OzelCNN_artirma", "OzelCNN_agirliksiz", grep("^OzelCNN_\\d+$", unique(kat_m$model), value = TRUE)),
                   unique(kat_m$model))
abl <- bind_rows(lapply(c("dogruluk", "makro_f1"), function(mt)
  bind_rows(lapply(abl_m, function(m) karsilastir(m, "orijinal", "OzelCNN", "orijinal", mt)))))
if (nrow(abl)) { print(bicim(abl)); write_out(bicim(abl), yol("tables", "T5_ablation.csv")) }

# ---- Table: per-class performance (mean +/- SD over repetitions) ------------------------
sb <- tum %>% filter(varyant == "orijinal", model %in% modeller) %>%
  group_by(model, tekrar) %>%
  summarise(m = list(metrikler(gercek, tahmin)), .groups = "drop") %>%
  rowwise() %>%
  mutate(d = list(data.frame(sinif = SINIF_ETIKET, kesinlik = m$kesinlik,
                             duyarlilik = m$duyarlilik, f1 = m$f1))) %>%
  select(model, tekrar, d) %>% unnest(d) %>%
  group_by(model, sinif) %>%
  summarise(across(c(kesinlik, duyarlilik, f1), ~ sprintf("%.3f \u00b1 %.3f", mean(.x), sd(.x))),
            .groups = "drop")
print(sb, n = 40)
write_out(sb, yol("tables", "T6_per_class.csv"))

# ---- Figure: accuracy by model x variant -------------------------------------------------
sekil_veri <- kat_m %>% filter(model %in% modeller) %>%
  mutate(varyant = factor(varyant, VARYANTLAR,
                          c("Original", "Leaf only", "Background only", "Silhouette only")),
         model = factor(model, modeller))
g <- ggplot(sekil_veri, aes(varyant, 100 * dogruluk, fill = model)) +
  geom_boxplot(outlier.size = 0.6, width = 0.75, position = position_dodge(0.8)) +
  geom_hline(yintercept = 100 * NIR, linetype = "dashed") +
  annotate("text", x = 4.4, y = 100 * NIR + 2, label = "No-information rate", size = 3, hjust = 1) +
  labs(x = NULL, y = "Accuracy (%)", fill = NULL) +
  theme_bw(base_size = 11) + theme(legend.position = "top")
ggsave(yol("figures", "S1_variant_accuracy.png"), g, width = 8, height = 4.8, dpi = 300)

# ---- Figure: pooled confusion matrices (repetition 1, original) ---------------------------
km <- tum %>% filter(tekrar == 1, varyant == "orijinal", model %in% modeller) %>%
  count(model, gercek, tahmin) %>%
  complete(model, gercek = seq_along(SINIFLAR), tahmin = seq_along(SINIFLAR), fill = list(n = 0)) %>%
  group_by(model, gercek) %>% mutate(oran = n / sum(n)) %>% ungroup() %>%
  mutate(gercek = factor(SINIF_ETIKET[gercek], rev(SINIF_ETIKET)),
         tahmin = factor(SINIF_ETIKET[tahmin], SINIF_ETIKET), model = factor(model, modeller))
g2 <- ggplot(km, aes(tahmin, gercek, fill = oran)) + geom_tile(colour = "white") +
  geom_text(aes(label = n, colour = oran > 0.5), size = 3) +
  scale_colour_manual(values = c(`TRUE` = "white", `FALSE` = "black"), guide = "none") +
  scale_fill_gradient(low = "#F4F7FB", high = "#1F4E79", limits = c(0, 1), name = "Row\nproportion") +
  facet_wrap(~ model, nrow = 1) + labs(x = "Predicted", y = "True") +
  theme_minimal(base_size = 10) + theme(axis.text.x = element_text(angle = 35, hjust = 1))
ggsave(yol("figures", "S2_confusion_matrices.png"), g2,
       width = 2.6 * length(modeller) + 1, height = 3.4, dpi = 300)

cat("\nNo-information rate (majority class):", round(NIR, 3), "\n05 done. Tables:", yol("tables"), "\n")
