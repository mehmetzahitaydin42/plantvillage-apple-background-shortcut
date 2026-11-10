# Background shortcuts in PlantVillage apple leaf disease classification

Code and results for the paper:

> Aydın, M. Z. *Is the Leaf Enough? Background Shortcuts and Fair Benchmarking of Lightweight Deep Learning and Classical Machine Learning for Apple Leaf Disease Classification.* (submitted)

The analysis asks whether models that reach 95–97% accuracy on the PlantVillage apple images actually recognise disease symptoms or exploit the image background. It compares a lightweight CNN trained from scratch, an ImageNet-pretrained MobileNetV3-Small and three classical classifiers (Random Forest, SVM, XGBoost on 40 colour/texture features) under identical repeated stratified cross-validation, in four controlled image variants (original, leaf only, background only, silhouette only). It adds a counterfactual background-swap test, minimal background probes, a quantitative Grad-CAM analysis and an external test on the Plant Pathology 2020 field images.

## Main results

| | Lightweight CNN | MobileNetV3-Small | Random Forest | SVM | XGBoost |
|---|---|---|---|---|---|
| Original images (CV accuracy, %) | 94.7 | 96.1 | 96.8 | 97.5 | 97.4 |
| Background only, leaf removed (%) | 86.7 | 86.8 | 95.6 | 95.6 | 96.7 |
| Silhouette only (%) | 72.6 | 71.0 | 58.2 | 57.9 | 49.1 |
| Background swap: follows background (%) | 25.0 | 15.7 | 32.4 | 47.6 | 38.5 |
| Plant Pathology 2020 field images (%) | 30.0 | 36.8 | 35.1 | 37.5 | 33.9 |

No-information rate: 51.8% (PlantVillage), 36.0% (Plant Pathology 2020). The mean background colour alone (three numbers) gives 80.0% accuracy. Full tables are in [`results/tables`](results/tables).

## Repository structure

```
RUN.R                          one-file entry point (installs, downloads, runs everything)
run_pipeline.R                 runs steps 01-06 in order, resumes if interrupted
config.R                       paths, experimental design and training settings
00_setup.R                     manual package installation (RUN.R does this automatically)
01_prepare_data.R              near-duplicate removal, leaf masks, image variants, CV folds
02_classical_models.R          40 features; Random Forest, SVM, XGBoost; 10 x 5 CV
03_deep_models.R               lightweight CNN and MobileNetV3-Small; 3 x 5 CV; ablation; 128 px
04_gradcam_quantitative.R      leaf attention ratio and pointing game for Grad-CAM
05_statistics_tables.R         corrected resampled t-test, Holm, TOST, McNemar; tables and figures
06_additional_analyses.R       background probes, background swap, Plant Pathology 2020 test
plantvillage_apple_file_list.csv   list of the 6,342 PlantVillage files used (colour + segmented)
results/                       tables and figures of the run reported in the paper
```

## Requirements

* R ≥ 4.3 (tested with R 4.6.1 on Windows 11 and R 4.3 on Linux)
* R packages: `magick`, `ranger`, `e1071`, `xgboost`, `dplyr`, `tidyr`, `readr`, `ggplot2`, `patchwork`, `keras3`, `reticulate`
* Python 3.9–3.11 with TensorFlow (tested with TensorFlow 2.21 and keras3 1.5.1). `RUN.R` creates the `r-keras` virtual environment automatically; on Windows it installs Python 3.11.9 per user if needed.
* No GPU is required. The full run took about 17.5 h on a 4-core laptop CPU (15.7 h of it CNN training).

## Data

* **PlantVillage** (Hughes and Salathé 2015; Mohanty et al. 2016): `RUN.R` downloads the 6,342 apple images (colour and segmented versions, ~90 MB) listed in `plantvillage_apple_file_list.csv` from <https://github.com/spMohanty/PlantVillage-Dataset> into `PlantVillage-Dataset/`.
* **Plant Pathology 2020** (Thapa et al. 2020), only for step C of `06_additional_analyses.R`: download from <https://www.kaggle.com/c/plant-pathology-2020-fgvc7/data> (Kaggle account and acceptance of the competition rules required) and extract to `PlantPathology2020/` so that it contains `train.csv` and `images/`. If the folder is absent, step C is skipped.

## How to run

1. Open R/RStudio with the repository folder as the working directory.
2. Open `RUN.R`, keep `MODE <- "test"` and source it. This installs everything, downloads the data and runs a 5–10 minute smoke test (output in `results_quick_test/`, not meaningful).
3. Set `MODE <- "full"` and source again. Output is written to `results/`. If the run is interrupted, source `RUN.R` again; finished folds and steps are skipped.

Settings (number of repetitions, image size, thresholds, training parameters) are in `config.R`. Without internet access to the ImageNet weights, `PV_IMAGENET=FALSE` initialises MobileNetV3 randomly (for code testing only).

## Outputs (`results/`)

| File | Content |
|---|---|
| `tables/T1_class_distribution.csv`, `data_summary.txt` | images per class after near-duplicate removal |
| `tables/fold_level_metrics.csv` | accuracy and macro F1 of every model, variant, repetition and fold |
| `tables/T2_main_results.csv` | mean ± SD and corrected 95% CI per model and variant |
| `tables/T3_model_comparison.csv` | pairwise model comparisons (original images) |
| `tables/T4_variant_effect.csv`, `T4b_chance_level.csv` | variant effects within models; tests against the no-information rate |
| `tables/T5_ablation.csv` | data augmentation, class weights, 128 px input |
| `tables/T6_per_class.csv` | per-class precision, recall and F1 |
| `tables/T_gradcam_quantitative.csv`, `gradcam_test.txt` | leaf attention ratio and pointing game |
| `tables/T7_simple_probes.csv` | mean background colour and corner-patch probes |
| `tables/T8_background_swap.csv`, `T8b_swap_class_pairs.csv` | counterfactual background swap |
| `tables/T9_external_test_pp2020.csv`, `T9b_*`, `T9c_*` | Plant Pathology 2020 external test |
| `tables/CNN_complexity.csv`, `RF_complexity.csv`, `training_times.csv`, `RF_feature_importance.csv` | complexity, timing and feature importance |
| `figures/` | figures of the paper |
| `run_log.txt` | start, end, warnings and errors of every step |

Intermediate files (`data/`, `predictions/`, `predictions_extra/`, `models/`) are created during the run and are not included in the repository.

## Statistical methods

Models are compared on identical repeated stratified five-fold partitions (classical models 10 × 5, deep models 3 × 5) with the Nadeau–Bengio corrected resampled t-test, Holm adjustment for multiple comparisons, two one-sided equivalence tests (margin ±2 accuracy points) and McNemar's test on the pooled out-of-fold predictions of the first repetition.

## Naming

Internal identifiers in the code are short Turkish words; all output tables are written with English column names and values. The main correspondences are:

| Code | Meaning |
|---|---|
| `orijinal`, `yaprak`, `arkaplan`, `siluet` | original, leaf only, background only, silhouette only |
| `OzelCNN` | lightweight CNN |
| `sinif`, `gercek`, `tahmin` | class, true class, predicted class |
| `tekrar`, `kat`, `KATLAR` | repetition, fold, fold assignment matrix |
| `dogruluk`, `makro_f1` | accuracy, macro F1 |
| `yol()`, `CIKTI` | output path helper, output folder |

## Reproducibility

All random partitions use fixed seeds (`TOHUM` in `config.R`), and all models share the same partitions. Near-duplicate removal (dHash, Hamming distance ≤ 4) is deterministic and removed 7 of 3,171 images. Deep-learning results can differ slightly between runs and hardware because TensorFlow training on CPU is not bit-for-bit deterministic.

## License

Code: MIT License (see `LICENSE`). The datasets are subject to their own licences: PlantVillage (CC0 / see the original repository) and Plant Pathology 2020 (Kaggle competition rules).

## Contact

Mehmet Zahit Aydın, Department of Horticulture, Faculty of Agriculture, Recep Tayyip Erdoğan University, Rize, Türkiye — mehmetzahit.aydin@erdogan.edu.tr
