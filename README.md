# Reassessing Lightweight CNN Architectures for Cotton Leaf Disease Recognition

Code and complete per-run results for the paper *Reassessing Lightweight CNN Architectures for Cotton Leaf Disease Recognition: Effects of Training Protocol, Data Duplication, and Explanation Faithfulness* (B. Ali, A. M. Soomro, W. Y. Leong, S. Bagchi, S. K. Debnath; submitted to IEEE Access).

The study compares **MobileNetV3Small**, **EfficientNetB0** and **DenseNet121** on the public [Cotton Disease Dataset](https://www.kaggle.com/datasets/janmejaybhoi/cotton-disease-dataset): 2,310 images in 4 classes. It covers two data splits × three backbones × five seeds, giving 30 independently trained models, with:

- a perceptual-hash duplicate audit and a de-duplicated split;
- hierarchical-bootstrap confidence intervals, image-level paired tests and GEE effect sizes;
- Grad-CAM deletion/insertion faithfulness for every test image;
- CPU and TensorFlow Lite latency.

## Main results

| Model | Original split | De-duplicated split | Params (M) | CPU latency (ms) | Grad-CAM insertion − deletion |
|---|---|---|---|---|---|
| MobileNetV3Small | 98.49 ± 0.52 % | 97.90 ± 0.80 % | 1.01 | 11.3 | 0.445 |
| EfficientNetB0 | 97.74 ± 0.52 % | 97.14 ± 1.17 % | 4.21 | 39.9 | 0.474 |
| DenseNet121 | 97.92 ± 0.42 % | 97.52 ± 0.52 % | 7.17 | 81.5 | 0.269 |

- The three backbones are statistically indistinguishable in accuracy on both splits. All Holm-adjusted pairwise tests give p = 1.0, and every GEE odds-ratio interval includes 1.
- Removing the 22 cross-split near-duplicate pairs changes accuracy by less than 0.6 percentage points.
- Architecture matters for cost and explanations. MobileNetV3Small is about 7× smaller and faster than DenseNet121. DenseNet121's Grad-CAM maps are significantly less faithful than those of the other two backbones.
- Two earlier versions of this benchmark had **silent training faults**, and each made MobileNetV3Small look clearly superior: callback state shared between training phases, and BatchNorm layers running in training mode. See below and `results/v3/analysis/figures/fig_protocol_history.png`.

## The Keras 3 BatchNorm pitfall

The Keras transfer-learning guide recommends calling a pretrained backbone with `training=False`, so that its BatchNorm layers stay in inference mode after unfreezing. In Keras 3.13.2, a functional model passes the `training` flag of its own call to every nested layer during `fit()`, which overrides that argument. Unfrozen BatchNorm layers therefore use batch statistics and overwrite their moving averages. Setting the BatchNorm layers to `trainable = False` fixes it.

Reproduce it without any dataset:

```bash
python scripts/check_bn_training_mode.py
```

## Repository layout

```
notebooks/cotton_disease_v3.ipynb      Kaggle GPU notebook: duplicate audit, 30 runs (one subprocess each),
                                       Grad-CAM evaluation, latency, TFLite
scripts/analysis_v3.py                 every statistic, table and data figure (no GPU needed)
scripts/make_diagrams.py               pipeline and architecture schematics
scripts/check_bn_training_mode.py      the BatchNorm demonstration above
results/v3/raw/                        per-run JSON records (image-level predictions, probabilities,
                                       histories, Grad-CAM scores), split manifest, duplicate audit,
                                       latency and TFLite benchmarks, package versions
results/v3/analysis/                   tables (CSV), figures (PNG + PDF), key_numbers.json
results/v2_repeated_seeds/raw/, archive/results/v3_run1_bn_training_mode/
                                       results of the two faulty earlier versions (protocol-history comparison)
```

## Reproducing

1. **Training (Kaggle, GPU, about 12 h).** Attach the Cotton Disease Dataset, upload `notebooks/cotton_disease_v3.ipynb` and run *Save Version → Save & Run All*. Each run is checkpointed: if a session stops, attach the previous version's output as a dataset, set `RESUME_FROM_INPUT` in the configuration cell, and run again. Finally, copy the output's `runs/`, `data/`, `figures/` and benchmark files to `results/v3/raw/`.
2. **Analysis (local).**
   ```bash
   pip install -r requirements.txt
   python scripts/analysis_v3.py
   python scripts/make_diagrams.py
   ```
   The analysis is deterministic, with a fixed bootstrap seed.

The 30 trained models (about 1.1 GB) are not stored here because of their size.

## Citation

The paper is under review; citation details will be added on publication (see `CITATION.cff`).

## License

Code: MIT (see `LICENSE`). Results and figures: CC BY 4.0. The Cotton Disease Dataset is subject to the license of its publisher on Kaggle.
