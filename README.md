# Spatial Mapping Standards — Interactive Demonstrator

> **[Live Demo →](https://j-cos.github.io/conservation-mapping-standards/)**

Companion to: **Schulte to Bühne, H., Williams, J., Byrne, A. & Pettorelli, N. (2026). A mapping standard for spatial data. *Methods in Ecology and Evolution*.**

An interactive, browser-based demonstrator that accompanies the proposed spatial mapping standard. Walk through the best-practice accuracy assessment pipeline on synthetic data — then see what happens when common shortcuts are taken.

All computation runs **entirely in the browser** — no server, no data uploads, no dependencies to install. Open the link and click through the pipeline.

---

## What This Demonstrator Does

Spatial maps are used to make critical real-world decisions about land management, carbon crediting, and resource monitoring. However, maps without rigorous accuracy assessment can be dangerously misleading. This tool demonstrates the *correct* way to assess map accuracy, implementing every step of the best-practice workflow:

1. **Generate** a realistic synthetic landscape with known ground truth
2. **Collect** reference data (simulating field surveys with clustered or random strategies)
3. **Partition** the landscape into spatial blocks to prevent data leakage
4. **Train & evaluate** random forests with spatially-blocked repeated cross-validation
5. **Assess** classification accuracy (or regression performance) with proper confidence intervals
6. **Compute** error-corrected area/biomass estimates with uncertainty quantification
7. **Score** the final map against the standard via an automated report card

The tool supports two mapping modes:
- **Categorical** — Land cover classification (5 classes: Dense Forest, Open Forest, Grassland, Water, Bare Soil)
- **Continuous** — Above-ground biomass estimation (0–500 Mg/ha)

---

## The 7-Step Pipeline

### Step 1 · Synthetic Data Generation

Generates a 1000×1000 pixel, 10-band synthetic raster with realistic spatial structure using **Perlin noise with fractal Brownian motion** (fBm). Each band is generated at different frequencies to simulate spectral diversity in real remote sensing imagery.

**Ground truth** is derived from nonlinear combinations of the spectral bands:
- *Categorical mode* — 5 land cover classes assigned by weighted spectral thresholds, producing spatially coherent patches
- *Continuous mode* — Biomass values from a nonlinear spectral index with spatial smoothing and log-normal noise

Training points are sampled randomly across the raster, and their spectral feature vectors are extracted.

**Key parameters:** Random seed (default: 42).

### Step 2 · Reference Data Collection

Simulates field data collection by sampling training points from the generated landscape. 

**Sampling Strategies:**
- **Clustered (Realistic):** Mimics real-world field plots, selecting points in spatially autocorrelated clusters.
- **Random (Ideal):** Selects completely independent random pixels across the landscape.

### Step 3 · Spatial Blocking

Partitions the raster into a grid of **200×200 pixel blocks** (25 blocks). Every training point is assigned to its containing block.

This is critical because spatially proximate pixels are autocorrelated — if nearby pixels end up in both training and validation sets, accuracy will be inflated. Spatial blocking ensures that training and validation data are spatially independent.

### Step 4 · Repeated Cross-Validation

Runs *B* bootstrap replicates (default: 100), each executing in a **Web Worker** for parallel processing:

1. **Sample blocks** with replacement — ~63.2% of blocks are drawn into the training set; ~36.8% are out-of-bag (OOB) and used for validation
2. **Train a random forest** (100 CART trees, `max_depth=10`, `mtry=√p`) on the training blocks
3. **Predict** on OOB pixels and compute accuracy metrics
4. **Predict** on the full raster to produce a wall-to-wall map

Each replicate yields a complete set of accuracy metrics and a full prediction map, enabling uncertainty quantification across all outputs.

**Parallelisation:** The app detects available CPU cores and spawns one Web Worker per core (capped at 8). Trees within each replicate are built using a seeded LCG random number generator for reproducibility.

### Step 5 · Accuracy Assessment

Aggregates metrics across all *B* replicates to produce distributions with 95% confidence intervals:

**Categorical mode:**
| Metric              | Description                                                 |
| ------------------- | ----------------------------------------------------------- |
| Overall Accuracy    | Proportion of correctly classified OOB pixels               |
| User's Accuracy     | Per-class precision (1 − commission error)                  |
| Producer's Accuracy | Per-class recall (1 − omission error)                       |
| Confusion Matrix    | Mean counts across all replicates, with marginal accuracies |

Also displayed: the **Mean Error Matrix** (manuscript terminology).

**Continuous mode:**
| Metric        | Description                                |
| ------------- | ------------------------------------------ |
| R²            | Coefficient of determination on OOB pixels |
| RMSE          | Root mean squared error (Mg/ha)            |
| Relative RMSE | RMSE normalised by mean observed value (%) |

All distributions are rendered as histograms with reference threshold lines (0.85 for accuracy, 0.80 for R²).

### Step 6 · Summary Statistics & Uncertainty

**Categorical — Olofsson area correction:**
Applies the [Olofsson et al. (2014)](https://doi.org/10.1016/j.rse.2014.02.015) error-adjusted area estimation. The confusion matrix is converted to proportions, and map class areas are corrected for commission and omission errors. Confidence intervals are derived from the bootstrap distribution of corrected areas.

**Continuous — Predicted vs. true total biomass:**
Each bootstrap replicate predicts biomass for the entire 1M-pixel raster. The distribution of predicted totals across replicates quantifies model uncertainty. This is compared directly against the known true total biomass, with the true value displayed as a reference line on the histogram. If the true total falls within the bootstrap distribution, the model is unbiased.

**Uncertainty maps:**
- *Categorical*: Pixel-level prediction standard deviation across replicates (high values = unstable classification)
- *Continuous*: Pixel-level biomass standard deviation (Mg/ha)

### Step 7 · Final Verdict & Report Card

Compares the internal cross-validation accuracy estimate against the **True Landscape Accuracy** (calculated across an independent 10,000-pixel evaluation set). This provides a final letter grade and clear feedback on whether the sampling and validation strategy successfully avoided data leakage and overestimation.

### ⚠️ Pitfall Comparison: Random Split vs Spatial Blocking

After the main pipeline completes, a **"Compare: What if we skipped spatial blocking?"** button appears. This runs 20 quick replicates with **random pixel-level splitting** (no spatial blocking) and displays the inflated accuracy metrics alongside the correct spatially-blocked results.

This directly demonstrates the manuscript's central warning: without spatial blocking, accuracy is artificially inflated because spatially correlated pixels leak between training and test sets. The comparison includes:
- Side-by-side accuracy distributions (red histograms for the random split)
- A quantified inflation figure (e.g., "+5.2 percentage points")
- A clear statement that such a map would **not meet the mapping standard**

### Naive vs Error-Corrected Area Estimates

In Step 6 (categorical mode), each class's area is shown three ways:
- **Corrected estimate** (Olofsson et al. approach) — the main value with 95% CI
- **Naive pixel count** (highlighted in red) — what you'd get by simply counting pixels
- **True area** — the known ground truth

This demonstrates why "counting pixels" is dangerous and error correction is essential.

---

## Architecture

```
index.html              Single-page UI with accordion steps
css/style.css           ZSL/PNAS-branded design system
js/
├── app.js              Main controller — state management, rendering, event handling
├── syntheticData.js    Perlin noise raster generation, ground truth, sampling
├── spatialBlocking.js  Block creation, point assignment, bootstrap sampling
├── charts.js           Chart.js wrapper enforcing PNAS figure style
├── rasterViz.js        Canvas-based raster rendering + colorbars
└── workers/
    └── rfWorker.js     Self-contained CART + Random Forest (runs in Web Workers)
lib/
└── chart.min.js        Chart.js v4 (vendored)
tests.js                Unit + integration test suite (Node.js)
```

### Key Design Decisions

- **Zero dependencies beyond Chart.js.** The random forest, CART trees, spatial blocking, Perlin noise, and raster rendering are all implemented from scratch in vanilla JavaScript
- **Web Workers for parallelism.** Each bootstrap replicate runs in its own thread, with progress reported back to the main thread
- **Typed arrays throughout.** Raster data, feature matrices, and predictions use `Float32Array`/`Uint32Array`/`Uint16Array` for memory efficiency
- **Seeded RNG everywhere.** All randomness uses deterministic seeded generators (LCG), making every replicate reproducible given the same seed
- **PNAS figure standards.** Charts follow the *Proceedings of the National Academy of Sciences* style: Helvetica Neue typography, muted colour palettes, minimal gridlines

---

## Running Locally

No build step. Just serve the directory:

```bash
# Python
python3 -m http.server 8080

# Node
npx -y serve .

# Then open http://localhost:8080
```

## Testing

The project includes a comprehensive test suite (`tests.js`) covering all core analytical functions. Run with:

```bash
node tests.js
```

Expected output: **267 assertions, 0 failures.**

### Test Sections

| Section                              | Tests | What's Verified                                                                                                                                                                                                                                     |
| ------------------------------------ | ----: | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **SyntheticData**                    |   135 | Raster dimensions, band generation, deterministic seeding, value ranges, categorical class validity, continuous ground truth, point sampling uniqueness, feature extraction                                                                         |
| **SpatialBlocking**                  |    38 | Block grid geometry, pixel-to-block assignment correctness, point assignment, bootstrap sampling, OOB fraction ≈ 36.8% (verified over 1,000 trials), deterministic seeding, weight consistency, `getPixelsInBlocks` counts, `getTrainingData` shape |
| **Random Forest**                    |    10 | Seeded RNG determinism, classification accuracy > 90% on linearly separable 2D data, regression R² > 0.85 on linear data, correct typed array output                                                                                                |
| **Accuracy Metrics**                 |    17 | Perfect classification (OA/UA/PA = 1.0), known misclassification with hand-computed confusion matrix, RMSE/R²/mean residual/total predicted against exact values                                                                                    |
| **Olofsson Area Correction**         |     6 | Corrected areas sum to total pixels, perfect classification preserves areas, reasonable values for imperfect matrices                                                                                                                               |
| **Summary Statistics**               |     9 | Mean/median of known sequences, CI bounds ordering, single-value edge case, odd-length median                                                                                                                                                       |
| **Integration**                      |     8 | Full pipeline: generate → block → bootstrap → train → predict → metrics → area correction, with OA > 50% and corrected areas summing to 1M                                                                                                          |
| **Sensor Noise & Hidden Gradient**   |    20 | Noisy vs clean band divergence, noise magnitude bounds, class balance realism (Water ≥ 1%, all 5 classes present, no class > 60%)                                                                                                                   |
| **Continuous Mode Metrics**          |     7 | RMSE, R², relative RMSE hand-computed against known values, totalPredicted sum                                                                                                                                                                      |
| **Pitfall Comparison Logic**         |     3 | Spatial blocking vs random-split accuracy on autocorrelated data, inflation ≥ −5pp (confirming data leakage effect)                                                                                                                                 |
| **Single-Split vs Repeated**         |     6 | Bootstrap distribution spread > 1pp, single replicate within range, CI bounds ordering, pairwise OA differences > 1pp                                                                                                                               |
| **Continuous Mode Integration**      |     8 | Full regression pipeline: generate → block → bootstrap → train → predict → regression metrics (R², RMSE, relRMSE, totalPredicted, n)                                                                                                                |

---

## Scientific References

- **Schulte to Bühne, H., Williams, J., Byrne, A. & Pettorelli, N. (2026).** A mapping standard for spatial data. *Methods in Ecology and Evolution*.
- **Olofsson, P. et al. (2014).** Good practices for estimating area and assessing accuracy of land use change. *Remote Sensing of Environment*, 148, 42–57. [doi:10.1016/j.rse.2014.02.015](https://doi.org/10.1016/j.rse.2014.02.015)
- **Roberts, D. R. et al. (2017).** Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913–929.
- **Breiman, L. (2001).** Random Forests. *Machine Learning*, 45(1), 5–32.

---

## Licence

This project is developed at the **ZSL Institute of Zoology**. See the repository for licence details.
