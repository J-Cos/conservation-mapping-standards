# Conservation Mapping Standards — Interactive Demonstrator

> **[Live Demo →](https://j-cos.github.io/conservation-mapping-standards/)**

An interactive, browser-based demonstration of the full accuracy assessment pipeline for conservation maps, following the Table 1 checklist standard from [Olofsson et al. (2014)](https://doi.org/10.1016/j.rse.2014.02.015). Built for the ZSL Institute of Zoology.

All computation runs **entirely in the browser** — no server, no data uploads, no dependencies to install. Open the link and click through the pipeline.

---

## What This Demonstrator Does

Conservation maps are used to make critical decisions about land protection, carbon crediting, and biodiversity monitoring. However, maps without rigorous accuracy assessment can be dangerously misleading. This tool demonstrates the *correct* way to assess map accuracy, implementing every step of the best-practice workflow:

1. **Generate** a realistic synthetic landscape with known ground truth
2. **Partition** the landscape into spatial blocks to prevent data leakage
3. **Train & evaluate** random forests with spatially-blocked bootstrap cross-validation
4. **Assess** classification accuracy (or regression performance) with proper confidence intervals
5. **Compute** error-corrected area/biomass estimates with uncertainty quantification

The tool supports two mapping modes:
- **Categorical** — Land cover classification (5 classes: Dense Forest, Open Forest, Grassland, Water, Bare Soil)
- **Continuous** — Above-ground biomass estimation (0–500 Mg/ha)

---

## The 5-Step Pipeline

### Step 1 · Synthetic Data Generation

Generates a 1000×1000 pixel, 10-band synthetic raster with realistic spatial structure using **Perlin noise with fractal Brownian motion** (fBm). Each band is generated at different frequencies to simulate spectral diversity in real remote sensing imagery.

**Ground truth** is derived from nonlinear combinations of the spectral bands:
- *Categorical mode* — 5 land cover classes assigned by weighted spectral thresholds, producing spatially coherent patches
- *Continuous mode* — Biomass values from a nonlinear spectral index with spatial smoothing and log-normal noise

Training points are sampled randomly across the raster, and their spectral feature vectors are extracted.

**Key parameters:** Random seed (default: 42), number of training points (default: 5,000).

### Step 2 · Spatial Blocking

Partitions the raster into a grid of **50×50 pixel blocks** (~400 blocks). Every training point is assigned to its containing block.

This is critical because spatially proximate pixels are autocorrelated — if nearby pixels end up in both training and validation sets, accuracy will be inflated. Blocking at the 50-pixel level ensures that training and validation data are spatially independent.

### Step 3 · Bootstrap Cross-Validation

Runs *B* bootstrap replicates (default: 1,000), each executing in a **Web Worker** for parallel processing:

1. **Sample blocks** with replacement — ~63.2% of blocks are drawn into the training set; ~36.8% are out-of-bag (OOB) and used for validation
2. **Train a random forest** (10 CART trees, `max_depth=12`, `mtry=√p`) on the training blocks
3. **Predict** on OOB pixels and compute accuracy metrics
4. **Predict** on the full raster to produce a wall-to-wall map

Each replicate yields a complete set of accuracy metrics and a full prediction map, enabling uncertainty quantification across all outputs.

**Parallelisation:** The app detects available CPU cores and spawns one Web Worker per core (capped at 8). Trees within each replicate are built using a seeded LCG random number generator for reproducibility.

### Step 4 · Accuracy Assessment

Aggregates metrics across all *B* replicates to produce distributions with 95% confidence intervals:

**Categorical mode:**
| Metric              | Description                                                 |
| ------------------- | ----------------------------------------------------------- |
| Overall Accuracy    | Proportion of correctly classified OOB pixels               |
| User's Accuracy     | Per-class precision (1 − commission error)                  |
| Producer's Accuracy | Per-class recall (1 − omission error)                       |
| Confusion Matrix    | Mean counts across all replicates, with marginal accuracies |

**Continuous mode:**
| Metric        | Description                                |
| ------------- | ------------------------------------------ |
| R²            | Coefficient of determination on OOB pixels |
| RMSE          | Root mean squared error (Mg/ha)            |
| Relative RMSE | RMSE normalised by mean observed value (%) |

All distributions are rendered as histograms with reference threshold lines (0.85 for accuracy, 0.80 for R²).

### Step 5 · Summary Statistics & Uncertainty

**Categorical — Olofsson area correction:**
Applies the [Olofsson et al. (2014)](https://doi.org/10.1016/j.rse.2014.02.015) error-adjusted area estimation. The confusion matrix is converted to proportions, and map class areas are corrected for commission and omission errors. Confidence intervals are derived from the bootstrap distribution of corrected areas.

**Continuous — Bias-corrected biomass totals:**
Computes bias-corrected total biomass using the OOB prediction residuals. The bootstrap distribution of corrected totals provides the confidence interval.

**Uncertainty maps:**
- *Categorical*: Pixel-level prediction standard deviation across replicates (high values = unstable classification)
- *Continuous*: Pixel-level biomass standard deviation (Mg/ha)

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

---

## Scientific References

- **Olofsson, P. et al. (2014).** Good practices for estimating area and assessing accuracy of land use change. *Remote Sensing of Environment*, 148, 42–57. [doi:10.1016/j.rse.2014.02.015](https://doi.org/10.1016/j.rse.2014.02.015)
- **Roberts, D. R. et al. (2017).** Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913–929.
- **Breiman, L. (2001).** Random Forests. *Machine Learning*, 45(1), 5–32.

---

## Licence

This project is developed at the **ZSL Institute of Zoology**. See the repository for licence details.
