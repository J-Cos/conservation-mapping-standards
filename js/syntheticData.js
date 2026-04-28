/* ============================================================
   Synthetic Data Generation Engine
   Generates 1000×1000 10-band raster with spatial autocorrelation,
   categorical (5 classes) and continuous (biomass) ground truth.
   ============================================================ */

const SyntheticData = (() => {
    const WIDTH = 1000;
    const HEIGHT = 1000;
    const NUM_BANDS = 10;
    const NUM_CLASSES = 5;

    const CLASS_NAMES = ['Dense Forest', 'Open Forest', 'Grassland', 'Water', 'Bare Soil'];
    const CLASS_COLORS = [
        [26, 94, 42],    // Dense Forest
        [106, 175, 76],  // Open Forest
        [200, 217, 111], // Grassland
        [47, 122, 191],  // Water
        [196, 149, 106], // Bare Soil
    ];

    /* --- Improved Perlin Noise (2D) --- */
    class PerlinNoise {
        constructor(seed) {
            this.perm = new Uint8Array(512);
            const p = new Uint8Array(256);
            // Seed-based permutation
            let s = seed | 0;
            for (let i = 0; i < 256; i++) p[i] = i;
            for (let i = 255; i > 0; i--) {
                s = (s * 1664525 + 1013904223) & 0xFFFFFFFF;
                const j = ((s >>> 0) % (i + 1));
                [p[i], p[j]] = [p[j], p[i]];
            }
            for (let i = 0; i < 512; i++) this.perm[i] = p[i & 255];
        }

        _fade(t) { return t * t * t * (t * (t * 6 - 15) + 10); }

        _grad(hash, x, y) {
            const h = hash & 3;
            const u = h < 2 ? x : y;
            const v = h < 2 ? y : x;
            return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
        }

        noise(x, y) {
            const X = Math.floor(x) & 255;
            const Y = Math.floor(y) & 255;
            const xf = x - Math.floor(x);
            const yf = y - Math.floor(y);
            const u = this._fade(xf);
            const v = this._fade(yf);
            const p = this.perm;
            const A = p[X] + Y, B = p[X + 1] + Y;
            return 0.5 + 0.5 * (
                this._lerp(v,
                    this._lerp(u, this._grad(p[A], xf, yf), this._grad(p[B], xf - 1, yf)),
                    this._lerp(u, this._grad(p[A + 1], xf, yf - 1), this._grad(p[B + 1], xf - 1, yf - 1))
                )
            );
        }

        _lerp(t, a, b) { return a + t * (b - a); }

        // Fractal Brownian motion for multi-scale spatial autocorrelation
        fbm(x, y, octaves = 4, lacunarity = 2, persistence = 0.5) {
            let val = 0, amp = 1, freq = 1, max = 0;
            for (let i = 0; i < octaves; i++) {
                val += this.noise(x * freq, y * freq) * amp;
                max += amp;
                amp *= persistence;
                freq *= lacunarity;
            }
            return val / max;
        }
    }

    /* --- Generate multi-band raster --- */
    function generateBands(seed = 42, sensorNoise = 0.04) {
        const totalPixels = WIDTH * HEIGHT;
        // Flat array: bands[band * totalPixels + pixelIndex]
        const bands = new Float32Array(NUM_BANDS * totalPixels);

        // Each band gets its own noise generator with different seed and scale
        const bandConfigs = [
            { seed: seed + 1, scale: 0.003, octaves: 5, name: 'Blue reflectance' },
            { seed: seed + 2, scale: 0.003, octaves: 5, name: 'Green reflectance' },
            { seed: seed + 3, scale: 0.003, octaves: 5, name: 'Red reflectance' },
            { seed: seed + 4, scale: 0.004, octaves: 5, name: 'NIR reflectance' },
            { seed: seed + 5, scale: 0.005, octaves: 4, name: 'SWIR-1' },
            { seed: seed + 6, scale: 0.005, octaves: 4, name: 'SWIR-2' },
            { seed: seed + 7, scale: 0.006, octaves: 3, name: 'Thermal' },
            { seed: seed + 8, scale: 0.004, octaves: 5, name: 'NDVI-like' },
            { seed: seed + 9, scale: 0.007, octaves: 3, name: 'NDWI-like' },
            { seed: seed + 10, scale: 0.008, octaves: 3, name: 'Texture' },
        ];

        for (let b = 0; b < NUM_BANDS; b++) {
            const cfg = bandConfigs[b];
            const perlin = new PerlinNoise(cfg.seed);
            const offset = b * totalPixels;
            for (let y = 0; y < HEIGHT; y++) {
                for (let x = 0; x < WIDTH; x++) {
                    bands[offset + y * WIDTH + x] = perlin.fbm(x * cfg.scale, y * cfg.scale, cfg.octaves);
                }
            }
        }

        // Post-process: create band relationships that mimic real remote sensing
        // Make NDVI-like = (NIR - Red) / (NIR + Red + 0.001) remapped
        // Make NDWI-like = (Green - SWIR1) / (Green + SWIR1 + 0.001) remapped
        const nirOff = 3 * totalPixels;
        const redOff = 2 * totalPixels;
        const greenOff = 1 * totalPixels;
        const swir1Off = 4 * totalPixels;
        const ndviOff = 7 * totalPixels;
        const ndwiOff = 8 * totalPixels;

        for (let i = 0; i < totalPixels; i++) {
            const nir = bands[nirOff + i];
            const red = bands[redOff + i];
            const grn = bands[greenOff + i];
            const swir1 = bands[swir1Off + i];

            // Blend noise with derived index (70% derived, 30% noise for added complexity)
            const ndviDerived = (nir - red) / (nir + red + 0.001);
            bands[ndviOff + i] = 0.7 * (ndviDerived * 0.5 + 0.5) + 0.3 * bands[ndviOff + i];

            const ndwiDerived = (grn - swir1) / (grn + swir1 + 0.001);
            bands[ndwiOff + i] = 0.7 * (ndwiDerived * 0.5 + 0.5) + 0.3 * bands[ndwiOff + i];
        }

        // Add realistic sensor noise to each band
        // This creates a gap between the "true" spectral values (used for ground truth)
        // and the "observed" values (used for classification), mimicking real sensor noise
        const noisyBands = new Float32Array(bands.length);
        noisyBands.set(bands); // copy first
        let noiseRng = seed + 999;
        const noiseRand = () => { noiseRng = (noiseRng * 1664525 + 1013904223) & 0x7FFFFFFF; return noiseRng / 0x7FFFFFFF; };
        const NOISE_LEVEL = sensorNoise; // sensor measurement error level
        for (let b = 0; b < NUM_BANDS; b++) {
            const offset = b * totalPixels;
            for (let i = 0; i < totalPixels; i++) {
                noisyBands[offset + i] = bands[offset + i] + (noiseRand() - 0.5) * NOISE_LEVEL;
            }
        }

        return { bands, noisyBands, bandConfigs, width: WIDTH, height: HEIGHT };
    }

    /* --- Generate categorical ground truth (5 classes) --- */
    // Ground truth uses CLEAN bands + a hidden environmental variable
    // The classifier only sees NOISY bands and has no access to the environment variable
    // This makes classification realistically imperfect
    function generateCategoricalTruth(bands, seed) {
        const totalPixels = WIDTH * HEIGHT;
        const classes = new Uint8Array(totalPixels);

        // Hidden environmental gradient — influences class assignment
        // but is NOT available as a feature to the classifier
        const envNoise = new PerlinNoise(seed + 500);

        for (let i = 0; i < totalPixels; i++) {
            const x = i % WIDTH;
            const y = Math.floor(i / WIDTH);
            const ndvi = bands[7 * totalPixels + i];
            const ndwi = bands[8 * totalPixels + i];
            const thermal = bands[6 * totalPixels + i];
            const nir = bands[3 * totalPixels + i];

            // Environment variable: soil moisture / topographic wetness
            // Shifts thresholds, making class boundaries location-dependent
            const env = envNoise.fbm(x * 0.004, y * 0.004, 3) * 0.02;

            // Decision rules with environment-shifted thresholds
            // The env variable shifts boundaries enough to cause real confusion
            // but thresholds are set so all 5 classes maintain reasonable proportions
            if (ndwi > (0.54 + env) && ndvi < (0.50 + env)) {
                classes[i] = 3; // Water
            } else if (ndvi > (0.52 + env) && nir > (0.50 - env * 0.5)) {
                classes[i] = 0; // Dense Forest
            } else if (ndvi > (0.44 + env) && nir > (0.42 - env * 0.5)) {
                classes[i] = 1; // Open Forest
            } else if (ndvi > (0.33 + env) && thermal < (0.62 - env * 0.5)) {
                classes[i] = 2; // Grassland
            } else {
                classes[i] = 4; // Bare Soil
            }
        }

        return classes;
    }

    /* --- Generate continuous ground truth (biomass 0-500 Mg/ha) --- */
    function generateContinuousTruth(bands, biomassNoise = 0.06) {
        const totalPixels = WIDTH * HEIGHT;
        const biomass = new Float32Array(totalPixels);
        // Simple LCG for reproducible noise
        let rng = 12345;
        const rand = () => { rng = (rng * 1103515245 + 12345) & 0x7FFFFFFF; return rng / 0x7FFFFFFF; };

        for (let i = 0; i < totalPixels; i++) {
            const ndvi = bands[7 * totalPixels + i];
            const nir = bands[3 * totalPixels + i];
            const swir1 = bands[4 * totalPixels + i];
            const thermal = bands[6 * totalPixels + i];
            const texture = bands[9 * totalPixels + i];

            // Nonlinear biomass model
            let b = 500 * (
                0.35 * Math.pow(ndvi, 1.5) +
                0.25 * nir +
                0.15 * (1 - swir1) +
                0.10 * (1 - thermal) +
                0.15 * texture
            );

            // Add heteroscedastic noise (more noise at higher biomass)
            const noise = (rand() - 0.5) * biomassNoise * b;
            b = Math.max(0, Math.min(500, b + noise));
            biomass[i] = b;
        }

        return biomass;
    }

    /* --- Sample training/validation points --- */
    // Strategy: 'clustered' (mimics real field data, causes leakage if not blocked)
    //           'random'    (true random sampling, no leakage)
    function samplePoints(numPoints, strategy = 'clustered', seed = 123) {
        const points = new Uint32Array(numPoints);
        const used = new Set();
        let rng = seed;
        const rand = () => { rng = (rng * 1664525 + 1013904223) & 0x7FFFFFFF; return rng / 0x7FFFFFFF; };

        let ptsGenerated = 0;

        if (strategy === 'random') {
            const totalPixels = WIDTH * HEIGHT;
            while (ptsGenerated < numPoints) {
                const pt = Math.floor(rand() * totalPixels);
                if (!used.has(pt)) {
                    points[ptsGenerated++] = pt;
                    used.add(pt);
                }
            }
        } else {
            // Clustered
            // Define clusters to mimic field plots (e.g. 10 points per cluster)
            const ptsPerCluster = 10;
            const numClusters = Math.ceil(numPoints / ptsPerCluster);

            for (let c = 0; c < numClusters; c++) {
                // Pick a random cluster center
                const cx = Math.floor(rand() * WIDTH);
                const cy = Math.floor(rand() * HEIGHT);

                for (let p = 0; p < ptsPerCluster; p++) {
                    if (ptsGenerated >= numPoints) break;

                    let px, py, pt;
                    let tries = 0;
                    do {
                        // Sample within a ~15-pixel radius (e.g., a 450m radius at 30m resolution)
                        px = Math.min(WIDTH - 1, Math.max(0, cx + Math.floor((rand() - 0.5) * 30)));
                        py = Math.min(HEIGHT - 1, Math.max(0, cy + Math.floor((rand() - 0.5) * 30)));
                        pt = py * WIDTH + px;
                        tries++;
                    } while (used.has(pt) && tries < 100);

                    points[ptsGenerated++] = pt;
                    used.add(pt);
                }
            }
        }

        return points;
    }

    /* --- Collect feature vectors for sampled points --- */
    function extractFeatures(bands, pointIndices) {
        const n = pointIndices.length;
        const features = new Float32Array(n * NUM_BANDS);
        const totalPixels = WIDTH * HEIGHT;

        for (let i = 0; i < n; i++) {
            const px = pointIndices[i];
            for (let b = 0; b < NUM_BANDS; b++) {
                features[i * NUM_BANDS + b] = bands[b * totalPixels + px];
            }
        }
        return features;
    }

    /* --- Main generation interface --- */
    function generateLandscape(seed = 42, noiseLevel = 'medium') {
        const noiseLevels = { low: 0.02, medium: 0.04, high: 0.08 };
        const biomassNoiseLevels = { low: 0.03, medium: 0.06, high: 0.12 };
        const sensorNoise = noiseLevels[noiseLevel] || 0.04;
        const biomassNoise = biomassNoiseLevels[noiseLevel] || 0.06;
        
        const { bands, noisyBands, bandConfigs } = generateBands(seed, sensorNoise);
        // Ground truth uses CLEAN bands (true spectral values)
        const categoricalTruth = generateCategoricalTruth(bands, seed);
        const continuousTruth = generateContinuousTruth(bands, biomassNoise);

        // Compute class distribution
        const classCounts = new Uint32Array(NUM_CLASSES);
        for (let i = 0; i < categoricalTruth.length; i++) classCounts[categoricalTruth[i]]++;

        // Generate a 10,000-pixel independent test set for "True Landscape Accuracy"
        const numTrueTest = 10000;
        const trueTestIndices = new Uint32Array(numTrueTest);
        let trng = 9999;
        const trand = () => { trng = (trng * 1664525 + 1013904223) & 0x7FFFFFFF; return trng / 0x7FFFFFFF; };
        for (let i = 0; i < numTrueTest; i++) {
            trueTestIndices[i] = Math.floor(trand() * (WIDTH * HEIGHT));
        }
        
        // Extract features directly within the function
        const trueTestFeatures = new Float32Array(numTrueTest * NUM_BANDS);
        const totalPixels = WIDTH * HEIGHT;
        for (let i = 0; i < numTrueTest; i++) {
            const px = trueTestIndices[i];
            for (let b = 0; b < NUM_BANDS; b++) {
                trueTestFeatures[i * NUM_BANDS + b] = noisyBands[b * totalPixels + px];
            }
        }
        
        const trueTestLabelsCat = new Uint8Array(numTrueTest);
        const trueTestLabelsCont = new Float32Array(numTrueTest);
        for(let i=0; i<numTrueTest; i++) {
            trueTestLabelsCat[i] = categoricalTruth[trueTestIndices[i]];
            trueTestLabelsCont[i] = continuousTruth[trueTestIndices[i]];
        }

        return {
            width: WIDTH,
            height: HEIGHT,
            numBands: NUM_BANDS,
            bands: noisyBands,          // Noisy bands for classification (what the RF sees)
            cleanBands: bands,           // Clean bands (for reference/display only)
            bandConfigs,
            categoricalTruth,
            continuousTruth,
            classCounts,
            classNames: CLASS_NAMES,
            classColors: CLASS_COLORS,
            numClasses: NUM_CLASSES,
            trueTestFeatures,
            trueTestLabelsCat,
            trueTestLabelsCont,
            // These will be populated by sampleReferenceData
            trainingIndices: null,
            trainingFeatures: null,
        };
    }

    function sampleReferenceData(data, numTrainingPoints = 500, strategy = 'clustered', seed = 123) {
        data.trainingIndices = samplePoints(numTrainingPoints, strategy, seed);
        data.trainingFeatures = extractFeatures(data.bands, data.trainingIndices);
        return data;
    }

    return {
        generateLandscape,
        sampleReferenceData,
        generateBands,
        generateCategoricalTruth,
        generateContinuousTruth,
        samplePoints,
        extractFeatures,
        WIDTH, HEIGHT, NUM_BANDS, NUM_CLASSES,
        CLASS_NAMES, CLASS_COLORS,
    };
})();

if (typeof module !== 'undefined') module.exports = SyntheticData;
