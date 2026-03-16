/* ============================================================
   Main Application Controller
   Orchestrates the 5-step conservation mapping workflow:
   1. Synthetic data generation
   2. Spatial blocking & sampling design
   3. Bootstrap SCV Random Forest
   4. Accuracy assessment
   5. Summary statistics with uncertainty
   ============================================================ */

const App = (() => {
    // State
    let state = {
        mode: 'categorical', // 'categorical' or 'continuous'
        data: null,          // SyntheticData output
        blocks: null,        // SpatialBlocking output
        blockPoints: null,   // points assigned to blocks
        bootstrapSamples: null,
        results: [],         // array of per-bootstrap results
        running: false,
        workers: [],
        charts: {},          // active Chart.js instances

        // Running aggregates for pixel-level uncertainty
        pixelSum: null,      // Float64Array
        pixelSumSq: null,    // Float64Array

        // Config
        config: {
            numBootstraps: 1000,
            nTrees: 50,
            maxDepth: 10,
            minLeafSamples: 5,
            blockSize: 50,
            numTrainingPoints: 5000,
            seed: 42,
            numWorkers: Math.min(navigator.hardwareConcurrency || 4, 8),
        },
    };

    /* --- Initialization --- */
    function init() {
        PNASCharts.applyDefaults();
        bindEvents();
        setMode('categorical');
        openStep(1);
    }

    function bindEvents() {
        // Mode buttons
        document.getElementById('btn-categorical').addEventListener('click', () => setMode('categorical'));
        document.getElementById('btn-continuous').addEventListener('click', () => setMode('continuous'));

        // Step headers (accordion)
        document.querySelectorAll('.step-panel__header').forEach(h => {
            h.addEventListener('click', () => {
                const panel = h.closest('.step-panel');
                const stepNum = parseInt(panel.dataset.step);
                toggleStep(stepNum);
            });
        });

        // Step 1: Generate data
        document.getElementById('btn-generate').addEventListener('click', generateData);

        // Step 2: Create blocks
        document.getElementById('btn-create-blocks').addEventListener('click', createBlocks);

        // Step 3: Run bootstrap
        document.getElementById('btn-run-bootstrap').addEventListener('click', runBootstrap);
        document.getElementById('btn-stop-bootstrap').addEventListener('click', stopBootstrap);

        // Config inputs
        document.getElementById('cfg-bootstraps').addEventListener('change', (e) => {
            state.config.numBootstraps = parseInt(e.target.value) || 1000;
        });
        document.getElementById('cfg-trees').addEventListener('change', (e) => {
            state.config.nTrees = parseInt(e.target.value) || 50;
        });
        document.getElementById('cfg-block-size').addEventListener('change', (e) => {
            state.config.blockSize = parseInt(e.target.value) || 50;
        });
        document.getElementById('cfg-training-points').addEventListener('change', (e) => {
            state.config.numTrainingPoints = parseInt(e.target.value) || 5000;
        });
    }

    /* --- Mode switching --- */
    function setMode(mode) {
        state.mode = mode;
        document.getElementById('btn-categorical').classList.toggle('active', mode === 'categorical');
        document.getElementById('btn-continuous').classList.toggle('active', mode === 'continuous');

        // Toggle visibility of mode-specific elements
        document.querySelectorAll('[data-mode]').forEach(el => {
            const modes = el.dataset.mode.split(',');
            el.classList.toggle('hidden', !modes.includes(mode));
        });

        // Re-render if data exists
        if (state.data) renderStep1();
        if (state.results.length > 0) {
            renderStep4();
            renderStep5();
        }
    }

    /* --- Step accordion --- */
    function openStep(num) {
        document.querySelectorAll('.step-panel').forEach(p => {
            p.classList.toggle('open', parseInt(p.dataset.step) === num);
        });
    }

    function toggleStep(num) {
        const panel = document.querySelector(`.step-panel[data-step="${num}"]`);
        if (panel) panel.classList.toggle('open');
    }

    function markStepDone(num) {
        const panel = document.querySelector(`.step-panel[data-step="${num}"]`);
        if (panel) {
            panel.classList.add('completed');
            const badge = panel.querySelector('.step-panel__badge');
            if (badge) { badge.className = 'step-panel__badge step-panel__badge--done'; badge.textContent = 'Complete'; }
        }
    }

    function markStepRunning(num) {
        const panel = document.querySelector(`.step-panel[data-step="${num}"]`);
        if (panel) {
            const badge = panel.querySelector('.step-panel__badge');
            if (badge) { badge.className = 'step-panel__badge step-panel__badge--running'; badge.textContent = 'Running'; }
        }
    }

    /* ============================================
       STEP 1: Generate Synthetic Data
       ============================================ */
    function generateData() {
        const btn = document.getElementById('btn-generate');
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span> Generating...';

        // Use setTimeout to allow UI to update
        setTimeout(() => {
            const seed = parseInt(document.getElementById('cfg-seed')?.value) || state.config.seed;
            const numPts = state.config.numTrainingPoints;

            state.data = SyntheticData.generate(seed, numPts);
            renderStep1();

            btn.disabled = false;
            btn.innerHTML = '✓ Regenerate Data';
            markStepDone(1);
            openStep(2);
        }, 50);
    }

    function renderStep1() {
        const d = state.data;
        if (!d) return;

        // RGB composite (bands 3=NIR, 2=Red, 1=Green → false color)
        const rgbCanvas = document.getElementById('canvas-rgb');
        RasterViz.renderRGB(rgbCanvas, d.bands, d.width, d.height, 3, 2, 1);

        // Ground truth
        const truthCanvas = document.getElementById('canvas-truth');
        const legendContainer = document.getElementById('legend-truth');

        if (state.mode === 'categorical') {
            RasterViz.renderClassMap(truthCanvas, d.categoricalTruth, d.width, d.height, d.classColors);
            RasterViz.createDiscreteLegend(legendContainer, d.classNames, d.classColors);
        } else {
            RasterViz.renderContinuous(truthCanvas, d.continuousTruth, d.width, d.height, 0, 500);
            RasterViz.createColorBar(legendContainer, 0, 500, 'Mg/ha');
        }

        // Band stats
        const statsEl = document.getElementById('data-stats');
        const totalPixels = d.width * d.height;
        let statsHTML = `<div class="stats-row">`;
        statsHTML += `<div class="stat-card"><div class="stat-card__label">Dimensions</div><div class="stat-card__value">${d.width}×${d.height}</div></div>`;
        statsHTML += `<div class="stat-card"><div class="stat-card__label">Bands</div><div class="stat-card__value">${d.numBands}</div></div>`;
        statsHTML += `<div class="stat-card"><div class="stat-card__label">Pixels</div><div class="stat-card__value">${(totalPixels / 1e6).toFixed(1)}M</div></div>`;
        statsHTML += `<div class="stat-card"><div class="stat-card__label">Training Points</div><div class="stat-card__value">${d.trainingIndices.length.toLocaleString()}</div></div>`;

        if (state.mode === 'categorical') {
            statsHTML += `<div class="stat-card"><div class="stat-card__label">Classes</div><div class="stat-card__value">${d.numClasses}</div></div>`;
        } else {
            let minB = Infinity, maxB = -Infinity;
            for (let i = 0; i < d.continuousTruth.length; i++) {
                if (d.continuousTruth[i] < minB) minB = d.continuousTruth[i];
                if (d.continuousTruth[i] > maxB) maxB = d.continuousTruth[i];
            }
            statsHTML += `<div class="stat-card"><div class="stat-card__label">Biomass Range</div><div class="stat-card__value">${minB.toFixed(0)}–${maxB.toFixed(0)}</div><div class="stat-card__ci">Mg/ha</div></div>`;
        }
        statsHTML += `</div>`;

        // Class distribution (categorical)
        if (state.mode === 'categorical') {
            statsHTML += `<div class="stats-row">`;
            for (let i = 0; i < d.numClasses; i++) {
                const pct = (d.classCounts[i] / totalPixels * 100).toFixed(1);
                statsHTML += `<div class="stat-card"><div class="stat-card__label">${d.classNames[i]}</div><div class="stat-card__value">${pct}%</div><div class="stat-card__ci">${d.classCounts[i].toLocaleString()} px</div></div>`;
            }
            statsHTML += `</div>`;
        }

        statsEl.innerHTML = statsHTML;
    }

    /* ============================================
       STEP 2: Spatial Blocking
       ============================================ */
    function createBlocks() {
        const d = state.data;
        if (!d) { alert('Generate data first (Step 1)'); return; }

        const blockSize = state.config.blockSize;
        state.blocks = SpatialBlocking.createBlocks(d.width, d.height, blockSize);
        state.blockPoints = SpatialBlocking.assignPointsToBlocks(
            d.trainingIndices, state.blocks.pixelBlockMap, state.blocks.numBlocks
        );

        renderStep2();
        markStepDone(2);
        openStep(3);
    }

    function renderStep2() {
        const d = state.data;
        const b = state.blocks;
        if (!d || !b) return;

        // Show blocking grid on top of RGB composite
        const blockCanvas = document.getElementById('canvas-blocks');
        RasterViz.renderRGB(blockCanvas, d.bands, d.width, d.height, 3, 2, 1);

        // Show one example bootstrap to illustrate OOB concept
        const exSample = SpatialBlocking.bootstrapSample(b.numBlocks, 42);
        RasterViz.renderBlockGrid(blockCanvas, b.blockSize, d.width, d.height, exSample.oobBlocks, b.blocksX);

        // Training points overlay
        const pointsCanvas = document.getElementById('canvas-points');
        if (state.mode === 'categorical') {
            RasterViz.renderClassMap(pointsCanvas, d.categoricalTruth, d.width, d.height, d.classColors);
        } else {
            RasterViz.renderContinuous(pointsCanvas, d.continuousTruth, d.width, d.height, 0, 500);
        }
        RasterViz.renderBlockGrid(pointsCanvas, b.blockSize, d.width, d.height, null, b.blocksX);
        RasterViz.renderPoints(pointsCanvas, d.trainingIndices, d.width, d.height,
            d.categoricalTruth, d.classColors);

        // Stats
        const statsEl = document.getElementById('block-stats');
        const oobPct = (exSample.oobBlocks.length / b.numBlocks * 100).toFixed(1);
        statsEl.innerHTML = `
      <div class="stats-row">
        <div class="stat-card"><div class="stat-card__label">Block Size</div><div class="stat-card__value">${b.blockSize}×${b.blockSize}</div><div class="stat-card__ci">pixels</div></div>
        <div class="stat-card"><div class="stat-card__label">Total Blocks</div><div class="stat-card__value">${b.numBlocks}</div><div class="stat-card__ci">${b.blocksX}×${b.blocksY} grid</div></div>
        <div class="stat-card"><div class="stat-card__label">OOB Blocks (example)</div><div class="stat-card__value">${exSample.oobBlocks.length}</div><div class="stat-card__ci">${oobPct}% ≈ 36.8% expected</div></div>
        <div class="stat-card"><div class="stat-card__label">Training Points</div><div class="stat-card__value">${d.trainingIndices.length.toLocaleString()}</div></div>
      </div>
    `;
    }

    /* ============================================
       STEP 3: Bootstrap SCV Random Forest
       ============================================ */
    function runBootstrap() {
        if (!state.data || !state.blocks) { alert('Complete Steps 1-2 first'); return; }
        if (state.running) return;

        state.running = true;
        state.results = [];
        state.pixelSum = null;
        state.pixelSumSq = null;

        // Destroy old charts
        Object.values(state.charts).forEach(c => PNASCharts.destroy(c));
        state.charts = {};

        const btn = document.getElementById('btn-run-bootstrap');
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span> Running...';
        document.getElementById('btn-stop-bootstrap').classList.remove('hidden');
        markStepRunning(3);

        // Pre-generate bootstrap samples
        const B = state.config.numBootstraps;
        state.bootstrapSamples = SpatialBlocking.generateAllBootstraps(state.blocks.numBlocks, B, state.config.seed);

        // Extract training data for each bootstrap
        const d = state.data;
        const totalPixels = d.width * d.height;
        const isClassification = state.mode === 'categorical';

        // Prepare labels array for training points
        const trainingLabels = isClassification
            ? new Uint8Array(d.trainingIndices.length)
            : new Float32Array(d.trainingIndices.length);

        for (let i = 0; i < d.trainingIndices.length; i++) {
            trainingLabels[i] = isClassification
                ? d.categoricalTruth[d.trainingIndices[i]]
                : d.continuousTruth[d.trainingIndices[i]];
        }

        // Spawn workers
        const numWorkers = state.config.numWorkers;
        state.workers = [];
        for (let w = 0; w < numWorkers; w++) {
            const worker = new Worker('js/workers/rfWorker.js');
            worker.onmessage = handleWorkerResult;
            state.workers.push(worker);
        }

        // Job queue
        let nextJob = 0;
        const startTime = performance.now();

        function dispatchJob(workerIdx) {
            if (nextJob >= B || !state.running) return;

            const bIdx = nextJob++;
            const sample = state.bootstrapSamples[bIdx];
            const { indices: trainIdx, weights: trainWeights } = SpatialBlocking.getTrainingData(
                sample.blockWeights, state.blockPoints
            );

            // Collect training features and labels for this bootstrap
            const nTrain = trainIdx.length;
            const trainFeatures = new Float32Array(nTrain * d.numBands);
            const trainLabels = isClassification ? new Uint8Array(nTrain) : new Float32Array(nTrain);

            for (let i = 0; i < nTrain; i++) {
                const ptIdx = trainIdx[i]; // index into trainingIndices
                for (let b = 0; b < d.numBands; b++) {
                    trainFeatures[i * d.numBands + b] = d.trainingFeatures[ptIdx * d.numBands + b];
                }
                trainLabels[i] = trainingLabels[ptIdx];
            }

            // Collect OOB features and labels
            const oobPixels = [];
            for (const blockId of sample.oobBlocks) {
                if (state.blockPoints[blockId]) {
                    for (const ptIdx of state.blockPoints[blockId]) {
                        oobPixels.push(ptIdx);
                    }
                }
            }

            const nOOB = oobPixels.length;
            const oobFeatures = new Float32Array(nOOB * d.numBands);
            const oobLabels = isClassification ? new Uint8Array(nOOB) : new Float32Array(nOOB);

            for (let i = 0; i < nOOB; i++) {
                const ptIdx = oobPixels[i];
                for (let b = 0; b < d.numBands; b++) {
                    oobFeatures[i * d.numBands + b] = d.trainingFeatures[ptIdx * d.numBands + b];
                }
                oobLabels[i] = trainingLabels[ptIdx];
            }

            // For every 10th bootstrap, also predict the full raster (for uncertainty maps)
            const computeFullMap = (bIdx % 10 === 0);
            let fullRasterFeatures = null;
            if (computeFullMap) {
                // Subsample: predict every 4th pixel for speed (250K pixels)
                const stride = 2;
                const subW = Math.ceil(d.width / stride);
                const subH = Math.ceil(d.height / stride);
                const nSub = subW * subH;
                fullRasterFeatures = new Float32Array(nSub * d.numBands);
                let k = 0;
                for (let y = 0; y < d.height; y += stride) {
                    for (let x = 0; x < d.width; x += stride) {
                        const px = y * d.width + x;
                        for (let b = 0; b < d.numBands; b++) {
                            fullRasterFeatures[k * d.numBands + b] = d.bands[b * totalPixels + px];
                        }
                        k++;
                    }
                }
            }

            state.workers[workerIdx].postMessage({
                type: 'bootstrap',
                bootstrapIndex: bIdx,
                trainingFeatures: trainFeatures,
                trainingLabels: trainLabels,
                trainingWeights: trainWeights,
                oobFeatures,
                oobLabels,
                fullRasterFeatures,
                numBands: d.numBands,
                numClasses: d.numClasses,
                isClassification,
                config: {
                    nTrees: state.config.nTrees,
                    maxDepth: state.config.maxDepth,
                    minLeafSamples: state.config.minLeafSamples,
                },
                seed: state.config.seed + bIdx * 13,
                computeFullMap,
            });
        }

        function handleWorkerResult(e) {
            if (!state.running) return;
            const result = e.data;
            state.results.push(result);

            // Aggregate pixel predictions for uncertainty map
            if (result.fullPredictions) {
                const stride = 2;
                const subW = Math.ceil(d.width / stride);
                const subH = Math.ceil(d.height / stride);
                const nSub = subW * subH;

                if (!state.pixelSum) {
                    state.pixelSum = new Float64Array(nSub);
                    state.pixelSumSq = new Float64Array(nSub);
                    state.pixelCount = 0;
                }

                for (let i = 0; i < nSub; i++) {
                    const v = result.fullPredictions[i];
                    state.pixelSum[i] += v;
                    state.pixelSumSq[i] += v * v;
                }
                state.pixelCount++;
            }

            // Update progress
            const completed = state.results.length;
            const elapsed = performance.now() - startTime;
            const rate = completed / (elapsed / 1000);
            const eta = (B - completed) / rate;

            updateProgress(completed, B, elapsed, eta);

            // Find which worker sent this and dispatch next job
            const workerIdx = state.workers.findIndex(w => e.target === w);
            if (workerIdx >= 0 && completed < B) {
                dispatchJob(workerIdx);
            }

            // Live update charts every 50 completions
            if (completed % 50 === 0 || completed === B) {
                renderStep4();
            }

            // All done
            if (completed >= B) {
                finishBootstrap();
            }
        }

        // Dispatch initial jobs
        for (let w = 0; w < numWorkers; w++) {
            dispatchJob(w);
        }
    }

    function stopBootstrap() {
        state.running = false;
        state.workers.forEach(w => w.terminate());
        state.workers = [];

        document.getElementById('btn-run-bootstrap').disabled = false;
        document.getElementById('btn-run-bootstrap').innerHTML = '▶ Run Bootstrap';
        document.getElementById('btn-stop-bootstrap').classList.add('hidden');

        if (state.results.length > 0) {
            renderStep4();
            renderStep5();
            markStepDone(3);
        }
    }

    function finishBootstrap() {
        state.running = false;
        state.workers.forEach(w => w.terminate());
        state.workers = [];

        document.getElementById('btn-run-bootstrap').disabled = false;
        document.getElementById('btn-run-bootstrap').innerHTML = '✓ Re-run Bootstrap';
        document.getElementById('btn-stop-bootstrap').classList.add('hidden');

        markStepDone(3);
        renderStep4();
        renderStep5();
        openStep(4);
    }

    function updateProgress(completed, total, elapsedMs, etaSeconds) {
        const pct = (completed / total * 100).toFixed(1);
        const fill = document.getElementById('progress-fill');
        if (fill) fill.style.width = pct + '%';

        const info = document.getElementById('progress-info');
        if (info) {
            const elapsedStr = (elapsedMs / 1000).toFixed(1);
            const etaStr = etaSeconds < 60
                ? `${Math.ceil(etaSeconds)}s`
                : `${Math.floor(etaSeconds / 60)}m ${Math.ceil(etaSeconds % 60)}s`;
            info.innerHTML = `
        <span>${completed} / ${total} replicates (${pct}%)</span>
        <span>Elapsed: ${elapsedStr}s | ETA: ${etaStr}</span>
      `;
        }
    }

    /* ============================================
       STEP 4: Accuracy Assessment
       ============================================ */
    function renderStep4() {
        if (state.results.length === 0) return;

        const isClassification = state.mode === 'categorical';

        if (isClassification) {
            renderClassificationAccuracy();
        } else {
            renderRegressionAccuracy();
        }
    }

    function renderClassificationAccuracy() {
        const results = state.results;
        const d = state.data;

        // Extract per-bootstrap accuracy arrays
        const overallAccuracies = results.map(r => r.metrics.overallAccuracy);
        const perClassUser = d.classNames.map((_, c) => results.map(r => r.metrics.userAccuracy[c]));
        const perClassProducer = d.classNames.map((_, c) => results.map(r => r.metrics.producerAccuracy[c]));

        // Overall accuracy histogram
        PNASCharts.destroy(state.charts.overallAcc);
        document.getElementById('chart1-container-title').textContent = 'Overall Accuracy Distribution';
        state.charts.overallAcc = PNASCharts.histogram('chart-overall-acc', overallAccuracies, {
            xLabel: 'Overall Accuracy',
            yLabel: 'Frequency',
            color: '#2D6A4F',
            bins: 25,
            thresholdLine: 0.85,
        });

        // Per-class user accuracy box summary
        const userStats = perClassUser.map(vals => PNASCharts.summaryStats(vals));
        PNASCharts.destroy(state.charts.userAcc);
        document.getElementById('chart2-container-title').textContent = "User's Accuracy by Class";
        state.charts.userAcc = PNASCharts.boxSummary('chart-user-acc', userStats, {
            xLabels: d.classNames,
            yLabel: 'Accuracy',
        });

        // Per-class producer accuracy box summary
        const prodStats = perClassProducer.map(vals => PNASCharts.summaryStats(vals));
        PNASCharts.destroy(state.charts.prodAcc);
        document.getElementById('chart3-container-title').textContent = "Producer's Accuracy by Class";
        state.charts.prodAcc = PNASCharts.boxSummary('chart-prod-acc', prodStats, {
            xLabels: d.classNames,
            yLabel: 'Accuracy',
        });

        // Summary stat cards
        const oaStats = PNASCharts.summaryStats(overallAccuracies);
        const statsEl = document.getElementById('accuracy-stats');
        if (statsEl) {
            let html = `<div class="stats-row">`;
            html += statCard('Overall Accuracy', oaStats.mean, oaStats.ci95, 0.85, true);
            for (let i = 0; i < d.numClasses; i++) {
                html += statCard(`${d.classNames[i]} (User)`, userStats[i].mean, userStats[i].ci95, 0.85, true);
            }
            html += `</div>`;
            statsEl.innerHTML = html;
        }

        // Mean confusion matrix
        document.getElementById('confmat-title').textContent = `Mean Confusion Matrix (${results.length} replicates)`;
        renderMeanConfusionMatrix(results, d);
    }

    function renderRegressionAccuracy() {
        const results = state.results;

        const rmseVals = results.map(r => r.metrics.rmse);
        const relRmseVals = results.map(r => r.metrics.relRmse);
        const r2Vals = results.map(r => r.metrics.r2);

        // R² distribution
        PNASCharts.destroy(state.charts.r2Hist);
        document.getElementById('chart1-container-title').textContent = 'R² Distribution';
        state.charts.r2Hist = PNASCharts.histogram('chart-overall-acc', r2Vals, {
            xLabel: 'R²',
            yLabel: 'Frequency',
            color: '#2D6A4F',
            bins: 25,
            thresholdLine: 0.8,
        });

        // RMSE distribution
        PNASCharts.destroy(state.charts.rmseHist);
        document.getElementById('chart2-container-title').textContent = 'RMSE Distribution';
        state.charts.rmseHist = PNASCharts.histogram('chart-user-acc', rmseVals, {
            xLabel: 'RMSE (Mg/ha)',
            yLabel: 'Frequency',
            color: '#4477AA',
            bins: 25,
        });

        // Relative RMSE
        PNASCharts.destroy(state.charts.relRmseHist);
        document.getElementById('chart3-container-title').textContent = 'Relative RMSE Distribution';
        state.charts.relRmseHist = PNASCharts.histogram('chart-prod-acc', relRmseVals, {
            xLabel: 'Relative RMSE',
            yLabel: 'Frequency',
            color: '#EE6677',
            bins: 25,
            thresholdLine: 0.2,
        });

        // Stats
        const r2Stats = PNASCharts.summaryStats(r2Vals);
        const rmseStats = PNASCharts.summaryStats(rmseVals);
        const relRmseStats = PNASCharts.summaryStats(relRmseVals);

        const statsEl = document.getElementById('accuracy-stats');
        if (statsEl) {
            let html = `<div class="stats-row">`;
            html += statCard('R²', r2Stats.mean, r2Stats.ci95, 0.8, true);
            html += statCard('RMSE', rmseStats.mean, rmseStats.ci95, null, false);
            html += statCard('Relative RMSE', relRmseStats.mean, relRmseStats.ci95, 0.2, false, true);
            html += `</div>`;
            statsEl.innerHTML = html;
        }
    }

    function renderMeanConfusionMatrix(results, d) {
        const el = document.getElementById('confusion-matrix-container');
        if (!el) return;

        const nc = d.numClasses;
        const meanConf = new Float64Array(nc * nc);
        for (const r of results) {
            for (let i = 0; i < nc * nc; i++) {
                meanConf[i] += r.metrics.confusionMatrix[i];
            }
        }
        const nResults = results.length;
        for (let i = 0; i < nc * nc; i++) meanConf[i] /= nResults;

        let html = `<table class="confusion-matrix"><thead><tr><th></th><th colspan="${nc}" style="text-align:center;font-style:italic">Predicted</th><th></th></tr><tr><th>Reference ↓</th>`;
        for (const name of d.classNames) html += `<th>${name.split(' ')[0]}</th>`;
        html += `<th>Prod. Acc.</th></tr></thead><tbody>`;

        for (let i = 0; i < nc; i++) {
            html += `<tr><th>${d.classNames[i]}</th>`;
            let rowSum = 0;
            for (let j = 0; j < nc; j++) {
                const val = meanConf[i * nc + j];
                rowSum += val;
                const cls = i === j ? ' class="diag"' : '';
                html += `<td${cls}>${val.toFixed(0)}</td>`;
            }
            const prodAcc = rowSum > 0 ? (meanConf[i * nc + i] / rowSum * 100).toFixed(1) : '0';
            html += `<td><strong>${prodAcc}%</strong></td></tr>`;
        }

        // User accuracy row
        html += `<tr><th>User Acc.</th>`;
        for (let j = 0; j < nc; j++) {
            let colSum = 0;
            for (let i = 0; i < nc; i++) colSum += meanConf[i * nc + j];
            const userAcc = colSum > 0 ? (meanConf[j * nc + j] / colSum * 100).toFixed(1) : '0';
            html += `<td><strong>${userAcc}%</strong></td>`;
        }
        html += `<td></td></tr>`;
        html += `</tbody></table>`;

        el.innerHTML = html;
    }

    /* ============================================
       STEP 5: Summary Statistics
       ============================================ */
    function renderStep5() {
        if (state.results.length === 0) return;
        markStepDone(4);

        const isClassification = state.mode === 'categorical';

        if (isClassification) {
            renderCategoricalSummary();
        } else {
            renderContinuousSummary();
        }

        // Render uncertainty map
        renderUncertaintyMap();
        openStep(5);
        markStepDone(5);
    }

    function renderCategoricalSummary() {
        const results = state.results;
        const d = state.data;
        const nc = d.numClasses;
        const totalPixels = d.width * d.height;

        // True class proportions
        const trueProps = d.classNames.map((_, c) => d.classCounts[c] / totalPixels);

        // Corrected area estimates from each bootstrap
        const correctedAreas = d.classNames.map((_, c) => results.map(r => r.metrics.correctedArea[c]));

        // Uncorrected (raw prediction counts)
        const rawAreas = d.classNames.map((_, c) => results.map(r => r.metrics.predictedCounts[c]));

        // Summary stats
        const statsEl = document.getElementById('summary-stats');
        let html = `<div class="info-alert"><strong>Area Estimation with Uncertainty:</strong> Each of the ${results.length} bootstrap replicates produces an independent map. Class areas are then corrected using the error matrix (Olofsson et al. approach). The distributions below show the uncertainty in these corrected estimates.</div>`;

        html += `<div class="stats-row">`;
        for (let c = 0; c < nc; c++) {
            const stats = PNASCharts.summaryStats(correctedAreas[c]);
            const truePct = (trueProps[c] * 100).toFixed(1);
            const estPct = (stats.mean / totalPixels * 100).toFixed(1);
            const ciPct = `[${(stats.ci95[0] / totalPixels * 100).toFixed(1)}, ${(stats.ci95[1] / totalPixels * 100).toFixed(1)}]`;
            html += `
        <div class="stat-card">
          <div class="stat-card__label">${d.classNames[c]}</div>
          <div class="stat-card__value">${estPct}%</div>
          <div class="stat-card__ci">95% CI: ${ciPct}</div>
          <div class="stat-card__ci">True: ${truePct}%</div>
        </div>
      `;
        }
        html += `</div>`;
        statsEl.innerHTML = html;

        // Area distribution for largest class
        const largestClass = trueProps.indexOf(Math.max(...trueProps));
        PNASCharts.destroy(state.charts.areaDist);
        document.getElementById('summary-chart-title').textContent = `Corrected Area: ${d.classNames[largestClass]}`;
        state.charts.areaDist = PNASCharts.histogram('chart-area-dist', correctedAreas[largestClass].map(v => v / totalPixels * 100), {
            xLabel: 'Corrected Area (%)',
            yLabel: 'Frequency',
            color: PNASCharts.CLASS_PALETTE[largestClass],
            bins: 25,
        });
    }

    function renderContinuousSummary() {
        const results = state.results;

        // Total biomass estimates from residual correction
        const totalBiomass = results.map(r => {
            // Simple: use mean prediction * total pixels, corrected by mean residual
            const correction = r.metrics.meanResidual || 0;
            return (r.metrics.totalPredicted || 0) - correction * (r.metrics.n || 1);
        });

        const stats = PNASCharts.summaryStats(totalBiomass);

        // True total biomass
        const d = state.data;
        let trueTotal = 0;
        for (let i = 0; i < d.continuousTruth.length; i++) trueTotal += d.continuousTruth[i];

        const statsEl = document.getElementById('summary-stats');
        let html = `<div class="info-alert"><strong>Biomass Estimation with Uncertainty:</strong> Each bootstrap replicate predicts OOB biomass values. The mean residual is used to bias-correct the population sum. The distribution shows uncertainty from spatial cross-validation.</div>`;

        html += `<div class="stats-row">`;
        html += `
      <div class="stat-card">
        <div class="stat-card__label">Bias-Corrected OOB Total</div>
        <div class="stat-card__value">${(stats.mean / 1e6).toFixed(2)}M</div>
        <div class="stat-card__ci">95% CI: [${(stats.ci95[0] / 1e6).toFixed(2)}, ${(stats.ci95[1] / 1e6).toFixed(2)}]M Mg</div>
      </div>
    `;
        html += `
      <div class="stat-card">
        <div class="stat-card__label">True Total (Full Raster)</div>
        <div class="stat-card__value">${(trueTotal / 1e6).toFixed(2)}M</div>
        <div class="stat-card__ci">Mg/ha summed over 1M pixels</div>
      </div>
    `;
        const r2Stats = PNASCharts.summaryStats(results.map(r => r.metrics.r2));
        html += statCard('Median R²', r2Stats.median, r2Stats.ci95, 0.8, true);
        html += `</div>`;
        statsEl.innerHTML = html;

        // Biomass distribution
        PNASCharts.destroy(state.charts.areaDist);
        document.getElementById('summary-chart-title').textContent = 'Bias-Corrected Total Biomass';
        state.charts.areaDist = PNASCharts.histogram('chart-area-dist', totalBiomass.map(v => v / 1e6), {
            xLabel: 'Total Biomass (M Mg)',
            yLabel: 'Frequency',
            color: '#228833',
            bins: 25,
        });
    }

    function renderUncertaintyMap() {
        if (!state.pixelSum || state.pixelCount < 2) return;

        const d = state.data;
        const stride = 2;
        const subW = Math.ceil(d.width / stride);
        const subH = Math.ceil(d.height / stride);
        const nSub = subW * subH;
        const n = state.pixelCount;

        // Compute std dev per pixel
        const uncertainty = new Float32Array(nSub);
        let maxUncertainty = 0;

        for (let i = 0; i < nSub; i++) {
            const mean = state.pixelSum[i] / n;
            const variance = state.pixelSumSq[i] / n - mean * mean;
            uncertainty[i] = Math.sqrt(Math.max(0, variance));
            if (uncertainty[i] > maxUncertainty) maxUncertainty = uncertainty[i];
        }

        // Render on canvas
        const canvas = document.getElementById('canvas-uncertainty');
        if (canvas) {
            RasterViz.renderContinuous(canvas, uncertainty, subW, subH, 0, maxUncertainty);
            const legendEl = document.getElementById('legend-uncertainty');
            if (legendEl) {
                RasterViz.createColorBar(legendEl, 0, maxUncertainty, state.mode === 'categorical' ? 'Prediction std. dev.' : 'Biomass std. dev. (Mg/ha)');
            }
        }

        // Also render mean prediction
        const meanCanvas = document.getElementById('canvas-mean-pred');
        if (meanCanvas) {
            const meanPred = new Float32Array(nSub);
            for (let i = 0; i < nSub; i++) meanPred[i] = state.pixelSum[i] / n;

            if (state.mode === 'categorical') {
                // Round to nearest class
                const classPred = new Uint8Array(nSub);
                for (let i = 0; i < nSub; i++) classPred[i] = Math.round(Math.max(0, Math.min(d.numClasses - 1, meanPred[i])));
                RasterViz.renderClassMap(meanCanvas, classPred, subW, subH, d.classColors);
            } else {
                RasterViz.renderContinuous(meanCanvas, meanPred, subW, subH, 0, 500);
            }
        }
    }

    /* --- Helper: stat card HTML --- */
    function statCard(label, value, ci95, threshold, higherBetter, lowerBetter) {
        let status = '';
        if (threshold !== null && threshold !== undefined) {
            if (higherBetter) {
                status = value >= threshold ? 'pass' : (ci95[1] >= threshold ? 'warn' : 'fail');
            }
            if (lowerBetter) {
                status = value <= threshold ? 'pass' : (ci95[0] <= threshold ? 'warn' : 'fail');
            }
        }

        const fmtVal = value < 1 ? value.toFixed(3) : value.toFixed(1);
        const fmtCI = ci95 ? `[${ci95[0] < 1 ? ci95[0].toFixed(3) : ci95[0].toFixed(1)}, ${ci95[1] < 1 ? ci95[1].toFixed(3) : ci95[1].toFixed(1)}]` : '';

        return `
      <div class="stat-card${status ? ' stat-card--' + status : ''}">
        <div class="stat-card__label">${label}</div>
        <div class="stat-card__value">${fmtVal}</div>
        ${fmtCI ? `<div class="stat-card__ci">95% CI: ${fmtCI}</div>` : ''}
        ${threshold != null ? `<div class="stat-card__ci">Threshold: ${threshold}</div>` : ''}
      </div>
    `;
    }

    return { init };
})();

// Boot
document.addEventListener('DOMContentLoaded', App.init);
