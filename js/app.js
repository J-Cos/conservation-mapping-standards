/* ============================================================
   Main Application Controller
   Orchestrates the 7-step mapping workflow:
   1. Synthetic data generation
   2. Reference Data Collection
   3. Spatial blocking & sampling design
   4. Bootstrap SCV Random Forest
   5. Accuracy assessment
   6. Summary statistics with uncertainty
   7. Final Verdict Report Card
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
            numBootstraps: 100,
            nTrees: 100,
            maxDepth: 10,
            minLeafSamples: 5,
            blockSize: 200,
            numTrainingPoints: 500,
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
        document.getElementById('btn-generate').addEventListener('click', generateLandscapeData);

        // Step 2: Collect Reference Data
        document.getElementById('btn-collect-data').addEventListener('click', collectReferenceData);

        // Step 3: Create blocks
        document.getElementById('btn-create-blocks').addEventListener('click', createBlocks);

        // Step 4: Run bootstrap
        document.getElementById('btn-run-bootstrap').addEventListener('click', runBootstrap);
        document.getElementById('btn-stop-bootstrap').addEventListener('click', stopBootstrap);

        // Config inputs
        document.getElementById('cfg-bootstraps').addEventListener('change', (e) => {
            state.config.numBootstraps = parseInt(e.target.value) || 100;
        });
        document.getElementById('cfg-trees').addEventListener('change', (e) => {
            state.config.nTrees = parseInt(e.target.value) || 100;
        });
        document.getElementById('cfg-block-size').addEventListener('change', (e) => {
            state.config.blockSize = parseInt(e.target.value) || 200;
        });
        document.getElementById('cfg-training-points').addEventListener('change', (e) => {
            state.config.numTrainingPoints = parseInt(e.target.value) || 500;
        });
        document.getElementById('cfg-sampling-strategy').addEventListener('change', (e) => {
            state.config.samplingStrategy = e.target.value;
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
        // Open the target step; keep completed steps open
        document.querySelectorAll('.step-panel').forEach(p => {
            const stepNum = parseInt(p.dataset.step);
            if (stepNum === num) {
                p.classList.add('open');
            } else if (!p.classList.contains('completed')) {
                // Close uncompleted panels that aren't the target
                p.classList.remove('open');
            }
        });
        // Scroll the newly opened step into view
        const target = document.querySelector(`.step-panel[data-step="${num}"]`);
        if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
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
       STEP 1: Generate Synthetic Landscape
       ============================================ */
    function generateLandscapeData() {
        const btn = document.getElementById('btn-generate');
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span> Generating...';

        // Use setTimeout to allow UI to update
        setTimeout(() => {
            const seed = parseInt(document.getElementById('cfg-seed')?.value) || state.config.seed;
            const noiseLevel = document.getElementById('cfg-noise')?.value || 'medium';
            state.config.noiseLevel = noiseLevel;

            state.data = SyntheticData.generateLandscape(seed, noiseLevel);
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

        if (d.trainingIndices) {
            // Also draw sampling points if we're rerendering Step 1 after Step 2
            renderStep2();
        }
    }

    /* ============================================
       STEP 2: Reference Data Collection
       ============================================ */
    function collectReferenceData() {
        if (!state.data) { alert('Complete Step 1 first'); return; }
        
        const btn = document.getElementById('btn-collect-data');
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span> Collecting...';

        setTimeout(() => {
            const numPts = state.config.numTrainingPoints;
            const strategy = state.config.samplingStrategy || 'clustered';
            const seed = state.config.seed + 100;

            SyntheticData.sampleReferenceData(state.data, numPts, strategy, seed);
            renderStep2();

            btn.disabled = false;
            btn.innerHTML = '✓ Re-collect Data';
            markStepDone(2);
            openStep(3);
        }, 50);
    }

    function renderStep2() {
        const d = state.data;
        if (!d || !d.trainingIndices) return;

        const canvas = document.getElementById('canvas-sampling');
        if (!canvas) return;

        // Render Truth as background
        if (state.mode === 'categorical') {
            RasterViz.renderClassMap(canvas, d.categoricalTruth, d.width, d.height, d.classColors);
        } else {
            RasterViz.renderContinuous(canvas, d.continuousTruth, d.width, d.height, 0, 500);
        }

        // Overlay points
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#000000';
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 0.5;

        for (let i = 0; i < d.trainingIndices.length; i++) {
            const pt = d.trainingIndices[i];
            const px = pt % d.width;
            const py = Math.floor(pt / d.width);
            ctx.beginPath();
            ctx.arc(px, py, 2.5, 0, 2 * Math.PI);
            ctx.fill();
            ctx.stroke();
        }
    }

    /* ============================================
       STEP 3: Spatial Blocking
       ============================================ */
    function createBlocks() {
        const d = state.data;
        if (!d) { alert('Generate and collect data first (Steps 1-2)'); return; }
        
        const btn = document.getElementById('btn-create-blocks');
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span> Creating Blocks...';

        setTimeout(() => {
            const blockSize = state.config.blockSize;
            state.blocks = SpatialBlocking.createBlocks(d.width, d.height, blockSize);
            state.blockPoints = SpatialBlocking.assignPointsToBlocks(
                d.trainingIndices, state.blocks.pixelBlockMap, state.blocks.numBlocks
            );

            renderStep3();

            btn.disabled = false;
            btn.innerHTML = '✓ Recreate Blocks';
            markStepDone(3);
            openStep(4);
        }, 50);
    }

    function renderStep3() {
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
       STEP 4: Bootstrap SCV Random Forest
       ============================================ */
    function runBootstrap() {
        if (!state.data || !state.blocks) { alert('Complete Steps 1-2 first'); return; }
        if (state.running) return;

        state.running = true;
        state.results = [];
        state.pixelSum = null;
        state.pixelSumSq = null;
        state.pixelCount = 0;

        // Hide pitfall comparison from previous run
        const pitfallPanel = document.getElementById('pitfall-comparison');
        if (pitfallPanel) pitfallPanel.classList.add('hidden');
        const pitfallBtn = document.getElementById('btn-pitfall-compare');
        if (pitfallBtn) pitfallBtn.remove();
        
        // Destroy old charts
        Object.values(state.charts).forEach(c => PNASCharts.destroy(c));
        state.charts = {};

        const btn = document.getElementById('btn-run-bootstrap');
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span> Running...';
        document.getElementById('btn-stop-bootstrap').classList.remove('hidden');
        markStepRunning(4);

        const d = state.data;
        const isClassification = state.mode === 'categorical';

        // Create an independent 10,000 point test set from the full raster
        // to represent the "True Population Accuracy"
        if (!state.data.trueTestFeatures) {
            const numTestPoints = 10000;
            const testFeatures = new Float32Array(numTestPoints * state.data.numBands);
            const testLabels = isClassification ? new Uint8Array(numTestPoints) : new Float32Array(numTestPoints);
            // Deterministic RNG for reproducible test set
            let trng = state.config.seed + 999;
            const trand = () => { trng = (trng * 1664525 + 1013904223) & 0x7FFFFFFF; return trng / 0x7FFFFFFF; };
            for(let i=0; i<numTestPoints; i++) {
                const p = Math.floor(trand() * (state.data.width * state.data.height));
                for(let b=0; b<state.data.numBands; b++) {
                    testFeatures[i * state.data.numBands + b] = state.data.bands[b * (state.data.width * state.data.height) + p];
                }
                testLabels[i] = isClassification ? state.data.categoricalTruth[p] : state.data.continuousTruth[p];
            }
            state.data.trueTestFeatures = testFeatures;
            state.data.trueTestLabels = testLabels;
        }

        // Pre-generate bootstrap samples
        const B = state.config.numBootstraps;
        state.bootstrapSamples = SpatialBlocking.generateAllBootstraps(state.blocks.numBlocks, B, state.config.seed);

        // Extract training data for each bootstrap
        const totalPixels = d.width * d.height;

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
                trueTestFeatures: state.data.trueTestFeatures,
                trueTestLabels: isClassification ? state.data.trueTestLabelsCat : state.data.trueTestLabelsCont,
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
        markStepDone(4);
        
        // Also trigger rendering of Steps 5 and 6
        renderStep5();
        renderStep6();
        renderReportCard();
        
        addPitfallButton();
        openStep(5);
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
        const trueAccuracies = results.filter(r => r.trueMetrics).map(r => r.trueMetrics.overallAccuracy);
        const perClassUser = d.classNames.map((_, c) => results.map(r => r.metrics.userAccuracy[c]));
        const perClassProducer = d.classNames.map((_, c) => results.map(r => r.metrics.producerAccuracy[c]));

        // Summary stat cards
        const oaStats = PNASCharts.summaryStats(overallAccuracies);
        const trueOAMean = trueAccuracies.length ? (trueAccuracies.reduce((a,b)=>a+b,0)/trueAccuracies.length) : null;

        // Overall accuracy histogram
        PNASCharts.destroy(state.charts.overallAcc);
        document.getElementById('chart1-container-title').textContent = 'Overall Accuracy Distribution';
        state.charts.overallAcc = PNASCharts.histogram('chart-overall-acc', overallAccuracies, {
            xLabel: 'Overall Accuracy',
            yLabel: 'Frequency',
            color: '#2D6A4F',
            bins: 25,
            thresholdLine: 0.85,
            thresholdLabel: 'Good (85%)',
            trueLine: trueOAMean,
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

        const statsEl = document.getElementById('accuracy-stats');
        if (statsEl) {
            let html = `<div class="stats-row">`;
            html += statCard('Overall Accuracy', oaStats.mean, oaStats.ci95, 0.85, true, false, trueOAMean);
            for (let i = 0; i < d.numClasses; i++) {
                const classTrueUser = trueAccuracies.length ? (results.reduce((sum, r) => sum + r.trueMetrics.userAccuracy[i], 0) / results.length) : null;
                html += statCard(`${d.classNames[i]}<br><span style="font-size:0.8em; font-weight:normal;">(USER)</span>`, userStats[i].mean, userStats[i].ci95, 0.85, true, false, classTrueUser);
            }
            html += `</div>`;
            statsEl.innerHTML = html;
        }

        // Mean confusion matrix
        document.getElementById('confmat-title').textContent = `Mean Error Matrix (${results.length} replicates)`;
        renderMeanConfusionMatrix(results, d);
    }

    function renderRegressionAccuracy() {
        const results = state.results;

        const rmseVals = results.map(r => r.metrics.rmse);
        const relRmseVals = results.map(r => r.metrics.relRmse);
        const r2Vals = results.map(r => r.metrics.r2);

        const trueAccuracies = results.filter(r => r.trueMetrics).map(r => r.trueMetrics);
        const trueR2Mean = trueAccuracies.length ? trueAccuracies.reduce((a,b)=>a+b.r2,0)/trueAccuracies.length : null;
        const trueRmseMean = trueAccuracies.length ? trueAccuracies.reduce((a,b)=>a+b.rmse,0)/trueAccuracies.length : null;
        const trueRelRmseMean = trueAccuracies.length ? trueAccuracies.reduce((a,b)=>a+b.relRmse,0)/trueAccuracies.length : null;

        // R² distribution
        PNASCharts.destroy(state.charts.r2Hist);
        document.getElementById('chart1-container-title').textContent = 'R² Distribution';
        state.charts.r2Hist = PNASCharts.histogram('chart-overall-acc', r2Vals, {
            xLabel: 'R²',
            yLabel: 'Frequency',
            color: '#2D6A4F',
            bins: 25,
            thresholdLine: 0.8,
            thresholdLabel: 'Good (0.8)',
            trueLine: trueR2Mean,
        });

        // RMSE distribution
        PNASCharts.destroy(state.charts.rmseHist);
        document.getElementById('chart2-container-title').textContent = 'RMSE Distribution';
        state.charts.rmseHist = PNASCharts.histogram('chart-user-acc', rmseVals, {
            xLabel: 'RMSE (Mg/ha)',
            yLabel: 'Frequency',
            color: '#4477AA',
            bins: 25,
            trueLine: trueRmseMean,
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
            thresholdLabel: 'Max (20%)',
            trueLine: trueRelRmseMean,
        });

        // Stats
        const r2Stats = PNASCharts.summaryStats(r2Vals);
        const rmseStats = PNASCharts.summaryStats(rmseVals);
        const relRmseStats = PNASCharts.summaryStats(relRmseVals);

        const statsEl = document.getElementById('accuracy-stats');
        if (statsEl) {
            let html = `<div class="stats-row">`;
            html += statCard('R²', r2Stats.mean, r2Stats.ci95, 0.8, true, false, trueR2Mean);
            html += statCard('RMSE', rmseStats.mean, rmseStats.ci95, null, false, false, trueRmseMean);
            html += statCard('Relative RMSE', relRmseStats.mean, relRmseStats.ci95, 0.2, false, true, trueRelRmseMean);
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
       STEP 5: Render Accuracy Assessment
       ============================================ */
    function renderStep5() {
        if (state.results.length === 0) return;

        const isClassification = state.mode === 'categorical';

        if (isClassification) {
            renderClassificationAccuracy();
        } else {
            renderRegressionAccuracy();
        }
    }

    /* ============================================
       STEP 6: Map Generation & Area Estimation
       ============================================ */
    function renderStep6() {
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
        markStepDone(6);
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

        // Summary stats cards (in collapsible)
        const statsEl = document.getElementById('summary-stats');
        let html = `<div class="info-alert"><strong>Key:</strong> The main value is the <span style="color:var(--zsl-green-dark);font-weight:600">error-corrected estimate</span> with 95% CI. <span style="color:#EE6677;font-weight:600">Naive</span> = uncorrected pixel count. <strong>True</strong> = known true value. Look for where naive estimates diverge from the truth.</div>`;

        html += `<div class="stats-row">`;
        for (let c = 0; c < nc; c++) {
            const stats = PNASCharts.summaryStats(correctedAreas[c]);
            const rawStats = PNASCharts.summaryStats(rawAreas[c]);
            const truePct = (trueProps[c] * 100).toFixed(1);
            const estPct = (stats.mean / totalPixels * 100).toFixed(1);
            const ciPct = `[${(stats.ci95[0] / totalPixels * 100).toFixed(1)}, ${(stats.ci95[1] / totalPixels * 100).toFixed(1)}]`;
            const meanRaw = rawAreas[c].reduce((a,b) => a+b, 0) / rawAreas[c].length;
            const totalOOBMean = d.classNames.reduce((sum, _, k) => {
                return sum + rawAreas[k].reduce((a,b) => a+b, 0) / rawAreas[k].length;
            }, 0);
            const naivePctVal = (meanRaw / totalOOBMean * 100).toFixed(1);
            html += `
        <div class="stat-card">
          <div class="stat-card__label">${d.classNames[c]}</div>
          <div class="stat-card__value">${estPct}%</div>
          <div class="stat-card__ci">95% CI: ${ciPct}</div>
          <div class="stat-card__ci" style="color:#EE6677">Naive: ${naivePctVal}%</div>
          <div class="stat-card__ci">True: ${truePct}%</div>
        </div>
      `;
        }
        html += `</div>`;
        statsEl.innerHTML = html;

        // Per-class histograms (H25/H27)
        const chartsContainer = document.getElementById('summary-charts-container');
        if (chartsContainer) {
            // Destroy old per-class charts
            if (state.charts.areaDistPerClass) {
                state.charts.areaDistPerClass.forEach(c => PNASCharts.destroy(c));
            }
            state.charts.areaDistPerClass = [];

            // Build dynamic grid
            let gridHtml = `<div class="viz-grid viz-grid--3">`;
            for (let c = 0; c < nc; c++) {
                gridHtml += `
                <div class="figure-container">
                    <div class="figure-container__title">Corrected Area: ${d.classNames[c]}</div>
                    <div class="chart-wrap"><canvas id="chart-area-class-${c}"></canvas></div>
                </div>`;
            }
            gridHtml += `</div>`;
            chartsContainer.innerHTML = gridHtml;

            // Render histograms
            for (let c = 0; c < nc; c++) {
                const areasPct = correctedAreas[c].map(v => v / totalPixels * 100);
                const truePct = trueProps[c] * 100;
                const chart = PNASCharts.histogram(`chart-area-class-${c}`, areasPct, {
                    xLabel: 'Corrected Area (%)',
                    yLabel: 'Frequency',
                    color: PNASCharts.CLASS_PALETTE[c],
                    bins: 20,
                    trueLine: truePct,
                });
                state.charts.areaDistPerClass.push(chart);
            }
        }

        // Legacy: also destroy old single chart if any
        PNASCharts.destroy(state.charts.areaDist);
    }

    function renderContinuousSummary() {
        const results = state.results;

        // Predicted total biomass from full-raster predictions per replicate
        const predictedTotals = results
            .filter(r => r.totalPredictedFull != null)
            .map(r => r.totalPredictedFull);

        // True total biomass (ground truth)
        const d = state.data;
        let trueTotal = 0;
        for (let i = 0; i < d.continuousTruth.length; i++) trueTotal += d.continuousTruth[i];

        const statsEl = document.getElementById('summary-stats');
        let html = `<div class="info-alert"><strong>Full-raster biomass estimation with uncertainty:</strong> Each bootstrap replicate predicts biomass for all 1M pixels. The distribution shows the spread of total predicted biomass across replicates. The dashed red line marks the true total. Overlap indicates an unbiased model.</div>`;

        html += `<div class="stats-row">`;

        if (predictedTotals.length > 0) {
            const predStats = PNASCharts.summaryStats(predictedTotals);
            html += `
      <div class="stat-card">
        <div class="stat-card__label">Predicted Total (Full Raster)</div>
        <div class="stat-card__value">${(predStats.mean / 1e6).toFixed(2)}M</div>
        <div class="stat-card__ci">95% CI: [${(predStats.ci95[0] / 1e6).toFixed(2)}, ${(predStats.ci95[1] / 1e6).toFixed(2)}]M Mg</div>
      </div>
    `;
        }

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

        // Predicted total biomass distribution with true total as reference line
        PNASCharts.destroy(state.charts.areaDist);
        const chartsContainer = document.getElementById('summary-charts-container');
        if (chartsContainer && predictedTotals.length > 0) {
            chartsContainer.innerHTML = `
                <div class="viz-grid viz-grid--1">
                    <div class="figure-container" style="max-width:600px; margin:0 auto;">
                        <div class="figure-container__title">Predicted Total Biomass Distribution</div>
                        <div class="chart-wrap"><canvas id="chart-area-dist-dynamic"></canvas></div>
                    </div>
                </div>`;
            state.charts.areaDist = PNASCharts.histogram('chart-area-dist-dynamic', predictedTotals.map(v => v / 1e6), {
                xLabel: 'Predicted Total Biomass (M Mg)',
                yLabel: 'Frequency',
                color: '#228833',
                bins: 25,
                trueLine: trueTotal / 1e6,
                thresholdLabel: 'True total',
            });
        }
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
    function statCard(label, value, ci95, threshold, higherBetter, lowerBetter, trueValue = null) {
        let status = '';
        if (threshold !== null && threshold !== undefined) {
            if (higherBetter) {
                status = value >= threshold ? 'pass' : (ci95[1] >= threshold ? 'warn' : 'fail');
            }
            if (lowerBetter) {
                status = value <= threshold ? 'pass' : (ci95[0] <= threshold ? 'warn' : 'fail');
            }
        }

        const fmtVal = value <= 1.0 ? value.toFixed(3) : value.toFixed(1);
        const fmtCI = ci95 ? `[${ci95[0] <= 1.0 ? ci95[0].toFixed(3) : ci95[0].toFixed(1)}, ${ci95[1] <= 1.0 ? ci95[1].toFixed(3) : ci95[1].toFixed(1)}]` : '';

        let trueHTML = '';
        if (trueValue !== null) {
            const tStr = trueValue <= 1.0 ? trueValue.toFixed(3) : trueValue.toFixed(1);
            const diff = (value - trueValue);
            // Highlight bias if it's more than 5pp (0.05 for percentages, 5.0 for absolute metrics like RMSE)
            const biasThreshold = value <= 1.0 ? 0.05 : 5.0; 
            const color = Math.abs(diff) > biasThreshold ? '#D32F2F' : '#666'; 
            const diffStr = (diff >= 0 ? '+' : '') + (value <= 1.0 ? diff.toFixed(3) : diff.toFixed(1));
            trueHTML = `<div style="font-size: 0.85rem; margin-top: 8px; padding-top: 6px; border-top: 1px solid #ddd; color: #555; line-height: 1.3;">
                True Landscape Accuracy:<br><b>${tStr}</b> 
                <span style="color:${color}; font-size:0.8rem;">(Bias: ${diffStr})</span>
            </div>`;
        }

        return `
      <div class="stat-card${status ? ' stat-card--' + status : ''}">
        <div class="stat-card__label">${label}</div>
        <div class="stat-card__value">${fmtVal}</div>
        ${fmtCI ? `<div class="stat-card__ci">95% CI: ${fmtCI}</div>` : ''}
        ${threshold != null ? `<div class="stat-card__ci">Threshold: ${threshold}</div>` : ''}
        ${trueHTML}
      </div>
    `;
    }

    /* ============================================
       VERDICT: Table 1 checklist assessment
       ============================================ */
    function renderVerdict() {
        const results = state.results;
        const d = state.data;
        if (!results || results.length === 0) return;

        const panel = document.getElementById('verdict-panel');
        if (!panel) return;
        panel.classList.remove('hidden');
        // Open it
        panel.classList.add('open');

        const isCategorical = state.mode === 'categorical';
        const checks = [];

        // 1. Thematic validity — synthetic data has known truth
        checks.push({
            checkpoint: 'Thematic accuracy of groundtruth',
            result: 'Synthetic data with perfect labels',
            status: 'pass'
        });

        // 2. Spatial coverage
        const nPoints = d.trainingIndices.length;
        checks.push({
            checkpoint: 'Spatial coverage of groundtruth',
            result: `${nPoints.toLocaleString()} points across 1M pixels`,
            status: 'pass'
        });

        // 3. Independence
        checks.push({
            checkpoint: 'Independence of training & validation',
            result: 'Spatial blocking ensures no data leakage',
            status: 'pass'
        });

        // 4. Spatial validity
        const blockSize = parseInt(document.getElementById('cfg-block-size').value);
        checks.push({
            checkpoint: 'Spatial validity of sampling design',
            result: `${blockSize}×${blockSize}px blocks, bootstrap resampling`,
            status: 'pass'
        });

        if (isCategorical) {
            // 5. Map accuracy
            const oas = results.map(r => r.metrics.overallAccuracy);
            const oaStats = PNASCharts.summaryStats(oas);
            const oaPass = oaStats.mean >= 0.85;
            checks.push({
                checkpoint: 'Map accuracy (overall)',
                result: `OA = ${(oaStats.mean * 100).toFixed(1)}% [${(oaStats.ci95[0] * 100).toFixed(1)}, ${(oaStats.ci95[1] * 100).toFixed(1)}]`,
                status: oaPass ? 'pass' : (oaStats.ci95[1] >= 0.85 ? 'warn' : 'fail')
            });

            // Per-class check
            const perClassUser = d.classNames.map((_, c) => results.map(r => r.metrics.userAccuracy[c]));
            const classFails = [];
            for (let c = 0; c < d.numClasses; c++) {
                const us = PNASCharts.summaryStats(perClassUser[c]);
                if (us.mean < 0.85) classFails.push(d.classNames[c]);
            }
            checks.push({
                checkpoint: 'Per-class accuracy (UA ≥ 85%)',
                result: classFails.length === 0 ? 'All classes pass' : `Failing: ${classFails.join(', ')}`,
                status: classFails.length === 0 ? 'pass' : 'fail'
            });
        } else {
            // Continuous
            const r2s = results.map(r => r.metrics.r2);
            const r2Stats = PNASCharts.summaryStats(r2s);
            checks.push({
                checkpoint: 'Map accuracy (R²)',
                result: `R² = ${r2Stats.mean.toFixed(3)} [${r2Stats.ci95[0].toFixed(3)}, ${r2Stats.ci95[1].toFixed(3)}]`,
                status: r2Stats.mean >= 0.8 ? 'pass' : (r2Stats.ci95[1] >= 0.8 ? 'warn' : 'fail')
            });

            const relRmses = results.map(r => r.metrics.relRmse);
            const rrStats = PNASCharts.summaryStats(relRmses);
            checks.push({
                checkpoint: 'Relative RMSE (≤ 20%)',
                result: `${(rrStats.mean).toFixed(1)}% [${rrStats.ci95[0].toFixed(1)}, ${rrStats.ci95[1].toFixed(1)}]`,
                status: rrStats.mean <= 20 ? 'pass' : 'fail'
            });
        }

        // 6. Precision
        checks.push({
            checkpoint: 'Precision of accuracy (CI width)',
            result: `${results.length} repeated assessments with 95% CIs`,
            status: 'pass'
        });

        // 7. Summary statistics
        checks.push({
            checkpoint: 'Error-corrected summary statistics',
            result: isCategorical ? 'Olofsson-corrected areas with CIs' : 'Predicted totals with bootstrap CIs',
            status: 'pass'
        });

        // Build table
        const checklistEl = document.getElementById('verdict-checklist');
        let html = `<table class="verdict-table">
            <thead><tr><th>#</th><th>Table 1 Checkpoint</th><th>Result</th><th>Status</th></tr></thead>
            <tbody>`;
        checks.forEach((c, i) => {
            const icon = c.status === 'pass' ? '🟢' : c.status === 'warn' ? '🟡' : '🔴';
            const cls = `verdict-${c.status}`;
            html += `<tr>
                <td>${i + 1}</td>
                <td>${c.checkpoint}</td>
                <td>${c.result}</td>
                <td class="${cls}">${icon} ${c.status.toUpperCase()}</td>
            </tr>`;
        });
        html += `</tbody></table>`;
        checklistEl.innerHTML = html;

        // Overall verdict
        const passes = checks.filter(c => c.status === 'pass').length;
        const fails = checks.filter(c => c.status === 'fail').length;
        const total = checks.length;

        const summaryEl = document.getElementById('verdict-summary');
        let verdictClass, verdictText;
        if (fails === 0) {
            verdictClass = 'verdict-banner--pass';
            verdictText = `✅ This map <strong>meets the mapping standard</strong>. All ${total} checkpoints pass.`;
        } else if (fails <= 2) {
            verdictClass = 'verdict-banner--partial';
            verdictText = `⚠️ This map <strong>partially meets</strong> the standard. ${passes}/${total} checkpoints pass, ${fails} fail. Review failing checkpoints before using this map for decisions.`;
        } else {
            verdictClass = 'verdict-banner--fail';
            verdictText = `❌ This map <strong>does not meet</strong> the mapping standard. ${fails}/${total} checkpoints fail.`;
        }
        summaryEl.innerHTML = `<div class="verdict-banner ${verdictClass}">${verdictText}</div>`;

        // Update badge
        const badge = panel.querySelector('.step-panel__badge');
        if (badge) {
            badge.textContent = 'Complete';
            badge.className = 'step-panel__badge step-panel__badge--done';
        }
    }

    /* ============================================
       PITFALL COMPARISONS
       ============================================ */
    function addPitfallButton() {
        // Only show if Step 4 has accuracy-stats rendered
        const statsEl = document.getElementById('accuracy-stats');
        if (!statsEl) return;

        // Remove any existing buttons
        const existing = document.getElementById('pitfall-buttons');
        if (existing) existing.remove();

        const wrap = document.createElement('div');
        wrap.id = 'pitfall-buttons';
        wrap.style.cssText = 'display:flex; gap:12px; flex-wrap:wrap; margin-top:16px;';

        const btn1 = document.createElement('button');
        btn1.id = 'btn-pitfall-compare';
        btn1.className = 'btn btn-compare';
        btn1.textContent = '⚠️ Compare: What if we skipped spatial blocking?';
        btn1.addEventListener('click', runPitfallComparison);

        const btn2 = document.createElement('button');
        btn2.id = 'btn-single-split';
        btn2.className = 'btn btn-compare';
        btn2.textContent = '⚠️ Compare: What if we used only a single split?';
        btn2.addEventListener('click', runSingleSplitComparison);

        wrap.appendChild(btn1);
        wrap.appendChild(btn2);
        statsEl.after(wrap);
    }

    function runPitfallComparison() {
        const btn = document.getElementById('btn-pitfall-compare');
        if (btn) {
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner"></span> Running random-split comparison…';
        }

        const d = state.data;
        const isClassification = state.mode === 'categorical';
        const totalPixels = d.width * d.height;
        const nPoints = d.trainingIndices.length;
        const numBands = d.numBands;

        // Prepare all labels
        const allLabels = isClassification
            ? new Uint8Array(nPoints)
            : new Float32Array(nPoints);
        for (let i = 0; i < nPoints; i++) {
            allLabels[i] = isClassification
                ? d.categoricalTruth[d.trainingIndices[i]]
                : d.continuousTruth[d.trainingIndices[i]];
        }

        // Run 20 replicates with random pixel-level splitting
        const nPitfallReplicates = 20;
        const pitfallResults = [];
        let pitfallCompleted = 0;
        const pitfallWorkers = [];
        const numWorkers = Math.min(state.config.numWorkers, 4);

        for (let w = 0; w < numWorkers; w++) {
            const worker = new Worker('js/workers/rfWorker.js');
            worker.onmessage = function(e) {
                pitfallResults.push(e.data);
                pitfallCompleted++;

                const workerIdx = pitfallWorkers.indexOf(worker);
                if (pitfallNextJob < nPitfallReplicates && workerIdx >= 0) {
                    dispatchPitfallJob(workerIdx);
                }

                if (pitfallCompleted >= nPitfallReplicates) {
                    pitfallWorkers.forEach(w => w.terminate());
                    renderPitfallResults(pitfallResults);
                }
            };
            pitfallWorkers.push(worker);
        }

        let pitfallNextJob = 0;

        function dispatchPitfallJob(workerIdx) {
            if (pitfallNextJob >= nPitfallReplicates) return;
            const bIdx = pitfallNextJob++;

            // Random split: randomly assign 63% to training, 37% to validation
            // This is the "common practice" pitfall — no spatial blocking
            const rng = mulberry32(state.config.seed + bIdx * 997);
            const trainMask = new Uint8Array(nPoints);
            let nTrain = 0;
            for (let i = 0; i < nPoints; i++) {
                if (rng() < 0.632) {
                    trainMask[i] = 1;
                    nTrain++;
                }
            }

            const nOOB = nPoints - nTrain;
            const trainFeatures = new Float32Array(nTrain * numBands);
            const trainLabels = isClassification ? new Uint8Array(nTrain) : new Float32Array(nTrain);
            const trainWeights = new Float32Array(nTrain).fill(1);
            const oobFeatures = new Float32Array(nOOB * numBands);
            const oobLabels = isClassification ? new Uint8Array(nOOB) : new Float32Array(nOOB);

            let ti = 0, oi = 0;
            for (let i = 0; i < nPoints; i++) {
                if (trainMask[i]) {
                    for (let b = 0; b < numBands; b++) {
                        trainFeatures[ti * numBands + b] = d.trainingFeatures[i * numBands + b];
                    }
                    trainLabels[ti] = allLabels[i];
                    ti++;
                } else {
                    for (let b = 0; b < numBands; b++) {
                        oobFeatures[oi * numBands + b] = d.trainingFeatures[i * numBands + b];
                    }
                    oobLabels[oi] = allLabels[i];
                    oi++;
                }
            }

            pitfallWorkers[workerIdx].postMessage({
                type: 'bootstrap',
                bootstrapIndex: bIdx,
                trainingFeatures: trainFeatures,
                trainingLabels: trainLabels,
                trainingWeights: trainWeights,
                oobFeatures,
                oobLabels,
                fullRasterFeatures: null,
                numBands,
                numClasses: d.numClasses,
                isClassification,
                config: {
                    nTrees: state.config.nTrees,
                    maxDepth: state.config.maxDepth,
                    minLeafSamples: state.config.minLeafSamples,
                },
                seed: state.config.seed + bIdx * 997 + 50000,
                computeFullMap: false,
                trueTestFeatures: state.data.trueTestFeatures,
                trueTestLabels: isClassification ? state.data.trueTestLabelsCat : state.data.trueTestLabelsCont,
            });
        }

        // Dispatch initial jobs
        for (let w = 0; w < numWorkers; w++) {
            dispatchPitfallJob(w);
        }
    }

    // Simple seeded RNG for pitfall comparison
    function mulberry32(a) {
        return function() {
            a |= 0; a = a + 0x6D2B79F5 | 0;
            let t = Math.imul(a ^ a >>> 15, 1 | a);
            t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
            return ((t ^ t >>> 14) >>> 0) / 4294967296;
        };
    }

    function renderPitfallResults(pitfallResults) {
        const btn = document.getElementById('btn-pitfall-compare');
        if (btn) btn.remove();

        const panel = document.getElementById('pitfall-comparison');
        if (!panel) return;
        panel.classList.remove('hidden');

        const isClassification = state.mode === 'categorical';
        const d = state.data;
        const spatialResults = state.results;

        // Destroy any existing pitfall charts
        if (state.charts.pitfall1) PNASCharts.destroy(state.charts.pitfall1);
        if (state.charts.pitfall2) PNASCharts.destroy(state.charts.pitfall2);
        if (state.charts.pitfall3) PNASCharts.destroy(state.charts.pitfall3);

        if (isClassification) {
            const spatialOA = spatialResults.map(r => r.metrics.overallAccuracy);
            const pitfallOA = pitfallResults.map(r => r.metrics.overallAccuracy);
            
            const spatialOAStats = PNASCharts.summaryStats(spatialOA);
            const pitfallOAStats = PNASCharts.summaryStats(pitfallOA);
            const pitfallTrueAccuracies = pitfallResults.filter(r => r.trueMetrics).map(r => r.trueMetrics.overallAccuracy);
            const trueOAMean = pitfallTrueAccuracies.length ? (pitfallTrueAccuracies.reduce((a,b)=>a+b,0)/pitfallTrueAccuracies.length) : null;

            // Chart 1: Side-by-side overall accuracy
            document.getElementById('pitfall-chart1-title').textContent = 'Overall Accuracy: Random Split (Inflated!)';
            state.charts.pitfall1 = PNASCharts.histogram('chart-pitfall-1', pitfallOA, {
                xLabel: 'Overall Accuracy',
                yLabel: 'Frequency',
                color: '#EE6677',
                bins: 15,
                thresholdLine: 0.85,
                thresholdLabel: 'Good (85%)',
                trueLine: trueOAMean,
            });

            // Chart 2 & 3: User/Producer accuracy comparison
            const spatialUserStats = d.classNames.map((_, c) => PNASCharts.summaryStats(spatialResults.map(r => r.metrics.userAccuracy[c])));
            const pitfallUserStats = d.classNames.map((_, c) => PNASCharts.summaryStats(pitfallResults.map(r => r.metrics.userAccuracy[c])));

            document.getElementById('pitfall-chart2-title').textContent = "User's Accuracy: Random Split";
            state.charts.pitfall2 = PNASCharts.boxSummary('chart-pitfall-2', pitfallUserStats, {
                xLabels: d.classNames,
                yLabel: 'Accuracy',
            });

            const spatialProdStats = d.classNames.map((_, c) => PNASCharts.summaryStats(spatialResults.map(r => r.metrics.producerAccuracy[c])));
            const pitfallProdStats = d.classNames.map((_, c) => PNASCharts.summaryStats(pitfallResults.map(r => r.metrics.producerAccuracy[c])));

            document.getElementById('pitfall-chart3-title').textContent = "Producer's Accuracy: Random Split";
            state.charts.pitfall3 = PNASCharts.boxSummary('chart-pitfall-3', pitfallProdStats, {
                xLabels: d.classNames,
                yLabel: 'Accuracy',
            });

            // Summary comparison stats
            const inflationVal = (pitfallOAStats.mean - spatialOAStats.mean) * 100;
            const inflation = inflationVal.toFixed(1);
            let trueText = '';
            if (trueOAMean !== null) {
                const trueOAPct = (trueOAMean * 100).toFixed(1);
                if (trueOAMean < pitfallOAStats.mean) {
                    trueText = `<br><br><em>Reality Check:</em> The <b>true</b> accuracy of the map across the entire landscape was actually only <b>${trueOAPct}%</b>. The spatial-blocking method safely covered this reality, but the random-split method overestimated it.`;
                } else {
                    trueText = `<br><br><em>Reality Check:</em> The <b>true</b> accuracy of the map across the entire landscape was <b>${trueOAPct}%</b>. In this case, the random-split estimate was not inflated, but the spatial-blocking method remains the methodologically correct approach.`;
                }
            }

            const statsEl = document.getElementById('pitfall-stats');
            if (statsEl) {
                let resultText;
                if (inflationVal > 0.5) {
                    resultText = `<strong>Result:</strong> Random pixel-level splitting inflated the estimated accuracy by
                        <strong>+${inflation} percentage points</strong>
                        (${(pitfallOAStats.mean * 100).toFixed(1)}% vs ${(spatialOAStats.mean * 100).toFixed(1)}% with spatial blocking).
                        This is because nearby pixels that are very similar to each other end up in both training
                        and test sets, making the model appear more accurate than it truly is.`;
                } else {
                    resultText = `<strong>Result:</strong> In this case, random splitting did not substantially inflate accuracy
                        (${(pitfallOAStats.mean * 100).toFixed(1)}% vs ${(spatialOAStats.mean * 100).toFixed(1)}% with spatial blocking).
                        However, this does not mean spatial blocking is unnecessary — with different data or sampling strategies,
                        random splitting routinely overestimates accuracy.`;
                }
                statsEl.innerHTML = `
                    <div class="info-alert info-alert--warning">
                        ${resultText}
                        ${trueText}
                    </div>
                `;
            }
        } else {
            // Continuous mode
            const spatialR2 = spatialResults.map(r => r.metrics.r2);
            const pitfallR2 = pitfallResults.map(r => r.metrics.r2);
            const pitfallTrueAccuracies = pitfallResults.filter(r => r.trueMetrics).map(r => r.trueMetrics.r2);
            const trueR2Mean = pitfallTrueAccuracies.length ? (pitfallTrueAccuracies.reduce((a,b)=>a+b,0)/pitfallTrueAccuracies.length) : null;

            document.getElementById('pitfall-chart1-title').textContent = 'R² Distribution: Random Split (Inflated!)';
            state.charts.pitfall1 = PNASCharts.histogram('chart-pitfall-1', pitfallR2, {
                xLabel: 'R²',
                yLabel: 'Frequency',
                color: '#EE6677',
                bins: 15,
                thresholdLine: 0.8,
                thresholdLabel: 'Good (0.8)',
                trueLine: trueR2Mean,
            });

            const spatialRMSE = spatialResults.map(r => r.metrics.rmse);
            const pitfallRMSE = pitfallResults.map(r => r.metrics.rmse);

            document.getElementById('pitfall-chart2-title').textContent = 'RMSE: Random Split';
            state.charts.pitfall2 = PNASCharts.histogram('chart-pitfall-2', pitfallRMSE, {
                xLabel: 'RMSE (Mg/ha)',
                yLabel: 'Frequency',
                color: '#EE6677',
                bins: 15,
            });

            const pitfallRelRMSE = pitfallResults.map(r => r.metrics.relRmse);

            document.getElementById('pitfall-chart3-title').textContent = 'Relative RMSE: Random Split';
            state.charts.pitfall3 = PNASCharts.histogram('chart-pitfall-3', pitfallRelRMSE, {
                xLabel: 'Relative RMSE',
                yLabel: 'Frequency',
                color: '#EE6677',
                bins: 15,
                thresholdLine: 0.2,
                thresholdLabel: 'Max (20%)',
            });

            const spatialR2Stats = PNASCharts.summaryStats(spatialR2);
            const pitfallR2Stats = PNASCharts.summaryStats(pitfallR2);
            const r2InflationVal = pitfallR2Stats.mean - spatialR2Stats.mean;
            const r2Inflation = (r2InflationVal * 100).toFixed(1);
            let trueText = '';
            if (trueR2Mean !== null) {
                if (trueR2Mean < pitfallR2Stats.mean) {
                    trueText = `<br><br><em>Reality Check:</em> The <b>true</b> R² of the map across the entire landscape was actually only <b>${trueR2Mean.toFixed(3)}</b>. The spatial-blocking method safely covered this reality, but the random-split method overestimated it.`;
                } else {
                    trueText = `<br><br><em>Reality Check:</em> The <b>true</b> R² of the map was <b>${trueR2Mean.toFixed(3)}</b>. In this case, the random-split estimate was not inflated, but the spatial-blocking method remains the methodologically correct approach.`;
                }
            }

            const statsEl = document.getElementById('pitfall-stats');
            if (statsEl) {
                let resultText;
                if (r2InflationVal > 0.005) {
                    resultText = `<strong>Result:</strong> Random pixel-level splitting inflated R² by
                        <strong>+${r2Inflation} percentage points</strong>
                        (${pitfallR2Stats.mean.toFixed(3)} vs ${spatialR2Stats.mean.toFixed(3)} with spatial blocking).
                        Without spatial blocking, the model memorises local spatial patterns rather than
                        learning generalisable spectral-biomass relationships.
                        <strong>A map validated this way would not meet the mapping standard.</strong>`;
                } else {
                    resultText = `<strong>Result:</strong> In this case, random splitting did not substantially inflate R²
                        (${pitfallR2Stats.mean.toFixed(3)} vs ${spatialR2Stats.mean.toFixed(3)} with spatial blocking).
                        However, this does not mean spatial blocking is unnecessary — with different data,
                        random splitting routinely overestimates accuracy.`;
                }
                statsEl.innerHTML = `
                    <div class="info-alert info-alert--warning">
                        ${resultText}
                        ${trueText}
                    </div>
                `;
            }
        }

        // Scroll to the pitfall panel
        panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    /* ============================================
       SINGLE-SPLIT COMPARISON
       ============================================ */
    function runSingleSplitComparison() {
        const btn = document.getElementById('btn-single-split');
        if (btn) {
            btn.disabled = true;
            btn.textContent = '✓ Single-split comparison shown below';
        }

        const panel = document.getElementById('single-split-comparison');
        if (!panel) return;
        panel.classList.remove('hidden');

        const results = state.results;
        const d = state.data;
        const isClassification = state.mode === 'categorical';

        // Pick a random replicate to illustrate "what if you only did one?"
        const rng = mulberry32(state.config.seed + 77777);
        const singleIdx = Math.floor(rng() * results.length);
        const single = results[singleIdx];

        const contentEl = document.getElementById('single-split-content');
        let html = '';

        if (isClassification) {
            const oas = results.map(r => r.metrics.overallAccuracy);
            const oaStats = PNASCharts.summaryStats(oas);
            const singleOA = single.metrics.overallAccuracy;

            // Show the single value vs the distribution
            html += `<div class="stats-row">`;
            html += `
              <div class="stat-card stat-card--warn">
                <div class="stat-card__label">Single Split OA</div>
                <div class="stat-card__value">${(singleOA * 100).toFixed(1)}%</div>
                <div class="stat-card__ci">No confidence interval</div>
                <div class="stat-card__ci" style="color:var(--text-muted)">Just one number</div>
              </div>
            `;
            html += `
              <div class="stat-card stat-card--pass">
                <div class="stat-card__label">Repeated Assessment OA</div>
                <div class="stat-card__value">${(oaStats.mean * 100).toFixed(1)}%</div>
                <div class="stat-card__ci">95% CI: [${(oaStats.ci95[0] * 100).toFixed(1)}, ${(oaStats.ci95[1] * 100).toFixed(1)}]</div>
                <div class="stat-card__ci" style="color:var(--text-muted)">From ${results.length} replicates</div>
              </div>
            `;

            // Show per-class comparison
            for (let c = 0; c < d.numClasses; c++) {
                const allUA = results.map(r => r.metrics.userAccuracy[c]);
                const uaStats = PNASCharts.summaryStats(allUA);
                const singleUA = single.metrics.userAccuracy[c];
                html += `
                  <div class="stat-card">
                    <div class="stat-card__label">${d.classNames[c]} (UA)</div>
                    <div class="stat-card__value">${(singleUA * 100).toFixed(1)}%</div>
                    <div class="stat-card__ci">Range: [${(uaStats.ci95[0] * 100).toFixed(1)}, ${(uaStats.ci95[1] * 100).toFixed(1)}]</div>
                  </div>
                `;
            }
            html += `</div>`;

            // Explanation
            const spread = ((oaStats.ci95[1] - oaStats.ci95[0]) * 100).toFixed(1);
            const deviation = ((singleOA - oaStats.mean) * 100).toFixed(1);
            const absDeviation = Math.abs(parseFloat(deviation)).toFixed(1);

            html += `
              <div class="info-alert info-alert--warning mt-4">
                <strong>Result:</strong> This single split gave an overall accuracy of
                <strong>${(singleOA * 100).toFixed(1)}%</strong>, which is
                ${parseFloat(deviation) >= 0 ? '+' : ''}${deviation} pp from the mean of ${(oaStats.mean * 100).toFixed(1)}%.
                The full distribution spans <strong>${spread} percentage points</strong> (95% CI).
                With only one split, you'd have no way to know where in this range your estimate falls —
                it could easily ${oaStats.ci95[0] < 0.85 ? 'fall below the 85% pass threshold or ' : ''}
                mislead a real-world decision.
                <strong>Repeated assessments are essential for knowing the precision of your accuracy estimate.</strong>
              </div>
            `;
        } else {
            // Continuous mode
            const r2s = results.map(r => r.metrics.r2);
            const r2Stats = PNASCharts.summaryStats(r2s);
            const singleR2 = single.metrics.r2;
            const rmses = results.map(r => r.metrics.rmse);
            const rmseStats = PNASCharts.summaryStats(rmses);
            const singleRMSE = single.metrics.rmse;

            html += `<div class="stats-row">`;
            html += `
              <div class="stat-card stat-card--warn">
                <div class="stat-card__label">Single Split R²</div>
                <div class="stat-card__value">${singleR2.toFixed(3)}</div>
                <div class="stat-card__ci">No confidence interval</div>
              </div>
            `;
            html += `
              <div class="stat-card stat-card--pass">
                <div class="stat-card__label">Repeated Assessment R²</div>
                <div class="stat-card__value">${r2Stats.mean.toFixed(3)}</div>
                <div class="stat-card__ci">95% CI: [${r2Stats.ci95[0].toFixed(3)}, ${r2Stats.ci95[1].toFixed(3)}]</div>
              </div>
            `;
            html += `
              <div class="stat-card stat-card--warn">
                <div class="stat-card__label">Single Split RMSE</div>
                <div class="stat-card__value">${singleRMSE.toFixed(1)}</div>
                <div class="stat-card__ci">No confidence interval</div>
              </div>
            `;
            html += `
              <div class="stat-card stat-card--pass">
                <div class="stat-card__label">Repeated Assessment RMSE</div>
                <div class="stat-card__value">${rmseStats.mean.toFixed(1)}</div>
                <div class="stat-card__ci">95% CI: [${rmseStats.ci95[0].toFixed(1)}, ${rmseStats.ci95[1].toFixed(1)}]</div>
              </div>
            `;
            html += `</div>`;

            const r2Spread = ((r2Stats.ci95[1] - r2Stats.ci95[0]) * 100).toFixed(1);
            html += `
              <div class="info-alert info-alert--warning mt-4">
                <strong>Result:</strong> This single split gave R² = <strong>${singleR2.toFixed(3)}</strong>,
                but the full distribution spans <strong>${r2Spread} percentage points</strong> (95% CI).
                A single estimate gives you no information about this variability.
                <strong>Repeated assessments are essential for knowing the precision of your accuracy estimate.</strong>
              </div>
            `;
        }

        contentEl.innerHTML = html;
        panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    /* ============================================
       REPORT CARD
       ============================================ */
    function renderReportCard() {
        const panel = document.getElementById('report-card-panel');
        if (!panel) return;
        panel.classList.remove('hidden');

        const metricsEl = document.getElementById('report-card-metrics');
        const conclusionEl = document.getElementById('report-card-conclusion');
        const gradeEl = document.getElementById('report-card-grade');
        
        const isClassification = state.mode === 'categorical';
        const results = state.results;
        const trueAccuracies = results.filter(r => r.trueMetrics).map(r => r.trueMetrics);
        
        if (trueAccuracies.length === 0) return;

        let html = '';
        let conclusion = '';
        let grade = 'PASS';
        
        const strategy = state.config.samplingStrategy || 'clustered';

        if (isClassification) {
            const oas = results.map(r => r.metrics.overallAccuracy);
            const estOA = PNASCharts.summaryStats(oas).mean * 100;
            const trueOA = (trueAccuracies.reduce((a,b)=>a+b.overallAccuracy,0)/trueAccuracies.length) * 100;
            const diff = trueOA - estOA;
            
            html += `
              <div class="stat-card">
                <div class="stat-card__label">Expected Map Accuracy</div>
                <div class="stat-card__value">${estOA.toFixed(1)}%</div>
                <div class="stat-card__ci">Estimated via Spatial Blocking</div>
              </div>
              <div class="stat-card ${diff < -5 ? 'stat-card--warn' : 'stat-card--pass'}">
                <div class="stat-card__label">True Landscape Accuracy</div>
                <div class="stat-card__value">${trueOA.toFixed(1)}%</div>
                <div class="stat-card__ci">Independent test (10,000 pixels)</div>
              </div>
            `;
            
            let lowAccNote = '';
            if (estOA < 70) {
                lowAccNote = ' The overall accuracy is quite low — this may reflect high sensor noise, insufficient reference data, or a genuinely difficult mapping problem.';
            }

            if (strategy === 'random') {
                conclusion = `<strong>Great!</strong> Because you used <strong>True Random</strong> sampling, your reference data was completely unclustered. As a result, your expected accuracy (${estOA.toFixed(1)}%) is almost perfectly aligned with the actual map accuracy (${trueOA.toFixed(1)}%). Any form of cross-validation works well with perfectly random data.${lowAccNote}`;
                grade = 'A+';
                gradeEl.style.backgroundColor = '#2D6A4F';
            } else {
                if (diff > 2.0) {
                    // Conservative: estimated < true (underestimate)
                    conclusion = `<strong>Good.</strong> Your spatial blocking gave a <strong>conservative</strong> accuracy estimate — the map actually performed better (${trueOA.toFixed(1)}%) than the cross-validation suggested (${estOA.toFixed(1)}%). This is the safe direction: the map is better than advertised.${lowAccNote}`;
                    grade = 'PASS';
                    gradeEl.style.backgroundColor = '#2D6A4F';
                } else if (Math.abs(diff) <= 2.0) {
                    // Close match
                    conclusion = `<strong>Success!</strong> Even though your field data was collected in realistic <strong>clusters</strong>, using <strong>Spatial Blocking</strong> successfully prevented data leakage. Your estimated accuracy (${estOA.toFixed(1)}%) closely matches the map's true performance (${trueOA.toFixed(1)}%).${lowAccNote}`;
                    grade = 'PASS';
                    gradeEl.style.backgroundColor = '#2D6A4F';
                } else if (diff >= -5.0) {
                    // Mild overestimate
                    conclusion = `<strong>Caution.</strong> Your estimated accuracy (${estOA.toFixed(1)}%) is somewhat higher than the true accuracy (${trueOA.toFixed(1)}%). Consider increasing block size or collecting more reference data.${lowAccNote}`;
                    grade = 'WARN';
                    gradeEl.style.backgroundColor = '#f6c23e';
                } else {
                    // Severe overestimate
                    conclusion = `<strong>Warning!</strong> Your estimated accuracy (${estOA.toFixed(1)}%) is significantly higher than the true accuracy (${trueOA.toFixed(1)}%). Because your data was highly <strong>clustered</strong>, the block size you chose (${state.config.blockSize}) might have been too small, allowing some spatial autocorrelation to leak between blocks and artificially inflate your estimate.${lowAccNote}`;
                    grade = 'FAIL';
                    gradeEl.style.backgroundColor = '#EE6677';
                }
            }
        } else {
            // Regression report card logic
            const r2s = results.map(r => r.metrics.r2);
            const estR2 = PNASCharts.summaryStats(r2s).mean;
            const trueR2 = trueAccuracies.reduce((a,b)=>a+b.r2,0)/trueAccuracies.length;
            const diff = trueR2 - estR2;

            html += `
              <div class="stat-card">
                <div class="stat-card__label">Expected Map R²</div>
                <div class="stat-card__value">${estR2.toFixed(3)}</div>
                <div class="stat-card__ci">Estimated via Spatial Blocking</div>
              </div>
              <div class="stat-card ${diff < -0.1 ? 'stat-card--warn' : 'stat-card--pass'}">
                <div class="stat-card__label">True Landscape R²</div>
                <div class="stat-card__value">${trueR2.toFixed(3)}</div>
                <div class="stat-card__ci">Independent test (10,000 pixels)</div>
              </div>
            `;
            
            let lowAccNote = '';
            if (estR2 < 0.5) {
                lowAccNote = ' The R² is quite low — this may reflect high sensor noise, insufficient reference data, or a genuinely difficult mapping problem.';
            }

            if (strategy === 'random') {
                conclusion = `<strong>Great!</strong> Because you used <strong>True Random</strong> sampling, your expected R² (${estR2.toFixed(3)}) closely matches reality (${trueR2.toFixed(3)}).${lowAccNote}`;
                grade = 'A+';
                gradeEl.style.backgroundColor = '#2D6A4F';
            } else {
                if (diff > 0.05) {
                    // Conservative: estimated < true (underestimate)
                    conclusion = `<strong>Good.</strong> Your spatial blocking gave a <strong>conservative</strong> R² estimate — the map actually performs better (${trueR2.toFixed(3)}) than the cross-validation suggested (${estR2.toFixed(3)}). This is the safe direction: the map is better than advertised.${lowAccNote}`;
                    grade = 'PASS';
                    gradeEl.style.backgroundColor = '#2D6A4F';
                } else if (Math.abs(diff) <= 0.05) {
                    // Close match
                    conclusion = `<strong>Success!</strong> Spatial Blocking successfully mitigated the spatial clustering in your data. Your estimated R² (${estR2.toFixed(3)}) closely matches reality (${trueR2.toFixed(3)}).${lowAccNote}`;
                    grade = 'PASS';
                    gradeEl.style.backgroundColor = '#2D6A4F';
                } else if (diff >= -0.1) {
                    // Mild overestimate
                    conclusion = `<strong>Caution.</strong> Your R² estimate (${estR2.toFixed(3)}) is somewhat higher than the true R² (${trueR2.toFixed(3)}). Consider increasing block size or collecting more reference data.${lowAccNote}`;
                    grade = 'WARN';
                    gradeEl.style.backgroundColor = '#f6c23e';
                } else {
                    // Severe overestimate
                    conclusion = `<strong>Warning!</strong> Your R² estimate (${estR2.toFixed(3)}) significantly overestimates the map's true performance (${trueR2.toFixed(3)}). Your block size might need increasing to prevent spatial leakage.${lowAccNote}`;
                    grade = 'FAIL';
                    gradeEl.style.backgroundColor = '#EE6677';
                }
            }
        }
        
        metricsEl.innerHTML = html;
        conclusionEl.innerHTML = conclusion;
        gradeEl.textContent = grade;
        
        markStepDone(7);
    }

    return { init };
})();

// Boot
document.addEventListener('DOMContentLoaded', App.init);
