/* ============================================================
   Unit Tests for Conservation Mapping Standards
   Tests all core analytical functions:
   - SyntheticData: raster generation, ground truth, sampling
   - SpatialBlocking: block creation, point assignment, bootstrap
   - Random Forest: CART trees, training, prediction, metrics
   - Olofsson area correction
   ============================================================ */

'use strict';

// ── Test infrastructure ──────────────────────────────────────
let passed = 0;
let failed = 0;
const failures = [];

function assert(condition, message) {
    if (condition) {
        passed++;
    } else {
        failed++;
        failures.push(message);
        console.error(`  ✗ FAIL: ${message}`);
    }
}

function assertApprox(actual, expected, tolerance, message) {
    const diff = Math.abs(actual - expected);
    if (diff <= tolerance) {
        passed++;
    } else {
        failed++;
        const msg = `${message} (expected ≈${expected}, got ${actual}, diff=${diff.toFixed(6)})`;
        failures.push(msg);
        console.error(`  ✗ FAIL: ${msg}`);
    }
}

function section(name) {
    console.log(`\n━━━ ${name} ━━━`);
}

// ── Load modules (Node.js) ───────────────────────────────────
const SyntheticData = require('./js/syntheticData.js');
const SpatialBlocking = require('./js/spatialBlocking.js');

// PNASCharts: extract summaryStats for testing
// charts.js expects a browser environment; we mock what we need
global.Chart = { defaults: { font: {}, plugins: { legend: {}, title: {} } }, register: () => { } };
const PNASCharts = require('./js/charts.js');

// rfWorker.js is a Web Worker script — we eval it to extract functions
const fs = require('fs');
const workerSource = fs.readFileSync('./js/workers/rfWorker.js', 'utf-8');
// Remove the self.onmessage handler and 'use strict', then eval
const workerCode = workerSource
    .replace("'use strict';", '')
    .replace(/self\.onmessage\s*=\s*function[\s\S]*$/, '');
// Create a sandbox with the worker functions
const workerSandbox = {};
const workerFn = new Function(
    workerCode +
    '\nreturn { SeededRNG, TreeNode, buildTree, trainForest, predictForest, ' +
    'computeClassificationMetrics, computeRegressionMetrics, correctAreaEstimates };'
);
const RF = workerFn();

// ══════════════════════════════════════════════════════════════
//  1. SYNTHETIC DATA TESTS
// ══════════════════════════════════════════════════════════════
section('SyntheticData');

// Test: Constants are correct
assert(SyntheticData.WIDTH === 1000, 'Raster width should be 1000');
assert(SyntheticData.HEIGHT === 1000, 'Raster height should be 1000');
assert(SyntheticData.NUM_BANDS === 10, 'Number of bands should be 10');
assert(SyntheticData.NUM_CLASSES === 5, 'Number of classes should be 5');
assert(SyntheticData.CLASS_NAMES.length === 5, 'Should have 5 class names');

// Test: Band generation
const bandResult = SyntheticData.generateBands(42);
assert(bandResult.bands instanceof Float32Array, 'bands should be Float32Array');
assert(bandResult.bands.length === 10 * 1000 * 1000, 'bands should have 10M values (10 bands × 1M pixels)');
assert(bandResult.width === 1000, 'width should be 1000');
assert(bandResult.height === 1000, 'height should be 1000');

// Test: Band values are in reasonable range
let bandMin = Infinity, bandMax = -Infinity;
for (let i = 0; i < bandResult.bands.length; i++) {
    if (bandResult.bands[i] < bandMin) bandMin = bandResult.bands[i];
    if (bandResult.bands[i] > bandMax) bandMax = bandResult.bands[i];
}
assert(bandMin >= -1, `Band min should be >= -1 (got ${bandMin.toFixed(3)})`);
assert(bandMax <= 2, `Band max should be <= 2 (got ${bandMax.toFixed(3)})`);

// Test: Deterministic generation (same seed = same output)
const bandResult2 = SyntheticData.generateBands(42);
let identical = true;
for (let i = 0; i < 100; i++) {
    if (bandResult.bands[i] !== bandResult2.bands[i]) { identical = false; break; }
}
assert(identical, 'Same seed should produce identical bands');

// Test: Different seeds produce different output
const bandResult3 = SyntheticData.generateBands(99);
let different = false;
for (let i = 0; i < 100; i++) {
    if (bandResult.bands[i] !== bandResult3.bands[i]) { different = true; break; }
}
assert(different, 'Different seeds should produce different bands');

// Test: Categorical ground truth
const catTruth = SyntheticData.generateCategoricalTruth(bandResult.bands);
assert(catTruth instanceof Uint8Array, 'Categorical truth should be Uint8Array');
assert(catTruth.length === 1000 * 1000, 'Categorical truth should have 1M pixels');
const classSet = new Set(catTruth);
assert(classSet.size > 1, 'Should contain multiple classes');
for (const c of classSet) {
    assert(c >= 0 && c < 5, `Class ${c} should be in range [0, 5)`);
}

// Test: Continuous ground truth
const contTruth = SyntheticData.generateContinuousTruth(bandResult.bands);
assert(contTruth instanceof Float32Array, 'Continuous truth should be Float32Array');
assert(contTruth.length === 1000 * 1000, 'Continuous truth should have 1M pixels');
let hasPositive = false;
for (let i = 0; i < contTruth.length; i++) {
    if (contTruth[i] > 0) { hasPositive = true; break; }
}
assert(hasPositive, 'Continuous truth should contain positive values');

// Test: Point sampling
const points = SyntheticData.samplePoints(100, 42);
assert(points.length === 100, 'Should sample exactly 100 points');
assert(points instanceof Uint32Array, 'Points should be Uint32Array');
const pointSet = new Set(points);
assert(pointSet.size === 100, 'All sampled points should be unique');
for (const p of points) {
    assert(p >= 0 && p < 1000000, `Point index ${p} should be in valid range`);
}

// Test: Feature extraction
const features = SyntheticData.extractFeatures(bandResult.bands, points);
assert(features instanceof Float32Array, 'Features should be Float32Array');
assert(features.length === 100 * 10, 'Features should have 100 × 10 = 1000 elements');

// Test: Full generate() returns all required fields
const fullData = SyntheticData.generate(42, 100);
assert(fullData.bands instanceof Float32Array, 'Full generate should include flat bands array');
assert(fullData.bands.length === 10 * 1000000, 'bands should have 10M values');
assert(fullData.categoricalTruth instanceof Uint8Array, 'Should include categorical truth');
assert(fullData.continuousTruth instanceof Float32Array, 'Should include continuous truth');
assert(fullData.trainingIndices.length === 100, 'Should have 100 training indices');
assert(fullData.trainingFeatures instanceof Float32Array, 'Should include training features');
console.log(`  ✓ SyntheticData: ${passed} assertions passed`);
const syntheticPassed = passed;

// ══════════════════════════════════════════════════════════════
//  2. SPATIAL BLOCKING TESTS
// ══════════════════════════════════════════════════════════════
section('SpatialBlocking');

// Test: Block creation
const blocks = SpatialBlocking.createBlocks(1000, 1000, 200);
assert(blocks.blocksX === 5, `blocksX should be 5 (got ${blocks.blocksX})`);
assert(blocks.blocksY === 5, `blocksY should be 5 (got ${blocks.blocksY})`);
assert(blocks.numBlocks === 25, `numBlocks should be 25 (got ${blocks.numBlocks})`);
assert(blocks.pixelBlockMap instanceof Uint16Array, 'pixelBlockMap should be Uint16Array');
assert(blocks.pixelBlockMap.length === 1000000, 'pixelBlockMap should cover 1M pixels');

// Test: Pixel-to-block assignment is correct
// Pixel (0,0) should be in block 0
assert(blocks.pixelBlockMap[0] === 0, 'Pixel (0,0) should be in block 0');
// Pixel (999,999) should be in block 24 (last block)
assert(blocks.pixelBlockMap[999 * 1000 + 999] === 24, 'Pixel (999,999) should be in block 24');
// Pixel (200, 0) should be in block 1 (next block in x)
assert(blocks.pixelBlockMap[0 * 1000 + 200] === 1, 'Pixel (200,0) should be in block 1');
// Pixel (0, 200) should be in block 5 (next block in y)
assert(blocks.pixelBlockMap[200 * 1000 + 0] === 5, 'Pixel (0,200) should be in block 5');

// Test: Block creation with different sizes
const blocks50 = SpatialBlocking.createBlocks(1000, 1000, 50);
assert(blocks50.numBlocks === 400, 'Block size 50 should create 400 blocks');

// Test: Point assignment to blocks
const testPoints = new Uint32Array([0, 500, 999999]); // corners
const blockPoints = SpatialBlocking.assignPointsToBlocks(testPoints, blocks.pixelBlockMap, blocks.numBlocks);
assert(Array.isArray(blockPoints), 'blockPoints should be an array');
assert(blockPoints.length === blocks.numBlocks, 'blockPoints should have entry per block');

// Verify point 0 (pixel 0 = block 0) is assigned correctly
assert(blockPoints[0].length > 0, 'Block 0 should contain point at pixel 0');
// Verify point 2 (pixel 999999 = block 24) is assigned correctly
assert(blockPoints[24].length > 0, 'Block 24 should contain point at pixel 999999');

// Test: Bootstrap sampling
const bs = SpatialBlocking.bootstrapSample(25, 42);
assert(bs.drawnBlocks instanceof Uint16Array, 'drawnBlocks should be Uint16Array');
assert(bs.drawnBlocks.length === 25, 'Should draw numBlocks samples');
assert(Array.isArray(bs.oobBlocks), 'oobBlocks should be an array');
assert(bs.blockWeights instanceof Uint16Array, 'blockWeights should be Uint16Array');

// OOB blocks should have weight 0
for (const b of bs.oobBlocks) {
    assert(bs.blockWeights[b] === 0, `OOB block ${b} should have weight 0`);
}

// Non-OOB blocks should have weight > 0
const totalWeight = Array.from(bs.blockWeights).reduce((a, b) => a + b, 0);
assert(totalWeight === 25, 'Total block weights should equal numBlocks');

// Test: OOB fraction should be approximately 36.8%
// Run many bootstrap samples and check average OOB fraction
let totalOOB = 0;
const nTrials = 1000;
for (let i = 0; i < nTrials; i++) {
    const trial = SpatialBlocking.bootstrapSample(100, i * 1000);
    totalOOB += trial.oobBlocks.length;
}
const avgOOBFraction = totalOOB / (nTrials * 100);
assertApprox(avgOOBFraction, 0.368, 0.02, 'Average OOB fraction should be ≈36.8%');

// Test: Deterministic bootstrap (same seed = same result)
const bs1 = SpatialBlocking.bootstrapSample(25, 42);
const bs2 = SpatialBlocking.bootstrapSample(25, 42);
let bsIdentical = true;
for (let i = 0; i < 25; i++) {
    if (bs1.drawnBlocks[i] !== bs2.drawnBlocks[i]) { bsIdentical = false; break; }
}
assert(bsIdentical, 'Same seed should produce identical bootstrap samples');

// Test: generateAllBootstraps
const allBS = SpatialBlocking.generateAllBootstraps(25, 10, 42);
assert(allBS.length === 10, 'Should generate 10 bootstrap samples');
assert(allBS[0].drawnBlocks.length === 25, 'Each sample should draw 25 blocks');

// Test: getPixelsInBlocks
const pixelsInBlock0 = SpatialBlocking.getPixelsInBlocks([0], blocks.pixelBlockMap, 1000000);
assert(pixelsInBlock0 instanceof Uint32Array, 'getPixelsInBlocks should return Uint32Array');
assert(pixelsInBlock0.length === 200 * 200, `Block 0 should contain ${200 * 200} pixels (got ${pixelsInBlock0.length})`);

// Test: getTrainingData
const trainData = SpatialBlocking.getTrainingData(bs.blockWeights, blockPoints);
assert(trainData.indices instanceof Uint32Array, 'Training indices should be Uint32Array');
assert(trainData.weights instanceof Float32Array, 'Training weights should be Float32Array');
assert(trainData.indices.length === trainData.weights.length, 'Indices and weights should match');

console.log(`  ✓ SpatialBlocking: ${passed - syntheticPassed} assertions passed`);
const blockingPassed = passed;

// ══════════════════════════════════════════════════════════════
//  3. RANDOM FOREST TESTS
// ══════════════════════════════════════════════════════════════
section('Random Forest');

// Test: SeededRNG
const rng = new RF.SeededRNG(42);
const v1 = rng.next();
assert(v1 > 0 && v1 < 1, `RNG should produce values in (0,1), got ${v1}`);
const v2 = rng.next();
assert(v1 !== v2, 'Consecutive RNG values should differ');

// Test RNG determinism
const rng1 = new RF.SeededRNG(42);
const rng2 = new RF.SeededRNG(42);
let rngIdentical = true;
for (let i = 0; i < 100; i++) {
    if (rng1.next() !== rng2.next()) { rngIdentical = false; break; }
}
assert(rngIdentical, 'Same seed RNG should produce identical sequences');

// Test: Classification with simple linearly separable data
// Create 2D data: class 0 when x < 0.5, class 1 when x >= 0.5
const nSamples = 200;
const numBands = 2;
const classX = new Float32Array(nSamples * numBands);
const classY = new Uint8Array(nSamples);
const classW = new Float32Array(nSamples).fill(1);

for (let i = 0; i < nSamples; i++) {
    const x = i / nSamples;
    const y = (i * 7 % nSamples) / nSamples;
    classX[i * numBands] = x;
    classX[i * numBands + 1] = y;
    classY[i] = x < 0.5 ? 0 : 1;
}

const classConfig = { nTrees: 10, maxDepth: 8, minLeafSamples: 2, mtry: 2 };
const classRng = new RF.SeededRNG(42);
const classTrees = RF.trainForest(classX, classY, classW, nSamples, numBands, classConfig, classRng, true);
assert(classTrees.length === 10, 'Should train 10 trees');

// Predict on training data
const classIndices = Array.from({ length: nSamples }, (_, i) => i);
const classPred = RF.predictForest(classTrees, classX, classIndices, numBands, true, 2);
assert(classPred instanceof Uint8Array, 'Classification predictions should be Uint8Array');
assert(classPred.length === nSamples, 'Should have prediction for each sample');

// Check accuracy on this trivially separable problem
let correctClass = 0;
for (let i = 0; i < nSamples; i++) {
    if (classPred[i] === classY[i]) correctClass++;
}
const classAcc = correctClass / nSamples;
assert(classAcc > 0.9, `Classification accuracy on separable data should be >90% (got ${(classAcc * 100).toFixed(1)}%)`);

// Test: Regression with linear relationship
// y = 2*x + noise
const regN = 300;
const regX = new Float32Array(regN * 2);
const regY = new Float32Array(regN);
const regW = new Float32Array(regN).fill(1);
const regRngSeed = new RF.SeededRNG(123);

for (let i = 0; i < regN; i++) {
    const x = i / regN;
    const noise = (regRngSeed.next() - 0.5) * 0.1;
    regX[i * 2] = x;
    regX[i * 2 + 1] = regRngSeed.next();
    regY[i] = 2 * x + noise;
}

const regConfig = { nTrees: 20, maxDepth: 10, minLeafSamples: 3, mtry: 2 };
const regRng = new RF.SeededRNG(42);
const regTrees = RF.trainForest(regX, regY, regW, regN, 2, regConfig, regRng, false);
assert(regTrees.length === 20, 'Should train 20 regression trees');

const regIndices = Array.from({ length: regN }, (_, i) => i);
const regPred = RF.predictForest(regTrees, regX, regIndices, 2, false, 0);
assert(regPred instanceof Float32Array, 'Regression predictions should be Float32Array');

// Compute R² manually to verify
let ssRes = 0, ssTot = 0;
let meanY = 0;
for (let i = 0; i < regN; i++) meanY += regY[i];
meanY /= regN;
for (let i = 0; i < regN; i++) {
    ssRes += (regPred[i] - regY[i]) ** 2;
    ssTot += (regY[i] - meanY) ** 2;
}
const r2 = 1 - ssRes / ssTot;
assert(r2 > 0.85, `R² on linear data should be >0.85 (got ${r2.toFixed(3)})`);

console.log(`  ✓ Random Forest: ${passed - blockingPassed} assertions passed`);
const rfPassed = passed;

// ══════════════════════════════════════════════════════════════
//  4. ACCURACY METRICS TESTS
// ══════════════════════════════════════════════════════════════
section('Accuracy Metrics');

// Test: Classification metrics with known confusion matrix
// Perfect classification
const perfectPred = new Uint8Array([0, 0, 1, 1, 2, 2]);
const perfectObs = new Uint8Array([0, 0, 1, 1, 2, 2]);
const perfectMetrics = RF.computeClassificationMetrics(perfectPred, perfectObs, 3);

assertApprox(perfectMetrics.overallAccuracy, 1.0, 1e-10, 'Perfect classification should have OA=1.0');
for (let c = 0; c < 3; c++) {
    assertApprox(perfectMetrics.userAccuracy[c], 1.0, 1e-10, `Perfect UA for class ${c}`);
    assertApprox(perfectMetrics.producerAccuracy[c], 1.0, 1e-10, `Perfect PA for class ${c}`);
}

// Test: Known misclassification
// Observed: [0,0,0, 1,1,1] → Predicted: [0,0,1, 1,1,0]
const knownPred = new Uint8Array([0, 0, 1, 1, 1, 0]);
const knownObs = new Uint8Array([0, 0, 0, 1, 1, 1]);
const knownMetrics = RF.computeClassificationMetrics(knownPred, knownObs, 2);
assertApprox(knownMetrics.overallAccuracy, 4 / 6, 1e-6, 'Known OA should be 4/6');
// Confusion matrix: [[2,1],[1,2]]
// UA class 0 = 2/3, UA class 1 = 2/3
assertApprox(knownMetrics.userAccuracy[0], 2 / 3, 1e-6, 'Known UA class 0 = 2/3');
assertApprox(knownMetrics.userAccuracy[1], 2 / 3, 1e-6, 'Known UA class 1 = 2/3');
// PA class 0 = 2/3, PA class 1 = 2/3
assertApprox(knownMetrics.producerAccuracy[0], 2 / 3, 1e-6, 'Known PA class 0 = 2/3');
assertApprox(knownMetrics.producerAccuracy[1], 2 / 3, 1e-6, 'Known PA class 1 = 2/3');

// Test: Regression metrics
const regPredMetric = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0]);
const regObsMetric = new Float32Array([1.1, 1.9, 3.2, 3.8, 5.1]);
const regrMetrics = RF.computeRegressionMetrics(regPredMetric, regObsMetric);

// RMSE = sqrt(mean((pred - obs)²))
const expectedRMSE = Math.sqrt((0.01 + 0.01 + 0.04 + 0.04 + 0.01) / 5);
assertApprox(regrMetrics.rmse, expectedRMSE, 1e-4, `RMSE should be ≈${expectedRMSE.toFixed(4)}`);

// R² should be high (nearly perfect predictions)
assert(regrMetrics.r2 > 0.95, `R² should be >0.95 (got ${regrMetrics.r2.toFixed(3)})`);

// Mean residual
const expectedMeanRes = ((-0.1 + 0.1 - 0.2 + 0.2 - 0.1) / 5);
assertApprox(regrMetrics.meanResidual, expectedMeanRes, 1e-4, `Mean residual should be ≈${expectedMeanRes.toFixed(4)}`);

// Total predicted
assertApprox(regrMetrics.totalPredicted, 15.0, 1e-6, 'Total predicted should be 15.0');
assert(regrMetrics.n === 5, 'n should be 5');

console.log(`  ✓ Accuracy Metrics: ${passed - rfPassed} assertions passed`);
const metricsPassed = passed;

// ══════════════════════════════════════════════════════════════
//  5. OLOFSSON AREA CORRECTION TESTS
// ══════════════════════════════════════════════════════════════
section('Olofsson Area Correction');

// Test with a known confusion matrix
// 2 classes, total pixels = 1000
// Conf matrix: [[80, 20], [10, 90]] (ref rows, pred cols)
// i.e. class 0: 80 correct, 20 confused with class 1
//      class 1: 10 confused with class 0, 90 correct
const confFlat = new Uint32Array([80, 20, 10, 90]);
const corrected = RF.correctAreaEstimates(Array.from(confFlat), 2, 1000);

assert(corrected.length === 2, 'Should return 2 corrected areas');
// Sum of corrected areas should equal total pixels
const correctedSum = corrected[0] + corrected[1];
assertApprox(correctedSum, 1000, 1, 'Corrected areas should sum to total pixels');

// With this confusion matrix:
// Map proportions: class 0 gets 90 predictions (col sum), class 1 gets 110
// Corrected class 0 proportion = (90/200)*(80/90) + (110/200)*(10/110) = 0.4*0.889 + 0.55*0.0909 = 0.4056
// Corrected class 0 pixels = 0.4056 * 1000 = 405.6 (approximate)
assert(corrected[0] > 0 && corrected[0] < 1000, 'Corrected area class 0 should be reasonable');
assert(corrected[1] > 0 && corrected[1] < 1000, 'Corrected area class 1 should be reasonable');

// Test: Perfect classification should not change areas
const perfectConf = [50, 0, 0, 50];
const perfectCorrected = RF.correctAreaEstimates(perfectConf, 2, 1000);
assertApprox(perfectCorrected[0], 500, 1, 'Perfect class 0 should keep 500 pixels');
assertApprox(perfectCorrected[1], 500, 1, 'Perfect class 1 should keep 500 pixels');

console.log(`  ✓ Olofsson Area Correction: ${passed - metricsPassed} assertions passed`);
const olofPassed = passed;

// ══════════════════════════════════════════════════════════════
//  6. SUMMARY STATISTICS TESTS
// ══════════════════════════════════════════════════════════════
section('Summary Statistics (PNASCharts)');

// Test: summaryStats with known data
const testData = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
const stats = PNASCharts.summaryStats(testData);

assertApprox(stats.mean, 5.5, 1e-6, 'Mean of 1..10 should be 5.5');
assertApprox(stats.median, 5.5, 1e-6, 'Median of 1..10 should be 5.5');
assert(stats.ci95[0] < stats.mean, 'CI lower bound should be < mean');
assert(stats.ci95[1] > stats.mean, 'CI upper bound should be > mean');
assert(stats.ci95[0] >= 1, 'CI lower should be >= min value');
assert(stats.ci95[1] <= 10, 'CI upper should be <= max value');

// Test: Single-value array
const singleStats = PNASCharts.summaryStats([42]);
assertApprox(singleStats.mean, 42, 1e-6, 'Mean of single value should be 42');
assertApprox(singleStats.median, 42, 1e-6, 'Median of single value should be 42');

// Test: Odd-length array median
const oddStats = PNASCharts.summaryStats([3, 1, 2]);
assertApprox(oddStats.median, 2, 1e-6, 'Median of [3,1,2] should be 2');

console.log(`  ✓ Summary Statistics: ${passed - olofPassed} assertions passed`);

// ══════════════════════════════════════════════════════════════
//  7. INTEGRATION TEST: END-TO-END PIPELINE
// ══════════════════════════════════════════════════════════════
section('Integration: End-to-End Pipeline');

// Generate data
const data = SyntheticData.generate(42, 500);
assert(data.bands instanceof Float32Array, 'Integration: should have flat bands array');

// Create blocks and assign points
const intBlocks = SpatialBlocking.createBlocks(1000, 1000, 200);
const intBlockPoints = SpatialBlocking.assignPointsToBlocks(
    data.trainingIndices, intBlocks.pixelBlockMap, intBlocks.numBlocks
);

// Bootstrap sample
const intBS = SpatialBlocking.bootstrapSample(intBlocks.numBlocks, 42);
assert(intBS.oobBlocks.length > 0, 'Integration: should have OOB blocks');

// Get training and OOB data
const intTrainData = SpatialBlocking.getTrainingData(intBS.blockWeights, intBlockPoints);
assert(intTrainData.indices.length > 0, 'Integration: should have training points');

// Prepare training features/labels
const trainFeatures = new Float32Array(intTrainData.indices.length * 10);
const trainLabels = new Uint8Array(intTrainData.indices.length);
for (let i = 0; i < intTrainData.indices.length; i++) {
    const ptIdx = intTrainData.indices[i];
    for (let b = 0; b < 10; b++) {
        trainFeatures[i * 10 + b] = data.trainingFeatures[ptIdx * 10 + b];
    }
    trainLabels[i] = data.categoricalTruth[data.trainingIndices[ptIdx]];
}

// Prepare OOB features/labels
const oobPointIndices = [];
for (const block of intBS.oobBlocks) {
    if (intBlockPoints[block]) {
        for (const ptIdx of intBlockPoints[block]) {
            oobPointIndices.push(ptIdx);
        }
    }
}
const oobFeatures = new Float32Array(oobPointIndices.length * 10);
const oobLabels = new Uint8Array(oobPointIndices.length);
for (let i = 0; i < oobPointIndices.length; i++) {
    const ptIdx = oobPointIndices[i];
    for (let b = 0; b < 10; b++) {
        oobFeatures[i * 10 + b] = data.trainingFeatures[ptIdx * 10 + b];
    }
    oobLabels[i] = data.categoricalTruth[data.trainingIndices[ptIdx]];
}

// Train RF
const intRng = new RF.SeededRNG(42);
const intConfig = { nTrees: 20, maxDepth: 10, minLeafSamples: 5, mtry: 3 };
const intTrees = RF.trainForest(trainFeatures, trainLabels, intTrainData.weights, intTrainData.indices.length, 10, intConfig, intRng, true);

// Predict OOB
const oobIndices = Array.from({ length: oobPointIndices.length }, (_, i) => i);
const oobPred = RF.predictForest(intTrees, oobFeatures, oobIndices, 10, true, 5);

// Compute metrics
const intMetrics = RF.computeClassificationMetrics(oobPred, oobLabels, 5);
assert(intMetrics.overallAccuracy > 0.5, `Integration OA should be >50% (got ${(intMetrics.overallAccuracy * 100).toFixed(1)}%)`);
assert(intMetrics.confusionMatrix.length === 25, 'Confusion matrix should be 5×5 = 25');
assert(intMetrics.n === oobPointIndices.length, 'Metrics n should equal OOB count');

// Area correction
const intCorrected = RF.correctAreaEstimates(intMetrics.confusionMatrix, 5, 1000000);
assert(intCorrected.length === 5, 'Should have 5 corrected areas');
const intCorrectedSum = intCorrected.reduce((a, b) => a + b, 0);
assertApprox(intCorrectedSum, 1000000, 100, 'Corrected areas should sum to ≈1M');

console.log(`  ✓ Integration: pipeline completed successfully`);

// ══════════════════════════════════════════════════════════════
//  RESULTS SUMMARY
// ══════════════════════════════════════════════════════════════
console.log('\n══════════════════════════════════════════');
console.log(`  TOTAL: ${passed} passed, ${failed} failed`);
if (failed > 0) {
    console.log('\n  Failures:');
    for (const f of failures) console.log(`    • ${f}`);
}
console.log('══════════════════════════════════════════\n');

process.exit(failed > 0 ? 1 : 0);
