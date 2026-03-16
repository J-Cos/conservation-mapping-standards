/* ============================================================
   Random Forest Web Worker
   Self-contained CART + RF implementation for both
   classification and regression, executing one bootstrap
   iteration per message.
   ============================================================ */

'use strict';

/* --- Lightweight seeded RNG --- */
class SeededRNG {
    constructor(seed) { this.s = seed | 0; }
    next() {
        this.s = (this.s * 1664525 + 1013904223) & 0x7FFFFFFF;
        return this.s / 0x7FFFFFFF;
    }
    nextInt(max) { return Math.floor(this.next() * max); }
}

/* --- CART Tree Node --- */
class TreeNode {
    constructor() {
        this.feature = -1;
        this.threshold = 0;
        this.left = null;
        this.right = null;
        this.value = null; // leaf: class (int) or mean (float)
    }
}

/* --- Build a single CART tree --- */
function buildTree(X, y, weights, nFeatures, nSamples, numBands, config, rng, isClassification) {
    const maxDepth = config.maxDepth || 10;
    const minLeaf = config.minLeafSamples || 5;
    const mtry = config.mtry || (isClassification ? Math.ceil(Math.sqrt(numBands)) : Math.max(1, Math.floor(numBands / 3)));

    function buildNode(indices, depth) {
        const n = indices.length;
        if (n <= minLeaf || depth >= maxDepth) {
            return makeLeaf(indices);
        }

        // Choose random feature subset
        const featureSubset = [];
        const avail = [];
        for (let i = 0; i < numBands; i++) avail.push(i);
        for (let i = 0; i < Math.min(mtry, numBands); i++) {
            const j = i + rng.nextInt(avail.length - i);
            [avail[i], avail[j]] = [avail[j], avail[i]];
            featureSubset.push(avail[i]);
        }

        let bestGain = -Infinity;
        let bestFeature = -1;
        let bestThreshold = 0;
        let bestLeftIdx = null;
        let bestRightIdx = null;

        const parentImpurity = isClassification ? giniImpurity(indices) : variance(indices);

        for (const feat of featureSubset) {
            // Get unique thresholds (sample up to 20 split points for speed)
            const vals = [];
            for (const idx of indices) vals.push(X[idx * numBands + feat]);
            vals.sort((a, b) => a - b);

            const step = Math.max(1, Math.floor(vals.length / 20));
            for (let t = step; t < vals.length; t += step) {
                const threshold = (vals[t - 1] + vals[t]) / 2;
                const leftIdx = [], rightIdx = [];

                for (const idx of indices) {
                    if (X[idx * numBands + feat] <= threshold) leftIdx.push(idx);
                    else rightIdx.push(idx);
                }

                if (leftIdx.length < minLeaf || rightIdx.length < minLeaf) continue;

                const leftImp = isClassification ? giniImpurity(leftIdx) : variance(leftIdx);
                const rightImp = isClassification ? giniImpurity(rightIdx) : variance(rightIdx);
                const gain = parentImpurity - (leftIdx.length / n) * leftImp - (rightIdx.length / n) * rightImp;

                if (gain > bestGain) {
                    bestGain = gain;
                    bestFeature = feat;
                    bestThreshold = threshold;
                    bestLeftIdx = leftIdx;
                    bestRightIdx = rightIdx;
                }
            }
        }

        if (bestGain <= 0 || !bestLeftIdx) {
            return makeLeaf(indices);
        }

        const node = new TreeNode();
        node.feature = bestFeature;
        node.threshold = bestThreshold;
        node.left = buildNode(bestLeftIdx, depth + 1);
        node.right = buildNode(bestRightIdx, depth + 1);
        return node;
    }

    function giniImpurity(indices) {
        const counts = {};
        let total = 0;
        for (const idx of indices) {
            const w = weights ? weights[idx] : 1;
            const c = y[idx];
            counts[c] = (counts[c] || 0) + w;
            total += w;
        }
        let gini = 1;
        for (const c in counts) {
            const p = counts[c] / total;
            gini -= p * p;
        }
        return gini;
    }

    function variance(indices) {
        let sum = 0, sumSq = 0, total = 0;
        for (const idx of indices) {
            const w = weights ? weights[idx] : 1;
            const v = y[idx];
            sum += v * w;
            sumSq += v * v * w;
            total += w;
        }
        if (total === 0) return 0;
        const mean = sum / total;
        return sumSq / total - mean * mean;
    }

    function makeLeaf(indices) {
        const node = new TreeNode();
        if (isClassification) {
            // Majority vote
            const counts = {};
            for (const idx of indices) {
                const w = weights ? weights[idx] : 1;
                const c = y[idx];
                counts[c] = (counts[c] || 0) + w;
            }
            let best = -1, bestCount = -1;
            for (const c in counts) {
                if (counts[c] > bestCount) { bestCount = counts[c]; best = parseInt(c); }
            }
            node.value = best;
        } else {
            // Mean
            let sum = 0, total = 0;
            for (const idx of indices) {
                const w = weights ? weights[idx] : 1;
                sum += y[idx] * w;
                total += w;
            }
            node.value = total > 0 ? sum / total : 0;
        }
        return node;
    }

    const allIndices = [];
    for (let i = 0; i < nSamples; i++) allIndices.push(i);
    return buildNode(allIndices, 0);
}

/* --- Predict with a single tree --- */
function predictTree(tree, X, startIdx, numBands) {
    let node = tree;
    while (node.value === null) {
        if (X[startIdx + node.feature] <= node.threshold) node = node.left;
        else node = node.right;
    }
    return node.value;
}

/* --- Random Forest --- */
function trainForest(X, y, weights, nSamples, numBands, config, rng, isClassification) {
    const nTrees = config.nTrees || 100;
    const trees = [];

    for (let t = 0; t < nTrees; t++) {
        // Bagging: sample with replacement from training data
        const bagSize = nSamples;
        const bagX = new Float32Array(bagSize * numBands);
        const bagY = isClassification ? new Uint8Array(bagSize) : new Float32Array(bagSize);
        const bagW = weights ? new Float32Array(bagSize) : null;

        for (let i = 0; i < bagSize; i++) {
            const src = rng.nextInt(nSamples);
            for (let b = 0; b < numBands; b++) {
                bagX[i * numBands + b] = X[src * numBands + b];
            }
            bagY[i] = y[src];
            if (bagW) bagW[i] = weights[src];
        }

        const tree = buildTree(bagX, bagY, bagW, numBands, bagSize, numBands, config, rng, isClassification);
        trees.push(tree);
    }

    return trees;
}

/* --- Predict with forest --- */
function predictForest(trees, X, pixelIndices, numBands, isClassification, numClasses) {
    const n = pixelIndices.length;
    const predictions = isClassification ? new Uint8Array(n) : new Float32Array(n);

    for (let i = 0; i < n; i++) {
        const startIdx = i * numBands;

        if (isClassification) {
            const votes = new Uint16Array(numClasses || 10);
            for (const tree of trees) {
                votes[predictTree(tree, X, startIdx, numBands)]++;
            }
            let bestClass = 0, bestVotes = 0;
            for (let c = 0; c < votes.length; c++) {
                if (votes[c] > bestVotes) { bestVotes = votes[c]; bestClass = c; }
            }
            predictions[i] = bestClass;
        } else {
            let sum = 0;
            for (const tree of trees) sum += predictTree(tree, X, startIdx, numBands);
            predictions[i] = sum / trees.length;
        }
    }

    return predictions;
}

/* --- Compute accuracy metrics --- */
function computeClassificationMetrics(predicted, observed, numClasses) {
    const confMatrix = Array.from({ length: numClasses }, () => new Uint32Array(numClasses));
    const n = predicted.length;

    for (let i = 0; i < n; i++) {
        confMatrix[observed[i]][predicted[i]]++;
    }

    // Overall accuracy
    let correct = 0;
    for (let c = 0; c < numClasses; c++) correct += confMatrix[c][c];
    const overallAccuracy = n > 0 ? correct / n : 0;

    // Per-class user & producer accuracy
    const userAccuracy = new Float32Array(numClasses);
    const producerAccuracy = new Float32Array(numClasses);

    for (let c = 0; c < numClasses; c++) {
        let rowSum = 0, colSum = 0;
        for (let j = 0; j < numClasses; j++) {
            rowSum += confMatrix[c][j];
            colSum += confMatrix[j][c];
        }
        producerAccuracy[c] = rowSum > 0 ? confMatrix[c][c] / rowSum : 0;
        userAccuracy[c] = colSum > 0 ? confMatrix[c][c] / colSum : 0;
    }

    // Flatten confusion matrix for transfer
    const flatConf = new Uint32Array(numClasses * numClasses);
    for (let i = 0; i < numClasses; i++) {
        for (let j = 0; j < numClasses; j++) {
            flatConf[i * numClasses + j] = confMatrix[i][j];
        }
    }

    // Class pixel counts from predictions (for area estimation)
    const predictedCounts = new Uint32Array(numClasses);
    for (let i = 0; i < n; i++) predictedCounts[predicted[i]]++;

    return {
        overallAccuracy,
        userAccuracy: Array.from(userAccuracy),
        producerAccuracy: Array.from(producerAccuracy),
        confusionMatrix: Array.from(flatConf),
        predictedCounts: Array.from(predictedCounts),
        n,
    };
}

function computeRegressionMetrics(predicted, observed) {
    const n = predicted.length;
    if (n === 0) return { rmse: 0, relRmse: 0, r2: 0, meanResidual: 0, residuals: [] };

    let sumObs = 0, sumPred = 0, sumSqErr = 0, sumSqObs = 0;
    const residuals = new Float32Array(n);

    for (let i = 0; i < n; i++) {
        const err = predicted[i] - observed[i];
        residuals[i] = err;
        sumObs += observed[i];
        sumPred += predicted[i];
        sumSqErr += err * err;
        sumSqObs += observed[i] * observed[i];
    }

    const meanObs = sumObs / n;
    let ssTot = 0;
    for (let i = 0; i < n; i++) ssTot += (observed[i] - meanObs) ** 2;

    const rmse = Math.sqrt(sumSqErr / n);
    const relRmse = meanObs > 0 ? rmse / meanObs : 0;
    const r2 = ssTot > 0 ? 1 - sumSqErr / ssTot : 0;
    const meanResidual = (sumPred - sumObs) / n;

    // Total sum prediction
    const totalPredicted = sumPred;

    return { rmse, relRmse, r2, meanResidual, totalPredicted, n, meanObs };
}

/* --- Olofsson area correction (categorical) --- */
function correctAreaEstimates(confMatrixFlat, numClasses, totalPixels) {
    // Reconstruct confusion matrix
    const C = Array.from({ length: numClasses }, (_, i) =>
        Array.from({ length: numClasses }, (_, j) => confMatrixFlat[i * numClasses + j])
    );

    // Mapped proportions (what the map says)
    const mapProps = new Float32Array(numClasses);
    for (let j = 0; j < numClasses; j++) {
        let colSum = 0;
        for (let i = 0; i < numClasses; i++) colSum += C[i][j];
        mapProps[j] = colSum;
    }
    const totalSampled = mapProps.reduce((a, b) => a + b, 0);
    if (totalSampled === 0) return new Float32Array(numClasses);

    // Corrected area proportions
    const corrected = new Float32Array(numClasses);
    for (let i = 0; i < numClasses; i++) {
        let sum = 0;
        for (let j = 0; j < numClasses; j++) {
            const colTotal = mapProps[j];
            if (colTotal > 0) {
                sum += (mapProps[j] / totalSampled) * (C[i][j] / colTotal);
            }
        }
        corrected[i] = sum * totalPixels;
    }

    return Array.from(corrected);
}

/* --- Worker message handler --- */
self.onmessage = function (e) {
    const msg = e.data;

    if (msg.type === 'bootstrap') {
        const {
            bootstrapIndex,
            trainingFeatures,   // Float32Array
            trainingLabels,     // Uint8Array or Float32Array
            trainingWeights,    // Float32Array (block weights)
            oobFeatures,        // Float32Array
            oobLabels,          // Uint8Array or Float32Array
            fullRasterFeatures, // Float32Array (subset or null)
            numBands,
            numClasses,
            isClassification,
            config,
            seed,
            computeFullMap,
        } = msg;

        const rng = new SeededRNG(seed);
        const nTrain = trainingFeatures.length / numBands;

        // Train RF
        const trees = trainForest(
            trainingFeatures, trainingLabels, trainingWeights,
            nTrain, numBands, config, rng, isClassification
        );

        // Predict OOB
        const nOOB = oobFeatures.length / numBands;
        const oobIndices = Array.from({ length: nOOB }, (_, i) => i);
        const oobPredictions = predictForest(trees, oobFeatures, oobIndices, numBands, isClassification, numClasses);

        // Compute metrics
        let metrics;
        if (isClassification) {
            metrics = computeClassificationMetrics(oobPredictions, oobLabels, numClasses);
            metrics.correctedArea = correctAreaEstimates(metrics.confusionMatrix, numClasses, 1000000);
        } else {
            metrics = computeRegressionMetrics(oobPredictions, oobLabels);
        }

        // Full raster prediction (optional, for generating maps)
        let fullPredictions = null;
        let totalPredictedFull = null;
        if (computeFullMap && fullRasterFeatures) {
            const nFull = fullRasterFeatures.length / numBands;
            const fullIndices = Array.from({ length: nFull }, (_, i) => i);
            fullPredictions = predictForest(trees, fullRasterFeatures, fullIndices, numBands, isClassification, numClasses);

            // Sum full-raster predictions for total biomass estimation
            if (!isClassification) {
                let sum = 0;
                for (let i = 0; i < nFull; i++) sum += fullPredictions[i];
                // Scale up: predictions are on subsampled raster (stride=2), so multiply by stride²
                totalPredictedFull = sum * 4;
            }
        }

        self.postMessage({
            type: 'result',
            bootstrapIndex,
            metrics,
            fullPredictions,
            totalPredictedFull,
            isClassification,
        });
    }
};
