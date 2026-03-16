/* ============================================================
   Spatial Blocking & Bootstrap Sampling
   Implements spatial CV with block-level bootstrap for
   independent training/validation splits.
   ============================================================ */

const SpatialBlocking = (() => {
    const DEFAULT_BLOCK_SIZE = 50; // pixels per block side

    /**
     * Create spatial blocks for a given raster.
     * @param {number} width - raster width in pixels
     * @param {number} height - raster height in pixels
     * @param {number} blockSize - block side length in pixels
     * @returns {Object} block metadata
     */
    function createBlocks(width, height, blockSize = DEFAULT_BLOCK_SIZE) {
        const blocksX = Math.ceil(width / blockSize);
        const blocksY = Math.ceil(height / blockSize);
        const numBlocks = blocksX * blocksY;

        // Assign each pixel to a block
        const pixelBlockMap = new Uint16Array(width * height);
        for (let y = 0; y < height; y++) {
            const by = Math.min(Math.floor(y / blockSize), blocksY - 1);
            for (let x = 0; x < width; x++) {
                const bx = Math.min(Math.floor(x / blockSize), blocksX - 1);
                pixelBlockMap[y * width + x] = by * blocksX + bx;
            }
        }

        return {
            blockSize,
            blocksX,
            blocksY,
            numBlocks,
            pixelBlockMap, // Uint16Array[width*height] → block ID per pixel
        };
    }

    /**
     * Assign training points to their spatial blocks.
     * @param {Uint32Array} pointIndices - pixel indices of training points
     * @param {Uint16Array} pixelBlockMap - block assignment per pixel
     * @param {number} numBlocks - total number of blocks
     * @returns {Object} mapping from blocks to point indices
     */
    function assignPointsToBlocks(pointIndices, pixelBlockMap, numBlocks) {
        // blockPoints[blockID] = array of indices into pointIndices
        const blockPoints = new Array(numBlocks);
        for (let b = 0; b < numBlocks; b++) blockPoints[b] = [];

        for (let i = 0; i < pointIndices.length; i++) {
            const blockId = pixelBlockMap[pointIndices[i]];
            blockPoints[blockId].push(i);
        }

        return blockPoints;
    }

    /**
     * Generate a single bootstrap sample of blocks (drawn with replacement).
     * Returns drawn block IDs and identifies OOB blocks.
     * @param {number} numBlocks - total blocks
     * @param {number} seed - random seed for this replicate
     * @returns {Object} { drawnBlocks, oobBlocks, blockWeights }
     */
    function bootstrapSample(numBlocks, seed) {
        let rng = seed;
        const rand = () => {
            rng = (rng * 1664525 + 1013904223) & 0x7FFFFFFF;
            return rng / 0x7FFFFFFF;
        };

        const drawnBlocks = new Uint16Array(numBlocks);
        const blockWeights = new Uint16Array(numBlocks); // how many times each block was drawn

        for (let i = 0; i < numBlocks; i++) {
            const drawn = Math.floor(rand() * numBlocks);
            drawnBlocks[i] = drawn;
            blockWeights[drawn]++;
        }

        // OOB blocks are those with weight 0 (~36.8% expected)
        const oobBlocks = [];
        for (let b = 0; b < numBlocks; b++) {
            if (blockWeights[b] === 0) oobBlocks.push(b);
        }

        return { drawnBlocks, oobBlocks, blockWeights };
    }

    /**
     * Get pixel indices belonging to a set of blocks.
     * @param {number[]} blockIds - array of block IDs
     * @param {Uint16Array} pixelBlockMap - pixel-to-block mapping
     * @param {number} totalPixels - total pixel count
     * @returns {Uint32Array} pixel indices in those blocks
     */
    function getPixelsInBlocks(blockIds, pixelBlockMap, totalPixels) {
        const blockSet = new Set(blockIds);
        const pixels = [];
        for (let i = 0; i < totalPixels; i++) {
            if (blockSet.has(pixelBlockMap[i])) pixels.push(i);
        }
        return new Uint32Array(pixels);
    }

    /**
     * Get training point indices for drawn blocks (with weights for duplication).
     * @param {Uint16Array} blockWeights - weight per block
     * @param {Array} blockPoints - blockPoints[blockID] = array of point indices
     * @returns {Object} { indices, weights } for weighted training
     */
    function getTrainingData(blockWeights, blockPoints) {
        const indices = [];
        const weights = [];

        for (let b = 0; b < blockWeights.length; b++) {
            const w = blockWeights[b];
            if (w > 0 && blockPoints[b]) {
                for (const ptIdx of blockPoints[b]) {
                    indices.push(ptIdx);
                    weights.push(w);
                }
            }
        }

        return {
            indices: new Uint32Array(indices),
            weights: new Float32Array(weights),
        };
    }

    /**
     * Pre-generate all bootstrap samples for B replicates.
     * @param {number} numBlocks
     * @param {number} B - number of bootstrap replicates
     * @param {number} baseSeed
     * @returns {Array} array of bootstrap sample objects
     */
    function generateAllBootstraps(numBlocks, B, baseSeed = 42) {
        const samples = new Array(B);
        for (let i = 0; i < B; i++) {
            samples[i] = bootstrapSample(numBlocks, baseSeed + i * 7919); // prime step
        }
        return samples;
    }

    return {
        createBlocks,
        assignPointsToBlocks,
        bootstrapSample,
        getPixelsInBlocks,
        getTrainingData,
        generateAllBootstraps,
        DEFAULT_BLOCK_SIZE,
    };
})();

if (typeof module !== 'undefined') module.exports = SpatialBlocking;
