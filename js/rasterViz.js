/* ============================================================
   Raster Visualization Module
   Canvas-based rendering for multi-band raster data,
   ground truth maps, predictions, and uncertainty.
   ============================================================ */

const RasterViz = (() => {
    /* --- Viridis colormap (256 entries) --- */
    const VIRIDIS = generateViridis();

    function generateViridis() {
        // Key control points for viridis (simplified 16-point LUT, interpolated to 256)
        const keys = [
            [68, 1, 84], [72, 35, 116], [64, 67, 135], [52, 94, 141],
            [41, 120, 142], [32, 144, 140], [34, 167, 132], [68, 190, 112],
            [121, 209, 81], [189, 222, 38], [253, 231, 37], [253, 231, 37],
        ];
        const cmap = new Array(256);
        for (let i = 0; i < 256; i++) {
            const t = (i / 255) * (keys.length - 1);
            const idx = Math.floor(t);
            const frac = t - idx;
            const c0 = keys[Math.min(idx, keys.length - 1)];
            const c1 = keys[Math.min(idx + 1, keys.length - 1)];
            cmap[i] = [
                Math.round(c0[0] + frac * (c1[0] - c0[0])),
                Math.round(c0[1] + frac * (c1[1] - c0[1])),
                Math.round(c0[2] + frac * (c1[2] - c0[2])),
            ];
        }
        return cmap;
    }

    /* --- Render a single band as greyscale --- */
    function renderBand(canvas, bandData, width, height) {
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        const img = ctx.createImageData(width, height);
        const d = img.data;

        // Find min/max for normalization
        let min = Infinity, max = -Infinity;
        for (let i = 0; i < bandData.length; i++) {
            if (bandData[i] < min) min = bandData[i];
            if (bandData[i] > max) max = bandData[i];
        }
        const range = max - min || 1;

        for (let i = 0; i < bandData.length; i++) {
            const v = Math.round(((bandData[i] - min) / range) * 255);
            const p = i * 4;
            d[p] = v; d[p + 1] = v; d[p + 2] = v; d[p + 3] = 255;
        }
        ctx.putImageData(img, 0, 0);
    }

    /* --- Render RGB false-color composite --- */
    function renderRGB(canvas, bands, width, height, rBand, gBand, bBand) {
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        const img = ctx.createImageData(width, height);
        const d = img.data;
        const totalPixels = width * height;

        // Normalize each band independently
        const normalize = (bandIdx) => {
            const offset = bandIdx * totalPixels;
            let min = Infinity, max = -Infinity;
            for (let i = 0; i < totalPixels; i++) {
                const v = bands[offset + i];
                if (v < min) min = v;
                if (v > max) max = v;
            }
            return { offset, min, range: max - min || 1 };
        };

        const rN = normalize(rBand);
        const gN = normalize(gBand);
        const bN = normalize(bBand);

        for (let i = 0; i < totalPixels; i++) {
            const p = i * 4;
            d[p] = Math.round(((bands[rN.offset + i] - rN.min) / rN.range) * 255);
            d[p + 1] = Math.round(((bands[gN.offset + i] - gN.min) / gN.range) * 255);
            d[p + 2] = Math.round(((bands[bN.offset + i] - bN.min) / bN.range) * 255);
            d[p + 3] = 255;
        }
        ctx.putImageData(img, 0, 0);
    }

    /* --- Render categorical class map --- */
    function renderClassMap(canvas, classData, width, height, classColors) {
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        const img = ctx.createImageData(width, height);
        const d = img.data;

        for (let i = 0; i < classData.length; i++) {
            const c = classColors[classData[i]] || [128, 128, 128];
            const p = i * 4;
            d[p] = c[0]; d[p + 1] = c[1]; d[p + 2] = c[2]; d[p + 3] = 255;
        }
        ctx.putImageData(img, 0, 0);
    }

    /* --- Render continuous data with viridis colormap --- */
    function renderContinuous(canvas, data, width, height, minVal, maxVal) {
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        const img = ctx.createImageData(width, height);
        const d = img.data;
        const range = maxVal - minVal || 1;

        for (let i = 0; i < data.length; i++) {
            const norm = Math.max(0, Math.min(255, Math.round(((data[i] - minVal) / range) * 255)));
            const c = VIRIDIS[norm];
            const p = i * 4;
            d[p] = c[0]; d[p + 1] = c[1]; d[p + 2] = c[2]; d[p + 3] = 255;
        }
        ctx.putImageData(img, 0, 0);
    }

    /* --- Render spatial block grid overlay on existing canvas --- */
    function renderBlockGrid(canvas, blockSize, width, height, oobBlocks, blocksX) {
        const ctx = canvas.getContext('2d');
        const scaleX = canvas.width / width;
        const scaleY = canvas.height / height;

        // Draw OOB blocks with highlight
        if (oobBlocks && oobBlocks.length > 0) {
            ctx.fillStyle = 'rgba(212, 168, 67, 0.25)';
            for (const blockId of oobBlocks) {
                const bx = blockId % blocksX;
                const by = Math.floor(blockId / blocksX);
                ctx.fillRect(bx * blockSize * scaleX, by * blockSize * scaleY,
                    blockSize * scaleX, blockSize * scaleY);
            }
        }

        // Draw grid
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.lineWidth = 0.5;
        for (let x = 0; x <= width; x += blockSize) {
            ctx.beginPath();
            ctx.moveTo(x * scaleX, 0);
            ctx.lineTo(x * scaleX, canvas.height);
            ctx.stroke();
        }
        for (let y = 0; y <= height; y += blockSize) {
            ctx.beginPath();
            ctx.moveTo(0, y * scaleY);
            ctx.lineTo(canvas.width, y * scaleY);
            ctx.stroke();
        }
    }

    /* --- Render training points on canvas --- */
    function renderPoints(canvas, pointIndices, width, height, classData, classColors) {
        const ctx = canvas.getContext('2d');
        const scaleX = canvas.width / width;
        const scaleY = canvas.height / height;

        for (let i = 0; i < pointIndices.length; i++) {
            const px = pointIndices[i] % width;
            const py = Math.floor(pointIndices[i] / width);
            const c = classColors ? classColors[classData[pointIndices[i]]] : [255, 255, 255];

            ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
            ctx.strokeStyle = 'rgba(0,0,0,0.5)';
            ctx.lineWidth = 0.5;
            ctx.beginPath();
            ctx.arc(px * scaleX, py * scaleY, 2, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
        }
    }

    /* --- Create a viridis color bar element --- */
    function createColorBar(container, minVal, maxVal, label) {
        container.innerHTML = '';
        const bar = document.createElement('div');
        bar.className = 'color-bar';

        const minLabel = document.createElement('span');
        minLabel.textContent = minVal.toFixed(0);

        const gradient = document.createElement('div');
        gradient.className = 'color-bar__gradient';
        // Build CSS gradient from viridis
        const stops = [];
        for (let i = 0; i <= 10; i++) {
            const idx = Math.round((i / 10) * 255);
            const c = VIRIDIS[idx];
            stops.push(`rgb(${c[0]},${c[1]},${c[2]}) ${i * 10}%`);
        }
        gradient.style.background = `linear-gradient(90deg, ${stops.join(', ')})`;

        const maxLabel = document.createElement('span');
        maxLabel.textContent = maxVal.toFixed(0);

        bar.appendChild(minLabel);
        bar.appendChild(gradient);
        bar.appendChild(maxLabel);

        if (label) {
            const lbl = document.createElement('span');
            lbl.textContent = label;
            lbl.style.marginLeft = '8px';
            lbl.style.fontWeight = '600';
            bar.appendChild(lbl);
        }

        container.appendChild(bar);
    }

    /* --- Create discrete legend --- */
    function createDiscreteLegend(container, classNames, classColors) {
        container.innerHTML = '';
        const legend = document.createElement('div');
        legend.className = 'discrete-legend';

        for (let i = 0; i < classNames.length; i++) {
            const item = document.createElement('div');
            item.className = 'discrete-legend__item';

            const swatch = document.createElement('div');
            swatch.className = 'discrete-legend__swatch';
            const c = classColors[i];
            swatch.style.background = `rgb(${c[0]},${c[1]},${c[2]})`;

            const label = document.createElement('span');
            label.textContent = classNames[i];

            item.appendChild(swatch);
            item.appendChild(label);
            legend.appendChild(item);
        }

        container.appendChild(legend);
    }

    return {
        renderBand,
        renderRGB,
        renderClassMap,
        renderContinuous,
        renderBlockGrid,
        renderPoints,
        createColorBar,
        createDiscreteLegend,
        VIRIDIS,
    };
})();

if (typeof module !== 'undefined') module.exports = RasterViz;
