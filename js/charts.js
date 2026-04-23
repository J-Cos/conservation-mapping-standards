/* ============================================================
   PNAS-Style Chart Configuration
   Wraps Chart.js with consistent PNAS figure styling.
   ============================================================ */

const PNASCharts = (() => {
    // PNAS color palette — colorblind friendly, muted
    const PALETTE = [
        '#4477AA', // blue
        '#EE6677', // red
        '#228833', // green
        '#CCBB44', // yellow
        '#66CCEE', // cyan
        '#AA3377', // purple
        '#BBBBBB', // grey
    ];

    const CLASS_PALETTE = [
        '#1A5E2A', // Dense Forest
        '#6AAF4C', // Open Forest
        '#C8D96F', // Grassland
        '#2F7ABF', // Water
        '#C4956A', // Bare Soil
    ];

    // PNAS defaults
    const FONT_FAMILY = "'Inter', sans-serif";
    const FONT_SIZE = 11;
    const TICK_COLOR = '#4B5563';
    const GRID_COLOR = 'rgba(0,0,0,0.06)';
    const AXIS_COLOR = '#9CA3AF';

    /* --- Base chart defaults --- */
    function applyDefaults() {
        if (typeof Chart === 'undefined') return;

        Chart.defaults.font.family = FONT_FAMILY;
        Chart.defaults.font.size = FONT_SIZE;
        Chart.defaults.color = TICK_COLOR;
        Chart.defaults.plugins.legend.labels.usePointStyle = true;
        Chart.defaults.plugins.legend.labels.pointStyle = 'circle';
        Chart.defaults.plugins.legend.labels.padding = 16;
        Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(27, 38, 59, 0.9)';
        Chart.defaults.plugins.tooltip.titleFont = { family: FONT_FAMILY, size: 12, weight: '600' };
        Chart.defaults.plugins.tooltip.bodyFont = { family: FONT_FAMILY, size: 11 };
        Chart.defaults.plugins.tooltip.cornerRadius = 4;
        Chart.defaults.plugins.tooltip.padding = 10;
    }

    /* --- Shared scale config (PNAS: minimal gridlines, no top/right spines) --- */
    function pnasScales(xLabel, yLabel, xType = 'linear') {
        return {
            x: {
                type: xType,
                title: { display: !!xLabel, text: xLabel, font: { size: 12, weight: '600' }, padding: { top: 6 } },
                grid: { display: false },
                border: { color: AXIS_COLOR },
                ticks: { color: TICK_COLOR, padding: 4 },
            },
            y: {
                title: { display: !!yLabel, text: yLabel, font: { size: 12, weight: '600' }, padding: { bottom: 6 } },
                grid: { color: GRID_COLOR, drawBorder: true },
                border: { color: AXIS_COLOR },
                ticks: { color: TICK_COLOR, padding: 4 },
            },
        };
    }

    /* --- Histogram --- */
    function histogram(canvasId, values, { title, xLabel, yLabel, color, bins = 20, thresholdLine = null, trueLine = null } = {}) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return null;

        // Compute histogram bins
        let min = Infinity, max = -Infinity;
        for (const v of values) { if (v < min) min = v; if (v > max) max = v; }
        
        if (trueLine !== null && trueLine !== undefined) {
            if (trueLine < min) min = trueLine;
            if (trueLine > max) max = trueLine;
        }

        const binWidth = (max - min) / bins || 1;
        const counts = new Array(bins).fill(0);
        const labels = new Array(bins);

        for (const v of values) {
            const idx = Math.min(Math.floor((v - min) / binWidth), bins - 1);
            counts[idx]++;
        }
        for (let i = 0; i < bins; i++) {
            labels[i] = (min + (i + 0.5) * binWidth).toFixed(2);
        }

        const datasets = [{
            data: counts,
            backgroundColor: (color || PALETTE[0]) + 'CC',
            borderColor: color || PALETTE[0],
            borderWidth: 1,
            barPercentage: 1.0,
            categoryPercentage: 1.0,
        }];

        const plugins = [];
        if (thresholdLine !== null) {
            plugins.push({
                id: 'thresholdLine',
                afterDatasetsDraw(chart) {
                    const { ctx, scales: { x } } = chart;
                    const binIdx = Math.max(0, Math.min(bins - 1, (thresholdLine - min) / binWidth));
                    const xPos = x.getPixelForValue(binIdx);
                    ctx.save();
                    ctx.strokeStyle = '#EE6677';
                    ctx.lineWidth = 2;
                    ctx.setLineDash([5, 3]);
                    ctx.beginPath();
                    ctx.moveTo(xPos, chart.chartArea.top);
                    ctx.lineTo(xPos, chart.chartArea.bottom);
                    ctx.stroke();
                    ctx.restore();
                }
            });
        }

        if (trueLine !== undefined && trueLine !== null) {
            plugins.push({
                id: 'trueLine',
                afterDatasetsDraw(chart) {
                    const { ctx, scales: { x } } = chart;
                    const binIdx = Math.max(0, Math.min(bins - 1, (trueLine - min) / binWidth));
                    const xPos = x.getPixelForValue(binIdx);
                    ctx.save();
                    ctx.strokeStyle = '#222222';
                    ctx.lineWidth = 3;
                    ctx.beginPath();
                    ctx.moveTo(xPos, chart.chartArea.top);
                    ctx.lineTo(xPos, chart.chartArea.bottom);
                    ctx.stroke();
                    // Draw a label at the top
                    ctx.fillStyle = '#222222';
                    ctx.font = 'bold 11px Inter, sans-serif';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'bottom';
                    ctx.fillText('TRUE', xPos, chart.chartArea.top - 4);
                    ctx.restore();
                }
            });
        }

        return new Chart(canvas, {
            type: 'bar',
            data: { labels, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                layout: { padding: { top: 15 } },
                plugins: {
                    legend: { display: false },
                    title: { display: !!title, text: title, font: { size: 13, weight: '700' }, padding: { bottom: 12 } },
                },
                scales: pnasScales(xLabel, yLabel || 'Count', 'category'),
            },
            plugins,
        });
    }

    /* --- Scatter plot (observed vs predicted) --- */
    function scatter(canvasId, observed, predicted, { title, xLabel, yLabel, color } = {}) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return null;

        const data = [];
        for (let i = 0; i < observed.length; i++) {
            data.push({ x: observed[i], y: predicted[i] });
        }

        // Subsample for performance (max 2000 points)
        const step = Math.max(1, Math.floor(data.length / 2000));
        const subsampled = data.filter((_, i) => i % step === 0);

        // 1:1 line
        let minV = Infinity, maxV = -Infinity;
        for (const d of data) {
            if (d.x < minV) minV = d.x;
            if (d.x > maxV) maxV = d.x;
            if (d.y < minV) minV = d.y;
            if (d.y > maxV) maxV = d.y;
        }

        return new Chart(canvas, {
            type: 'scatter',
            data: {
                datasets: [
                    {
                        label: 'Data',
                        data: subsampled,
                        backgroundColor: (color || PALETTE[0]) + '55',
                        borderColor: (color || PALETTE[0]) + '99',
                        borderWidth: 0.5,
                        pointRadius: 2,
                    },
                    {
                        label: '1:1 line',
                        data: [{ x: minV, y: minV }, { x: maxV, y: maxV }],
                        type: 'line',
                        borderColor: '#4B5563',
                        borderWidth: 1.5,
                        borderDash: [5, 3],
                        pointRadius: 0,
                        fill: false,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: { display: false },
                    title: { display: !!title, text: title, font: { size: 13, weight: '700' }, padding: { bottom: 12 } },
                },
                scales: {
                    ...pnasScales(xLabel || 'Observed', yLabel || 'Predicted'),
                    x: { ...pnasScales(xLabel || 'Observed', '').x, min: minV, max: maxV },
                    y: { ...pnasScales('', yLabel || 'Predicted').y, min: minV, max: maxV },
                },
            },
        });
    }

    /* --- Box plot (using bar chart with error bars via custom plugin) --- */
    function boxSummary(canvasId, data, { title, xLabels, yLabel, colors } = {}) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return null;

        // data: array of { median, q1, q3, min, max, mean }
        const medians = data.map(d => d.median);
        const errorBars = data.map(d => ({
            yMin: d.q1,
            yMax: d.q3,
        }));

        return new Chart(canvas, {
            type: 'bar',
            data: {
                labels: xLabels || data.map((_, i) => `Class ${i}`),
                datasets: [{
                    label: 'Median',
                    data: medians,
                    backgroundColor: (colors || CLASS_PALETTE).map(c => c + 'CC'),
                    borderColor: colors || CLASS_PALETTE,
                    borderWidth: 1,
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: { display: false },
                    title: { display: !!title, text: title, font: { size: 13, weight: '700' }, padding: { bottom: 12 } },
                },
                scales: pnasScales('', yLabel || 'Accuracy', 'category'),
            },
            plugins: [{
                id: 'errorBars',
                afterDatasetsDraw(chart) {
                    const { ctx, scales: { y } } = chart;
                    const meta = chart.getDatasetMeta(0);
                    ctx.save();
                    ctx.strokeStyle = '#1F2937';
                    ctx.lineWidth = 1.5;

                    data.forEach((d, i) => {
                        const bar = meta.data[i];
                        if (!bar) return;
                        const x = bar.x;
                        const yMin = y.getPixelForValue(d.q1);
                        const yMax = y.getPixelForValue(d.q3);
                        const yLow = y.getPixelForValue(d.low);
                        const yHigh = y.getPixelForValue(d.high);

                        // Whiskers
                        ctx.beginPath();
                        ctx.moveTo(x, yLow);
                        ctx.lineTo(x, yHigh);
                        ctx.stroke();

                        // Caps
                        const capW = 6;
                        ctx.beginPath();
                        ctx.moveTo(x - capW, yLow);
                        ctx.lineTo(x + capW, yLow);
                        ctx.stroke();
                        ctx.beginPath();
                        ctx.moveTo(x - capW, yHigh);
                        ctx.lineTo(x + capW, yHigh);
                        ctx.stroke();
                    });

                    ctx.restore();
                }
            }],
        });
    }

    /* --- Simple line chart (e.g. for cumulative distributions) --- */
    function line(canvasId, xValues, yValues, { title, xLabel, yLabel, color } = {}) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return null;

        const data = xValues.map((x, i) => ({ x, y: yValues[i] }));

        return new Chart(canvas, {
            type: 'line',
            data: {
                datasets: [{
                    data,
                    borderColor: color || PALETTE[0],
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: {
                        target: 'origin',
                        above: (color || PALETTE[0]) + '15',
                    },
                    tension: 0.2,
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: { display: false },
                    title: { display: !!title, text: title, font: { size: 13, weight: '700' }, padding: { bottom: 12 } },
                },
                scales: pnasScales(xLabel, yLabel),
            },
        });
    }

    /* --- Destroy chart if exists --- */
    function destroy(chart) {
        if (chart && typeof chart.destroy === 'function') chart.destroy();
    }

    /* --- Compute summary stats for an array --- */
    function summaryStats(values) {
        const sorted = Float64Array.from(values).sort();
        const n = sorted.length;
        if (n === 0) return { mean: 0, median: 0, q1: 0, q3: 0, low: 0, high: 0, ci95: [0, 0] };

        const mean = sorted.reduce((a, b) => a + b, 0) / n;
        const median = n % 2 ? sorted[(n - 1) / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
        const q1 = sorted[Math.floor(n * 0.25)];
        const q3 = sorted[Math.floor(n * 0.75)];
        const low = sorted[Math.floor(n * 0.025)];
        const high = sorted[Math.floor(n * 0.975)];

        return { mean, median, q1, q3, low, high, ci95: [low, high] };
    }

    return {
        applyDefaults,
        histogram,
        scatter,
        boxSummary,
        line,
        destroy,
        summaryStats,
        PALETTE,
        CLASS_PALETTE,
    };
})();

if (typeof module !== 'undefined') module.exports = PNASCharts;
