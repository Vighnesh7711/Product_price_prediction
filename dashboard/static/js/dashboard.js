/* ══════════════════════════════════════════════════════════════
   PriceOracle  —  Dashboard JS  (v4)
   ══════════════════════════════════════════════════════════════ */

// ── DOM refs ─────────────────────────────────────────────────────
const $ = (s) => document.querySelector(s);
const urlForm = $('#url-form');
const urlInput = $('#url-input');
const analyzeBtn = $('#analyze-btn');
const manualForm = $('#manual-form');
const manualToggle = $('#manual-toggle');
const errorMsg = $('#error-msg');
const resultsEl = $('#results');
const heroEl = $('#hero');

let priceChart = null;
let categoryChart = null;
let gaugeChart = null;

// ── chart colours ────────────────────────────────────────────────
const C = {
  indigo: 'rgba(129,140,248,', violet: 'rgba(167,139,250,',
  sky: 'rgba(56,189,248,', green: 'rgba(52,211,153,',
  amber: 'rgba(251,191,36,', red: 'rgba(248,113,113,',
  pink: 'rgba(244,114,182,',
};

// ── events ───────────────────────────────────────────────────────
manualToggle.addEventListener('click', e => {
  e.preventDefault();
  manualForm.classList.toggle('hidden');
});

urlForm.addEventListener('submit', async e => {
  e.preventDefault();
  const url = urlInput.value.trim();
  if (!url) return;
  await analyze('/api/analyze', { url });
});

manualForm.addEventListener('submit', async e => {
  e.preventDefault();
  await analyze('/api/manual-analyze', {
    name: $('#m-name').value.trim(),
    price: parseFloat($('#m-price').value),
    description: $('#m-desc').value.trim(),
  });
});

// ── main analyse call ────────────────────────────────────────────
async function analyze(endpoint, body) {
  showLoading(true);
  hideError();
  resultsEl.classList.add('hidden');

  try {
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json();

    // Handle scrape failure gracefully
    if (data.scrape_failed) {
      showScrapeFailed(data.fail_reason);
      return;
    }

    if (!res.ok) throw new Error(data.error || 'Something went wrong.');
    render(data);
  } catch (err) {
    showError(err.message);
  } finally {
    showLoading(false);
  }
}

function showScrapeFailed(reason) {
  showLoading(false);
  errorMsg.innerHTML = `
    <div style="display:flex;align-items:flex-start;gap:.75rem">
      <span style="font-size:1.4rem;flex-shrink:0">🛡️</span>
      <div>
        <strong>Scraping blocked</strong><br/>
        <span style="opacity:.85">${reason || 'The site blocked our automated request.'}</span><br/>
        <span style="opacity:.85">Enter the product details below and we'll still predict the price!</span>
      </div>
    </div>`;
  errorMsg.classList.remove('hidden');
  errorMsg.style.background = 'rgba(251,191,36,.10)';
  errorMsg.style.borderColor = 'rgba(251,191,36,.3)';
  errorMsg.style.color = '#fbbf24';
  // Auto-open manual form
  manualForm.classList.remove('hidden');
  manualForm.scrollIntoView({ behavior: 'smooth', block: 'center' });
}


// ── render everything ────────────────────────────────────────────
function render(data) {
  const { product, predictions, similar_products, category_stats, category_comparison } = data;

  // shrink hero
  heroEl.style.minHeight = 'auto';
  heroEl.style.paddingTop = '3rem';
  heroEl.style.paddingBottom = '2rem';

  // ── Product Card ───────────────────────────────
  const img = product.image || '';
  $('#product-img').src = img || 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200"><rect fill="%23111827" width="200" height="200"/><text x="50%" y="50%" fill="%23475569" font-family="sans-serif" font-size="14" text-anchor="middle" dy=".35em">No Image</text></svg>';
  $('#product-img').alt = product.name || '';
  $('#product-name').textContent = product.name || 'Unknown Product';
  $('#product-desc').textContent = product.description || '';
  $('#product-price').textContent = product.price ? `₹${fmt(product.price)}` : '';
  $('#product-original').textContent = product.original_price ? `₹${fmt(product.original_price)}` : '';
  $('#product-category').textContent = product.detected_category || '';
  $('#product-rating').textContent = product.rating ? `★ ${product.rating}` : '';
  $('#source-badge').textContent = product.source || '';

  // ── Extra product details ──────────────────────
  renderProductExtra(product);

  // ── Specifications ─────────────────────────────
  renderSpecs(product.specifications || {});

  // ── Metrics + Charts ──────────────────────────
  if (predictions && !predictions.error) {
    $('#mv-7d').textContent = `₹${fmt(predictions.predicted_7d)}`;
    $('#mc-7d-pct').textContent = `${predictions.week_change_pct > 0 ? '+' : ''}${predictions.week_change_pct}%`;
    $('#mc-7d-pct').className = `metric-change ${predictions.week_change_pct > 0 ? 'up' : predictions.week_change_pct < 0 ? 'down' : 'flat'}`;

    $('#mv-30d').textContent = `₹${fmt(predictions.predicted_30d)}`;
    $('#mc-30d-pct').textContent = `${predictions.month_change_pct > 0 ? '+' : ''}${predictions.month_change_pct}%`;
    $('#mc-30d-pct').className = `metric-change ${predictions.month_change_pct > 0 ? 'up' : predictions.month_change_pct < 0 ? 'down' : 'flat'}`;

    const trendIcons = { up: '📈 Rising', down: '📉 Falling', stable: '📊 Stable' };
    $('#mv-trend').textContent = trendIcons[predictions.trend] || predictions.trend;

    const confMap = { high: '🟢 High', medium: '🟡 Medium', low: '🔴 Low' };
    $('#mv-conf').textContent = confMap[predictions.confidence] || predictions.confidence;

    // Model used
    $('#mv-model').textContent = predictions.model_used || '—';

    // Buy / Wait indicator
    renderBuyWait(predictions, product);

    drawPriceChart(predictions);
    renderInsights(predictions.insights || []);
  }

  // Price position gauge
  if (category_stats && category_stats.avg_price && product.price) {
    drawGaugeChart(product.price, category_stats);
  }

  // similar products
  renderSimilar(similar_products || []);

  // category comparison chart
  if (category_comparison && Object.keys(category_comparison).length) {
    drawCategoryChart(category_comparison);
  }

  // category stats
  if (category_stats && Object.keys(category_stats).length) {
    renderCatStats(category_stats, product.detected_category);
  }

  resultsEl.classList.remove('hidden');
  resultsEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── render extra product fields ──────────────────────────────────
function renderProductExtra(product) {
  const el = $('#product-extra');
  if (!el) return;
  const items = [];

  if (product.brand) items.push(`<span class="extra-chip"><strong>Brand:</strong> ${esc(product.brand)}</span>`);
  if (product.reviews) items.push(`<span class="extra-chip"><strong>Reviews:</strong> ${Number(product.reviews).toLocaleString()}</span>`);
  if (product.category_path) items.push(`<span class="extra-chip"><strong>Category:</strong> ${esc(product.category_path)}</span>`);
  if (product.category_confidence) items.push(`<span class="extra-chip"><strong>Confidence:</strong> ${(product.category_confidence * 100).toFixed(0)}%</span>`);
  if (product.url && product.source !== 'manual') items.push(`<span class="extra-chip"><a href="${esc(product.url)}" target="_blank" rel="noopener" style="color:var(--accent);">🔗 View on ${esc(product.source || 'website')}</a></span>`);

  el.innerHTML = items.length ? items.join('') : '';
}

// ── render specifications table ──────────────────────────────────
function renderSpecs(specs) {
  const section = $('#specs-section');
  const body = $('#specs-body');
  if (!body || !section) return;

  const entries = Object.entries(specs || {});
  if (!entries.length) {
    section.style.display = 'none';
    return;
  }

  section.style.display = '';
  body.innerHTML = entries.map(([k, v]) => `
    <tr>
      <td>${esc(k)}</td>
      <td>${esc(v)}</td>
    </tr>`).join('');
}

// ══════════════════════════════════════════════════════════════════
//  BUY / WAIT INDICATOR
// ══════════════════════════════════════════════════════════════════
function renderBuyWait(predictions, product) {
  const card = $('#buy-wait-card');
  const iconEl = $('#bw-icon');
  const labelEl = $('#bw-label');
  const reasonEl = $('#bw-reason');
  const confEl = $('#bw-confidence');
  if (!card) return;

  const monthPct = predictions.month_change_pct || 0;
  const weekPct  = predictions.week_change_pct || 0;
  const trend    = predictions.trend || 'stable';
  const conf     = predictions.confidence || 'low';

  // Decision logic: if prices are expected to rise → buy now, if dropping → wait
  let isBuy = false;
  let reason = '';

  if (trend === 'up' || monthPct > 2) {
    isBuy = true;
    reason = `Prices are predicted to rise ~${Math.abs(monthPct).toFixed(1)}% over the next 30 days. Buying now could save you ₹${fmt(Math.abs(predictions.predicted_30d - product.price))}.`;
  } else if (trend === 'down' || monthPct < -2) {
    isBuy = false;
    reason = `Prices are predicted to drop ~${Math.abs(monthPct).toFixed(1)}% over the next 30 days. Waiting could save you ₹${fmt(Math.abs(product.price - predictions.predicted_30d))}.`;
  } else {
    // Stable — check week trend
    if (weekPct > 1) {
      isBuy = true;
      reason = 'Short-term prices show a slight upward trend. Buying now is a safe bet.';
    } else if (weekPct < -1) {
      isBuy = false;
      reason = 'Short-term prices show a slight dip. Waiting a week might get you a better deal.';
    } else {
      isBuy = true;
      reason = 'Prices are stable with no significant change expected. Good time to buy at current price.';
    }
  }

  // Update card
  card.classList.remove('bw-buy', 'bw-wait');
  card.classList.add(isBuy ? 'bw-buy' : 'bw-wait');
  iconEl.textContent = isBuy ? '🛒' : '⏳';
  labelEl.textContent = isBuy ? 'BUY NOW' : 'WAIT';
  labelEl.className = `bw-label ${isBuy ? 'buy' : 'wait'}`;
  reasonEl.textContent = reason;

  const confColors = { high: '#34d399', medium: '#fbbf24', low: '#f87171' };
  const confLabels = { high: '🟢 High', medium: '🟡 Medium', low: '🔴 Low' };
  confEl.textContent = confLabels[conf] || conf;
  confEl.style.color = confColors[conf] || '#94a3b8';
}

// ══════════════════════════════════════════════════════════════════
//  PRICE POSITION GAUGE CHART
// ══════════════════════════════════════════════════════════════════
function drawGaugeChart(currentPrice, catStats) {
  if (gaugeChart) gaugeChart.destroy();

  const avg = catStats.avg_price || 0;
  const min = catStats.min_price || avg * 0.3;
  const max = catStats.max_price || avg * 3;
  const range = max - min || 1;

  // Normalize current price to 0-100 within the range
  const pct = Math.max(0, Math.min(100, ((currentPrice - min) / range) * 100));

  // Determine color segments: Low (green) | Fair (amber) | High (red)
  const lowEnd = 35;
  const midEnd = 65;

  // Segments: what % of the gauge is filled vs empty
  const segments = [lowEnd, midEnd - lowEnd, 100 - midEnd];
  const segColors = ['rgba(52,211,153,.6)', 'rgba(251,191,36,.6)', 'rgba(248,113,113,.6)'];

  // Position label
  let posLabel = 'Fair Price';
  let posColor = '#fbbf24';
  if (pct < lowEnd) { posLabel = 'Below Average — Great Deal!'; posColor = '#34d399'; }
  else if (pct > midEnd) { posLabel = 'Above Average — Pricey'; posColor = '#f87171'; }

  const ctx = $('#gauge-chart').getContext('2d');

  gaugeChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Low Range', 'Mid Range', 'High Range'],
      datasets: [
        {
          data: segments,
          backgroundColor: segColors,
          borderWidth: 0,
          circumference: 180,
          rotation: 270,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      cutout: '75%',
      plugins: {
        legend: { display: false },
        tooltip: { enabled: false },
      },
    },
    plugins: [{
      id: 'gaugeNeedle',
      afterDatasetDraw(chart) {
        const { ctx: c, chartArea } = chart;
        const cx = (chartArea.left + chartArea.right) / 2;
        const cy = chartArea.bottom - 10;
        const outerR = Math.min(chartArea.right - chartArea.left, chartArea.bottom - chartArea.top) * 0.48;

        // Needle angle: 180° arc from left (π) to right (0), mapped to pct
        const angle = Math.PI + (pct / 100) * Math.PI;
        const needleLen = outerR * 0.85;
        const nx = cx + Math.cos(angle) * needleLen;
        const ny = cy + Math.sin(angle) * needleLen;

        // Draw needle
        c.save();
        c.beginPath();
        c.moveTo(cx - 2, cy);
        c.lineTo(nx, ny);
        c.lineTo(cx + 2, cy);
        c.fillStyle = '#e2e8f0';
        c.fill();

        // Center dot
        c.beginPath();
        c.arc(cx, cy, 6, 0, Math.PI * 2);
        c.fillStyle = '#818cf8';
        c.fill();

        // Price label in center
        c.textAlign = 'center';
        c.fillStyle = posColor;
        c.font = 'bold 14px Inter, sans-serif';
        c.fillText(posLabel, cx, cy - 22);
        c.fillStyle = '#e2e8f0';
        c.font = 'bold 20px Inter, sans-serif';
        c.fillText(`₹${Number(currentPrice).toLocaleString('en-IN')}`, cx, cy - 2);
        c.restore();
      },
    }],
  });

  // Labels
  $('#gauge-min').textContent = `₹${fmt(min)}`;
  $('#gauge-current').textContent = '';
  $('#gauge-max').textContent = `₹${fmt(max)}`;
}

// ── price chart ──────────────────────────────────────────────────
function drawPriceChart(pred) {
  if (priceChart) priceChart.destroy();

  const histDates = (pred.historical?.dates || []).slice(-60);
  const histPrices = (pred.historical?.prices || []).slice(-60);
  const futureDates = pred.dates || [];
  const futurePrices = pred.predictions || [];
  const upperBound = pred.upper_bound || [];
  const lowerBound = pred.lower_bound || [];

  const allLabels = [...histDates, ...futureDates].map(d => {
    const dt = new Date(d);
    return dt.toLocaleDateString('en-IN', { day: 'numeric', month: 'short' });
  });

  const histData = [...histPrices, ...new Array(futureDates.length).fill(null)];
  const futureData = [...new Array(histDates.length).fill(null), pred.current_price, ...futurePrices.slice(1)];
  const upData = [...new Array(histDates.length).fill(null), ...upperBound];
  const loData = [...new Array(histDates.length).fill(null), ...lowerBound];

  // connect line: last hist point joins first future point
  if (histPrices.length && futurePrices.length) {
    futureData[histDates.length - 1] = histPrices[histPrices.length - 1] || pred.current_price;
  }

  const ctx = $('#price-chart').getContext('2d');
  priceChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: allLabels,
      datasets: [
        {
          label: 'Historical Price',
          data: histData,
          borderColor: C.indigo + '1)',
          backgroundColor: C.indigo + '.08)',
          fill: true,
          tension: .35,
          pointRadius: 0,
          borderWidth: 2,
        },
        {
          label: 'Predicted Price',
          data: futureData,
          borderColor: C.green + '1)',
          backgroundColor: C.green + '.08)',
          fill: true,
          tension: .35,
          pointRadius: 0,
          borderWidth: 2.5,
          borderDash: [6, 3],
        },
        {
          label: 'Upper Bound',
          data: upData,
          borderColor: C.amber + '.3)',
          backgroundColor: 'transparent',
          fill: false,
          tension: .35,
          pointRadius: 0,
          borderWidth: 1,
          borderDash: [3, 3],
        },
        {
          label: 'Lower Bound',
          data: loData,
          borderColor: C.sky + '.3)',
          backgroundColor: 'transparent',
          fill: '-1',
          tension: .35,
          pointRadius: 0,
          borderWidth: 1,
          borderDash: [3, 3],
        },
      ],
    },
    options: chartOpts('Price (₹)'),
  });
}

// ── category comparison chart ────────────────────────────────────
function drawCategoryChart(comp) {
  if (categoryChart) categoryChart.destroy();
  const colors = [C.indigo, C.violet, C.sky, C.green, C.amber];
  const datasets = [];
  let labels = [];

  Object.entries(comp).forEach(([cat, d], i) => {
    const c = colors[i % colors.length];
    const vals = (d.price_index || []).filter(v => v !== null);
    const dates = (d.dates || []).slice(-vals.length);
    if (dates.length > labels.length) {
      labels = dates.map(x => {
        const dt = new Date(x);
        return dt.toLocaleDateString('en-IN', { month: 'short', year: '2-digit' });
      });
    }
    datasets.push({
      label: cat,
      data: vals.slice(-labels.length),
      borderColor: c + '1)',
      backgroundColor: 'transparent',
      tension: .35,
      pointRadius: 0,
      borderWidth: 2,
    });
  });

  const ctx = $('#category-chart').getContext('2d');
  categoryChart = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: chartOpts('Price Index (base = 100)'),
  });
}

function chartOpts(yLabel) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { labels: { color: '#94a3b8', font: { family: "'Inter'" } } },
      tooltip: {
        backgroundColor: '#1e1b4b',
        titleColor: '#e2e8f0',
        bodyColor: '#cbd5e1',
        borderColor: 'rgba(99,102,241,.3)',
        borderWidth: 1,
        padding: 12,
        callbacks: {
          label: ctx => {
            const v = ctx.parsed.y;
            return v != null ? `${ctx.dataset.label}: ₹${fmt(v)}` : '';
          },
        },
      },
    },
    scales: {
      x: { ticks: { color: '#64748b', maxTicksLimit: 12, font: { size: 10 } }, grid: { color: 'rgba(255,255,255,.04)' } },
      y: { ticks: { color: '#64748b', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,.06)' }, title: { display: true, text: yLabel, color: '#94a3b8' } },
    },
  };
}

// ── insights ─────────────────────────────────────────────────────
function renderInsights(arr) {
  const el = $('#insights-grid');
  el.innerHTML = arr.map((ins, i) => `
    <div class="insight-card ${ins.type}" style="animation-delay:${i * .08}s">
      <span class="icon">${ins.icon}</span>
      <span class="text">${ins.text}</span>
    </div>`).join('');
}

// ── similar products ─────────────────────────────────────────────
function renderSimilar(products) {
  const el = $('#similar-grid');
  if (!products.length) { el.innerHTML = '<p style="color:var(--text-dim)">No similar products found.</p>'; return; }
  el.innerHTML = products.map(p => {
    const disc = p.original_price && p.price ? Math.round((1 - p.price / p.original_price) * 100) : 0;
    const priceChange = p.first_price && p.last_price ? ((p.last_price - p.first_price) / p.first_price * 100).toFixed(1) : null;
    return `
    <div class="similar-card glass-card">
      <div class="s-name">${esc(p.name)}</div>
      <div class="s-brand">${esc(p.brand)} · ${esc(p.category)}</div>
      <div class="s-price">₹${fmt(p.price)}${disc > 0 ? ` <span style="font-size:.75rem;color:var(--green)">(${disc}% off)</span>` : ''}</div>
      <div class="s-meta">
        <span>${p.rating ? '★ ' + p.rating : ''}</span>
        <span>${priceChange !== null ? (priceChange > 0 ? '↑' : '↓') + ' ' + Math.abs(priceChange) + '% trend' : ''}</span>
      </div>
    </div>`;
  }).join('');
}

// ── category stats ───────────────────────────────────────────────
function renderCatStats(stats, catName) {
  const el = $('#cat-stats');
  el.innerHTML = `
    <div class="metric-card glass-card">
      <span class="metric-label">Products Analysed</span>
      <span class="metric-value">${(stats.total_products || 0).toLocaleString()}</span>
    </div>
    <div class="metric-card glass-card">
      <span class="metric-label">Avg Price (${catName})</span>
      <span class="metric-value">₹${fmt(stats.avg_price)}</span>
    </div>
    <div class="metric-card glass-card">
      <span class="metric-label">Avg Discount</span>
      <span class="metric-value">${stats.avg_discount_pct || 0}%</span>
    </div>
    <div class="metric-card glass-card">
      <span class="metric-label">Avg Rating</span>
      <span class="metric-value">★ ${stats.avg_rating || '—'}</span>
    </div>`;
}

// ── helpers ──────────────────────────────────────────────────────
function fmt(n) { return n != null ? Number(n).toLocaleString('en-IN', { maximumFractionDigits: 0 }) : '—'; }
function esc(s) { const d = document.createElement('div'); d.textContent = s || ''; return d.innerHTML; }

function showLoading(on) {
  analyzeBtn.querySelector('.btn-text').classList.toggle('hidden', on);
  analyzeBtn.querySelector('.btn-loader').classList.toggle('hidden', !on);
  analyzeBtn.disabled = on;
}
function showError(msg) {
  errorMsg.textContent = msg;
  errorMsg.classList.remove('hidden');
}
function hideError() { errorMsg.classList.add('hidden'); }
