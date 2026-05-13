/**
 * dashboard.js — populates the metrics dashboard from /api/dashboard-metrics
 */
"use strict";

(async function () {
  const summaryEl   = document.getElementById('summaryCards');
  const tableBody   = document.querySelector('#modelTable tbody');
  const featureBars = document.getElementById('featureBars');
  const classGrid   = document.getElementById('classGrid');
  const datasetGrid = document.getElementById('datasetGrid');
  const generatedAt = document.getElementById('generatedAt');

  if (!summaryEl) return;  // not on the dashboard page

  let data;
  try {
    const res = await fetch('/api/dashboard-metrics');
    if (!res.ok) {
      summaryEl.innerHTML = '<p style="color:var(--danger)">Could not load metrics. Train the model first to generate dashboard_metrics.json.</p>';
      return;
    }
    data = await res.json();
  } catch (err) {
    summaryEl.innerHTML = '<p style="color:var(--danger)">Network error loading metrics.</p>';
    return;
  }

  const best = data.model_results[data.best_model];

  // ── Summary cards ────────────────────────────────────────────
  summaryEl.innerHTML = `
    <div class="dash-stat-card">
      <div class="stat-label">Best Model</div>
      <div class="stat-value" style="font-size:1.3rem">${data.best_model}</div>
      <div class="stat-sub">Selected by macro-F1</div>
    </div>
    <div class="dash-stat-card">
      <div class="stat-label">Test Accuracy</div>
      <div class="stat-value">${best.accuracy.toFixed(1)}%</div>
      <div class="stat-sub">on ${data.dataset.test_samples} held-out samples</div>
    </div>
    <div class="dash-stat-card">
      <div class="stat-label">F1 Score</div>
      <div class="stat-value">${best.f1.toFixed(1)}%</div>
      <div class="stat-sub">macro average</div>
    </div>
    <div class="dash-stat-card">
      <div class="stat-label">CV Score</div>
      <div class="stat-value">${best.cv_mean.toFixed(1)}%</div>
      <div class="stat-sub">5-fold ±${best.cv_std.toFixed(2)}%</div>
    </div>
  `;

  // ── Model comparison table ───────────────────────────────────
  const modelEntries = Object.entries(data.model_results);
  tableBody.innerHTML = modelEntries.map(([name, r]) => {
    const cls = name === data.best_model ? 'best' : '';
    return `<tr class="${cls}">
      <td><strong>${name}</strong></td>
      <td>${r.accuracy.toFixed(2)}%</td>
      <td>${r.precision.toFixed(2)}%</td>
      <td>${r.recall.toFixed(2)}%</td>
      <td>${r.f1.toFixed(2)}%</td>
      <td>${r.cv_mean.toFixed(2)}% ± ${r.cv_std.toFixed(2)}%</td>
    </tr>`;
  }).join('');

  // ── Feature importance bars ──────────────────────────────────
  const labels = {
    'N': 'Nitrogen', 'P': 'Phosphorus', 'K': 'Potassium',
    'temperature': 'Temperature', 'humidity': 'Humidity',
    'ph': 'Soil pH', 'rainfall': 'Rainfall'
  };
  const maxImp = Math.max(...data.feature_importance.map(f => f.importance));
  featureBars.innerHTML = data.feature_importance.map(f => `
    <div class="feat-bar-row">
      <span class="feat-bar-label">${labels[f.feature] || f.feature}</span>
      <div class="feat-bar-track">
        <div class="feat-bar-fill" style="width:0%" data-target="${(f.importance / maxImp) * 100}"></div>
      </div>
      <span class="feat-bar-pct">${(f.importance * 100).toFixed(1)}%</span>
    </div>
  `).join('');
  // Animate fills
  requestAnimationFrame(() => {
    featureBars.querySelectorAll('.feat-bar-fill').forEach(el => {
      el.style.width = el.dataset.target + '%';
    });
  });

  // ── Per-class cards ──────────────────────────────────────────
  const cropEmojis = {
    apple: '🍎', banana: '🍌', blackgram: '🫘', chickpea: '🫘', coconut: '🥥',
    coffee: '☕', cotton: '🌱', grapes: '🍇', jute: '🌿', kidneybeans: '🫘',
    lentil: '🫘', maize: '🌽', mango: '🥭', mothbeans: '🫘', mungbean: '🫘',
    muskmelon: '🍈', orange: '🍊', papaya: '🍑', pigeonpeas: '🫘',
    pomegranate: '🍎', rice: '🌾', watermelon: '🍉'
  };
  classGrid.innerHTML = data.per_class
    .sort((a, b) => b.f1 - a.f1)
    .map(c => {
      const pct = c.f1 * 100;
      const colour = c.f1 >= 0.99 ? '#3A6B2D' : c.f1 >= 0.95 ? '#5C8A44' : '#C4692A';
      return `
        <div class="class-card" title="Precision ${(c.precision*100).toFixed(1)}% · Recall ${(c.recall*100).toFixed(1)}%">
          <div class="class-name">${cropEmojis[c.crop] || ''} ${c.crop}</div>
          <div class="class-bar-bg"><div class="class-bar-fill" style="width:${pct}%; background:${colour}"></div></div>
          <div class="class-stats">
            <span>F1 ${c.f1.toFixed(2)}</span>
            <span>n=${c.support}</span>
          </div>
        </div>
      `;
    }).join('');

  // ── Dataset summary ─────────────────────────────────────────
  const ds = data.dataset;
  datasetGrid.innerHTML = `
    <div class="dataset-stat"><div class="dataset-stat-label">Total Records</div><div class="dataset-stat-value">${ds.total_records.toLocaleString()}</div></div>
    <div class="dataset-stat"><div class="dataset-stat-label">Crop Classes</div><div class="dataset-stat-value">${ds.n_classes}</div></div>
    <div class="dataset-stat"><div class="dataset-stat-label">Features</div><div class="dataset-stat-value">${ds.n_features}</div></div>
    <div class="dataset-stat"><div class="dataset-stat-label">Train / Test</div><div class="dataset-stat-value">${ds.train_samples} / ${ds.test_samples}</div></div>
    <div class="dataset-stat"><div class="dataset-stat-label">Class Balance</div><div class="dataset-stat-value" style="font-size:.95rem">${ds.class_balance}</div></div>
  `;

  // ── Generated timestamp ──────────────────────────────────────
  if (generatedAt && data.generated_at) {
    const dt = new Date(data.generated_at);
    generatedAt.textContent = `Metrics generated ${dt.toLocaleString()}`;
  }
})();
