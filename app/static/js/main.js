/**
 * main.js — CropSense Frontend Logic v3
 * Adds: theme toggle, SHAP panel rendering, confidence-tier banner
 */
"use strict";

/* ── Theme handling ─────────────────────────────────────────────── */
const THEME_KEY = 'cropsense-theme';
function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  try { localStorage.setItem(THEME_KEY, theme); } catch (e) {}
}
function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem(THEME_KEY); } catch (e) {}
  const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  applyTheme(saved || (prefersDark ? 'dark' : 'light'));
}
initTheme();

const themeToggle = document.getElementById('themeToggle');
if (themeToggle) {
  themeToggle.addEventListener('click', () => {
    const cur = document.documentElement.getAttribute('data-theme');
    applyTheme(cur === 'dark' ? 'light' : 'dark');
  });
}

/* ── Sample data presets ────────────────────────────────────────── */
const SAMPLES = {
  rice:       { N: 90,  P: 42, K: 43,  temperature: 20.87, humidity: 82.0, ph: 6.5, rainfall: 202.93 },
  maize:      { N: 77,  P: 52, K: 17,  temperature: 22.61, humidity: 65.0, ph: 5.8, rainfall: 82.10  },
  cotton:     { N: 118, P: 48, K: 20,  temperature: 25.39, humidity: 80.0, ph: 6.9, rainfall: 73.10  },
  mango:      { N: 20,  P: 27, K: 30,  temperature: 31.50, humidity: 50.0, ph: 5.7, rainfall: 97.56  },
  // Edge case: ambiguous input near class boundary → triggers low-confidence banner
  ambiguous:  { N: 50,  P: 40, K: 40,  temperature: 27.00, humidity: 70.0, ph: 6.5, rainfall: 100.00 },
};

const CROP_EMOJI = {
  rice: '🌾', maize: '🌽', wheat: '🌾', chickpea: '🫘',
  kidneybeans: '🫘', pigeonpeas: '🫘', mothbeans: '🫘',
  mungbean: '🫘', blackgram: '🫘', lentil: '🫘',
  pomegranate: '🍎', banana: '🍌', mango: '🥭', grapes: '🍇',
  watermelon: '🍉', muskmelon: '🍈', apple: '🍎', orange: '🍊',
  papaya: '🍑', coconut: '🥥', cotton: '🌱', jute: '🌿', coffee: '☕',
};

/* ── DOM refs (only on the predictor page) ─────────────────────── */
const form = document.getElementById('predictionForm');

if (form) {
  const submitBtn      = document.getElementById('submitBtn');
  const btnLabel       = submitBtn.querySelector('.btn-label');
  const btnIcon        = submitBtn.querySelector('.btn-icon');
  const btnLoader      = submitBtn.querySelector('.btn-loader');
  const resultPanel    = document.getElementById('resultPanel');
  const resultSuccess  = document.getElementById('resultSuccess');
  const resultError    = document.getElementById('resultError');
  const resultCrop     = document.getElementById('resultCrop');
  const resultEmoji    = document.getElementById('resultEmoji');
  const confidencePct  = document.getElementById('confidencePct');
  const confidenceFill = document.getElementById('confidenceFill');
  const confidenceMeta = document.getElementById('confidenceMeta');
  const top3List       = document.getElementById('top3List');
  const errorMsg       = document.getElementById('errorMsg');
  const banner         = document.getElementById('confidenceBanner');
  const bannerEmoji    = document.getElementById('bannerEmoji');
  const bannerMessage  = document.getElementById('bannerMessage');
  const shapSection    = document.getElementById('shapSection');
  const shapBars       = document.getElementById('shapBars');

  /* ── Sample buttons ──────────────────────────────────────────── */
  document.querySelectorAll('.sample-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const data = SAMPLES[btn.dataset.sample];
      if (!data) return;
      Object.entries(data).forEach(([field, val]) => {
        const input = document.getElementById(field);
        if (input) { input.value = val; input.classList.remove('is-invalid'); }
      });
    });
  });

  /* ── Validation ──────────────────────────────────────────────── */
  const BOUNDS = {
    N: [0,200], P: [0,200], K: [0,250],
    temperature: [0,50], humidity: [0,100], ph: [0,14], rainfall: [0,3000],
  };
  function validateForm() {
    let valid = true;
    const payload = {};
    Object.keys(BOUNDS).forEach(f => {
      const input = document.getElementById(f);
      const val = parseFloat(input.value);
      const [lo, hi] = BOUNDS[f];
      if (isNaN(val) || val < lo || val > hi) { input.classList.add('is-invalid'); valid = false; }
      else { input.classList.remove('is-invalid'); payload[f] = val; }
    });
    return valid ? payload : null;
  }

  /* ── Loading state ───────────────────────────────────────────── */
  function setLoading(on) {
    submitBtn.disabled = on;
    btnLabel.textContent = on ? 'Analysing...' : 'Predict Crop';
    btnIcon.hidden = on;
    btnLoader.hidden = !on;
  }

  /* ── Render: confidence banner ───────────────────────────────── */
  function renderBanner(tier) {
    banner.className = 'confidence-banner tier-' + tier.tier;
    bannerEmoji.textContent = tier.emoji;
    bannerMessage.textContent = tier.message;
  }

  /* ── Render: SHAP bars ───────────────────────────────────────── */
  function renderShap(contributions) {
    shapBars.innerHTML = '';
    if (!contributions || !contributions.length) {
      shapSection.hidden = true;
      return;
    }
    shapSection.hidden = false;

    const maxAbs = Math.max(...contributions.map(c => Math.abs(c.contribution))) || 1;

    contributions.forEach(c => {
      const pct = (Math.abs(c.contribution) / maxAbs) * 50;  // 50% of bar = 100% of max contribution
      const isPos = c.direction === 'positive';
      const sign = isPos ? '+' : '';
      const row = document.createElement('div');
      row.className = 'shap-bar-row';
      row.setAttribute('role', 'listitem');
      row.innerHTML = `
        <span class="shap-bar-feature">${c.label}</span>
        <div class="shap-bar-track">
          <div class="shap-bar-center"></div>
          <div class="shap-bar-fill ${c.direction}" style="width: ${pct}%"></div>
        </div>
        <span class="shap-bar-value ${c.direction}">${sign}${c.contribution.toFixed(3)}</span>
      `;
      shapBars.appendChild(row);
    });
  }

  /* ── Render: result success ──────────────────────────────────── */
  function showResult(data) {
    resultPanel.hidden = false;
    resultSuccess.hidden = false;
    resultError.hidden = true;

    const crop = data.crop || 'Unknown';
    resultCrop.textContent = crop;
    const emojiKey = crop.toLowerCase().replace(/\s+/g, '');
    resultEmoji.textContent = CROP_EMOJI[emojiKey] || '🌱';

    // Confidence + margin
    const conf = data.confidence ?? null;
    if (conf !== null) {
      confidencePct.textContent = conf.toFixed(2) + '%';
      confidenceFill.style.width = '0%';
      requestAnimationFrame(() => { confidenceFill.style.width = Math.min(conf, 100) + '%'; });

      if (data.margin !== undefined && data.margin !== null) {
        confidenceMeta.textContent = `${data.margin.toFixed(2)}% gap to second-place prediction`;
      }
    } else {
      confidencePct.textContent = 'N/A';
      confidenceMeta.textContent = '';
    }

    // Confidence banner (NEW)
    if (data.confidence_tier) {
      banner.style.display = '';
      renderBanner(data.confidence_tier);
    } else {
      banner.style.display = 'none';
    }

    // SHAP feature contributions (NEW)
    renderShap(data.feature_contributions);

    // Top-3
    top3List.innerHTML = '';
    (data.top3 || []).forEach((item, idx) => {
      const el = document.createElement('div');
      el.className = 'top3-item';
      el.setAttribute('role', 'listitem');
      el.innerHTML = `
        <span class="top3-rank">#${idx + 1}</span>
        <span class="top3-name">${item.crop}</span>
        <div class="top3-bar-wrap"><div class="top3-bar-fill" style="width:0%"></div></div>
        <span class="top3-pct">${item.probability.toFixed(2)}%</span>
      `;
      top3List.appendChild(el);
      requestAnimationFrame(() => {
        el.querySelector('.top3-bar-fill').style.width = Math.min(item.probability, 100) + '%';
      });
    });

    resultPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function showError(msg) {
    resultPanel.hidden = false;
    resultSuccess.hidden = true;
    resultError.hidden = false;
    errorMsg.textContent = msg;
    resultPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  /* ── Form submit ─────────────────────────────────────────────── */
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const payload = validateForm();
    if (!payload) return;

    setLoading(true);
    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (res.ok && data.status === 'success') {
        showResult(data);
      } else {
        showError(data.message || 'Prediction failed.');
      }
    } catch (err) {
      console.error(err);
      showError('Cannot reach the server. Make sure the Flask app is running.');
    } finally {
      setLoading(false);
    }
  });

  /* ── Clear invalid state on input ────────────────────────────── */
  document.querySelectorAll('input[type=number]').forEach(input => {
    input.addEventListener('input', () => input.classList.remove('is-invalid'));
  });

  /* ── Keyboard: press Enter anywhere to submit ────────────────── */
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.target.tagName !== 'BUTTON' && !e.target.disabled) {
      const tgt = e.target.tagName;
      if (tgt === 'INPUT') { e.preventDefault(); form.requestSubmit(); }
    }
  });
}
