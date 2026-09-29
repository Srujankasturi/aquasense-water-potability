/**
 * AquaSense AI - Shared Client-Side Application Logic
 *
 * - Live predictions come from the Flask backend (/api/predict): the verdict and probability
 *   are produced by the trained ML model.
 * - When the backend is unreachable, a rules-only OFFLINE ESTIMATE is used instead. It is built
 *   from the same WHO/EPA thresholds as the backend and is clearly labelled as "not ML".
 */

// WHO / EPA reference guidelines. Mirrors WATER_STANDARDS in app.py (the backend is the source of
// truth and overrides this copy through loadLiveMetadata()).
//   critical_*   -> breach (strictly beyond the limit)
//   acceptable_* -> marginal, not a breach
const WATER_STANDARDS = {
  ph: { name: 'pH Level', short_name: 'pH Level', unit: 'pH', decimals: 2,
        critical_min: 6.5, critical_max: 8.5, acceptable_min: 6.8, acceptable_max: 8.2,
        critical_penalty: 15, acceptable_penalty: 8 },
  hardness: { name: 'Hardness', short_name: 'Hardness', unit: 'mg/L', decimals: 1,
        critical_max: 330, acceptable_max: 280, critical_penalty: 15, acceptable_penalty: 8,
        critical_label: 'Extreme Hardness (> 330 mg/L)' },
  solids: { name: 'Total Solids (TDS)', short_name: 'Total Solids', unit: 'ppm', decimals: 0,
        critical_max: 26000, acceptable_max: 20000, critical_penalty: 24, acceptable_penalty: 10,
        critical_label: 'High Total Dissolved Solids (> 26,000 ppm)' },
  chloramines: { name: 'Chloramines', short_name: 'Chloramines', unit: 'ppm', decimals: 2,
        critical_max: 8.5, acceptable_max: 7.5, critical_penalty: 22, acceptable_penalty: 8,
        critical_label: 'High Chloramines / Disinfection Excess (> 8.5 ppm)' },
  sulfate: { name: 'Sulfate', short_name: 'Sulfate', unit: 'mg/L', decimals: 1,
        critical_max: 450, acceptable_max: 350, critical_penalty: 22, acceptable_penalty: 10,
        critical_label: 'Elevated Sulfates (> 450 mg/L)' },
  conductivity: { name: 'Conductivity', short_name: 'Conductivity', unit: 'μS/cm', decimals: 1,
        critical_max: 650, acceptable_max: null, critical_penalty: 14, acceptable_penalty: 0,
        critical_label: 'High Electrical Conductivity (> 650 μS/cm)' },
  organic: { name: 'Organic Carbon', short_name: 'Org. Carbon', unit: 'ppm', decimals: 2,
        critical_max: 20.0, acceptable_max: 16.0, critical_penalty: 16, acceptable_penalty: 8,
        critical_label: 'Elevated Total Organic Carbon (> 20 ppm)' },
  trihalo: { name: 'Trihalomethanes', short_name: 'Trihalomethanes', unit: 'μg/L', decimals: 2,
        critical_max: 80.0, acceptable_max: null, critical_penalty: 18, acceptable_penalty: 0,
        critical_label: 'Exceeds Trihalomethanes Limit (EPA 80 μg/L)' },
  turbidity: { name: 'Turbidity', short_name: 'Turbidity', unit: 'NTU', decimals: 2,
        critical_max: 5.0, acceptable_max: 4.0, critical_penalty: 22, acceptable_penalty: 8,
        critical_label: 'High Turbidity / Cloudiness (> 5.0 NTU)' }
};

const RO_PARAMS = ['solids', 'sulfate', 'conductivity'];
const GAC_PARAMS = ['turbidity', 'organic', 'trihalo'];

// Champion model key. The backend chooses it (highest cross-validated F1) and reports it in /api/metadata;
// this constant is only the last-resort default when the backend has never been reachable.
const FALLBACK_CHAMPION_KEY = 'random_forest';

function cachedChampionKey() {
  try {
    const meta = JSON.parse(localStorage.getItem('aquasense_metadata') || 'null');
    if (meta && meta.model_key) return meta.model_key;
  } catch (e) { /* storage unavailable */ }
  return FALLBACK_CHAMPION_KEY;
}

let championPromise = null;
/** Resolves to the backend's champion model key (cached metadata or the fallback if offline). */
function getChampionKey() {
  if (!championPromise) {
    championPromise = loadLiveMetadata().then(meta => (meta && meta.model_key) || cachedChampionKey());
  }
  return championPromise;
}

const OFFLINE_MODEL_NAME = 'Offline estimate (rules only – not ML)';
const DEFAULT_DISCLAIMER = 'This prediction is an ML-based screening aid and should not replace certified laboratory water testing.';
const OFFLINE_DISCLAIMER = 'Backend unreachable: this is a rules-only estimate from WHO/EPA thresholds, NOT an ML prediction. Start the server for model predictions.';

// Preset samples. 'clean' and 'tap' are real rows from dataset/water_potability.csv (Potability = 1) that the
// trained models classify as potable; 'industrial' is a synthetic worst case.
const SAMPLE_PRESETS = {
  clean: {
    name: 'Clean River Reserve',
    desc: 'Real dataset sample (labelled potable), all parameters within guidelines',
    ph: 7.72,
    hardness: 208.44,
    solids: 17248.62,
    chloramines: 7.69,
    sulfate: 286.4,
    conductivity: 269.01,
    organic: 11.76,
    trihalo: 56.88,
    turbidity: 3.22
  },
  tap: {
    name: 'Treated Municipal Tap',
    desc: 'Real dataset sample (labelled potable), marginal TDS and chloramines',
    ph: 7.17,
    hardness: 203.41,
    solids: 20401.1,
    chloramines: 7.68,
    sulfate: 287.09,
    conductivity: 315.55,
    organic: 14.53,
    trihalo: 74.41,
    turbidity: 3.94
  },
  industrial: {
    name: 'Industrial Runoff Stream',
    desc: 'Elevated solids, chloramines & turbidity',
    ph: 4.80,
    hardness: 380,
    solids: 42000,
    chloramines: 11.8,
    sulfate: 520,
    conductivity: 890,
    organic: 26.5,
    trihalo: 124.0,
    turbidity: 8.6
  }
};

/**
 * Parse a number, treating only blank / non-numeric input as missing.
 * Unlike `parseFloat(v) || fallback`, an entered 0 is kept as 0.
 */
function parseNum(value, fallback) {
  if (value === null || value === undefined || String(value).trim() === '') return fallback;
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

/**
 * Classify one parameter against WATER_STANDARDS.
 * Returns { status: 'optimal' | 'acceptable' | 'critical', label } (label only for critical).
 */
function evaluateParameter(key, value) {
  const std = WATER_STANDARDS[key];
  const v = Number(value);
  if (!std || !Number.isFinite(v)) return { status: 'unknown', label: null };

  const below = std.critical_min !== undefined && v < std.critical_min;
  if (below || v > std.critical_max) {
    const label = key === 'ph' ? (below ? 'Acidic pH (< 6.5)' : 'Alkaline pH (> 8.5)') : std.critical_label;
    return { status: 'critical', label };
  }
  const belowAcc = std.acceptable_min !== undefined && v < std.acceptable_min;
  const aboveAcc = std.acceptable_max !== null && std.acceptable_max !== undefined && v > std.acceptable_max;
  if (belowAcc || aboveAcc) return { status: 'acceptable', label: null };
  return { status: 'optimal', label: null };
}

/**
 * Get threshold status for any single water parameter (live form validation badges).
 */
function getParameterStatus(key, val) {
  const empty = String(val ?? '').trim() === '';
  const v = Number(val);
  if (empty || !Number.isFinite(v) || !WATER_STANDARDS[key]) {
    return { status: 'unknown', text: 'Enter value', color: 'text-on-surface-variant', bg: 'bg-surface-container-high' };
  }
  const { status, label } = evaluateParameter(key, v);
  if (status === 'critical') {
    return { status, text: label, color: 'text-error', bg: 'bg-error-container text-on-error-container' };
  }
  if (status === 'acceptable') {
    return { status, text: 'Marginal', color: 'text-secondary', bg: 'bg-surface-container-high text-secondary' };
  }
  return { status, text: 'Optimal', color: 'text-tertiary', bg: 'bg-tertiary-fixed text-on-tertiary-fixed' };
}

/**
 * Guideline evaluation shared by the offline estimate (same rules as evaluate_parameters in app.py).
 * Returns { violations, parameters, score, flagged }.
 */
function evaluateGuidelines(inputs) {
  const violations = [];
  const parameters = [];
  const flagged = new Set();
  let score = 94;

  for (const [key, std] of Object.entries(WATER_STANDARDS)) {
    const value = inputs[key];
    const { status, label } = evaluateParameter(key, value);
    if (status === 'critical') {
      violations.push(label);
      flagged.add(key);
      score -= key === 'ph' ? Math.min(35, std.critical_penalty + Math.abs(7.0 - value) * 10) : std.critical_penalty;
    } else if (status === 'acceptable') {
      score -= std.acceptable_penalty;
    }
    parameters.push({
      key,
      name: std.short_name,
      value: Number(value.toFixed(std.decimals)),
      unit: std.unit,
      status
    });
  }

  // Synergistic chemical hazard penalties
  if (inputs.chloramines > 7.2 && (inputs.ph < 6.8 || inputs.ph > 8.2)) {
    violations.push('Chloramine / pH Equilibrium Disruption');
    flagged.add('ph');
    score -= 10;
  }
  if (inputs.solids > 22000 && inputs.sulfate > 360) {
    violations.push('Synergistic Mineral Salinity (TDS + Sulfate)');
    flagged.add('solids'); flagged.add('sulfate');
    score -= 12;
  }
  if (inputs.organic > 15 && inputs.trihalo > 65) {
    violations.push('Disinfection Byproduct Precursor Hazard (TOC + THM)');
    flagged.add('organic'); flagged.add('trihalo');
    score -= 14;
  }

  score = Math.max(5, Math.min(98, Math.round(score)));
  return { violations, parameters, score, flagged };
}

/**
 * Treatment recommendations (mirrors generate_recommendations in app.py).
 */
function buildRecommendations(isPotable, isOffline, violations, flagged) {
  const recs = [];
  const has = (list) => list.some(k => flagged.has(k));

  if (isPotable && violations.length === 0) {
    recs.push({
      title: 'Conforms to Reference Guidelines',
      desc: isOffline
        ? 'All parameters conform to WHO & EPA reference thresholds (rules-only estimate; verify with certified lab testing).'
        : 'The model classifies this sample as potable and all parameters conform to WHO & EPA reference thresholds (laboratory confirmation advised).',
      icon: 'task_alt',
      type: 'success'
    });
    recs.push({
      title: 'Routine Disinfection Monitoring',
      desc: 'Maintain residual disinfectant between 0.2 and 4.0 ppm to prevent secondary bacterial contamination during storage.',
      icon: 'sanitizer',
      type: 'info'
    });
    return recs;
  }

  if (isPotable) {
    recs.push({
      title: 'Guideline Warning · Model Predicts Potable',
      desc: `The model predicts this sample is potable, but ${violations.length} WHO/EPA guideline threshold(s) are exceeded. Confirm with laboratory testing before consumption.`,
      icon: 'warning',
      type: 'warning'
    });
  } else {
    recs.push({
      title: 'Screening Alert · Corrective Treatment Advised',
      desc: violations.length
        ? `${isOffline ? 'The estimate flags this sample as not potable' : 'The model classifies this sample as not potable'} and ${violations.length} guideline threshold(s) are exceeded. Treatment recommended before consumption.`
        : 'This sample is classified as not potable, although no single guideline threshold is exceeded. Laboratory testing and filtration recommended.',
      icon: 'dangerous',
      type: 'danger'
    });
  }

  if (has(RO_PARAMS)) {
    recs.push({
      title: 'Reverse Osmosis (RO) Demineralization',
      desc: 'Total Dissolved Solids, sulfates or conductivity exceed guideline levels. Demineralize using multi-stage RO membranes.',
      icon: 'filter_alt',
      type: 'warning'
    });
  }
  if (has(GAC_PARAMS)) {
    recs.push({
      title: 'Coagulation & Activated Carbon Filtering',
      desc: 'Reduce suspended colloidal particles and organic carbon precursors with granulated activated carbon (GAC) filtering.',
      icon: 'layers',
      type: 'warning'
    });
  }
  if (flagged.has('ph')) {
    recs.push({
      title: 'pH Neutralization Buffering',
      desc: 'Inject stabilizing buffering agents (calcite/soda ash) to return pH to the reference 6.5 – 8.5 range.',
      icon: 'tune',
      type: 'warning'
    });
  }
  return recs;
}

/**
 * Persist the latest result for cross-page use and append it to the audit history (deduped by runId).
 */
function saveResult(result) {
  try {
    localStorage.setItem('aquasense_prediction', JSON.stringify(result));
    sessionStorage.setItem('aquasense_prediction', JSON.stringify(result));

    const history = JSON.parse(localStorage.getItem('aquasense_history') || '[]')
      .filter(h => h.runId !== result.runId);
    history.unshift({
      runId: result.runId,
      timestamp: result.timestamp,
      isPotable: result.isPotable,
      score: result.score,
      confidence: result.confidence,
      potablePct: result.potablePct,
      model: result.modelName,
      ph: result.params?.ph,
      solids: result.params?.solids,
      turbidity: result.params?.turbidity
    });
    localStorage.setItem('aquasense_history', JSON.stringify(history.slice(0, 15)));
  } catch (e) {
    console.warn('Storage unavailable:', e);
  }
}

/**
 * Rules-only OFFLINE ESTIMATE from WHO/EPA thresholds. NOT an ML prediction: used only when the
 * backend cannot be reached. Verdict = no critical breach and guideline score >= 60; the
 * "potable %" shown is the guideline compliance score.
 */
function evaluateWaterQuality(inputs, modelKey = null, saveToStorage = true) {
  modelKey = modelKey || cachedChampionKey();
  const values = {
    ph: parseNum(inputs.ph, 7.0),
    hardness: parseNum(inputs.hardness, 180),
    solids: parseNum(inputs.solids, 15000),
    chloramines: parseNum(inputs.chloramines, 7.0),
    sulfate: parseNum(inputs.sulfate, 300),
    conductivity: parseNum(inputs.conductivity, 400),
    organic: parseNum(inputs.organic ?? inputs.carbon, 10.0),
    trihalo: parseNum(inputs.trihalo ?? inputs.thm, 60.0),
    turbidity: parseNum(inputs.turbidity, 3.0)
  };

  const { violations, parameters, score, flagged } = evaluateGuidelines(values);
  const criticalCount = parameters.filter(p => p.status === 'critical').length;
  const isPotable = criticalCount === 0 && score >= 60;
  const potablePct = score;
  const riskPct = 100 - score;

  const result = {
    engine: 'offline-rules',
    isLiveBackend: false,
    isOffline: true,
    inputs: values,
    params: values,
    prediction: isPotable ? 1 : 0,
    label: isPotable ? 'Potable' : 'Not Potable',
    probability: potablePct / 100,
    isPotable,
    score,
    confidence: isPotable ? potablePct : riskPct,
    risk: riskPct,
    potablePct,
    riskPct,
    probabilities: { potable: potablePct, non_potable: riskPct },
    flags: violations,
    violations,
    parameters,
    recommendations: buildRecommendations(isPotable, true, violations, flagged),
    modelName: OFFLINE_MODEL_NAME,
    modelKey,
    disclaimer: OFFLINE_DISCLAIMER,
    timestamp: new Date().toISOString(),
    runId: 'OFFLINE-' + Math.floor(1000 + Math.random() * 9000)
  };

  if (saveToStorage) saveResult(result);
  return result;
}

/**
 * Toast Notification UI Helper
 */
function showToast(message, type = 'info') {
  let toastContainer = document.getElementById('aquasense-toast-container');
  if (!toastContainer) {
    toastContainer = document.createElement('div');
    toastContainer.id = 'aquasense-toast-container';
    toastContainer.className = 'fixed bottom-6 right-6 z-50 flex flex-col gap-2 pointer-events-none';
    document.body.appendChild(toastContainer);
  }

  const toast = document.createElement('div');
  const bgClass = type === 'success'
    ? 'bg-tertiary text-on-tertiary'
    : type === 'error'
    ? 'bg-error text-on-error'
    : 'bg-surface-container-highest text-on-surface';

  const iconName = type === 'success' ? 'check_circle' : type === 'error' ? 'error' : 'info';

  toast.className = `${bgClass} shadow-xl backdrop-blur-xl px-4 py-3 rounded-2xl flex items-center gap-2.5 font-title-md text-body-sm pointer-events-auto transition-all duration-300 transform translate-y-4 opacity-0`;

  const icon = document.createElement('span');
  icon.className = 'material-symbols-outlined text-[20px]';
  icon.textContent = iconName;
  const text = document.createElement('span');
  text.textContent = message;
  toast.append(icon, text);

  toastContainer.appendChild(toast);

  // Trigger enter animation
  setTimeout(() => {
    toast.classList.remove('translate-y-4', 'opacity-0');
  }, 10);

  // Remove after delay
  setTimeout(() => {
    toast.classList.add('translate-y-4', 'opacity-0');
    setTimeout(() => { toast.remove(); }, 300);
  }, 3200);
}

/**
 * Setup Theme Toggle (Dark / Light). Safe to call more than once: the click handler is bound a
 * single time per button (previously a double bind made each click toggle the theme twice).
 */
function setupThemeToggle() {
  const toggleBtn = document.querySelector('[aria-label="Toggle light mode"]');
  if (!toggleBtn || toggleBtn.dataset.themeBound === 'true') return;
  toggleBtn.dataset.themeBound = 'true';

  const iconHtml = (isDark) =>
    `<span class="material-symbols-outlined text-[20px]">${isDark ? 'dark_mode' : 'light_mode'}</span>`;

  const currentTheme = localStorage.getItem('theme');
  const startDark = currentTheme === 'dark' || (!currentTheme && window.matchMedia('(prefers-color-scheme: dark)').matches);
  document.documentElement.classList.toggle('dark', startDark);
  toggleBtn.innerHTML = iconHtml(startDark);

  toggleBtn.addEventListener('click', () => {
    const isDark = document.documentElement.classList.toggle('dark');
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
    toggleBtn.innerHTML = iconHtml(isDark);
    showToast(`Switched to ${isDark ? 'Dark' : 'Light'} Mode`, 'info');
  });
}

// Auto-initialize theme on script execution
document.addEventListener('DOMContentLoaded', () => {
  setupThemeToggle();
});

/**
 * Flask API Integration
 * Calls /api/predict. Only network failures, timeouts and 5xx responses fall back to the offline
 * estimate; a 400 (invalid input) is returned to the caller as { error: true, fieldErrors }.
 */
const API_BASE_URL = (window.location.protocol.startsWith('http'))
  ? window.location.origin
  : 'http://127.0.0.1:5000';

async function predictWaterQualityLive(inputs, modelKey = null, opts = {}) {
  modelKey = modelKey || await getChampionKey();
  const saveToStorage = opts.saveToStorage !== false;
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 8000);

    const response = await fetch(`${API_BASE_URL}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...inputs, model: modelKey }),
      signal: controller.signal
    });

    clearTimeout(timeoutId);

    if (response.status === 400) {
      const body = await response.json().catch(() => ({}));
      return { error: true, message: body.error || 'Invalid input', fieldErrors: body.errors || {} };
    }

    if (response.ok) {
      const data = await response.json();
      if (data.success) {
        const diag = data.diagnostic || {};
        const isPotable = data.label === 'Potable';
        const probs = diag.probabilities || {};
        const potablePct = probs.potable ?? Math.round((data.probability ?? 0) * 100);
        const riskPct = probs.non_potable ?? (100 - potablePct);

        const result = {
          engine: data.engine || 'ml',
          isLiveBackend: true,
          isOffline: false,
          inputs: data.inputs || inputs,
          params: data.inputs || inputs,
          prediction: isPotable ? 1 : 0,
          label: isPotable ? 'Potable' : 'Not Potable',
          probability: data.probability ?? (potablePct / 100),
          isPotable,
          score: diag.score,
          confidence: diag.confidence ?? (isPotable ? potablePct : riskPct),
          risk: diag.risk ?? riskPct,
          potablePct,
          riskPct,
          probabilities: { potable: potablePct, non_potable: riskPct },
          flags: data.violations || [],
          violations: data.violations || [],
          parameters: data.parameters || [],
          recommendations: data.recommendations || [],
          modelName: data.model_name || 'ML classifier',
          modelKey: data.model_key || modelKey,
          disclaimer: data.disclaimer || DEFAULT_DISCLAIMER,
          timestamp: data.timestamp,
          runId: data.run_id
        };
        if (saveToStorage) saveResult(result);
        return result;
      }
    }
    console.info('[AquaSense] Backend returned status', response.status, '- using offline estimate');
  } catch (err) {
    // Backend offline or running purely as a static file - fall back to the labelled offline estimate
    console.info('[AquaSense] Backend unreachable, using offline estimate:', err.message);
  }

  return evaluateWaterQuality(inputs, modelKey, saveToStorage);
}

/**
 * Fetch and cache live model metadata from backend /api/metadata.
 * Also adopts the backend's WATER_STANDARDS so thresholds stay in sync.
 */
async function loadLiveMetadata() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/metadata`);
    if (res.ok) {
      const data = await res.json();
      if (data.success && data.metadata) {
        if (data.metadata.standards) {
          for (const [key, std] of Object.entries(data.metadata.standards)) {
            if (WATER_STANDARDS[key]) Object.assign(WATER_STANDARDS[key], std);
          }
        }
        localStorage.setItem('aquasense_metadata', JSON.stringify(data.metadata));
        return data.metadata;
      }
    }
  } catch (e) {
    console.warn('[AquaSense] Could not fetch live metadata:', e);
  }
  return null;
}
