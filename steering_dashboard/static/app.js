const $ = id => document.getElementById(id);
let current = null;
let ratings = {};
let polling = false;
const ratingKey = id => `steering-ratings-${id}`;
async function request(path, options) {
  const response = await fetch(path, options);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `Request failed (${response.status})`);
  return body;
}
function showError(message) { $('error').textContent = message; $('error').hidden = false; }
function saveRatings() {
  try { sessionStorage.setItem(ratingKey(current.id), JSON.stringify(ratings)); } catch (_) { /* Export still includes ratings. */ }
  $('ratedCount').textContent = String(Object.values(ratings).filter(Boolean).length).padStart(2, '0');
}
function render(job) {
  current = job;
  $('export').disabled = !job.results.length;
  $('responseCount').textContent = String(job.results.length).padStart(2, '0');
  $('progress').textContent = job.status === 'loading' ? 'Loading model and steering direction… First load may take a few minutes.'
    : job.status === 'running' ? `Generating · ${job.results.length} of ${job.config.alphas.length} responses complete`
    : job.status === 'failed' ? `Sweep failed · ${job.results.length} responses retained`
    : `Sweep complete · ${job.results.length} responses · ${new Date(job.created_at).toLocaleTimeString()}`;
  if (job.error) showError(`${job.error} Check the terminal, model access, and direction files, then retry.`);
  const container = $('results');
  // Append completed cards without replacing selectors or moving keyboard focus.
  for (const result of job.results) {
    if (container.querySelector(`[data-alpha="${result.alpha}"]`)) continue;
    const empty = container.querySelector('.empty');
    if (empty) empty.remove();
    const card = document.createElement('article');
    card.className = `card${result.alpha === 0 ? ' baseline' : ''}`;
    card.dataset.alpha = result.alpha;
    card.innerHTML = '<div class="card-top"><span class="alpha"></span><span class="tag"></span></div><p class="response"></p><div class="metrics"></div><div class="rating"><label>Manual assessment</label><select aria-label="Manual assessment"><option value="">Unrated</option><option value="useful">Useful / coherent</option><option value="refusal">Refusal</option><option value="poor">Poor / incoherent</option><option value="mixed">Mixed / unsure</option></select></div>';
    card.querySelector('.alpha').textContent = `α = ${result.alpha}`;
    card.querySelector('.tag').textContent = result.alpha === 0 ? 'BASELINE' : 'STEERED';
    card.querySelector('.response').textContent = result.response || '(Empty response)';
    card.querySelector('.metrics').textContent = `${result.tokens == null ? '—' : result.tokens} tokens · ${result.seconds.toFixed(2)}s${result.truncated ? ' · Token limit reached' : ''}${job.demo ? ' · DEMO' : ''}`;
    const select = card.querySelector('select');
    select.setAttribute('aria-label', `Manual assessment for alpha ${result.alpha}`);
    select.value = ratings[result.alpha] || '';
    card.dataset.rating = select.value;
    select.onchange = () => { ratings[result.alpha] = select.value; card.dataset.rating = select.value; saveRatings(); };
    container.append(card);
  }
  saveRatings();
}
async function poll(id) {
  polling = true;
  $('run').disabled = true;
  try {
    while (true) {
      const job = await request(`/api/runs/${id}`);
      render(job);
      if (['complete', 'failed'].includes(job.status)) break;
      await new Promise(resolve => setTimeout(resolve, 800));
    }
  } catch (error) { showError(`${error.message} Reload this page to reconnect to the latest run.`); }
  finally { polling = false; $('run').disabled = false; }
}
let outsideRecommended = false;
function sweepAlphas() {
  const lo = Number($('alphaMin').value), hi = Number($('alphaMax').value);
  const n = Number($('points').value);
  if (!Number.isInteger(n) || n < 2 || n > 12) throw new Error('Choose between 2 and 12 points.');
  if (lo >= hi) throw new Error('Minimum alpha must be below maximum alpha.');
  return Array.from({length:n}, (_, i) => Number((lo + (hi - lo) * i / (n - 1)).toFixed(8)));
}
function updateRange(event) {
  const loInput = $('alphaMin'), hiInput = $('alphaMax');
  if (Number(loInput.value) >= Number(hiInput.value)) {
    if (event && event.target === loInput) loInput.value = (Number(hiInput.value) - .01).toFixed(2);
    else hiInput.value = (Number(loInput.value) + .01).toFixed(2);
  }
  const lo = Number(loInput.value), hi = Number(hiInput.value);
  $('minValue').textContent = lo;
  $('maxValue').textContent = hi;
  $('rangeFill').style.left = `${(lo + 2) * 25}%`;
  $('rangeFill').style.width = `${(hi - lo) * 25}%`;
  const warning = Math.abs(lo) > .5 || Math.abs(hi) > .5;
  $('alphaRange').classList.toggle('warning', warning);
  $('rangeWarning').classList.toggle('warning-text', warning);
  $('rangeWarning').textContent = warning ? 'Beyond ±0.5 · generation quality may drop.' : 'Within ±0.5 · recommended range';
  if (warning && !outsideRecommended) {
    $('alphaRange').classList.remove('shake');
    void $('alphaRange').offsetWidth;
    $('alphaRange').classList.add('shake');
  }
  outsideRecommended = warning;
  try {
    const alphas = sweepAlphas();
    const hasZero = alphas.includes(0);
    $('alphaPreview').textContent = `α: ${alphas.map(a => Number(a.toFixed(4))).join(' · ')}${hasZero ? '' : ' (+ 0 baseline)'}`;
  } catch (error) { $('alphaPreview').textContent = error.message; }
}
$('alphaMin').oninput = updateRange;
$('alphaMax').oninput = updateRange;
$('points').oninput = updateRange;
$('reset').onclick = () => {
  $('alphaMin').value = '-0.5'; $('alphaMax').value = '0.5'; $('points').value = '5'; updateRange();
};
document.addEventListener('keydown', event => {
  if (event.ctrlKey && event.key === 'Enter') {
    event.preventDefault();
    if (!polling && !$('run').disabled) $('sweepForm').requestSubmit();
  }
});
updateRange();
$('sweepForm').onsubmit = async event => {
  event.preventDefault();
  if (polling || $('run').disabled) return;
  $('error').hidden = true;
  let alphas;
  try { alphas = sweepAlphas(); } catch (error) { return showError(error.message); }
  $('run').disabled = true;
  try {
    const job = await request('/api/runs', { method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({prompt: $('prompt').value, alphas, max_tokens: Number($('tokens').value)}) });
    ratings = {};
    $('results').replaceChildren();
    render(job);
    await poll(job.id);
  } catch (error) { showError(error.message); $('run').disabled = false; }
};
$('export').onclick = () => {
  if (!current) return;
  const blob = new Blob([JSON.stringify({...current, results: current.results.map(r => ({...r, manual_rating: ratings[r.alpha] || null})),
    evaluation: {jailbreak_success_rate: null, status: 'not_connected'}}, null, 2)], {type: 'application/json'});
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a'); link.href = url; link.download = `steering-${current.id}.json`; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
async function init() {
  try {
    const status = await request('/api/status');
    $('connection').textContent = status.demo ? '● DEMO MODE' : '● LOCAL MODEL';
    $('model').textContent = status.model;
    $('demoBanner').hidden = !status.demo;
    if (status.latest_job_id) {
      const job = await request(`/api/runs/${status.latest_job_id}`);
      $('prompt').value = job.config.prompt;
      const values = job.config.alphas;
      $('alphaMin').value = Math.min(...values);
      $('alphaMax').value = Math.max(...values);
      $('points').value = Math.min(12, Math.max(2, values.length));
      updateRange();
      $('tokens').value = job.config.max_tokens;
      try { ratings = JSON.parse(sessionStorage.getItem(ratingKey(job.id)) || '{}'); } catch (_) { ratings = {}; }
      render(job);
      if (!['complete', 'failed'].includes(job.status)) await poll(job.id);
    }
  } catch (error) { $('connection').textContent = '● DISCONNECTED'; showError(error.message); }
}
init();
