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
  if (event.ctrlKey && event.key === 'Enter' && !$('collectedDialog').open) {
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

let collected = null;
let savedVariants = [];
function renderSavedVariant() {
  const index = Number($('savedPrompt').value);
  const variant = savedVariants[index];
  if (!variant) return;
  $('previousSaved').disabled = index === 0;
  $('nextSaved').disabled = index === savedVariants.length - 1;
  $('savedPromptText').textContent = variant.prompt;
  const reference = (variant.source_records || [])[0] || {};
  $('savedPromptMeta').textContent = `Variant ${index + 1} of ${savedVariants.length} · Row ${reference.row_index == null ? '—' : reference.row_index} · Suffix ${reference.suffix_id == null ? '—' : reference.suffix_id} · ${variant.results.length} saved responses`;
  $('savedCards').replaceChildren();
  for (const result of variant.results.slice().sort((a, b) => a.alpha - b.alpha)) {
    const card = document.createElement('article');
    card.className = `card${result.alpha === 0 ? ' baseline' : ''}`;
    card.innerHTML = '<div class="card-top"><span class="alpha"></span><span class="tag"></span></div><p class="response"></p><div class="metrics"></div>';
    card.querySelector('.alpha').textContent = `α = ${result.alpha}`;
    card.querySelector('.tag').textContent = result.alpha === 0 ? 'BASELINE' : 'STEERED';
    card.querySelector('.response').textContent = result.response || '(Empty response)';
    const evaluation = result.refusal_evaluation || {label:'unscored'};
    card.dataset.rating = evaluation.label === 'refusal' ? 'refusal' : '';
    card.querySelector('.tag').textContent += ` · ${evaluation.label === 'non_refusal' ? 'EST. SUCCESS' : evaluation.label.toUpperCase()}`;
    card.querySelector('.metrics').textContent = `${result.tokens} tokens · ${result.seconds.toFixed(2)}s${result.truncated ? ' · Token limit reached' : ''}${evaluation.matched_prefix ? ` · Matched “${evaluation.matched_prefix}”` : ' · No refusal prefix detected'}`;
    $('savedCards').append(card);
  }
}
function renderCollected(data) {
  collected = data;
  const groups = new Map();
  for (const result of data.results) {
    if (!groups.has(result.prompt_id)) groups.set(result.prompt_id, {...result, results: []});
    groups.get(result.prompt_id).results.push(result);
  }
  savedVariants = Array.from(groups.values());
  $('savedVariants').textContent = savedVariants.length;
  $('savedResponses').textContent = data.results.length;
  const seconds = data.collection_report && data.collection_report.invocation_seconds;
  $('savedTime').textContent = seconds == null ? '—' : `${(seconds / 60).toFixed(2)} min`;
  $('savedTruncated').textContent = data.results.filter(r => r.truncated).length;
  const messageIds = new Set();
  for (const variant of savedVariants) for (const ref of variant.source_records || []) {
    if (ref.message_id != null) messageIds.add(ref.message_id);
  }
  $('collectedStatus').textContent = `${data.complete ? 'Complete' : 'Partial collection'} · ${data.config.model} · ${data.config.max_tokens}-token cap · ${data.config.decoding} decoding`;
  $('savedContext').textContent = `These ${savedVariants.length} prompt/suffix variants${messageIds.size ? ` cover ${messageIds.size} base question${messageIds.size === 1 ? '' : 's'} and` : ''} were selected for success in an earlier baseline review. Responses are scored by the refusal-prefix heuristic. Non-refusal is estimated success, not verified compliance. Rates describe this selected subset. Token-capped responses may be incomplete.`;
  $('savedAlphaStats').replaceChildren();
  for (const alpha of data.config.alphas.slice().sort((a, b) => a - b)) {
    const rows = data.results.filter(r => r.alpha === alpha);
    const mean = field => rows.length ? (rows.reduce((sum, row) => sum + row[field], 0) / rows.length).toFixed(1) : '—';
    const tr = document.createElement('tr');
    const scored = rows.filter(r => r.refusal_evaluation && r.refusal_evaluation.label !== 'unscored');
    const rate = scored.length ? `${(100 * scored.filter(r => r.refusal_evaluation.label === 'non_refusal').length / scored.length).toFixed(1)}%` : '—';
    for (const value of [`α = ${alpha}`, rows.length, rate, mean('tokens'), rows.length ? `${mean('seconds')}s` : '—', rows.filter(r => r.truncated).length]) {
      const td = document.createElement('td'); td.textContent = value; tr.append(td);
    }
    $('savedAlphaStats').append(tr);
  }
  $('savedPrompt').replaceChildren();
  savedVariants.forEach((variant, index) => {
    const reference = (variant.source_records || [])[0] || {};
    const option = document.createElement('option'); option.value = index;
    option.textContent = `Variant ${index + 1}${reference.row_index == null ? '' : ` · row ${reference.row_index}`}${reference.suffix_id == null ? '' : ` · suffix ${reference.suffix_id}`}`;
    $('savedPrompt').append(option);
  });
  $('collectedContent').hidden = false;
  renderSavedVariant();
}
$('viewCollected').onclick = async () => {
  const dialog = $('collectedDialog');
  if (!dialog.open) dialog.showModal();
  $('collectedTitle').focus();
  $('collectedStatus').textContent = 'Loading collected results…';
  $('collectedContent').hidden = true;
  try { renderCollected(await request('/api/collected-results')); }
  catch (error) { $('collectedStatus').textContent = error.message; }
};
$('closeCollected').onclick = () => $('collectedDialog').close();
$('savedPrompt').onchange = renderSavedVariant;
$('previousSaved').onclick = () => { $('savedPrompt').value = Number($('savedPrompt').value) - 1; renderSavedVariant(); };
$('nextSaved').onclick = () => { $('savedPrompt').value = Number($('savedPrompt').value) + 1; renderSavedVariant(); };

$('downloadCollected').onclick = () => {
  if (!collected) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(collected, null, 2)], {type: 'application/json'}));
  const link = document.createElement('a'); link.href = url; link.download = 'collected-steering-results.json'; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};

async function loadSuccessComparison() {
  try {
    const data = await request('/api/jailbreak-comparison');
    const rows = data.by_alpha;
    if (!data.paired_variants) throw new Error('No complete, nonempty response pairs yet. Collect the same variants at alpha 0, 0.1, and 0.2.');
    $('successStatus').textContent = `${data.paired_variants} paired prompt/suffix variants · ${data.model} · ${data.max_tokens}-token cap${data.excluded_variants ? ` · ${data.excluded_variants} incomplete variants excluded` : ''}`;
    $('successBars').replaceChildren();
    $('successTable').replaceChildren();
    const descriptions = [];
    for (const row of rows) {
      const percent = (row.estimated_success_rate * 100).toFixed(1);
      const label = row.alpha === 0 ? 'Baseline' : `Alpha ${row.alpha}`;
      const bar = document.createElement('div');
      bar.className = `success-bar-column${row.alpha === 0 ? ' control-bar' : ''}`;
      bar.innerHTML = '<div class="bar-track"><div class="success-bar"><strong></strong></div></div><div class="bar-label"></div><div class="bar-count"></div>';
      bar.querySelector('.success-bar').style.height = `${percent}%`;
      bar.querySelector('strong').textContent = `${percent}%`;
      bar.querySelector('.bar-label').textContent = label;
      bar.querySelector('.bar-count').textContent = `${row.estimated_successes} / ${row.total}`;
      $('successBars').append(bar);
      const delta = row.alpha === 0 ? 'Reference' : `${row.change_from_baseline_pp > 0 ? '+' : ''}${row.change_from_baseline_pp.toFixed(1)} pp`;
      const tr = document.createElement('tr');
      for (const value of [label, `${row.estimated_successes} / ${row.total}`, `${percent}%`, delta]) {
        const td = document.createElement('td'); td.textContent = value; tr.append(td);
      }
      $('successTable').append(tr);
      descriptions.push(`${label}: ${percent}% (${row.estimated_successes} of ${row.total})`);
    }
    $('successChart').setAttribute('aria-label', `Estimated jailbreak success by refusal-prefix check. ${descriptions.join('; ')}.`);
    $('successCaption').textContent = 'Same 27 previously approved suffix variants of one base question at each alpha. Non-refusal proxy; selected subset, not an overall jailbreak success rate.'.replace('27', String(data.paired_variants));
    $('scorePrefixes').textContent = `Refusal openings: ${data.refusal_prefixes.join(' · ')}`;
    const baseline = rows.find(r => r.alpha === 0);
    $('baselineSuccess').textContent = baseline ? `${(baseline.estimated_success_rate * 100).toFixed(1)}%` : '—';
    $('successFigure').hidden = false;
    $('successTableWrap').hidden = false;
  } catch (error) { $('successStatus').textContent = error.message; }
}
loadSuccessComparison();
