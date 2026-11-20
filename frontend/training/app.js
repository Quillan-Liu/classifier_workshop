'use strict';
const form = document.querySelector('#training-form');
// 原生表单校验发现折叠区中的必填项时，先展开，便于定位并修正。
form.addEventListener('invalid', event => {
  let panel = event.target.closest('details');
  while (panel) { panel.open = true; panel = panel.parentElement.closest('details'); }
}, true);
const model = document.querySelector('#model');
const purpose = document.querySelector('#purpose');
const parameters = document.querySelector('#parameters');
const start = document.querySelector('#start');
const reset = document.querySelector('#reset');
const status = document.querySelector('#status');
const retry = document.querySelector('#retry');
const names = {n_estimators:'决策树数量',n_jobs:'并行任务数',random_state:'随机种子',tokenizer:'分词方式',epochs:'训练轮数',lr:'学习率',dim:'向量维度',word_ngrams:'词 N-gram',bucket:'哈希桶数量',minn:'最小字符 N-gram',maxn:'最大字符 N-gram',batch_size:'训练批大小',eval_batch_size:'评估批大小',max_length:'最大序列长度',warmup_steps:'预热步数',weight_decay:'权重衰减',logging_steps:'日志间隔',seed:'随机种子'};
Object.assign(names, {autotune:'是否自动调超参', autotune_duration:'自动搜索时长（秒）', thread:'训练线程数（1 为单线程）', verbose:'日志详细程度', autotuneModelSize:'目标模型大小（如 100M）'});
Object.assign(names, {tfidf_max_features:'TF-IDF 最大特征数', train_validation_size:'训练集中的验证比例', trees_per_step:'每批新增树数', max_rounds:'最大训练轮数', early_stop_accuracy:'早停验证准确率'});
const rfHints = {n_estimators:'每轮新增树数；总上限 = 每轮树数 × 最大轮数。', tfidf_max_features:'留空表示不限制，保留全部非停用词特征。', train_validation_size:'从训练数据划分验证集；若数据已有 dev 分组，则沿用已有分组。', trees_per_step:'每批加树并更新进度；每轮结束计算指标。', max_rounds:'持续增加树，保存验证准确率最高轮次的模型。', early_stop_accuracy:'轮末验证准确率达到此值即停止；0.92 表示 92%。', random_state:'同时用于数据划分和随机森林。', n_jobs:'-1 使用全部 CPU 核心；1 为单任务。'};
let defaults = {}, options = [], version = 0, activeJob = null, busy = false, ready = false;
function trainingModeControls() {
  const hint = document.querySelector('#parameter-hint');
  const automatic = model.value === 'fasttext' && parameters.querySelector('[data-key="autotune"]')?.checked;
  hint.hidden = !busy && !automatic;
  hint.textContent = busy ? '当前任务已提交，参数已锁定。训练结束后可修改并再次提交。' : automatic ? '自动调参已开启，手动超参数不参与训练。取消勾选“自动调参”即可手动修改并使用这些参数。' : '';
  if (model.value !== 'fasttext') return;
  for (const input of parameters.querySelectorAll('[data-key]')) {
    const key = input.dataset.key;
    input.disabled = busy || (['autotune_duration','thread','verbose','autotuneModelSize'].includes(key) ? !automatic : automatic && !['autotune','tokenizer'].includes(key));
  }
}
function message(text, error = false) { status.textContent = text; status.className = error ? 'error' : ''; }
async function api(url, init) {
  const response = await fetch(url, init);
  let data;
  try { data = await response.json(); } catch { throw new Error(`服务返回异常（HTTP ${response.status}），请通过训练服务打开此页面。`); }
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail || data));
  return data;
}
function controls() {
  start.disabled = busy || !ready;
  model.disabled = busy || !options.length;
  purpose.disabled = busy || !purpose.options.length;
  reset.disabled = busy || !ready;
  for (const input of form.querySelectorAll('input')) input.disabled = busy;
  for (const input of parameters.querySelectorAll('select')) input.disabled = busy;
  trainingModeControls();
}
function render() {
  parameters.replaceChildren();
  form.elements.dev_ratio.closest('details').hidden = model.value === 'rf';
  for (const [key, value] of Object.entries(defaults)) {
    const label = document.createElement('label');
    label.append(`${model.value === 'rf' && key === 'n_estimators' ? '每轮新增树数' : names[key] || key} · ${key}`);
    const input = document.createElement(key === 'tokenizer' ? 'select' : 'input');
    input.dataset.key = key;
    if (key === 'tokenizer') {
      input.add(new Option('字符级（char）', 'char')); input.add(new Option('词级（word）', 'word')); input.value = value;
    } else if (typeof value === 'boolean') { input.type = 'checkbox'; input.checked = value; }
    else {
      input.type = typeof value === 'number' || key === 'tfidf_max_features' ? 'number' : 'text'; input.value = value ?? ''; input.required = key !== 'tfidf_max_features';
      if (input.type === 'number') {
        input.step = ['lr','weight_decay'].includes(key) || (key === 'epochs' && model.value === 'bert') ? 'any' : '1';
        input.min = key === 'n_jobs' ? '-1' : ['lr','epochs'].includes(key) ? '0.000000001' : ['n_estimators','dim','bucket','batch_size','eval_batch_size','max_length','logging_steps','word_ngrams','autotune_duration','thread'].includes(key) ? '1' : '0';
      }
      if (key === 'autotuneModelSize') { input.pattern = '[1-9][0-9]*[kKmMgG]?'; input.title = '输入正整数大小，如 100M'; }
      if (['train_validation_size','early_stop_accuracy'].includes(key)) {
        input.step = 'any'; input.min = key === 'train_validation_size' ? '0.000000001' : '0';
        input.max = key === 'train_validation_size' ? '0.999999999' : '1';
      }
      if (['tfidf_max_features','trees_per_step','max_rounds'].includes(key)) input.min = '1';
      if (key === 'random_state') input.max = '4294967295';
    }
    label.append(input); parameters.append(label);
    if (model.value === 'rf' && rfHints[key]) {
      const hint = document.createElement('small'); hint.textContent = rfHints[key]; label.append(hint);
    }
  }
  document.querySelector('#summary-params').textContent = `${Object.keys(defaults).length} 项 · config 默认`;
  trainingModeControls();
}
async function loadDefaults() {
  const current = ++version; ready = false; parameters.replaceChildren(); controls();
  const selected = options.find(item => item.value === model.value);
  document.querySelector('#summary-model').textContent = selected?.label || model.value;
  document.querySelector('#model-description').textContent = selected?.description || '';
  message('正在读取默认超参数…');
  try {
    const data = await api(`/models/${encodeURIComponent(model.value)}/defaults`);
    if (current !== version) return;
    defaults = data.params; render(); ready = true; message('配置已就绪，选择训练集、测试集和类别映射文件后即可开始训练。');
  } catch (error) { if (current === version) message(`加载失败：${error.message}。请切换模型重试或刷新页面。`, true); }
  finally { if (current === version) controls(); }
}
model.addEventListener('change', loadDefaults);
reset.addEventListener('click', () => { render(); message('已恢复当前模型的默认超参数。'); });
parameters.addEventListener('input', () => {
  trainingModeControls();
  const count = [...parameters.querySelectorAll('[data-key]')].filter(input => (input.dataset.key === 'tfidf_max_features' && input.value === '' ? null : input.type === 'checkbox' ? input.checked : input.type === 'number' ? Number(input.value) : input.value) !== defaults[input.dataset.key]).length;
  document.querySelector('#summary-params').textContent = `${Object.keys(defaults).length} 项 · ${count ? `已修改 ${count} 项` : 'config 默认'}`;
});
for (const key of ['train_data', 'test_data', 'label_mapping']) {
  form.elements[key].addEventListener('change', event => {
    const file = event.target.files[0];
    event.target.nextElementSibling.textContent = file ? `${file.name} · ${(file.size / 1024).toFixed(1)} KB` : '尚未选择文件';
    document.querySelector({train_data:'#summary-train', test_data:'#summary-test', label_mapping:'#summary-mapping'}[key]).textContent = file?.name || '未选择';
  });
}
function showProgress(job) {
  const panel = document.querySelector('#training-progress');
  const bar = document.querySelector('#training-progress-bar');
  const label = document.querySelector('#training-progress-label');
  const progress = job.progress || {};
  const finished = progress.finished || job.status === 'succeeded';
  panel.hidden = finished || job.status === 'failed';
  const hasIterations = progress.total_iterations > 0;
  const percent = hasIterations ? 100 * progress.iteration / progress.total_iterations : progress.percent;
  bar.max = 100;
  if (finished) bar.value = 100;
  else if (typeof percent === 'number') bar.value = Math.max(0, Math.min(100, percent));
  else bar.removeAttribute('value');
  const count = hasIterations ? `${percent.toFixed(1)}%` : typeof percent === 'number' ? `原生训练进度 ${percent.toFixed(1)}%` : '';
  const phase = job.status === 'failed' ? '训练失败' : finished ? (job.status === 'succeeded' ? '训练完成' : '训练已结束，正在评估和保存') : progress.phase === 'autotune' ? '自动调参搜索中' : count ? '训练中' : job.status === 'queued' ? '排队中' : '正在上传或准备数据';
  label.textContent = `${phase}${count ? ` · ${count}` : ''}`;
  return percent;
}
function showJob(job) {
  const percent = showProgress(job);
  const labels = {queued:'排队中',running:'训练中',succeeded:'训练完成',failed:'训练失败'};
  const state = job.status === 'running' && job.progress?.finished ? '正在评估和保存' : job.status === 'running' && typeof percent === 'number' ? `训练中 ${percent.toFixed(1)}%` : labels[job.status] || job.status;
  message(`${state} · 任务 ${job.id}${job.error ? `：${job.error}` : ''}`, job.status === 'failed');
  document.querySelector('#logs-panel').hidden = !job.log?.length;
  const logs = document.querySelector('#logs');
  const atBottom = logs.scrollHeight - logs.scrollTop - logs.clientHeight < 40;
  logs.replaceChildren();
  for (const entry of job.log || []) {
    const item = document.createElement('li');
    const lines = entry.split('\n');
    const title = document.createElement('strong'); title.textContent = lines[0]; item.append(title);
    for (const line of lines.slice(1)) {
      const detail = document.createElement('p'); detail.textContent = line.trim(); item.append(detail);
    }
    logs.append(item);
  }
  if (atBottom) logs.scrollTop = logs.scrollHeight;
  if (typeof job.metrics?.selected_epoch === 'number') {
    status.textContent += ` · 最终产物：第 ${job.metrics.selected_epoch} 轮训练的模型`;
  }
  const metrics = document.querySelector('#metrics'); metrics.replaceChildren();
  const columns = Object.entries({accuracy:'准确率',precision:'精确率',recall:'召回率',f1:'F1 分数',loss:'Loss'})
    .filter(([key]) => ['train', 'dev', 'test'].some(split => typeof (job.metrics?.[split] || (split === 'test' ? job.metrics : null))?.[key] === 'number'));
  if (!columns.length) return;
  for (const [split, label] of Object.entries({train:'训练集',dev:'验证集',test:'测试集'})) {
    const scores = job.metrics?.[split] || (split === 'test' ? job.metrics : null);
    if (!columns.some(([key]) => typeof scores?.[key] === 'number')) continue;
    const group = document.createElement('div'); group.className = 'metric-group';
    const heading = document.createElement('h3'); heading.textContent = label; group.append(heading);
    const values = document.createElement('dl');
    for (const [key, title] of columns) {
      const item = document.createElement('div');
      const term = document.createElement('dt'); term.textContent = title;
      const value = document.createElement('dd');
      value.textContent = typeof scores?.[key] !== 'number' ? '—' : key === 'loss' ? scores[key].toFixed(4) : `${(scores[key] * 100).toFixed(2)}%`;
      item.append(term); item.append(value); values.append(item);
    }
    group.append(values); metrics.append(group);
  }
}
async function poll() {
  retry.hidden = true;
  try {
    const job = await api(`/jobs/${encodeURIComponent(activeJob)}`); showJob(job);
    if (['queued','running'].includes(job.status)) setTimeout(poll, 2000);
    else { busy = false; controls(); }
  } catch (error) { message(`任务 ${activeJob} 状态查询失败：${error.message}。可重新查询。`, true); retry.hidden = false; }
}
retry.addEventListener('click', poll);
form.addEventListener('submit', async event => {
  event.preventDefault();
  if (busy || !ready || !form.reportValidity()) return;
  const params = {};
  for (const input of parameters.querySelectorAll('[data-key]')) {
    if (input.disabled) continue;
    const key = input.dataset.key;
    params[key] = key === 'tfidf_max_features' ? (input.value === '' ? null : Number(input.value)) : input.type === 'checkbox' ? input.checked : typeof defaults[key] === 'number' ? Number(input.value) : input.value;
    if (key === 'n_jobs' && params[key] === 0) { message('并行任务数不能为 0；使用 -1 表示全部 CPU 核心。', true); input.focus(); return; }
  }
  if ('minn' in params && params.maxn < params.minn) { message('最大字符 N-gram 不能小于最小字符 N-gram。', true); return; }
  for (const key of ['train_data','test_data']) {
    const file = form.elements[key].files[0];
    if (!file?.size || !/\.(arrow|jsonl|txt|tsv|csv|zip)$/i.test(file.name)) { message('请选择非空且受支持的数据集文件。', true); return; }
  }
  const mapping = form.elements.label_mapping.files[0];
  if (!mapping?.size || !/\.(json|txt|tsv)$/i.test(mapping.name)) { message('请选择非空的 JSON、TXT 或 TSV 类别映射文件。', true); return; }
  const body = new FormData(form); body.set('params', JSON.stringify(params)); body.set('sync', 'false');
  busy = true; controls(); retry.hidden = true;
  document.querySelector('#metrics').replaceChildren(); document.querySelector('#logs-panel').hidden = true;
  showProgress({status: 'uploading'});
  message('正在上传数据集并提交训练，请稍候…');
  try {
    const job = await api('/train', {method:'POST', body}); activeJob = job.id; showJob(job); poll();
  } catch (error) { showProgress({status: 'failed'}); message(`提交失败：${error.message}`, true); busy = false; controls(); }
});
async function init() {
  try {
    const [models, registry] = await Promise.all([api('/models'), api('/model-registry-options')]);
    options = models.options;
    for (const item of options) model.add(new Option(item.label, item.value));
    purpose.add(new Option('请选择训练用途', ''));
    for (const item of registry.purposes) purpose.add(new Option(item.label, item.value));
    await loadDefaults();
  } catch (error) { message(`初始化失败：${error.message}。请刷新重试。`, true); }
}
init();
