const $ = id => document.getElementById(id);

// 软碰撞只改变色斑的运动，背景层不接收输入事件。
function startWallpaper() {
  const layer = document.querySelector('.ambient-wallpaper');
  if (!layer) return;
  const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const spots = [...layer.querySelectorAll('span')].map((element, index) => ({
    element, x: 0, y: 0, size: 0, r: 0,
    vx: Math.cos(index * 2.4 + .7) * 22, vy: Math.sin(index * 2.4 + .7) * 22,
  }));
  let width, height, frame = 0, previous = 0;
  function paint() {
    for (const spot of spots) spot.element.style.transform = `translate3d(${spot.x - spot.size / 2}px,${spot.y - spot.size / 2}px,0)`;
  }
  function resize() {
    width = layer.clientWidth; height = layer.clientHeight;
    spots.forEach((spot, index) => {
      spot.size = spot.element.getBoundingClientRect().width;
      spot.r = Math.min(spot.size * .30, width * .24, height * .24);
      spot.x = width * (index < 6 ? .18 + (index % 3) * .32 : index === 6 ? .38 : .62);
      spot.y = height * (index < 6 ? index < 3 ? .26 : .74 : .5);
    });
    paint();
  }
  function tick(time) {
    const dt = previous ? Math.min((time - previous) / 1000, .04) : 0;
    previous = time;
    advanceSpots(spots, width, height, dt * 1.875);
    paint(); frame = requestAnimationFrame(tick);
  }
  function play() {
    cancelAnimationFrame(frame); previous = 0;
    if (!motion.matches) frame = requestAnimationFrame(tick);
  }
  resize(); play();
  window.addEventListener('resize', resize);
  motion.addEventListener('change', play);
}

function advanceSpots(spots, width, height, dt) {
  for (let index = 0; index < spots.length; index++) {
    const a = spots[index];
    for (const b of spots.slice(index + 1)) {
      const dx = b.x - a.x, dy = b.y - a.y, distance = Math.hypot(dx, dy) || .001;
      const overlap = a.r + b.r - distance;
      if (overlap <= 0) continue;
      const nx = distance === .001 ? 1 : dx / distance, ny = dy / distance;
      const closing = (b.vx - a.vx) * nx + (b.vy - a.vy) * ny;
      const push = Math.max(0, -closing) * .5 + overlap * dt * .8;
      a.vx -= nx * push; a.vy -= ny * push;
      b.vx += nx * push; b.vy += ny * push;
    }
  }
  for (const spot of spots) {
    const speed = Math.hypot(spot.vx, spot.vy) || 1;
    const scale = Math.max(12, Math.min(26, speed)) / speed;
    spot.vx *= scale; spot.vy *= scale;
    spot.x += spot.vx * dt; spot.y += spot.vy * dt;
    if (spot.x < spot.r) { spot.x = spot.r; spot.vx = Math.abs(spot.vx); }
    if (spot.x > width - spot.r) { spot.x = width - spot.r; spot.vx = -Math.abs(spot.vx); }
    if (spot.y < spot.r) { spot.y = spot.r; spot.vy = Math.abs(spot.vy); }
    if (spot.y > height - spot.r) { spot.y = height - spot.r; spot.vy = -Math.abs(spot.vy); }
  }
}

startWallpaper();
let data = {items: [], purposes: [], model_types: {}}, selected = null, busy = false;
async function api(url, options) {
  const response = await fetch(url, options);
  const body = await response.json();
  if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : '请求失败，请稍后重试');
  return body;
}
function closeMenu() {
  $('model-menu').hidden = true;
  $('model-toggle').setAttribute('aria-expanded', 'false');
}
function controls() {
  $('submit').disabled = busy || !selected || !$('text').value.trim();
  $('model-toggle').disabled = busy;
  $('refresh').disabled = busy;
}
function branch(title) {
  const details = document.createElement('details');
  const summary = document.createElement('summary'); summary.textContent = title;
  const children = document.createElement('div'); children.className = 'menu-children';
  details.append(summary, children);
  return {details, children};
}
function renderModels() {
  $('model-options').replaceChildren();
  for (const purpose of data.purposes) {
    const group = branch(purpose.label);
    const items = data.items.filter(item => item.purpose === purpose.value);
    for (const kind of [...new Set(items.map(item => item.model))]) {
      const type = branch(data.model_types[kind] || kind);
      for (const record of items.filter(item => item.model === kind)) {
        const button = document.createElement('button');
        button.type = 'button'; button.className = 'model-choice'; button.textContent = record.name;
        button.title = `${record.name} · ${record.created_at}`;
        button.setAttribute('aria-pressed', String(selected?.id === record.id));
        button.onclick = () => {
          selected = record;
          $('selected-model').textContent = record.name;
          $('model-toggle').title = record.name;
          $('state').textContent = `${purpose.label} · ${data.model_types[kind] || kind}`;
          for (const choice of $('model-options').querySelectorAll('button')) choice.setAttribute('aria-pressed', String(choice === button));
          closeMenu(); controls(); $('text').focus();
        };
        type.children.append(button);
      }
      group.children.append(type.details);
    }
    $('model-options').append(group.details);
  }
}
async function load() {
  $('refresh').disabled = true;
  $('menu-state').textContent = '正在加载模型…';
  try {
    data = await api('/inference/models');
    selected = data.items.find(item => item.id === selected?.id) || null;
    $('selected-model').textContent = selected?.name || '选择模型';
    renderModels();
    $('menu-state').textContent = data.items.length ? '用途 → 模型类型 → 模型名称' : '暂无可用模型，请先完成训练。';
    $('state').textContent = selected ? '已选择模型，可以发送文本。' : data.items.length ? '选择模型后即可发送文本' : '暂无已登记模型，请先完成训练。';
  } catch (error) { $('menu-state').textContent = error.message; $('state').textContent = error.message; }
  finally { controls(); }
}
function renderResult(state, text) {
  const panel = $('messages');
  panel.className = `messages result-panel result-${state}`;
  panel.hidden = state === 'idle';
  panel.setAttribute('aria-busy', String(state === 'pending'));
  const heading = document.createElement('span');
  heading.className = 'result-heading'; heading.textContent = state === 'error' ? '推理失败' : '推测结果';
  const content = document.createElement('div'); content.className = 'result-content';
  if (state === 'pending') {
    const dots = document.createElement('div'); dots.className = 'thinking-dots'; dots.setAttribute('aria-hidden', 'true');
    for (let i = 0; i < 3; i++) dots.append(document.createElement('span'));
    content.append(dots);
  }
  const body = document.createElement('p'); body.className = 'result-value'; body.textContent = text;
  content.append(body);
  panel.replaceChildren(heading, content);
}
async function renderProbabilities(items) {
  const chart = $('probability-chart');
  const previous = chart.querySelector('svg');
  if (!previous) chart.textContent = '正在生成概率图…';
  const response = await fetch('/inference/chart', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({items}),
  });
  if (!response.ok) throw new Error('概率图生成失败，请确认后端已更新并重启。');
  const svg = await response.text();
  const graphic = new DOMParser().parseFromString(svg, 'image/svg+xml').documentElement;
  if (graphic.localName !== 'svg') throw new Error('概率图格式异常');
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (previous) {
    // 新图已准备好后再退场，推理或绘图请求期间始终保留旧图。
    if (chart.motionFrame) cancelAnimationFrame(chart.motionFrame);
    chart.motionFrame = 0;
    chart.onpointermove = chart.onpointerleave = null;
    if (!reducedMotion) await previous.animate([
      {opacity: 1, transform: 'translateY(0)'},
      {opacity: 0, transform: 'translateY(36px)'},
    ], {duration: 500, easing: 'ease-in', fill: 'forwards'}).finished;
  }
  chart.replaceChildren(graphic);
  attachGraphMotion(graphic);
  if (!reducedMotion) await graphic.animate([
    {opacity: 0, transform: 'translateY(-10px)'},
    {opacity: 1, transform: 'translateY(0)'},
  ], {duration: 650, easing: 'ease-out'}).finished;
}
// 概率控制圆大小，深度控制视差；线端点跟随移动后的圆边缘。
function attachGraphMotion(svg) {
  const host = $('probability-chart');
  if (host.motionFrame) cancelAnimationFrame(host.motionFrame);
  host.motionFrame = 0;
  const nodes = [...svg.querySelectorAll('[data-node-index]')].map(group => {
    const circle = group.querySelector('circle[data-node-body]');
    return {group, x: +circle.getAttribute('cx'), y: +circle.getAttribute('cy'),
      r: +circle.getAttribute('r'), depth: +group.getAttribute('data-depth')};
  });
  const edges = [...svg.querySelectorAll('line[data-source]')];
  const maxRadius = Math.max(...nodes.map(node => node.r), 1);
  let x = 0, y = 0, targetX = 0, targetY = 0;
  function tick() {
    x += (targetX - x) * .09; y += (targetY - y) * .09;
    const positions = nodes.map(node => {
      const amount = 4 + 12 * (node.r / maxRadius) ** 2 + node.depth * 2;
      const dx = x * amount, dy = y * amount;
      node.group.setAttribute('transform', `translate(${dx} ${dy})`);
      return {x: node.x + dx, y: node.y + dy, r: node.r};
    });
    for (const edge of edges) {
      const a = positions[+edge.getAttribute('data-source')], b = positions[+edge.getAttribute('data-target')];
      const distance = Math.hypot(b.x - a.x, b.y - a.y) || 1;
      const ux = (b.x - a.x) / distance, uy = (b.y - a.y) / distance;
      edge.setAttribute('x1', a.x + ux * a.r); edge.setAttribute('y1', a.y + uy * a.r);
      edge.setAttribute('x2', b.x - ux * b.r); edge.setAttribute('y2', b.y - uy * b.r);
    }
    host.motionFrame = Math.abs(targetX - x) + Math.abs(targetY - y) > .001 ? requestAnimationFrame(tick) : 0;
  }
  function move(nextX, nextY) {
    targetX = nextX; targetY = nextY;
    if (!host.motionFrame) host.motionFrame = requestAnimationFrame(tick);
  }
  host.onpointermove = event => {
    if (event.pointerType === 'touch' || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const rect = host.getBoundingClientRect();
    move(Math.max(-1, Math.min(1, (event.clientX - rect.left) / rect.width * 2 - 1)),
         Math.max(-1, Math.min(1, (event.clientY - rect.top) / rect.height * 2 - 1)));
  };
  host.onpointerleave = () => move(0, 0);
}
$('model-toggle').onclick = () => {
  const open = $('model-menu').hidden;
  $('model-menu').hidden = !open;
  $('model-toggle').setAttribute('aria-expanded', String(open));
};
document.addEventListener('click', event => { if (!$('model-picker').contains(event.target)) closeMenu(); });
document.addEventListener('keydown', event => { if (event.key === 'Escape' && !$('model-menu').hidden) { closeMenu(); $('model-toggle').focus(); } });
$('refresh').onclick = load;
$('reset-page').onclick = () => window.location.reload();
$('text').oninput = () => {
  $('text').style.height = 'auto'; $('text').style.height = `${Math.min(180, $('text').scrollHeight)}px`; controls();
};
$('text').onkeydown = event => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); if (!$('submit').disabled) $('form').requestSubmit(); }
};
$('form').onsubmit = async event => {
  event.preventDefault();
  const text = $('text').value.trim(), record = selected;
  if (busy || !record || !text) return;
  busy = true; closeMenu(); controls();
  renderResult('pending', '正在推理…');

  try {
    const result = await api('/inference/predict', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({purpose: record.purpose, model_id: record.id, text})});
    renderResult('complete', result.class_name);
    // 布局移动与绘图独立：缺少概率或绘图失败也不能静默跳过结果布局。
    document.querySelector('.chat-shell').classList.add('has-result');
    $('probability-panel').setAttribute('aria-hidden', 'false');
    try {
      if (!Array.isArray(result.probabilities) || !result.probabilities.length) {
        throw new Error('当前后端未返回标签概率，请重启后端后重新推理。');
      }
      await renderProbabilities(result.probabilities);
    } catch (error) {
      if ($('probability-chart').querySelector('svg')) $('state').textContent = `${error.message}（保留上次图表）`;
      else $('probability-chart').textContent = error.message;
    }
  } catch (error) { renderResult('error', error.message); }
  finally { busy = false; controls(); $('messages').scrollTop = $('messages').scrollHeight; $('text').focus(); }
};
$('text').value = '';
$('text').style.height = '';
renderResult('idle', '等待推理');
load();
