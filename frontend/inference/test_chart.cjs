const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
function element() {
  return {children: [], attrs: {}, style: {}, value: '', classList: {add() {}, remove() {}},
    setAttribute(key, value) { this.attrs[key] = value; },
    append(...children) { this.children.push(...children); },
    replaceChildren(...children) { this.children = children; },
    querySelector(selector) { return this.children.find(child => child.localName === selector) || null; },
    animate() { return {finished: Promise.resolve()}; }};
}
const elements = new Map();
const document = {getElementById(id) { if (!elements.has(id)) elements.set(id, element()); return elements.get(id); },
  createElementNS() { return element(); }, createElement() { return element(); }, addEventListener() {},
  querySelector(selector) { return selector === '.ambient-wallpaper' ? null : element(); }};
const graphic = element();
graphic.localName = 'svg'; graphic.querySelectorAll = () => [];
const context = vm.createContext({document, fetch: () => new Promise(() => {}),
  window: {matchMedia: () => ({matches: false})},
  DOMParser: class { parseFromString() { return {documentElement: graphic}; } }});
vm.runInContext(fs.readFileSync(__dirname + '/app.js', 'utf8'), context);
context.spots = [{x: 20, y: 100, r: 20, vx: -22, vy: 0}];
vm.runInContext('advanceSpots(spots, 500, 300, .04)', context);
assert.ok(context.spots[0].vx > 0);
assert.ok(context.spots[0].x >= 20);
context.spots = [{x: 100, y: 100, r: 40, vx: 22, vy: 0}, {x: 170, y: 100, r: 40, vx: -22, vy: 0}];
vm.runInContext('advanceSpots(spots, 500, 300, .04)', context);
assert.ok(context.spots[0].vx < 0 && context.spots[1].vx > 0);
for (let i = 0; i < 1000; i++) vm.runInContext('advanceSpots(spots, 500, 300, .04)', context);
assert.ok(context.spots.every(spot => spot.x >= spot.r && spot.x <= 500 - spot.r && Number.isFinite(spot.y)));
for (const state of ['idle', 'pending', 'complete', 'error']) {
  vm.runInContext(`renderResult('${state}', '测试结果', '测试模型')`, context);
  const panel = elements.get('messages');
  assert.equal(panel.hidden, state === 'idle');
  assert.equal(panel.children.length, 2);
  assert.equal(panel.attrs['aria-busy'], String(state === 'pending'));
  assert.equal(panel.children[1].children.at(-1).textContent, '测试结果');
  assert.equal(panel.children[1].children.length, state === 'pending' ? 2 : 1);
}
context.fetch = async () => ({ok: true, text: async () => '<svg xmlns="http://www.w3.org/2000/svg"/>'});
(async () => {
  await vm.runInContext('renderProbabilities([{class_name: "体育", probability: 0.9}])', context);
  assert.equal(elements.get('probability-chart').children[0].localName, 'svg');
  // 新请求等待时保留旧图，新 SVG 就绪后依次执行旧图下移退场、新图入场。
  const chart = elements.get('probability-chart');
  let resolveFetch, finishExit;
  const animations = [];
  graphic.animate = keyframes => {
    animations.push(keyframes);
    return {finished: animations.length === 1 ? new Promise(resolve => { finishExit = resolve; }) : Promise.resolve()};
  };
  context.fetch = () => new Promise(resolve => { resolveFetch = resolve; });
  const replacement = vm.runInContext('renderProbabilities([])', context);
  assert.equal(chart.children[0], graphic);
  assert.equal(animations.length, 0);
  resolveFetch({ok: true, text: async () => '<svg/>'});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(chart.children[0], graphic);
  assert.equal(animations.length, 1);
  assert.equal(animations[0][1].transform, 'translateY(36px)');
  finishExit(); await replacement;
  assert.equal(animations.length, 2);
  assert.equal(animations[1][0].opacity, 0);
  const frames = [];
  context.requestAnimationFrame = callback => { frames.push(callback); return frames.length; };
  context.cancelAnimationFrame = () => {};
  context.window = {matchMedia: () => ({matches: false})};
  const nodes = [72, 36].map((radius, index) => {
    const group = element(), circle = element();
    circle.attrs = {cx: index * 180, cy: 0, r: radius};
    circle.getAttribute = key => circle.attrs[key];
    group.attrs['data-depth'] = 0;
    group.getAttribute = key => group.attrs[key]; group.querySelector = () => circle;
    return group;
  });
  const edge = element(); edge.attrs = {'data-source': '0', 'data-target': '1'};
  edge.getAttribute = key => edge.attrs[key];
  context.motionSvg = {querySelectorAll: selector => selector === '[data-node-index]' ? nodes : [edge]};
  vm.runInContext('attachGraphMotion(motionSvg)', context);
  const host = elements.get('probability-chart');
  host.getBoundingClientRect = () => ({left: 0, top: 0, width: 400, height: 400});
  host.onpointermove({clientX: 400, clientY: 200, pointerType: 'mouse'});
  for (let i = 0; i < 200 && frames.length; i++) frames.shift()();
  const offset = group => Number(group.attrs.transform.match(/translate\(([^ ]+)/)[1]);
  assert.ok(offset(nodes[0]) > offset(nodes[1]));
  assert.ok(Number(edge.attrs.x1) > 72);
  host.onpointerleave();
  for (let i = 0; i < 200 && frames.length; i++) frames.shift()();
  assert.ok(Math.abs(offset(nodes[0])) < 0.02);
  context.fetch = async () => ({ok: false});
  await assert.rejects(vm.runInContext('renderProbabilities([])', context), /概率图生成失败/);
  console.log('Result states and backend SVG rendering passed.');
})().catch(error => { console.error(error); process.exitCode = 1; });
