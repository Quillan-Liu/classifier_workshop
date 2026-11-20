const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
class Element {
  constructor(tag = 'div') { this.tag = tag; this.children = []; this.attrs = {}; this.disabled = false; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  setAttribute(key, value) { this.attrs[key] = value; }
  addEventListener() {}
  querySelectorAll(selector) {
    return this.children.flatMap(child => {
      const match = selector === 'button' ? child.tag === 'button'
        : selector === 'input[type=checkbox]' ? child.tag === 'input' && child.type === 'checkbox'
        : child.className === selector.slice(1);
      return [...(match ? [child] : []), ...child.querySelectorAll(selector)];
    });
  }
}
const elements = new Map();
const get = id => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); };
const heading = new Element('th'); heading.className = 'selection-cell';
const document = {
  querySelector: get, createElement: tag => new Element(tag),
  querySelectorAll: selector => selector === '.selection-cell' ? [heading, ...get('#rows').querySelectorAll(selector)] : [],
};
let records = Array.from({length: 12}, (_, i) => ({id: `m${i + 1}`, name: `model${i + 1}`, test_result: {}}));
let failOnce = true;
const batches = [];
let allCalls = 0;
const fetch = async (url, options) => {
  const target = new URL(url, 'http://test');
  if (target.pathname === '/model-registry/delete-all') {
    allCalls++;
    assert.equal(options.body, undefined);
    const deleted = records.map(item => ({id: item.id, is_deleted: true}));
    records = [];
    return {ok: true, json: async () => ({deleted, failed: []})};
  }
  if (target.pathname === '/model-registry/batch-delete') {
    const ids = JSON.parse(options.body).ids;
    batches.push(ids);
    const failed = failOnce && ids.includes('m11') ? [{id: 'm11', error: 'locked'}] : [];
    failOnce = false;
    const deleted = ids.filter(id => !failed.some(item => item.id === id)).map(id => ({id, is_deleted: true}));
    records = records.filter(item => !deleted.some(row => row.id === item.id));
    return {ok: true, json: async () => ({deleted, failed})};
  }
  const page = Number(target.searchParams.get('page'));
  return {ok: true, json: async () => ({items: records.slice((page - 1) * 10, page * 10), page, total: records.length})};
};
const context = vm.createContext({document, fetch, Set});
vm.runInContext(fs.readFileSync(__dirname + '/app.js', 'utf8'), context);
(async () => {
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(get('#rows').children.length, 10);
  assert.equal(get('#rows').children[9].children[1].textContent, '10');
  await get('#bulk-delete').onclick();
  assert.equal(heading.hidden, false);
  assert.equal(get('#bulk-delete').disabled, true);
  const first = get('#rows').querySelectorAll('input[type=checkbox]')[0];
  first.checked = true; first.onchange();
  assert.equal(get('#select-all').indeterminate, true);
  get('#select-all').onchange({target: {checked: true}});
  assert.equal(get('#select-all').checked, true);
  await get('#next').onclick();
  assert.equal(get('#rows').children.length, 2);
  assert.equal(get('#bulk-delete').disabled, true);
  assert.ok(get('#rows').querySelectorAll('input[type=checkbox]').every(checkbox => !checkbox.checked));
  for (const checkbox of get('#rows').querySelectorAll('input[type=checkbox]')) {
    checkbox.checked = true; checkbox.onchange();
  }
  await get('#bulk-delete').onclick();
  assert.equal(batches.length, 1);
  assert.equal(batches[0].length, 2);
  assert.deepEqual(batches[0], ['m11', 'm12']);
  assert.equal(records.length, 11);
  assert.ok(records.some(item => item.id === 'm11'));
  assert.equal(get('#select-all').checked, false);
  assert.equal(get('#select-all').indeterminate, true);
  assert.equal(heading.hidden, false);
  assert.match(get('#state').textContent, /locked/);
  await get('#bulk-delete').onclick();
  assert.equal(records.length, 10);
  assert.equal(heading.hidden, true);
  assert.equal(get('#bulk-delete').disabled, false);
  // 单选只删除当前页的一条记录，不会误删整页。
  await get('#bulk-delete').onclick();
  const one = get('#rows').querySelectorAll('input[type=checkbox]')[0];
  one.checked = true; one.onchange();
  await get('#bulk-delete').onclick();
  assert.equal(batches[2].length, 1);
  assert.equal(records.length, 9);
  assert.equal(allCalls, 0);
  await get('#bulk-delete').onclick();
  get('#select-all').onchange({target: {checked: true}});
  assert.match(get('#bulk-delete').textContent, /确认删除全部/);
  await get('#bulk-delete').onclick();
  assert.equal(allCalls, 1);
  assert.equal(records.length, 0);
  console.log('Page-only selection, partial-failure retry, and explicit global select-all passed.');
})().catch(error => { console.error(error); process.exitCode = 1; });
