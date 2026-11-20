'use strict';
const $ = selector => document.querySelector(selector);
let page = 1, total = 0, loading = false, editing = null, deleting = null;
let selecting = false, bulkPending = false, deleteAll = false, visibleItems = [];
const selectedIds = new Set();
async function request(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail || data));
  return data;
}
function navigation() {
  $('#previous').disabled = loading || bulkPending || page <= 1;
  $('#next').disabled = loading || bulkPending || page >= Math.max(1, Math.ceil(total / 10));
  $('#refresh').disabled = loading || bulkPending;
  $('#page-info').textContent = `第 ${page} / ${Math.max(1, Math.ceil(total / 10))} 页`;
  $('#bulk-delete').textContent = selecting ? deleteAll ? `确认删除全部（${total}）` : `确认删除${selectedIds.size ? `（${selectedIds.size}）` : ''}` : '批量删除';
  $('#bulk-delete').disabled = loading || bulkPending || (selecting ? !deleteAll && !selectedIds.size : !total);
  $('#bulk-cancel').hidden = !selecting;
  $('#bulk-cancel').disabled = loading || bulkPending;
  for (const cell of document.querySelectorAll('.selection-cell')) cell.hidden = !selecting;
  $('#select-all').checked = deleteAll;
  $('#select-all').indeterminate = !deleteAll && selectedIds.size > 0;
  $('#select-all').disabled = loading || bulkPending || !visibleItems.length;
  for (const checkbox of $('#rows').querySelectorAll('input[type=checkbox]')) {
    checkbox.checked = selectedIds.has(checkbox.value);
    checkbox.disabled = loading || bulkPending;
  }
  for (const button of $('#rows').querySelectorAll('button')) button.disabled = loading || bulkPending || selecting;
}
function action(text, callback, className = 'secondary') {
  const button = document.createElement('button'); button.type = 'button'; button.textContent = text; button.className = className;
  button.addEventListener('click', async () => {
    button.disabled = true;
    try { await callback(); } catch (error) { $('#state').textContent = error.message; $('#state').className = 'error'; }
    finally { button.disabled = false; }
  }); return button;
}
async function detail(id) {
  const record = await request(`/model-registry/${encodeURIComponent(id)}`);
  const meta = $('#detail-meta'); meta.replaceChildren();
  for (const [label, value] of [['名称',record.name],['模型类型',record.model_label],['产物训练轮次',record.test_result?.training_metrics?.selected_epoch == null ? '未记录' : `第 ${record.test_result.training_metrics.selected_epoch} 轮`],['用途',record.purpose_label],['登记时间',record.created_at],['存储路径',record.path],['标签映射路径',record.label_path],['训练集',record.train_source],['测试集',record.test_source]]) {
    const dt = document.createElement('dt'), dd = document.createElement('dd'); dt.textContent = label; dd.textContent = value || '—'; meta.append(dt, dd);
  }
  $('#detail-params').textContent = JSON.stringify(record.params, null, 2);
  const comparison = $('#detail-metrics'); comparison.replaceChildren();
  for (const [split, label] of Object.entries({train:'训练集',dev:'验证集',test:'测试集'})) {
    const scores = record.test_result?.training_metrics?.[split] || (split === 'test' ? record.test_result?.scores : null);
    const row = document.createElement('p');
    row.textContent = `${label}：` + (scores ? ['accuracy','precision','recall','f1'].map((key, i) => `${['准确率','精确率','召回率','F1'][i]} ${typeof scores[key] === 'number' ? (scores[key] * 100).toFixed(2) + '%' : '—'}`).join(' · ') : '暂无数据');
    comparison.append(row);
  }
  $('#detail-result').textContent = JSON.stringify(record.test_result, null, 2);
  $('#detail').showModal();
}
async function load(target = page) {
  if (loading || bulkPending) return;
  loading = true; navigation(); $('#state').textContent = '正在加载…'; $('#state').className = '';
  try {
    let data = await request(`/model-registry?page=${target}&page_size=10`);
    const last = Math.max(1, Math.ceil(data.total / 10));
    if (target > last) data = await request(`/model-registry?page=${last}&page_size=10`);
    page = data.page; total = data.total; $('#rows').replaceChildren();
    selectedIds.clear();
    deleteAll = false;
    visibleItems = data.items;
    for (const [index, record] of data.items.entries()) {
      const tr = document.createElement('tr');
      const selection = document.createElement('td'); selection.className = 'selection-cell'; selection.hidden = !selecting;
      const checkbox = document.createElement('input'); checkbox.type = 'checkbox'; checkbox.value = record.id;
      checkbox.setAttribute('aria-label', `选择模型 ${record.name}`);
      checkbox.onchange = () => {
        deleteAll = false;
        checkbox.checked ? selectedIds.add(record.id) : selectedIds.delete(record.id);
        $('#state').textContent = '仅删除当前页勾选的模型。';
        navigation();
      };
      selection.append(checkbox); tr.append(selection);
      const number = document.createElement('td'); number.className = 'row-number'; number.textContent = String(index + 1); tr.append(number);
      const name = document.createElement('td'); name.className = 'model-name'; name.append(action(record.name, () => detail(record.id), 'link')); tr.append(name);
      const scores = record.test_result?.scores || {};
      const values = [`${record.model_label} / ${record.purpose_label}`, ...['accuracy','precision','recall','f1'].map(key => typeof scores[key] === 'number' ? `${(scores[key] * 100).toFixed(2)}%` : '—'), record.created_at];
      for (const value of values) { const td = document.createElement('td'); td.textContent = value; tr.append(td); }
      const td = document.createElement('td'), actions = document.createElement('div'); actions.className = 'actions';
      actions.append(action('查看', () => detail(record.id)), action('改名', () => {
        editing = record.id; $('#new-name').value = record.name; $('#edit-error').textContent = ''; $('#edit').showModal(); $('#new-name').focus();
      }), action('删除', () => {
        deleting = record.id; $('#delete-description').textContent = `确定删除「${record.name}」？`; $('#delete-error').textContent = ''; $('#delete').showModal();
      }, 'danger'));
      td.append(actions); tr.append(td); $('#rows').append(tr);
    }
    $('#count').textContent = '模型列表';
    $('#state').textContent = total ? '' : '暂无模型，完成训练和测试后将自动登记。';
  } catch (error) { $('#state').textContent = `加载失败：${error.message}`; $('#state').className = 'error'; }
  finally { loading = false; navigation(); }
}
$('#previous').onclick = () => load(page - 1);
$('#next').onclick = () => load(page + 1);
$('#refresh').onclick = () => load();
$('#select-all').onchange = event => {
  deleteAll = event.target.checked;
  for (const item of visibleItems) event.target.checked ? selectedIds.add(item.id) : selectedIds.delete(item.id);
  $('#state').textContent = deleteAll ? `已全选全部 ${total} 个模型（包含所有页面），确认后将全部软删除。` : '';
  navigation();
};
$('#bulk-cancel').onclick = () => { selecting = false; deleteAll = false; selectedIds.clear(); $('#state').textContent = ''; navigation(); };
$('#bulk-delete').onclick = async () => {
  if (loading || bulkPending) return;
  if (!selecting) { selecting = true; deleteAll = false; selectedIds.clear(); navigation(); return; }
  if (!deleteAll && !selectedIds.size) return;
  bulkPending = true; navigation();
  try {
    const result = await request(deleteAll ? '/model-registry/delete-all' : '/model-registry/batch-delete', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: deleteAll ? undefined : JSON.stringify({ids: visibleItems.filter(item => selectedIds.has(item.id)).map(item => item.id)}),
    });
    for (const item of result.deleted) selectedIds.delete(item.id);
    if (!selectedIds.size) selecting = false;
    bulkPending = false;
    await load();
    for (const item of visibleItems) if (result.failed.some(failed => failed.id === item.id)) selectedIds.add(item.id);
    selecting = selectedIds.size > 0;
    $('#state').textContent = result.failed.length
      ? `已删除 ${result.deleted.length} 个，${result.failed.length} 个未删除：${result.failed.map(item => item.error).join('；')}`
      : `已删除 ${result.deleted.length} 个模型。`;
    $('#state').className = result.failed.length ? 'error' : '';
  } catch (error) { $('#state').textContent = error.message; $('#state').className = 'error'; }
  finally { bulkPending = false; navigation(); }
};
for (const button of document.querySelectorAll('[data-close]')) button.onclick = () => document.getElementById(button.dataset.close).close();
$('#edit-form').onsubmit = async event => {
  event.preventDefault(); const name = $('#new-name').value.trim();
  if (!name) { $('#edit-error').textContent = '请输入非空的模型名称。'; return; }
  $('#save-name').disabled = true;
  try {
    await request(`/model-registry/${encodeURIComponent(editing)}`, {method:'PATCH', headers:{'Content-Type':'application/json'}, body:JSON.stringify({name})});
    $('#edit').close(); await load();
  } catch (error) { $('#edit-error').textContent = error.message; }
  finally { $('#save-name').disabled = false; }
};
$('#confirm-delete').onclick = async () => {
  $('#confirm-delete').disabled = true;
  try { await request(`/model-registry/${encodeURIComponent(deleting)}`, {method:'DELETE'}); $('#delete').close(); await load(); }
  catch (error) { $('#delete-error').textContent = error.message; }
  finally { $('#confirm-delete').disabled = false; }
};
load();
