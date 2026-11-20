const form = document.querySelector('#classify-form');
const result = document.querySelector('#result');
const button = document.querySelector('#submit');

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const text = form.elements.title.value.trim();
  if (!text) return;

  result.hidden = false;
  result.className = '';
  result.textContent = '正在分类…';
  button.disabled = true;

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: form.elements.model.value, text }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || '分类失败');
    result.textContent = `分类结果：${data.class_name}`;
  } catch (error) {
    result.className = 'error';
    result.textContent = error.message;
  } finally {
    button.disabled = false;
  }
});
