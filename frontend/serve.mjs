// Dependency-free development server: static frontend + streaming API proxy.
import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';

const ports = { training: 3001, model_management: 3002, inference: 3003 };
export function createFrontend(kind, backend = 'http://127.0.0.1:8008') {
  if (!(kind in ports)) throw new Error('Frontend must be training, model_management or inference');
  const api = new URL(backend);
  if (api.protocol !== 'http:') throw new Error('Backend must use http://');
  return http.createServer(async (request, response) => {
    const pathname = new URL(request.url, 'http://localhost').pathname;
    if (request.method === 'GET' && (pathname === '/' || pathname === '/training' && kind === 'training' || pathname === '/inference' && kind === 'inference' || pathname === '/model-management' && kind === 'model_management')) {
      try {
        let html = await readFile(new URL(`./${kind}/index.html`, import.meta.url), 'utf8');
        // 用唯一占位符一次替换，将导航路径映射到对应的独立前端端口。
        html = html.replace(/href="(\/inference|\/training|\/model-management)"/g, (_, path) => {
          const target = path === '/inference' ? 'inference' : path === '/training' ? 'training' : 'model_management';
          return `href="${target === kind ? '/' : `http://127.0.0.1:${ports[target]}/`}"`;
        });
        response.writeHead(200, {'Content-Type': 'text/html; charset=utf-8'}); response.end(html);
      } catch { response.writeHead(500); response.end('Unable to load frontend'); }
      return;
    }
    const assets = new Map([
      ['/ui/shared.css', ['shared.css', 'text/css']],
      [`/ui/${kind}/app.js`, [`${kind}/app.js`, 'text/javascript']],
    ]);
    assets.set(`/ui/${kind}/style.css`, [`${kind}/style.css`, 'text/css']);
    if (assets.has(pathname) && request.method === 'GET') {
      try {
        const [file, mime] = assets.get(pathname);
        const data = await readFile(new URL(`./${file}`, import.meta.url));
        response.writeHead(200, {'Content-Type': `${mime}; charset=utf-8`}); response.end(data);
      } catch { response.writeHead(404); response.end('Asset not found'); }
      return;
    }
    // Stream multipart uploads without buffering entire datasets in memory.
    const upstream = http.request(new URL(request.url, api), {
      method: request.method, headers: {...request.headers, host: api.host},
    }, result => {
      response.writeHead(result.statusCode, result.headers); result.pipe(response);
      result.on('error', () => response.destroy());
    });
    upstream.on('error', () => {
      if (response.headersSent) { response.destroy(); return; }
      response.writeHead(502, {'Content-Type': 'application/json; charset=utf-8'});
      response.end(JSON.stringify({detail: `后端服务无法连接，请先启动 ${backend}`}));
    });
    request.on('aborted', () => upstream.destroy());
    request.pipe(upstream);
  });
}
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const kind = process.argv[2];
  const port = Number(process.argv[3] || ports[kind]);
  const server = createFrontend(kind, process.env.TMF_API_URL);
  server.on('error', error => { console.error(error.message); process.exitCode = 1; });
  server.listen(port, '127.0.0.1', () => console.log(`${kind}: http://127.0.0.1:${server.address().port}`));
}
