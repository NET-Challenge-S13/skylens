import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

// VWorld dev proxy credentials. The key lives OUTSIDE the repo (../.env.vworld)
// on purpose — never commit it. The proxy injects key+domain server-side so the
// browser (and the bundle) never sees them. Building layer degrades gracefully
// when the file is absent (terrain-only scene).
function loadVworld(): { key: string; domain: string } | null {
  try {
    const txt = readFileSync(
      fileURLToPath(new URL('../.env.vworld', import.meta.url)),
      'utf8',
    );
    const key = /^VWORLD_KEY=(.+)$/m.exec(txt)?.[1]?.trim();
    const domain =
      /^VWORLD_DOMAIN=(.+)$/m.exec(txt)?.[1]?.trim() ?? 'http://localhost:5173';
    return key ? { key, domain } : null;
  } catch {
    return null;
  }
}
const vworld = loadVworld();

// Multi-page build: a landing page plus the two role pages that each run on a
// separate computer (SIM and RECON).
export default defineConfig({
  server: {
    // Expose on the LAN so the other computer can load the page.
    host: true,
    proxy: {
      // VWorld has no CORS headers, so the browser can't call it directly.
      // /vworld/wfs?… → https://api.vworld.kr/req/wfs?…&key=…&domain=…
      '/vworld': {
        target: 'https://api.vworld.kr',
        changeOrigin: true,
        rewrite: (path) => {
          // WMTS wants the key IN the path:
          // /vworld/wmts/{z}/{y}/{x}.jpeg → /req/wmts/1.0.0/KEY/Satellite/{z}/{y}/{x}.jpeg
          const wmts = /^\/vworld\/wmts\/(\d+)\/(\d+)\/(\d+)\.jpeg$/.exec(path);
          if (wmts && vworld) {
            return `/req/wmts/1.0.0/${vworld.key}/Satellite/${wmts[1]}/${wmts[2]}/${wmts[3]}.jpeg`;
          }
          const p = path.replace(/^\/vworld/, '/req');
          if (!vworld) return p;
          const sep = p.includes('?') ? '&' : '?';
          return `${p}${sep}key=${vworld.key}&domain=${encodeURIComponent(vworld.domain)}`;
        },
      },
    },
  },
  build: {
    rollupOptions: {
      input: {
        main: 'index.html',
        sim: 'sim.html',
        recon: 'recon.html',
      },
    },
  },
});
