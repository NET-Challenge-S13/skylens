import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

// VWorld dev proxy credentials. Prefer the key OUTSIDE the repo (../.env.vworld)
// so it can never be committed, but also accept it at the repo root
// (.env.vworld) for convenience — .gitignore's `.env.*` rule keeps that ignored
// too. The proxy injects key+domain server-side so the browser (and the bundle)
// never see them. The map layer degrades gracefully when no key is found.
function loadVworld(): { key: string; domain: string } | null {
  for (const rel of ['../.env.vworld', '.env.vworld']) {
    try {
      const txt = readFileSync(fileURLToPath(new URL(rel, import.meta.url)), 'utf8');
      const key = /^VWORLD_KEY=(.+)$/m.exec(txt)?.[1]?.trim();
      const domain =
        /^VWORLD_DOMAIN=(.+)$/m.exec(txt)?.[1]?.trim() ?? 'http://localhost:5173';
      if (key) return { key, domain };
    } catch {
      // try the next location
    }
  }
  return null;
}
const vworld = loadVworld();

// Multi-page build: a landing page plus the two role pages that each run on a
// separate computer (SIM and RECON).
export default defineConfig({
  publicDir: 'res/public',
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
        main: 'res/static/index.html',
        sim: 'res/static/sim.html',
        recon: 'res/static/recon.html',
      },
    },
  },
});
