import { defineConfig } from 'vite';

// Multi-page build: a landing page plus the two role pages that each run on a
// separate computer (SIM and RECON).
export default defineConfig({
  publicDir: 'res/public',
  server: {
    // Expose on the LAN so the other computer can load the page.
    host: true,
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
