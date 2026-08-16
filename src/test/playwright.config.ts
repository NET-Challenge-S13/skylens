import { defineConfig, devices } from '@playwright/test';

// E2E smoke config. Boots the Vite dev server and drives it in headless Chromium
// with a GPU-enabled flag set so WebGL actually initializes (SwiftShader fallback).
export default defineConfig({
  // This config lives in tests/, so the test dir is the config's own directory.
  testDir: '.',
  // Nested html entries (res/static/sim.html …) make Vite's first cold
  // transform heavy (~25s); give the first navigation room before the server warms.
  timeout: 45_000,
  expect: { timeout: 10_000 },
  fullyParallel: false,
  reporter: [['list']],
  use: {
    baseURL: 'http://localhost:5173',
    trace: 'retain-on-failure',
    launchOptions: {
      // Ensure WebGL works in headless CI-like environments.
      args: [
        '--use-gl=angle',
        '--use-angle=swiftshader',
        '--ignore-gpu-blocklist',
        '--enable-unsafe-swiftshader',
      ],
    },
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
  ],
  webServer: {
    command: 'npm run dev',
    // Run the dev server from the repo root (this config is in src/test/).
    cwd: '../..',
    // The entry html live under res/static, so `/` returns 404 — poll a real page
    // (Playwright treats 404 as "not ready" and would time out on `/`).
    url: 'http://localhost:5173/res/static/index.html',
    reuseExistingServer: true,
    timeout: 90_000,
  },
});
