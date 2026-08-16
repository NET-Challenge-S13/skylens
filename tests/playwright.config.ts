import { defineConfig, devices } from '@playwright/test';

// E2E smoke config. Boots the Vite dev server and drives it in headless Chromium
// with a GPU-enabled flag set so WebGL actually initializes (SwiftShader fallback).
export default defineConfig({
  // This config lives in tests/, so the test dir is the config's own directory.
  testDir: '.',
  timeout: 30_000,
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
    // Run the dev server from the repo root (this config is in tests/).
    cwd: '..',
    url: 'http://localhost:5173',
    reuseExistingServer: true,
    timeout: 60_000,
  },
});
