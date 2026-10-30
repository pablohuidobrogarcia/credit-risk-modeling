const { defineConfig } = require("@playwright/test");

module.exports = defineConfig({
  testDir: "tests/e2e",
  timeout: 45_000,
  expect: { timeout: 7_000 },
  fullyParallel: false,
  workers: 1,
  reporter: process.env.CI ? [["line"], ["html", { open: "never" }]] : "line",
  use: {
    baseURL: "http://127.0.0.1:8765",
    headless: true,
    viewport: { width: 1280, height: 900 },
    actionTimeout: 10_000,
    launchOptions: {
      args: ["--use-gl=swiftshader", "--enable-unsafe-swiftshader"],
    },
  },
  webServer: {
    command: "python -m http.server 8765 --directory docs",
    url: "http://127.0.0.1:8765",
    reuseExistingServer: !process.env.CI,
    timeout: 30_000,
  },
});
