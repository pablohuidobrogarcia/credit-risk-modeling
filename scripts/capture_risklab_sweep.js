/* Capture the Risk Lab tail-by-grade panel while rho moves. Invoked by
 * scripts/make_risklab_sweep_gif.py. */
const { chromium } = require("@playwright/test");
const path = require("path");
const fs = require("fs");

const BASE = process.env.RISKLAB_URL || "http://127.0.0.1:8766/";
const OUT = process.env.RISKLAB_FRAMES || path.resolve("artifacts", "risklab-sweep-frames");

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch({
    args: ["--use-gl=swiftshader", "--enable-unsafe-swiftshader"],
  });
  const page = await browser.newPage({
    viewport: { width: 1600, height: 1100 },
    deviceScaleFactor: 1,
    colorScheme: "light",
  });
  await page.goto(BASE, { waitUntil: "networkidle" });
  await page.waitForTimeout(1200);
  if (await page.locator("#welcome-start").isVisible().catch(() => false)) {
    await page.locator("#welcome-start").click();
    await page.waitForTimeout(400);
  }
  await page.locator("#view-tab-risk").click();
  await page.waitForTimeout(1600);
  await page.locator('#tail-metric button[data-tail-metric="diff"]').click();
  await page.locator("#surface-chart canvas").waitFor({ state: "visible", timeout: 20_000 });
  await page.evaluate(() => {
    const section = document.querySelector('section[aria-labelledby="surface-title"]');
    if (section) window.scrollTo(0, section.getBoundingClientRect().top + window.scrollY - 20);
  });
  await page.waitForTimeout(700);
  const mode = await page.locator("#surface-chart").getAttribute("data-mode");
  if (mode !== "surface") throw new Error(`expected 3D surface branch, got ${mode}`);

  // Slider indexes: 0=.02, 2=.08, 4=.12, 6=.18, 8=.22, 10=.30, 11=.35.
  const values = [0, 2, 4, 6, 8, 10, 11, 10, 8, 6, 5, 4];
  for (let i = 0; i < values.length; i += 1) {
    await page.evaluate((value) => {
      const slider = document.querySelector("#risk-rho");
      slider.value = String(value);
      slider.dispatchEvent(new Event("input", { bubbles: true }));
    }, values[i]);
    await page.waitForTimeout(260);
    await page.locator('section[aria-labelledby="surface-title"]').screenshot({
      path: path.join(OUT, `${String(i + 1).padStart(2, "0")}-rho-${values[i]}.png`),
    });
  }
  await browser.close();
})();
