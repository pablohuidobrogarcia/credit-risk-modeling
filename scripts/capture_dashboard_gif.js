/* Capture three still frames of the shipped dashboard for the README GIF.
 * Invoked by scripts/make_dashboard_gif.py, which starts the local server and
 * assembles the PNGs with Pillow. */
const { chromium } = require("@playwright/test");
const path = require("path");

const BASE = process.env.DASHBOARD_URL || "http://127.0.0.1:8766/";
const OUT = process.env.DASHBOARD_FRAMES || path.resolve("artifacts", "gif-frames");

(async () => {
  const browser = await chromium.launch({
    args: ["--use-gl=swiftshader", "--enable-unsafe-swiftshader"],
  });
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 }, deviceScaleFactor: 1 });
  await page.goto(BASE, { waitUntil: "networkidle" });
  await page.waitForTimeout(1500);
  if (await page.locator("#welcome-start").isVisible().catch(() => false)) {
    await page.locator("#welcome-start").click();
    await page.waitForTimeout(500);
  }
  await page.locator("#view-tab-explorer").click();
  await page.waitForTimeout(500);
  await page.locator('#dimension-tabs button[data-dim="total"]').click();
  await page.waitForTimeout(500);

  // 1. Total folder overlay with sheet 3 (expected loss, EAD at default).
  await page.locator("#total-folder-open").click();
  await page.waitForTimeout(700);
  await page.locator("#folder-tab-2").click();
  await page.waitForTimeout(500);
  await page.screenshot({ path: path.join(OUT, "01-total-folder.png") });
  await page.locator("[data-folder-close]").click();
  await page.waitForTimeout(400);

  // 2. State map, expected loss.
  await page.locator('#dimension-tabs button[data-dim="state"]').click();
  await page.waitForTimeout(700);
  await page.screenshot({ path: path.join(OUT, "02-state-map.png") });

  // 3. Risk Lab top: scenario cards, gauge and loss distribution.
  await page.locator("#view-tab-risk").click();
  await page.waitForTimeout(2200);
  await page.screenshot({ path: path.join(OUT, "03-risk-lab.png") });

  await browser.close();
})();
