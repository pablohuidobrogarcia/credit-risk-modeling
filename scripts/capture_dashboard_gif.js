/* Capture five still frames of the shipped dashboard for the README GIF.
 * Invoked by scripts/make_dashboard_gif.py, which starts the local server and
 * assembles the PNGs with Pillow. */
const { chromium } = require("@playwright/test");
const path = require("path");
const fs = require("fs");

const BASE = process.env.DASHBOARD_URL || "http://127.0.0.1:8766/";
const OUT = process.env.DASHBOARD_FRAMES || path.resolve("artifacts", "gif-frames");
const CHECKS = path.join(OUT, "checks");

async function dismissWelcome(page) {
  await page.waitForTimeout(1500);
  if (await page.locator("#welcome-start").isVisible().catch(() => false)) {
    await page.locator("#welcome-start").click();
    await page.waitForTimeout(500);
  }
}

async function scrollElementToTop(page, selector) {
  await page.evaluate((selector) => {
    const el = document.querySelector(selector);
    if (el) window.scrollTo(0, el.getBoundingClientRect().top + window.scrollY);
  }, selector);
  await page.waitForTimeout(700);
}

(async () => {
  fs.mkdirSync(CHECKS, { recursive: true });
  const browser = await chromium.launch({
    args: ["--use-gl=swiftshader", "--enable-unsafe-swiftshader"],
  });
  const page = await browser.newPage({
    viewport: { width: 1280, height: 800 },
    deviceScaleFactor: 1,
    colorScheme: "light",
  });
  await page.goto(BASE, { waitUntil: "networkidle" });
  await dismissWelcome(page);
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
  await page.waitForTimeout(500);
  await page.locator("#metric-tabs button[data-metric='el_pct_ead']").click();
  await page.waitForTimeout(700);
  await page.screenshot({ path: path.join(OUT, "02-state-map.png") });

  // 3. Risk Lab top: scenario controls, KPI row, gauge and loss histogram.
  await page.locator("#view-tab-risk").click();
  await page.waitForTimeout(2200);
  await page.evaluate(() => { document.body.style.zoom = "0.58"; });
  await page.evaluate(() => {
    const view = document.querySelector("#view-risk");
    if (view) window.scrollTo(0, view.getBoundingClientRect().top + window.scrollY - 10);
  });
  await page.waitForTimeout(700);
  await page.screenshot({ path: path.join(OUT, "03-risk-lab-top.png") });
  await page.evaluate(() => { document.body.style.zoom = "1"; });
  await page.waitForTimeout(500);

  // 4. Risk Lab, Tail loss by grade, Difference view at rho = 0.12.
  await page.locator('#tail-metric button[data-tail-metric="diff"]').click();
  await page.locator("#rho-output").filter({ hasText: "0.120" }).waitFor({ timeout: 10_000 }).catch(() => {});
  await page.locator("#surface-chart canvas").waitFor({ state: "visible", timeout: 20_000 });
  await page.waitForTimeout(1200);
  const surfaceState = await page.evaluate(() => {
    const host = document.querySelector("#surface-chart");
    const canvas = host?.querySelector("canvas");
    const rect = canvas?.getBoundingClientRect();
    return {
      mode: host?.dataset.mode || null,
      canvas: !!canvas,
      width: rect?.width || 0,
      height: rect?.height || 0,
      zaxis: host?._fullLayout?.scene?.zaxis?.title?.text || null,
    };
  });
  if (surfaceState.mode !== "surface" || !surfaceState.canvas || surfaceState.width < 10 || surfaceState.height < 10) {
    throw new Error(`Tail chart did not render in the 3D surface branch: ${JSON.stringify(surfaceState)}`);
  }
  await page.evaluate(() => {
    document.body.style.zoom = "0.90";
    const section = document.querySelector('section[aria-labelledby="surface-title"]');
    if (section) window.scrollTo(0, section.getBoundingClientRect().top + window.scrollY - 40);
  });
  await page.waitForTimeout(900);
  await page.screenshot({ path: path.join(OUT, "04-tail-by-grade.png") });
  await page.locator("#surface-chart").screenshot({ path: path.join(CHECKS, "04-chart-only.png") });
  await page.evaluate(() => { document.body.style.zoom = "1"; });
  await page.waitForTimeout(500);

  // 5. Evidence & limits, model comparison table and benchmark note.
  await page.locator("#view-tab-evidence").click();
  await page.waitForTimeout(800);
  await page.evaluate(() => {
    const h = Array.from(document.querySelectorAll("h2")).find((el) => el.textContent.includes("The numbers"));
    if (h) window.scrollTo(0, h.getBoundingClientRect().top + window.scrollY - 80);
  });
  await page.waitForTimeout(700);
  await page.screenshot({ path: path.join(OUT, "05-evidence-model-comparison.png") });

  await browser.close();
})();
