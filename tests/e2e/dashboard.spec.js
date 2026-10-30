const { test, expect } = require("@playwright/test");
const fs = require("node:fs");
const path = require("node:path");

const ROOT = path.resolve(__dirname, "../..");
const DATA = JSON.parse(fs.readFileSync(path.join(ROOT, "docs/data.json"), "utf8"));
const PE = DATA.portfolio_explorer;
const PLOTLY_PATH = path.join(ROOT, "node_modules/plotly.js-dist-min/plotly.min.js");

const pct = (v, nd = 2) => `${(v * 100).toFixed(nd)}%`;
const dims = ["total", "state", "grade", "purpose", "vintage", "term", "home_ownership"];

async function open(page, { blockCdn = false } = {}) {
  const errors = [];
  page.on("pageerror", (err) => errors.push(`pageerror: ${err.message}`));
  page.on("console", (msg) => {
    if (msg.type() !== "error") return;
    const text = msg.text();
    if (text.includes("Plotly")) return;
    if (blockCdn && (text.includes("ERR_FAILED") || text.includes("Failed to load resource"))) return;
    errors.push(`console: ${text}`);
  });
  await page.route("https://cdn.jsdelivr.net/**", async (route) => {
    if (blockCdn) return route.abort();
    return route.fulfill({ path: PLOTLY_PATH, contentType: "application/javascript" });
  });
  await page.goto("/");
  await expect(page.locator("#app-content")).toBeVisible();
  await expect(page.locator("#tile-map .tile").first()).toBeVisible();
  await page.waitForTimeout(700);
  return errors;
}

async function expectNoErrors(errors) {
  expect(errors, errors.join("\n")).toEqual([]);
}

test("loads the linked explorer without console or page errors", async ({ page }) => {
  const errors = await open(page);
  await expect(page.locator("#summary-title")).toHaveText("Portfolio");
  await expect(page.locator("#tile-map .tile:not(.empty)")).toHaveCount(PE.dimensions.state.length);
  await expect(page.locator("#rank-list .rank-row")).toHaveCount(PE.dimensions.state.length);
  await expect(page.locator('[data-testid="summary-default-rate"]')).toHaveText(pct(PE.dimensions.total[0].default_rate));
  await expectNoErrors(errors);
});

test("every dimension tab changes the whole explorer and the URL", async ({ page }) => {
  const errors = await open(page);
  for (const dim of dims) {
    const button = page.locator(`#dimension-tabs button[data-dim="${dim}"]`);
    await button.click();
    await expect(button).toHaveAttribute("aria-selected", "true");
    await expect(page).toHaveURL(new RegExp(`dim=${dim}`));
    const items = PE.dimensions[dim];
    await expect(page.locator("#tile-map .tile:not(.empty)")).toHaveCount(items.length);
    await expect(page.locator("#rank-list .rank-row")).toHaveCount(items.length);
    if (dim !== "total") {
      await expect(page.locator("#summary-title")).toHaveText("Portfolio");
    }
  }
  await expectNoErrors(errors);
});

test("every state tile selects the linked list row, summary and hash", async ({ page }) => {
  test.setTimeout(120_000);
  const errors = await open(page);
  const states = PE.dimensions.state;
  for (const item of states) {
    await page.locator(`#tile-map .tile[data-id="${item.id}"]`).click();
    await expect(page.locator(`#tile-map .tile[data-id="${item.id}"]`)).toHaveClass(/selected/);
    await expect(page.locator(`#rank-list .rank-row[data-id="${item.id}"]`)).toHaveAttribute("aria-current", "true");
    await expect(page.locator("#summary-title")).toHaveText(item.name);
    await expect(page.locator('[data-testid="summary-default-rate"]')).toHaveText(pct(item.default_rate));
    await expect(page).toHaveURL(new RegExp(`sel=${item.id}`));
  }
  await expectNoErrors(errors);
});

test("every ranked row selects the linked tile and summary", async ({ page }) => {
  test.setTimeout(120_000);
  const errors = await open(page);
  for (const dim of ["state", "grade", "purpose", "vintage", "home_ownership"]) {
    await page.locator(`#dimension-tabs button[data-dim="${dim}"]`).click();
    const items = PE.dimensions[dim];
    for (const item of items) {
      await page.locator(`#rank-list .rank-row[data-id="${item.id}"]`).click();
      await expect(page.locator("#summary-title")).toHaveText(item.name);
      await expect(page.locator('[data-testid="summary-default-rate"]')).toHaveText(pct(item.default_rate));
      await expect(page).toHaveURL(new RegExp(`sel=${encodeURIComponent(item.id)}`));
    }
  }
  await expectNoErrors(errors);
});

test("metric tabs recolour the map and re-sort the list", async ({ page }) => {
  const errors = await open(page);
  const firstTile = page.locator("#tile-map .tile[data-id='CA']");
  const before = await firstTile.evaluate((el) => getComputedStyle(el).backgroundColor);
  await page.locator("#metric-tabs button[data-metric='el_pct_ead']").click();
  await expect(page).toHaveURL(/metric=el_pct_ead/);
  const after = await firstTile.evaluate((el) => getComputedStyle(el).backgroundColor);
  expect(after).not.toBe(before);
  const firstRow = page.locator("#rank-list .rank-row").first();
  const topId = await firstRow.getAttribute("data-id");
  const topItem = PE.dimensions.state.find((x) => x.id === topId);
  await expect(firstRow.locator(".rank-value")).toHaveText(pct(topItem.el_pct_ead));
  await expectNoErrors(errors);
});

test("search filters the ranked list and clears safely", async ({ page }) => {
  const errors = await open(page);
  const input = page.locator("#segment-search");
  await input.fill("CA");
  await expect(page.locator("#rank-list .rank-row")).toHaveCount(1);
  await expect(page.locator("#list-count")).toContainText("1 of");
  await input.fill("not-a-state");
  await expect(page.locator("#list-empty")).toBeVisible();
  await input.fill("");
  await expect(page.locator("#rank-list .rank-row")).toHaveCount(PE.dimensions.state.length);
  await expectNoErrors(errors);
});

test("every quick-action chip changes global state", async ({ page }) => {
  const errors = await open(page);
  await page.getByRole("button", { name: "View by grade" }).click();
  await expect(page).toHaveURL(/dim=grade/);
  await expect(page.locator("#summary-title")).toHaveText("Portfolio");
  await page.getByRole("button", { name: "Highest loss segment" }).click();
  await expect(page).not.toHaveURL(/sel=(?:&|$)/);
  await expect(page.locator("#summary-title")).not.toHaveText("Portfolio");
  await page.getByRole("button", { name: "Compare to portfolio" }).click();
  await expect(page).toHaveURL(/dim=total/);
  await page.getByRole("button", { name: "How do we know?" }).first().click();
  await expect(page.locator("#provenance-drawer")).toBeVisible();
  await page.locator("#drawer-close").click();
  await expect(page.locator("#provenance-drawer")).toBeHidden();
  await expectNoErrors(errors);
});

test("back and forward restore explorer state", async ({ page }) => {
  const errors = await open(page);
  await page.locator("#tile-map .tile[data-id='CA']").click();
  await expect(page).toHaveURL(/sel=CA/);
  await page.locator("#tile-map .tile[data-id='NY']").click();
  await expect(page).toHaveURL(/sel=NY/);
  await page.goBack();
  await expect(page).toHaveURL(/sel=CA/);
  await expect(page.locator("#summary-title")).toHaveText("CA");
  await page.goForward();
  await expect(page.locator("#summary-title")).toHaveText("NY");
  await expectNoErrors(errors);
});

test("view tabs switch panels and support keyboard arrows", async ({ page }) => {
  const errors = await open(page);
  await page.locator("#view-tab-risk").click();
  await expect(page.locator("#view-risk")).toBeVisible();
  await expect(page.locator("#view-explorer")).toBeHidden();
  await expect(page).toHaveURL(/view=risk/);
  await page.locator("#view-tab-evidence").click();
  await expect(page.locator("#view-evidence")).toBeVisible();
  await expect(page.locator("#limitations-list li")).toHaveCount(DATA.limitations.length);
  await page.locator("#view-tab-explorer").focus();
  await page.keyboard.press("ArrowRight");
  await expect(page.locator("#view-risk")).toBeVisible();
  await expectNoErrors(errors);
});

test("risk sliders, play sweep and surface control the shared state", async ({ page }) => {
  const errors = await open(page);
  await page.locator("#view-tab-risk").click();
  await expect(page.locator("#loss-chart .plot-container")).toBeVisible({ timeout: 20_000 });
  const rho = page.locator("#risk-rho");
  const before = await page.locator("#rho-output").innerText();
  await rho.fill("5");
  await expect(page.locator("#rho-output")).not.toHaveText(before);
  await expect(page).toHaveURL(/rho=/);
  const pd = page.locator("#risk-pd");
  await pd.fill("2");
  await expect(page).toHaveURL(/pd=/);
  const lgd = page.locator("#risk-lgd");
  await lgd.fill("1");
  await expect(page).toHaveURL(/lgd=/);
  await expect(page.locator("#loss-chart .plot-container")).toBeVisible();
  await page.locator("#surface-metrics button[data-metric='es_0.999']").click();
  await expect(page.locator("#surface-metrics button[data-metric='es_0.999']")).toHaveAttribute("aria-selected", "true");
  await expect(page.locator("#surface-chart canvas")).toBeVisible({ timeout: 20_000 });
  const beforeRho = await page.locator("#rho-output").innerText();
  await page.locator("#play-rho").click();
  await expect(page.locator("#play-rho")).toHaveAttribute("aria-pressed", "true");
  await page.waitForTimeout(900);
  await expect(page.locator("#rho-output")).not.toHaveText(beforeRho);
  await page.locator("#play-rho").click();
  await expect(page.locator("#play-rho")).toHaveAttribute("aria-pressed", "false");
  await expectNoErrors(errors);
});

test("clicking the surface point moves the linked sliders", async ({ page }) => {
  const errors = await open(page);
  await page.locator("#view-tab-risk").click();
  const surface = page.locator("#surface-chart");
  await expect(surface.locator("canvas")).toBeVisible({ timeout: 20_000 });
  await surface.scrollIntoViewIfNeeded();
  await page.waitForTimeout(300);
  const before = await page.locator("#scenario-badge").innerText();
  const box = await surface.boundingBox();
  await page.mouse.click(box.x + box.width * 0.52, box.y + box.height * 0.48);
  await page.waitForTimeout(400);
  const after = await page.locator("#scenario-badge").innerText();
  expect(after).not.toBe(before);
  await expect(page).toHaveURL(/rho=/);
  await expectNoErrors(errors);
});

test("applicant simulator recomputes score, PD and percentile", async ({ page }) => {
  const errors = await open(page);
  await page.locator("#view-tab-risk").click();
  const select = page.locator("#sim-controls select").first();
  const before = await page.locator("#sim-score").innerText();
  await select.selectOption({ index: 1 });
  await expect(page.locator("#sim-score")).not.toHaveText(before);
  await expect(page.locator("#sim-pd")).toContainText("%");
  await expect(page.locator("#sim-pct")).toContainText("th");
  await expectNoErrors(errors);
});

test("cutoff slider updates approval, bad rate and hash", async ({ page }) => {
  const errors = await open(page);
  await page.locator("#view-tab-risk").click();
  const slider = page.locator("#cutoff-slider");
  const before = await page.locator('[data-testid="cut-approval"]').innerText();
  await slider.fill("560");
  await expect(page.locator('[data-testid="cut-approval"]')).not.toHaveText(before);
  await expect(page).toHaveURL(/cut=560/);
  await expect(page.locator("#cutoff-chart .plot-container")).toBeVisible({ timeout: 20_000 });
  await expectNoErrors(errors);
});

test("theme toggle cycles auto, light and dark", async ({ page }) => {
  const errors = await open(page);
  const initial = await page.locator("html").getAttribute("data-theme");
  await page.locator("#theme-toggle").click();
  await expect(page.locator("#theme-label")).toHaveText("Light");
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await page.locator("#theme-toggle").click();
  await expect(page.locator("#theme-label")).toHaveText("Dark");
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await page.locator("#theme-toggle").click();
  await expect(page.locator("#theme-label")).toHaveText("Auto");
  await expect(page.locator("html")).toHaveAttribute("data-theme", initial);
  await expectNoErrors(errors);
});

test("keyboard navigation reaches tiles, rows, chips and the drawer", async ({ page }) => {
  const errors = await open(page);
  const firstTile = page.locator("#tile-map .tile[data-id='AK']");
  await firstTile.focus();
  await page.keyboard.press("ArrowRight");
  const focused = await page.evaluate(() => document.activeElement?.getAttribute("data-id"));
  expect(focused).not.toBe("AK");
  await page.locator("#rank-list .rank-row").first().focus();
  await page.keyboard.press("ArrowDown");
  const rowFocus = await page.evaluate(() => document.activeElement?.getAttribute("data-id"));
  expect(rowFocus).toBeTruthy();
  await page.locator("#provenance-btn").click();
  await expect(page.locator("#provenance-drawer")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.locator("#provenance-drawer")).toBeHidden();
  await expectNoErrors(errors);
});

test("mobile 390px stacks panels without horizontal scrolling and falls back to heatmap", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  const errors = await open(page);
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow).toBeLessThanOrEqual(1);
  await page.locator("#view-tab-risk").click();
  await expect(page.locator("#loss-chart .plot-container")).toBeVisible({ timeout: 20_000 });
  await expect(page.locator("#surface-chart")).toHaveAttribute("data-mode", "heatmap");
  const overflowRisk = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflowRisk).toBeLessThanOrEqual(1);
  await expectNoErrors(errors);
});

test("blocked CDN falls back to static charts without page errors", async ({ page }) => {
  const errors = await open(page, { blockCdn: true });
  await page.locator("#view-tab-risk").click();
  await expect(page.locator("#loss-chart img")).toBeVisible({ timeout: 15_000 });
  await expect(page.locator("#surface-chart img")).toBeVisible({ timeout: 15_000 });
  await expect(page.locator("#data-error")).toBeHidden();
  await expectNoErrors(errors);
});

test("reduced motion keeps count-ups instant and turns the sweep into one step", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  const errors = await open(page);
  await page.locator("#view-tab-risk").click();
  await page.locator("#play-rho").click();
  await expect(page.locator("#play-rho")).toHaveAttribute("aria-pressed", "false");
  await expect(page.locator("#rho-output")).not.toHaveText("â€”");
  const duration = await page.evaluate(() => getComputedStyle(document.querySelector(".tile")).transitionDuration);
  expect(parseFloat(duration)).toBeLessThan(0.01);
  await expectNoErrors(errors);
});




