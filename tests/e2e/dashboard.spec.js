const { test, expect } = require("@playwright/test");
const fs = require("node:fs");
const path = require("node:path");

const ROOT = path.resolve(__dirname, "../..");
const DATA = JSON.parse(fs.readFileSync(path.join(ROOT, "docs/data.json"), "utf8"));
const PE = DATA.portfolio_explorer;
const PLOTLY_PATH = path.join(ROOT, "node_modules/plotly.js-dist-min/plotly.min.js");

const pct = (v, nd = 2) => `${(v * 100).toFixed(nd)}%`;
const dims = ["total", "state", "grade", "purpose", "vintage", "home_ownership"];

async function childBoxes(locator) {
  return locator.evaluate((el) => {
    const parent = el.getBoundingClientRect();
    return [...el.children].map((child) => {
      const rect = child.getBoundingClientRect();
      return { top: rect.top - parent.top, height: rect.height };
    });
  });
}

async function open(page, { blockCdn = false, keepWelcome = false, url = "/" } = {}) {
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
  await page.goto(url);
  await expect(page.locator("#app-content")).toBeVisible();
  await expect(page.locator("#tile-map .tile").first()).toBeVisible();
  await page.waitForTimeout(700);
  if (!keepWelcome && await page.locator("#welcome-start").isVisible().catch(() => false)) {
    await page.locator("#welcome-start").click();
    await expect(page.locator("#view-learn")).toBeVisible();
    await page.locator("#view-tab-explorer").click();
    await expect(page.locator("#view-explorer")).toBeVisible();
  }
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

test("single-valued Term dimension is hidden and the hash falls back safely", async ({ page }) => {
  const errors = await open(page);
  await expect(page.locator('#dimension-tabs button[data-dim="term"]')).toHaveCount(0);
  await page.goto("/#view=explorer&dim=term&metric=default_rate");
  await page.waitForTimeout(500);
  await expect(page.locator("#dimension-tabs button[data-dim='state']")).toHaveAttribute("aria-selected", "true");
  await expect(page).toHaveURL(/dim=state/);
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

test("metric tabs recolor the map and re-sort the list", async ({ page }) => {
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

test("map hover shows no tooltip and clicking still updates the linked details", async ({ page }) => {
  const errors = await open(page);
  await expect(page.locator("#map-tooltip")).toHaveCount(0);
  await expect(page.locator(".map-tooltip")).toHaveCount(0);
  const tile = page.locator("#tile-map .tile[data-id='CA']");
  const label = await tile.getAttribute("aria-label");
  expect(label).toContain("CA");
  expect(label).toContain("n=");
  await tile.hover();
  await page.waitForTimeout(150);
  await expect(page.locator(".map-tooltip")).toHaveCount(0);
  await expect(page.locator("#map-tooltip")).toHaveCount(0);
  await tile.click();
  await expect(page.locator("#summary-title")).toHaveText("CA");
  await expect(page.locator('[data-testid="summary-default-rate"]')).toHaveText(pct(PE.dimensions.state.find((x) => x.id === "CA").default_rate));
  await expect(page).toHaveURL(/sel=CA/);
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

test("removed header pill, hero note and insight chips stay out of the DOM", async ({ page }) => {
  const errors = await open(page);
  await expect(page.locator("#window-pill")).toHaveCount(0);
  await expect(page.locator(".window-pill")).toHaveCount(0);
  await expect(page.locator("#hero-window")).toHaveCount(0);
  await expect(page.locator("#hero-note-copy")).toHaveCount(0);
  await expect(page.locator(".hero-note")).toHaveCount(0);
  await expect(page.locator("#insight-chips")).toHaveCount(0);
  await expect(page.locator(".chip-row")).toHaveCount(0);
  await expect(page.locator("#insight-text")).toContainText("observed default");
  await page.locator("#tile-map .tile[data-id='CA']").click();
  await expect(page.locator("#summary-title")).toHaveText("CA");
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

test("view tabs are Learn first, default to Explorer and support keyboard arrows", async ({ page }) => {
  const errors = await open(page, { keepWelcome: true });
  await expect(page.locator(".view-tab")).toHaveText(["Learn", "Portfolio Explorer", "Risk Lab", "Evidence & limits"]);
  await expect(page.locator("#view-explorer")).toBeVisible();
  await expect(page.locator("#view-tab-explorer")).toHaveAttribute("aria-selected", "true");
  await expect(page.locator("#view-tab-explorer")).toHaveAttribute("tabindex", "0");
  await expect(page.locator("#view-tab-learn")).toHaveAttribute("tabindex", "-1");
  await page.locator("#welcome-provenance").click();
  await expect(page.locator("#welcome-overlay")).toBeHidden();
  await expect(page.locator("#view-explorer")).toBeVisible();
  await page.locator("#drawer-close").click();
  await page.locator("#view-tab-risk").click();
  await expect(page.locator("#view-risk")).toBeVisible();
  await expect(page.locator("#view-explorer")).toBeHidden();
  await expect(page).toHaveURL(/view=risk/);
  await expect(page.locator("#view-tab-risk")).toHaveAttribute("tabindex", "0");
  await page.locator("#view-tab-evidence").click();
  await expect(page.locator("#view-evidence")).toBeVisible();
  await expect(page.locator("#limitations-list li")).toHaveCount(DATA.limitations.length);
  await page.locator("#view-tab-risk").focus();
  await page.keyboard.press("Home");
  await expect(page.locator("#view-learn")).toBeVisible();
  await expect(page.locator("#view-tab-learn")).toBeFocused();
  await page.keyboard.press("End");
  await expect(page.locator("#view-evidence")).toBeVisible();
  await expect(page.locator("#view-tab-evidence")).toBeFocused();
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
  const before = await page.locator("#sim-score").innerText();
  await page.locator('#sim-presets [data-preset="higher"]').click();
  await expect(page.locator("#sim-score")).not.toHaveText(before);
  await expect(page.locator("#sim-pd")).toContainText("%");
  await expect(page.locator("#sim-pct")).toContainText("%");
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

test("Risk Lab side-by-side panels keep independent heights", async ({ page }) => {
  const errors = await open(page);
  await page.locator("#view-tab-risk").click();
  await expect(page.locator("#loss-chart .plot-container")).toBeVisible({ timeout: 20_000 });
  await expect(page.locator(".risk-grid").first()).toHaveCSS("align-items", "start");
  await expect(page.locator(".risk-grid.lower")).toHaveCSS("align-items", "start");
  const lowerGrid = page.locator(".risk-grid.lower");
  const [simBefore, cutoffBefore] = await childBoxes(lowerGrid);
  await page.locator("#sim-breakdown-details summary").click();
  await expect(page.locator("#sim-breakdown-details")).toHaveAttribute("open", "");
  await page.waitForTimeout(250);
  const [simAfter, cutoffAfter] = await childBoxes(lowerGrid);
  expect(simAfter.height).toBeGreaterThan(simBefore.height);
  expect(Math.abs(cutoffAfter.height - cutoffBefore.height)).toBeLessThanOrEqual(1);
  expect(Math.abs(cutoffAfter.top - cutoffBefore.top)).toBeLessThanOrEqual(1);
  await page.locator("#sim-breakdown-details summary").click();
  await expect(page.locator("#sim-breakdown-details")).not.toHaveAttribute("open", "");
  await page.waitForTimeout(250);
  const [, cutoffCollapsed] = await childBoxes(lowerGrid);
  expect(Math.abs(cutoffCollapsed.height - cutoffBefore.height)).toBeLessThanOrEqual(1);
  expect(Math.abs(cutoffCollapsed.top - cutoffBefore.top)).toBeLessThanOrEqual(1);
  await expectNoErrors(errors);
});

test("VaR gauge moves above the loss distribution and tracks every slider", async ({ page }) => {
  const errors = await open(page);
  await page.locator("#view-tab-risk").click();
  await expect(page.locator("#loss-chart .plot-container")).toBeVisible({ timeout: 20_000 });
  const scenario = page.locator(".scenario-panel");
  const gaugePanel = page.locator(".gauge-panel");
  await expect(scenario.locator("#risk-gauge")).toHaveCount(0);
  await expect(scenario.locator("#gauge-caption")).toHaveCount(0);
  await expect(scenario.locator(".slider-field")).toHaveCount(3);
  await expect(scenario.locator("#risk-kpis .summary-kpi")).toHaveCount(6);
  await expect(gaugePanel).toContainText("VaR 99.9% gauge");
  await expect(gaugePanel.locator("#risk-gauge")).toBeVisible();
  await expect(gaugePanel.locator('[data-info="gauge"]')).toBeVisible();
  await expect(gaugePanel.locator("#gauge-caption")).toContainText("VaR 99.9% on a 60% loss scale");
  const gaugeBeforeLoss = await page.evaluate(() => {
    const gauge = document.querySelector(".gauge-panel").getBoundingClientRect();
    const loss = document.querySelector(".risk-stack > .panel:nth-child(2)").getBoundingClientRect();
    return gauge.top < loss.top;
  });
  expect(gaugeBeforeLoss).toBe(true);

  const gaugeValue = page.locator("#risk-gauge text:not(.scale)");
  let before = await gaugeValue.textContent();
  await page.locator("#risk-rho").fill("5");
  await expect.poll(() => gaugeValue.textContent()).not.toBe(before);
  before = await gaugeValue.textContent();
  await page.locator("#risk-pd").fill("2");
  await expect.poll(() => gaugeValue.textContent()).not.toBe(before);
  before = await gaugeValue.textContent();
  await page.locator("#risk-lgd").fill("1");
  await expect.poll(() => gaugeValue.textContent()).not.toBe(before);
  await expectNoErrors(errors);
});

test("Risk Lab gauge row balances at 1360 and 1000 and stacks in order at 390", async ({ page }) => {
  const errors = await open(page);
  for (const [width, height] of [[1360, 900], [1000, 900], [390, 844]]) {
    for (const scheme of ["light", "dark"]) {
      await page.setViewportSize({ width, height });
      await page.emulateMedia({ colorScheme: scheme });
      await page.goto("/#view=risk");
      await expect(page.locator("#app-content")).toBeVisible();
      await expect(page.locator("#loss-chart .plot-container")).toBeVisible({ timeout: 20_000 });
      await expect(page.locator("#risk-gauge")).toBeVisible();
      const layout = await page.evaluate(() => {
        const box = (selector) => document.querySelector(selector).getBoundingClientRect();
        const gauge = box(".gauge-panel");
        const loss = box(".risk-stack > .panel:nth-child(2)");
        const stack = box(".risk-stack");
        const surface = box(".risk-grid > .panel");
        const scenario = box(".scenario-panel");
        return {
          gauge: { top: gauge.top + window.scrollY, height: gauge.height },
          loss: { top: loss.top + window.scrollY, height: loss.height },
          stack: { top: stack.top + window.scrollY, height: stack.height },
          surface: { top: surface.top + window.scrollY, height: surface.height },
          scenario: { top: scenario.top + window.scrollY },
          overflow: document.documentElement.scrollWidth - window.innerWidth,
        };
      });
      expect(layout.overflow, `${width}px ${scheme}`).toBeLessThanOrEqual(1);
      if (width === 390) {
        expect(layout.scenario.top).toBeLessThan(layout.gauge.top);
        expect(layout.gauge.top).toBeLessThan(layout.loss.top);
        expect(layout.loss.top).toBeLessThan(layout.surface.top);
      } else {
        expect(Math.abs(layout.stack.height - layout.surface.height), `${width}px ${scheme}`).toBeLessThan(40);
      }
    }
  }
  await expectNoErrors(errors);
});

test("theme follows the operating system and old theme links are ignored", async ({ page }) => {
  await page.emulateMedia({ colorScheme: "light" });
  const errors = await open(page, { url: "/#theme=dark" });
  await expect(page.locator("#theme-toggle")).toHaveCount(0);
  await expect(page.locator("#theme-label")).toHaveCount(0);
  await expect(page.locator('[data-info="theme-toggle"]')).toHaveCount(0);
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await expect(page).not.toHaveURL(/theme=/);
  await page.locator("#view-tab-risk").click();
  await expect(page.locator("#loss-chart .plot-container")).toBeVisible({ timeout: 20_000 });
  await expect(page.locator("#surface-chart .plot-container")).toBeVisible({ timeout: 20_000 });

  await page.emulateMedia({ colorScheme: "dark" });
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await expect.poll(() => page.locator("#loss-chart").evaluate((el) => el.layout?.font?.color)).toBe("#e2e8f0");
  await expect.poll(() => page.locator("#surface-chart").evaluate((el) => el.layout?.font?.color)).toBe("#e2e8f0");

  await page.emulateMedia({ colorScheme: "light" });
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await expect.poll(() => page.locator("#loss-chart").evaluate((el) => el.layout?.font?.color)).toBe("#1f2937");
  await expect.poll(() => page.locator("#surface-chart").evaluate((el) => el.layout?.font?.color)).toBe("#1f2937");
  await expectNoErrors(errors);
});

test("1360, 1000 and 390px stay clean in light and dark", async ({ page }) => {
  const errors = await open(page);
  for (const [width, height] of [[1360, 900], [1000, 900], [390, 844]]) {
    for (const scheme of ["light", "dark"]) {
      await page.setViewportSize({ width, height });
      await page.emulateMedia({ colorScheme: scheme });
      await page.reload();
      await expect(page.locator("#app-content")).toBeVisible();
      await expect(page.locator("#view-explorer")).toBeVisible();
      await expect(page.locator("html")).toHaveAttribute("data-theme", scheme);
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
      expect(overflow, `${width}px ${scheme}`).toBeLessThanOrEqual(1);
    }
  }
  await expectNoErrors(errors);
});

test("welcome overlay presents the introduction, fits both widths and traps focus", async ({ page }) => {
  await page.setViewportSize({ width: 1360, height: 900 });
  await page.emulateMedia({ colorScheme: "light" });
  const errors = await open(page, { keepWelcome: true });
  const overlay = page.locator("#welcome-overlay");
  const dialog = page.locator("#welcome-dialog");
  await expect(overlay).toBeVisible();
  await expect(dialog).toHaveAttribute("role", "dialog");
  await expect(dialog).toHaveAttribute("aria-modal", "true");
  await expect(page.locator("#welcome-title")).toHaveText("Credit Risk Studio");
  await expect(page.locator("#welcome-problem")).toHaveText("When a bank lends money, some borrowers will not pay it back. The bank does not know which ones, but it must still decide who to approve, at what price, and how much capital to keep in reserve.");
  await expect(dialog).toContainText("Underestimate the losses and the bank can run out of capital in a bad year. Overestimate them and it rejects good customers.");
  await expect(page.locator(".welcome-list li")).toHaveText([
    "Who is likely to default? A scorecard turns a borrower's profile into a probability of default (PD).",
    "How much could we lose? That probability becomes expected loss, and a Monte Carlo simulation measures bad-year losses (VaR).",
    "Can we trust it? Everything is tested on later loans the model never saw (2014-2015), not on the data it learned from.",
  ]);
  await expect(page.locator(".welcome-data")).toHaveCount(0);
  await expect(page.locator(".welcome-cards")).toHaveCount(0);
  await expect(page.locator("[data-welcome-target]")).toHaveCount(0);
  await expect(dialog).toBeFocused();

  const desktop = await dialog.evaluate((el) => {
    const box = el.getBoundingClientRect();
    return { x: box.x, y: box.y, right: box.right, bottom: box.bottom, client: el.clientHeight, scroll: el.scrollHeight };
  });
  expect(desktop.x).toBeGreaterThanOrEqual(0);
  expect(desktop.y).toBeGreaterThanOrEqual(0);
  expect(desktop.right).toBeLessThanOrEqual(1360);
  expect(desktop.bottom).toBeLessThanOrEqual(900);
  expect(desktop.scroll).toBeLessThanOrEqual(desktop.client + 1);
  const centers = await page.evaluate(() => {
    const copy = document.querySelector(".welcome-copy").getBoundingClientRect();
    const side = document.querySelector(".welcome-side").getBoundingClientRect();
    return { copy: copy.top + copy.height / 2, side: side.top + side.height / 2 };
  });
  expect(Math.abs(centers.copy - centers.side)).toBeLessThanOrEqual(1);
  for (const button of await page.locator("#welcome-dialog button").all()) {
    const box = await button.boundingBox();
    expect(box.height).toBeGreaterThanOrEqual(44);
  }

  await page.keyboard.press("Shift+Tab");
  expect(await page.evaluate(() => Boolean(document.activeElement?.closest("#welcome-dialog")))).toBe(true);

  const darkBg = await dialog.evaluate((el) => getComputedStyle(el).backgroundColor);
  await page.emulateMedia({ colorScheme: "dark" });
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  const lightOnDark = await dialog.evaluate((el) => getComputedStyle(el).backgroundColor);
  expect(lightOnDark).not.toBe(darkBg);

  await page.setViewportSize({ width: 390, height: 844 });
  await expect(overlay).toBeVisible();
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow).toBeLessThanOrEqual(1);
  const mobile = await dialog.evaluate((el) => {
    const box = el.getBoundingClientRect();
    return { x: box.x, right: box.right, bottom: box.bottom };
  });
  expect(mobile.x).toBeGreaterThanOrEqual(0);
  expect(mobile.right).toBeLessThanOrEqual(391);
  expect(mobile.bottom).toBeLessThanOrEqual(845);
  const startBox = await page.locator("#welcome-start").boundingBox();
  expect(startBox.y).toBeGreaterThanOrEqual(0);
  expect(startBox.y + startBox.height).toBeLessThanOrEqual(844);
  for (const button of await page.locator("#welcome-dialog button").all()) {
    const box = await button.boundingBox();
    expect(box.height).toBeGreaterThanOrEqual(44);
  }

  await page.keyboard.press("Escape");
  await expect(overlay).toBeHidden();
  await expect(page.locator("#view-learn")).toBeVisible();
  await expect(page.locator("#view-learn")).toBeFocused();
  await expect(page).toHaveURL(/view=learn/);
  await expectNoErrors(errors);
});

test("welcome dismissal opens Learn while provenance keeps the current tab", async ({ page }) => {
  await page.setViewportSize({ width: 1360, height: 900 });
  const errors = await open(page, { keepWelcome: true });
  await expect(page.locator("#welcome-overlay")).toBeVisible();
  await page.locator("#welcome-start").click();
  await expect(page.locator("#welcome-overlay")).toBeHidden();
  await expect(page.locator("#view-learn")).toBeVisible();
  await expect(page).toHaveURL(/view=learn/);

  await page.goto("/");
  await expect(page.locator("#app-content")).toBeVisible();
  await expect(page.locator("#welcome-overlay")).toBeVisible();
  await page.mouse.click(5, 5);
  await expect(page.locator("#welcome-overlay")).toBeHidden();
  await expect(page.locator("#view-learn")).toBeVisible();
  await expect(page).toHaveURL(/view=learn/);

  await page.goto("/");
  await expect(page.locator("#app-content")).toBeVisible();
  await expect(page.locator("#welcome-overlay")).toBeVisible();
  const before = page.url();
  await page.locator("#welcome-provenance").click();
  await expect(page.locator("#welcome-overlay")).toBeHidden();
  await expect(page.locator("#provenance-drawer")).toBeVisible();
  await expect(page.locator("#view-explorer")).toBeVisible();
  expect(page.url()).toBe(before);
  await page.locator("#drawer-close").click();
  await expect(page.locator("#provenance-drawer")).toBeHidden();
  await expectNoErrors(errors);
});

test("welcome overlay persists a dismissal and still opens Learn", async ({ page }) => {
  const errors = await open(page, { keepWelcome: true });
  await expect(page.locator("#welcome-overlay")).toBeVisible();
  await page.locator("#welcome-dismiss").check();
  await page.locator("#welcome-start").click();
  await expect(page.locator("#welcome-overlay")).toBeHidden();
  await expect(page.locator("#view-learn")).toBeVisible();
  await expect(page).toHaveURL(/view=learn/);
  expect(await page.evaluate(() => localStorage.getItem("credit-risk-studio:welcome-dismissed"))).toBe("1");
  await page.goto("/");
  await expect(page.locator("#app-content")).toBeVisible();
  await expect(page.locator("#welcome-overlay")).toBeHidden();
  await expect(page.locator("#view-explorer")).toBeVisible();
  await expectNoErrors(errors);
});

test("welcome overlay is suppressed by deep links", async ({ page }) => {
  const errors = await open(page, { keepWelcome: true, url: "/#view=explorer" });
  await expect(page.locator("#welcome-overlay")).toBeHidden();
  await page.goto("/#view=learn&term=pd");
  await expect(page.locator("#app-content")).toBeVisible();
  await expect(page.locator("#welcome-overlay")).toBeHidden();
  await expect(page.locator("#view-learn")).toBeVisible();
  await expect(page.locator("#term-pd")).toHaveAttribute("open", "");
  await expect(page.locator("#glossary-list")).toHaveCSS("align-items", "start");
  await expectNoErrors(errors);
});

test("deep links still select every view", async ({ page }) => {
  const errors = await open(page);
  for (const view of ["explorer", "risk", "evidence", "learn"]) {
    await page.goto(`/#view=${view}`);
    await expect(page.locator("#app-content")).toBeVisible();
    await expect(page.locator(`#view-${view}`)).toBeVisible();
    await expect(page.locator(`#view-tab-${view}`)).toHaveAttribute("aria-selected", "true");
    await expect(page.locator("#welcome-overlay")).toBeHidden();
  }
  await expectNoErrors(errors);
});

test("keyboard navigation reaches tiles, rows and the drawer", async ({ page }) => {
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
  const offenders = await page.evaluate(() => Array.from(document.querySelectorAll("*"))
    .map((el) => ({ el, box: el.getBoundingClientRect() }))
    .filter(({ box }) => box.right > window.innerWidth + 1 || box.left < -1)
    .map(({ el, box }) => `${el.tagName}#${el.id}.${String(el.className).slice(0, 40)} right=${box.right.toFixed(1)} width=${box.width.toFixed(1)}`));
  expect(overflow, offenders.join("\n")).toBeLessThanOrEqual(1);
  const mapOverflow = await page.locator("#tile-map").evaluate((el) => el.scrollWidth - el.clientWidth);
  expect(mapOverflow).toBeLessThanOrEqual(1);
  await page.locator("#view-tab-risk").click();
  await expect(page.locator("#loss-chart .plot-container")).toBeVisible({ timeout: 20_000 });
  await expect(page.locator("#surface-chart")).toHaveAttribute("data-mode", "heatmap");
  const overflowRisk = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  const riskOffenders = await page.evaluate(() => Array.from(document.querySelectorAll("*"))
    .map((el) => ({ el, box: el.getBoundingClientRect() }))
    .filter(({ box }) => box.right > window.innerWidth + 1 || box.left < -1)
    .map(({ el, box }) => `${el.tagName}#${el.id}.${String(el.className).slice(0, 40)} right=${box.right.toFixed(1)} width=${box.width.toFixed(1)}`));
  expect(overflowRisk, riskOffenders.join("\n")).toBeLessThanOrEqual(1);
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

test("every info button maps to content and opens one popover at a time", async ({ page }) => {
  const INFO = JSON.parse(fs.readFileSync(path.join(ROOT, "docs/content/info.json"), "utf8"));
  const errors = await open(page);
  await page.locator("#view-tab-risk").click();
  await page.waitForTimeout(600);
  const LEARN = JSON.parse(fs.readFileSync(path.join(ROOT, "docs/content/learn.json"), "utf8"));
  const ids = await page.evaluate(() => Array.from(new Set(Array.from(document.querySelectorAll("[data-info]")).map((el) => el.dataset.info))));
  for (const id of ids) {
    expect(INFO[id], `missing info content for ${id}`).toBeTruthy();
    expect(LEARN.terms.some((term) => term.id === INFO[id].term), `missing Learn term for ${id}`).toBeTruthy();
  }
  await page.locator("#view-tab-explorer").click();
  await page.waitForTimeout(300);
  const mapButton = page.locator('[data-info="map"]').first();
  await mapButton.click();
  await expect(page.locator("#info-popover")).toBeVisible();
  await expect(mapButton).toHaveAttribute("aria-expanded", "true");
  const popBox = await page.locator("#info-popover").boundingBox();
  expect(popBox.x).toBeGreaterThanOrEqual(0);
  expect(popBox.y).toBeGreaterThanOrEqual(0);
  expect(popBox.x + popBox.width).toBeLessThanOrEqual(1360);
  expect(popBox.y + popBox.height).toBeLessThanOrEqual(1000);
  await expect(page.locator("#info-popover .info-close")).toBeFocused();
  const second = page.locator('[data-info="metric-tabs"]').first();
  await second.click();
  await expect(second).toHaveAttribute("aria-expanded", "true");
  await expect(mapButton).toHaveAttribute("aria-expanded", "false");
  await page.keyboard.press("Escape");
  await expect(page.locator("#info-popover")).toBeHidden();
  await expect(second).toBeFocused();
  await mapButton.click();
  await page.locator("#summary-title").click();
  await expect(page.locator("#info-popover")).toBeHidden();
  await mapButton.focus();
  await page.keyboard.press("Enter");
  await expect(page.locator("#info-popover")).toBeVisible();
  await page.keyboard.press("Escape");
  await mapButton.focus();
  await page.keyboard.press("Space");
  await expect(page.locator("#info-popover")).toBeVisible();
  await page.keyboard.press("Escape");
  for (const view of ["explorer", "risk", "evidence", "learn"]) {
    await page.locator(`#view-tab-${view}`).click();
    await page.waitForTimeout(250);
    const buttons = page.locator("[data-info]:visible");
    const count = await buttons.count();
    for (let i = 0; i < count; i += 1) {
      const button = buttons.nth(i);
      await button.click();
      await expect(page.locator("#info-popover")).toBeVisible();
      await page.keyboard.press("Escape");
      await expect(page.locator("#info-popover")).toBeHidden();
    }
  }
  await expectNoErrors(errors);
});

test("info popovers work on mobile and do not overflow 390px", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  const errors = await open(page);
  await page.locator('[data-info="map"]').first().click();
  await expect(page.locator("#info-popover")).toBeVisible();
  const box = await page.locator("#info-popover").boundingBox();
  expect(box.x).toBeGreaterThanOrEqual(0);
  expect(box.x + box.width).toBeLessThanOrEqual(391);
  expect(box.y + box.height).toBeLessThanOrEqual(845);
  await page.keyboard.press("Escape");
  for (const view of ["explorer", "risk", "evidence", "learn"]) {
    await page.locator(`#view-tab-${view}`).click();
    await page.waitForTimeout(200);
    const buttons = page.locator("[data-info]:visible");
    const count = await buttons.count();
    for (let i = 0; i < count; i += 1) {
      await buttons.nth(i).click();
      const box = await page.locator("#info-popover").boundingBox();
      expect(box.x).toBeGreaterThanOrEqual(0);
      expect(box.x + box.width).toBeLessThanOrEqual(391);
      expect(box.y + box.height).toBeLessThanOrEqual(845);
      await page.keyboard.press("Escape");
    }
  }
  await expectNoErrors(errors);
});

test("Learn guide, glossary, categories, search and hash round-trip", async ({ page }) => {
  const errors = await open(page);
  const LEARN = JSON.parse(fs.readFileSync(path.join(ROOT, "docs/content/learn.json"), "utf8"));
  await page.locator("#view-tab-learn").click();
  await expect(page.locator("#learn-guide .guide-card")).toHaveCount(5);
  await expect(page.locator("#glossary-list")).toHaveCSS("align-items", "start");
  const glossaryEntries = page.locator("#glossary-list .glossary-entry");
  const firstEntry = glossaryEntries.first();
  const [firstBefore, secondBefore] = await childBoxes(page.locator("#glossary-list"));
  await firstEntry.locator("summary").click();
  await expect(firstEntry).toHaveAttribute("open", "");
  const [firstExpanded, secondExpanded] = await childBoxes(page.locator("#glossary-list"));
  expect(firstExpanded.height).toBeGreaterThan(firstBefore.height);
  expect(Math.abs(secondExpanded.height - secondBefore.height)).toBeLessThanOrEqual(1);
  expect(Math.abs(secondExpanded.top - secondBefore.top)).toBeLessThanOrEqual(1);
  await firstEntry.locator("summary").click();
  await expect(firstEntry).not.toHaveAttribute("open", "");
  const [, secondCollapsed] = await childBoxes(page.locator("#glossary-list"));
  expect(Math.abs(secondCollapsed.height - secondBefore.height)).toBeLessThanOrEqual(1);
  expect(Math.abs(secondCollapsed.top - secondBefore.top)).toBeLessThanOrEqual(1);
  const required = ["loan", "default", "resolved-loan", "grade", "dti", "revolving-utilization", "annual-income", "home-ownership", "loan-purpose", "term", "vintage", "pd", "lgd", "ead", "el", "ul", "approval-rate", "bad-rate", "cutoff", "percentile", "scorecard", "score", "points", "pdo", "woe", "iv", "logistic-regression", "lightgbm", "champion", "features", "leakage", "calibration", "brier", "citl", "auc", "gini", "ks", "psi", "roc", "confusion-matrix", "out-of-time", "primary-window", "right-censoring", "small-segment-masking", "wilson-ci", "selection-bias", "monte-carlo", "vasicek", "rho", "systematic-factor", "stress-test", "var", "es", "loss-distribution", "importance-sampling", "basel-correlation", "not-a-decision", "descriptive"];
  for (const id of required) {
    const entry = LEARN.terms.find((t) => t.id === id);
    expect(entry, `missing glossary term ${id}`).toBeTruthy();
    expect(entry.definition.length).toBeGreaterThan(10);
  }
  for (const entry of LEARN.terms) {
    expect(entry.term.length).toBeGreaterThan(1);
    expect(entry.definition.length).toBeGreaterThan(10);
    expect(["explorer", "risk", "evidence"]).toContain(entry.see.view);
    if (entry.see.dim) expect(DATA.portfolio_explorer.dimensions[entry.see.dim]).toBeTruthy();
    if (entry.see.sim && entry.see.sim !== "score") expect(DATA.interactive.applicant_simulator.var_meta[entry.see.sim]).toBeTruthy();
  }
  await page.locator("#glossary-search").fill("value at risk");
  await expect(page.locator("#glossary-list .glossary-entry")).toHaveCount(1);
  await expect(page.locator("#term-var")).toBeVisible();
  await page.locator('#glossary-categories button[data-category="Portfolio risk"]').click();
  await expect(page.locator("#glossary-list .glossary-entry")).not.toHaveCount(0);
  await page.goto("/#view=learn&term=pd");
  await page.waitForTimeout(900);
  await expect(page.locator("#term-pd")).toHaveAttribute("open", "");
  await page.reload();
  await page.waitForTimeout(900);
  await expect(page.locator("#term-pd")).toHaveAttribute("open", "");
  await expectNoErrors(errors);
});

test("info Learn more and glossary See it links resolve", async ({ page }) => {
  const errors = await open(page);
  await page.locator('[data-info="map"]').first().click();
  await expect(page.locator("#info-popover .info-learn")).toBeVisible();
  await page.locator("#info-popover .info-learn").click();
  await expect(page.locator("#view-learn")).toBeVisible();
  await expect(page.locator("#term-small-segment-masking")).toHaveAttribute("open", "");
  await page.locator("#glossary-search").fill("value at risk");
  await page.locator("#term-var").click();
  await page.locator('#term-var [data-see-term="var"]').click();
  await expect(page.locator("#view-risk")).toBeVisible();
  await expectNoErrors(errors);
});

test("simulator defaults to a typical applicant, uses readable bins and its breakdown sums", async ({ page }) => {
  const errors = await open(page);
  await page.locator("#view-tab-risk").click();
  await page.waitForTimeout(500);
  const app = DATA.interactive.applicant_simulator;
  let points = 0;
  for (const [variable, key] of Object.entries(app.presets.typical)) {
    const opt = app.var_meta[variable].options.find((o) => o.key === key);
    points += Number(opt.points);
  }
  const expectedScore = app.base_points + points;
  const displayed = Number(await page.locator("#sim-score").innerText());
  expect(Math.abs(displayed - expectedScore)).toBeLessThan(1);
  const factor = app.scaler.factor;
  const expectedPd = 1 / (1 + Math.exp(points / factor));
  const pdText = await page.locator("#sim-pd").innerText();
  expect(Math.abs(parseFloat(pdText) - expectedPd * 100)).toBeLessThan(0.01);
  const options = await page.locator("#sim-controls select option").allInnerTexts();
  for (const text of options) {
    expect(text).not.toMatch(/__|inf|e\+|\(|\)|\[|\]/i);
  }
  const labels = await page.locator("#sim-controls .sim-range-label").allInnerTexts();
  expect(labels.every((x) => x.trim())).toBeTruthy();
  for (const [variable, meta] of Object.entries(app.var_meta)) {
    for (const option of meta.options) {
      const jsLabel = await page.evaluate(({ variable, key }) => window.__formatBin(variable, key), { variable, key: option.key });
      expect(jsLabel).toBe(option.label);
    }
  }
  await page.locator("#sim-breakdown-details summary").click();
  const rows = await page.locator("#sim-breakdown tbody tr").allInnerTexts();
  const totalRow = rows[rows.length - 1];
  expect(totalRow).toContain(String(Math.round(expectedScore)));
  await page.locator("#sim-presets [data-preset='higher']").click();
  await page.waitForTimeout(700);
  const high = Number(await page.locator("#sim-score").innerText());
  expect(high).toBeLessThan(displayed);
  await page.locator("#sim-presets [data-preset='reset']").click();
  await expect(page.locator("#sim-score")).toHaveText(String(Math.round(expectedScore)));
  await expectNoErrors(errors);
});

test("loss markers use a readable legend and the surface card is large with a base-case label", async ({ page }) => {
  const errors = await open(page);
  await page.locator("#view-tab-risk").click();
  await page.waitForTimeout(2500);
  await expect(page.locator("#loss-legend .legend-item")).toHaveCount(4);
  const legend = await page.locator("#loss-legend").innerText();
  for (const label of ["EL", "VaR 99%", "VaR 99.9%", "ES 99%"]) expect(legend).toContain(label);
  for (const item of await page.locator("#loss-legend .legend-item").all()) {
    const box = await item.boundingBox();
    expect(box.width).toBeGreaterThan(20);
    expect(await item.evaluate((el) => el.scrollWidth - el.clientWidth)).toBeLessThanOrEqual(1);
  }
  const surfaceHeight = await page.locator("#surface-chart").evaluate((el) => el.getBoundingClientRect().height);
  expect(surfaceHeight).toBeGreaterThanOrEqual(520);
  const surfaceData = await page.locator("#surface-chart").evaluate((el) => el.data);
  expect(JSON.stringify(surfaceData)).toContain("Base case");
  await expect(page.locator("#surface-sub")).toContainText("base case");
  const toggle = page.locator("#surface-metrics");
  for (const button of await toggle.locator("button").all()) {
    const box = await button.boundingBox();
    expect(box.width).toBeGreaterThan(20);
    expect(await button.evaluate((el) => el.scrollWidth - el.clientWidth)).toBeLessThanOrEqual(1);
  }
  await expectNoErrors(errors);
});

test("no-JS fallback contains KPIs, model table and limitations", async ({ browser }) => {
  const context = await browser.newContext({ javaScriptEnabled: false, viewport: { width: 1360, height: 1000 } });
  const page = await context.newPage();
  await page.goto("/");
  await expect(page.locator("#app-content")).toBeVisible();
  await expect(page.locator("#welcome-overlay")).toBeHidden();
  await expect(page.locator("#kpi-loans")).toHaveText(DATA.display.n_rows);
  await expect(page.locator("#kpi-auc")).toHaveText(DATA.display.champion_auc);
  await expect(page.locator("#model-table tbody tr")).toHaveCount(7);
  await expect(page.locator("#limitations-list li")).toHaveCount(DATA.limitations.length);
  await context.close();
});

test("every docs asset is referenced", async () => {
  const assets = fs.readdirSync(path.join(ROOT, "docs/assets"));
  const refs = ["docs/index.html", "docs/assets/site.js", "docs/assets/site.css", "README.md"]
    .map((file) => fs.readFileSync(path.join(ROOT, file), "utf8"))
    .join("\n");
  for (const asset of assets) expect(refs, `unreferenced asset ${asset}`).toContain(asset);
});

test("reduced motion keeps count-ups instant and turns the sweep into one step", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  const errors = await open(page, { keepWelcome: true });
  await expect(page.locator("#welcome-overlay")).toBeVisible();
  const welcomeDuration = await page.locator("#welcome-dialog").evaluate((el) => getComputedStyle(el).animationDuration);
  expect(parseFloat(welcomeDuration)).toBeLessThan(0.01);
  await page.locator("#welcome-start").click();
  await page.locator("#view-tab-risk").click();
  await page.locator("#play-rho").click();
  await expect(page.locator("#play-rho")).toHaveAttribute("aria-pressed", "false");
  await expect(page.locator("#rho-output")).not.toHaveText("â€”");
  const duration = await page.evaluate(() => getComputedStyle(document.querySelector(".tile")).transitionDuration);
  expect(parseFloat(duration)).toBeLessThan(0.01);
  await expectNoErrors(errors);
});






