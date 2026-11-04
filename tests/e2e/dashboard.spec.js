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
  await expect(page.locator("#tile-map .tile:visible, #tile-map .us-map-svg .st:visible, #tile-map .state-chip:visible").first()).toBeVisible();
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
  await expect(page.locator("#tile-map .us-map-svg .st")).toHaveCount(PE.dimensions.state.length);
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
    if (dim === "state") await expect(page.locator("#tile-map .us-map-svg .st")).toHaveCount(items.length);
    else if (dim === "grade") await expect(page.locator("#tile-map .grade-step")).toHaveCount(items.length);
    else await expect(page.locator("#tile-map .tile:not(.empty)")).toHaveCount(items.length);
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

test("every selectable state shape selects the linked list row, summary and hash", async ({ page }) => {
  test.setTimeout(120_000);
  const errors = await open(page);
  const states = PE.dimensions.state;
  for (const item of states.filter((entry) => !entry.insufficient)) {
    await page.locator(`#tile-map .us-map-svg .st[data-state="${item.id}"]`).evaluate((element) => element.dispatchEvent(new MouseEvent("click", { bubbles: true })));
    await expect(page.locator(`#tile-map .us-map-svg .st[data-state="${item.id}"]`)).toHaveClass(/is-selected/);
    await expect(page.locator(`#rank-list .rank-row[data-id="${item.id}"]`)).toHaveAttribute("aria-current", "true");
    await expect(page.locator("#summary-title")).toHaveText(item.name);
    await expect(page.locator('[data-testid="summary-default-rate"]')).toHaveText(pct(item.default_rate));
    await expect(page).toHaveURL(new RegExp(`sel=${item.id}`));
  }
  await expectNoErrors(errors);
});

test("state map exposes small-state buttons and blocks masked states", async ({ page }) => {
  const errors = await open(page);
  const selectable = page.locator("#tile-map .us-map-svg .st:not(.is-masked)");
  const masked = page.locator("#tile-map .us-map-svg .st.is-masked");
  await expect(selectable).toHaveCount(47);
  await expect(masked).toHaveCount(4);
  const maskedIowa = page.locator('#tile-map .us-map-svg .st[data-state="IA"]');
  await expect(maskedIowa).toHaveAttribute("aria-disabled", "true");
  await expect(maskedIowa).toHaveAttribute("tabindex", "-1");
  const before = page.url();
  await maskedIowa.click({ force: true });
  expect(page.url()).toBe(before);
  await maskedIowa.hover();
  await expect(page.locator(".geo-tooltip")).toHaveText("IA");

  const rhodeIsland = page.locator('.state-small-strip .state-chip[data-id="RI"]');
  await rhodeIsland.hover();
  await expect(page.locator('.us-map-svg .st[data-state="RI"]')).toHaveClass(/is-hovered/);
  await rhodeIsland.click();
  await expect(page.locator("#summary-title")).toHaveText("RI");
  await expect(page.locator('.us-map-svg .st[data-state="RI"]')).toHaveClass(/is-selected/);
  await expect(rhodeIsland).toHaveClass(/is-selected/);
  await expect(page).toHaveURL(/sel=RI/);
  await page.setViewportSize({ width: 1100, height: 900 });
  await page.goto("/#view=explorer&dim=state");
  await expect(page.locator("#tile-map .us-map-svg .st")).toHaveCount(51);
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - innerWidth);
  expect(overflow).toBeLessThanOrEqual(1);
  await expectNoErrors(errors);
});

test("state map follows the metric selector", async ({ page }) => {
  const errors = await open(page);
  await page.locator('#tile-map .us-map-svg .st[data-state="CA"]').evaluate((element) => element.dispatchEvent(new MouseEvent("click", { bubbles: true })));
  await page.waitForTimeout(250);
  const fills = () => page.locator("#tile-map .us-map-svg .st:not(.is-masked)").evaluateAll((elements) => elements.map((element) => getComputedStyle(element).fill));
  const legend = () => page.locator("#map-legend .legend-num").allTextContents();
  const chipFill = () => page.locator('.state-small-strip .state-chip[data-id="RI"]').evaluate((element) => getComputedStyle(element).backgroundColor);
  const defaultFills = await fills();
  const defaultLegend = await legend();
  const defaultChip = await chipFill();
  await expect(page.locator("#map-subtitle")).toContainText("color = observed default");

  await page.locator("#metric-tabs button[data-metric='pred_pd']").click();
  await expect(page.locator("#map-subtitle")).toContainText("color = predicted PD");
  await page.waitForTimeout(250);
  const predFills = await fills();
  expect(predFills.filter((fill, index) => fill !== defaultFills[index]).length).toBeGreaterThanOrEqual(10);
  expect(await legend()).not.toEqual(defaultLegend);
  expect(await chipFill()).not.toBe(defaultChip);
  await expect(page.locator('#tile-map .us-map-svg .st[data-state="CA"]')).toHaveClass(/is-selected/);
  await page.locator('#tile-map .us-map-svg .st[data-state="CA"]').hover();
  await expect(page.locator(".geo-tooltip")).toHaveText("CA");

  await page.locator("#metric-tabs button[data-metric='el_pct_ead']").click();
  await expect(page.locator("#map-subtitle")).toContainText("color = expected loss");
  await page.waitForTimeout(250);
  expect(await fills()).not.toEqual(defaultFills);
  await expect(page.locator('#tile-map .us-map-svg .st[data-state="CA"]')).toHaveClass(/is-selected/);
  await page.locator('#tile-map .us-map-svg .st[data-state="CA"]').hover();
  await expect(page.locator(".geo-tooltip")).toHaveText("CA");

  await page.locator("#metric-tabs button[data-metric='default_rate']").click();
  await expect(page.locator("#map-subtitle")).toContainText("color = observed default");
  await page.waitForTimeout(250);
  expect(await fills()).toEqual(defaultFills);
  expect(await legend()).toEqual(defaultLegend);
  expect(await chipFill()).toBe(defaultChip);
  await expect(page.locator('#tile-map .us-map-svg .st[data-state="CA"]')).toHaveClass(/is-selected/);

  for (const [width, height] of [[1100, 900], [390, 844]]) {
    await page.setViewportSize({ width, height });
    await page.goto("/#view=explorer&dim=state&metric=pred_pd");
    await expect(page.locator("#map-subtitle")).toContainText("color = predicted PD");
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - innerWidth);
    expect(overflow, `${width}px`).toBeLessThanOrEqual(1);
  }
  await expectNoErrors(errors);
});

test("all info content resolves and every visible info button opens and closes", async ({ page }) => {
  const INFO = JSON.parse(fs.readFileSync(path.join(ROOT, "docs/content/info.json"), "utf8"));
  const LEARN = JSON.parse(fs.readFileSync(path.join(ROOT, "docs/content/learn.json"), "utf8"));
  const texts = Object.values(INFO).map((entry) => entry.text);
  expect(new Set(texts).size, "duplicate info text").toBe(texts.length);
  expect(JSON.stringify({ INFO, LEARN })).not.toContain("How do we know?");

  const errors = await open(page);
  for (const view of ["explorer", "risk", "evidence", "learn"]) {
    await page.locator(`#view-tab-${view}`).click();
    await page.waitForTimeout(300);
    expect(await page.locator("body").innerText()).not.toMatch(/\{[a-z0-9_]+\}/i);
    const buttons = page.locator("[data-info]");
    const count = await buttons.count();
    for (let index = 0; index < count; index += 1) {
      const button = buttons.nth(index);
      if (!(await button.isVisible())) continue;
      await button.click();
      const popover = page.locator("#info-popover");
      await expect(popover).toBeVisible();
      expect(await popover.innerText()).not.toMatch(/\{[a-z0-9_]+\}/i);
      await page.keyboard.press("Escape");
      await expect(popover).toBeHidden();
      await button.focus();
      await page.keyboard.press("Enter");
      await expect(popover).toBeVisible();
      await page.keyboard.press("Escape");
      await expect(popover).toBeHidden();
    }
  }
  await expectNoErrors(errors);
});

test("state subtitles, small-state strip and expected-loss note stay correct", async ({ page }) => {
  const errors = await open(page);
  const subtitle = page.locator("#map-subtitle");
  const note = page.locator("#map-note");
  const panelHeight = () => page.locator(".map-panel").evaluate((element) => element.getBoundingClientRect().height);

  await expect(subtitle).toContainText("50 states and DC");
  await expect(subtitle).not.toContainText("51 states");
  const stripIds = await page.locator("#tile-map .state-small-strip .state-chip").evaluateAll((nodes) => nodes.map((node) => node.dataset.id));
  expect(stripIds).toEqual(["RI", "DE", "DC", "CT", "NJ", "MD", "MA", "NH", "VT"]);

  const before = await panelHeight();
  await expect(note).not.toContainText("Expected loss is roughly");
  await page.locator("#metric-tabs button[data-metric='el_pct_ead']").click();
  await page.waitForTimeout(200);
  const lgd = PE.meta.mean_lgd.toFixed(3);
  await expect(note).toHaveText(`Expected loss is roughly PD x the portfolio LGD (${lgd}), so it ranks segments almost like Predicted PD.`);
  expect(await note.innerText()).not.toMatch(/\{[a-z0-9_]+\}/i);
  await expect(subtitle).toContainText("color = expected loss");
  await expect(subtitle).not.toContainText("51 states");
  expect(Math.abs(before - await panelHeight())).toBeLessThanOrEqual(1);

  await page.locator("#metric-tabs button[data-metric='default_rate']").click();
  await expect(note).not.toContainText("Expected loss is roughly");
  await page.locator("#dimension-tabs button[data-dim='grade']").click();
  await page.waitForTimeout(200);
  await expect(note).not.toContainText("Expected loss is roughly");
  await page.locator("#metric-tabs button[data-metric='el_pct_ead']").click();
  await expect(note).toContainText(`portfolio LGD (${lgd})`);
  await expectNoErrors(errors);
});

test("state map and grade staircase keep the same camera across metrics", async ({ page }) => {
  const errors = await open(page);
  const metrics = ["default_rate", "pred_pd", "el_pct_ead"];
  const measure = () => page.evaluate(() => {
    const box = (selector) => {
      const rect = document.querySelector(selector).getBoundingClientRect();
      return { x: rect.x, y: rect.y, width: rect.width, height: rect.height };
    };
    const svg = document.querySelector(".us-map-svg, .grade-svg");
    return {
      svg: box(svg.matches(".us-map-svg") ? ".us-map-svg" : ".grade-svg"),
      legend: box("#map-legend"),
      panel: box(".map-panel"),
      viewBox: svg.getAttribute("viewBox"),
    };
  });
  const compare = (first, next, label) => {
    for (const key of ["x", "y", "width", "height"]) {
      expect(Math.abs(first.svg[key] - next.svg[key]), `${label} svg ${key}`).toBeLessThanOrEqual(1);
      expect(Math.abs(first.legend[key] - next.legend[key]), `${label} legend ${key}`).toBeLessThanOrEqual(1);
      expect(Math.abs(first.panel[key] - next.panel[key]), `${label} panel ${key}`).toBeLessThanOrEqual(1);
    }
    expect(first.viewBox).toBe(next.viewBox);
  };
  for (const width of [1360, 1100]) {
    await page.setViewportSize({ width, height: 950 });
    for (const dim of ["state", "grade"]) {
      await page.goto(`/#view=explorer&dim=${dim}`);
      await page.waitForTimeout(500);
      await page.locator("#metric-tabs button[data-metric='default_rate']").click();
      await page.waitForTimeout(150);
      const first = await measure();
      for (const metric of metrics.slice(1)) {
        await page.locator(`#metric-tabs button[data-metric='${metric}']`).click();
        await page.waitForTimeout(150);
        compare(first, await measure(), `${width} ${dim} ${metric}`);
      }
    }
  }
  await expectNoErrors(errors);
});

test("state map zoom, pan and keyboard controls work", async ({ page }) => {
  const errors = await open(page);
  const svg = page.locator(".us-map-svg").first();
  const viewBox = () => svg.getAttribute("viewBox");
  const initial = await viewBox();
  await page.locator('[data-map-zoom="in"]').click();
  const zoomed = await viewBox();
  expect(zoomed).not.toBe(initial);
  await page.locator('[data-map-zoom="reset"]').click();
  expect(await viewBox()).toBe(initial);

  await page.locator(".state-map-wrap").focus();
  await page.keyboard.press("+");
  expect(await viewBox()).not.toBe(initial);
  await page.keyboard.press("0");
  expect(await viewBox()).toBe(initial);

  const mapBox = await svg.boundingBox();
  await page.mouse.move(mapBox.x + mapBox.width / 2, mapBox.y + mapBox.height / 2);
  await page.mouse.wheel(0, -160);
  await page.waitForTimeout(120);
  expect(await viewBox()).toBe(initial);
  await page.locator(".state-map-wrap").evaluate((element) => {
    const rect = element.getBoundingClientRect();
    element.dispatchEvent(new WheelEvent("wheel", {
      deltaY: -160,
      ctrlKey: true,
      clientX: rect.left + rect.width / 2,
      clientY: rect.top + rect.height / 2,
      bubbles: true,
      cancelable: true,
    }));
  });
  await page.waitForTimeout(120);
  const ctrlZoomed = await viewBox();
  expect(ctrlZoomed).not.toBe(initial);

  await page.locator('[data-map-zoom="reset"]').click();
  const ca = page.locator('.st[data-state="CA"]');
  await ca.click();
  await expect(page.locator("#summary-title")).toHaveText("CA");
  await page.locator('[data-map-zoom="in"]').click();
  const beforeDrag = await viewBox();
  const svgBox = await svg.boundingBox();
  await page.mouse.move(svgBox.x + svgBox.width / 2, svgBox.y + svgBox.height / 2);
  await page.mouse.down();
  await page.mouse.move(svgBox.x + svgBox.width / 2 + 28, svgBox.y + svgBox.height / 2 + 14, { steps: 8 });
  await page.mouse.up();
  await page.waitForTimeout(120);
  expect(await viewBox()).not.toBe(beforeDrag);
  await expect(page.locator("#summary-title")).toHaveText("CA");

  await page.locator("#metric-tabs button[data-metric='pred_pd']").click();
  await page.waitForTimeout(200);
  expect(await viewBox()).not.toBe(initial);
  await page.locator('.st[data-state="ND"]').click({ force: true });
  await expect(page.locator("#summary-title")).toHaveText("CA");
  await page.locator('#tile-map .state-small-strip .state-chip[data-id="MA"]').click();
  await expect(page.locator("#summary-title")).toHaveText("MA");

  await page.locator("#dimension-tabs button[data-dim='grade']").click();
  await page.waitForTimeout(200);
  await page.locator("#dimension-tabs button[data-dim='state']").click();
  await page.waitForTimeout(500);
  expect(await viewBox()).toBe(initial);
  await expectNoErrors(errors);
});

test("grade staircase renders A to G and selects a grade", async ({ page }) => {
  const errors = await open(page);
  await page.locator('#dimension-tabs button[data-dim="grade"]').click();
  const steps = page.locator(".grade-step");
  await expect(steps).toHaveCount(7);
  const ids = await steps.evaluateAll((elements) => elements.map((element) => element.dataset.id));
  expect(ids).toEqual(["A", "B", "C", "D", "E", "F", "G"]);
  const stepG = page.locator('.grade-step[data-id="G"]');
  await expect(stepG).toHaveAttribute("aria-disabled", "true");
  await expect(stepG).toHaveAttribute("tabindex", "-1");
  const stepB = page.locator('.grade-step[data-id="B"]');
  await stepB.hover();
  await expect(page.locator(".geo-tooltip")).toHaveText("Grade B");
  await stepB.click();
  await expect(page.locator("#summary-title")).toHaveText("B");
  await expect(stepB).toHaveClass(/is-selected/);
  await expect(page).toHaveURL(/sel=B/);
  const before = page.url();
  await stepG.click({ force: true });
  expect(page.url()).toBe(before);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/#view=explorer&dim=grade");
  await expect(page.locator(".grade-step")).toHaveCount(7);
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - innerWidth);
  expect(overflow).toBeLessThanOrEqual(1);
  await expectNoErrors(errors);
});

test("grade staircase follows the metric selector", async ({ page }) => {
  const errors = await open(page);
  await page.locator('#dimension-tabs button[data-dim="grade"]').click();
  const stepData = () => page.locator(".grade-step").evaluateAll((elements) => elements.map((step) => ({
    id: step.dataset.id,
    height: Math.round(step.querySelector(".grade-step-face").getBoundingClientRect().height * 100) / 100,
    label: step.querySelector(".grade-step-rate").textContent,
    fill: getComputedStyle(step.querySelector(".grade-step-face")).fill,
  })));
  const defaultData = await stepData();
  const defaultLegend = await page.locator("#map-legend .legend-num").allTextContents();
  await page.locator('.grade-step[data-id="B"]').click();
  await expect(page.locator("#map-subtitle")).toContainText("color = observed default");

  await page.locator("#metric-tabs button[data-metric='pred_pd']").click();
  await expect(page.locator("#map-subtitle")).toContainText("color = predicted PD");
  const predData = await stepData();
  expect(predData.map((step) => step.height)).not.toEqual(defaultData.map((step) => step.height));
  expect(predData.map((step) => step.label)).not.toEqual(defaultData.map((step) => step.label));
  expect(await page.locator("#map-legend .legend-num").allTextContents()).not.toEqual(defaultLegend);
  const gradeB = PE.dimensions.grade.find((item) => item.id === "B");
  await expect(page.locator('.grade-step[data-id="B"] .grade-step-rate')).toHaveText(pct(gradeB.pred_pd, 1));
  await expect(page.locator('.grade-step[data-id="B"]')).toHaveClass(/is-selected/);
  await page.locator('.grade-step[data-id="B"]').hover();
  await expect(page.locator(".geo-tooltip")).toHaveText("Grade B");

  await page.locator("#metric-tabs button[data-metric='el_pct_ead']").click();
  await expect(page.locator("#map-subtitle")).toContainText("color = expected loss");
  const elData = await stepData();
  expect(elData.map((step) => step.height)).not.toEqual(predData.map((step) => step.height));
  await expect(page.locator('.grade-step[data-id="B"] .grade-step-rate')).toHaveText(pct(gradeB.el_pct_ead, 1));
  await expect(page.locator('.grade-step[data-id="B"]')).toHaveClass(/is-selected/);

  await page.locator("#metric-tabs button[data-metric='default_rate']").click();
  await expect(page.locator("#map-subtitle")).toContainText("color = observed default");
  const restored = await stepData();
  expect(restored.map((step) => step.height)).toEqual(defaultData.map((step) => step.height));
  expect(restored.map((step) => step.label)).toEqual(defaultData.map((step) => step.label));
  expect(await page.locator("#map-legend .legend-num").allTextContents()).toEqual(defaultLegend);

  for (const [width, height] of [[1100, 900], [390, 844]]) {
    await page.setViewportSize({ width, height });
    await page.goto("/#view=explorer&dim=grade&metric=el_pct_ead");
    await expect(page.locator("#map-subtitle")).toContainText("color = expected loss");
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - innerWidth);
    expect(overflow, `${width}px`).toBeLessThanOrEqual(1);
  }
  await expectNoErrors(errors);
});

test("state map switches to a full state-button grid at 390px", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  const errors = await open(page);
  await expect(page.locator(".state-map-wrap")).toBeHidden();
  await expect(page.locator(".state-all-grid")).toBeVisible();
  await expect(page.locator(".state-all-grid .state-chip")).toHaveCount(51);
  await page.locator('.state-all-grid .state-chip[data-id="CA"]').click();
  await expect(page.locator("#summary-title")).toHaveText("CA");
  await expect(page).toHaveURL(/sel=CA/);
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - innerWidth);
  expect(overflow).toBeLessThanOrEqual(1);
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
  await page.locator('#dimension-tabs button[data-dim="purpose"]').click();
  const tile = page.locator('#tile-map .tile[data-id="debt_consolidation"]');
  const before = await tile.evaluate((el) => getComputedStyle(el).backgroundColor);
  await page.locator("#metric-tabs button[data-metric='el_pct_ead']").click();
  await expect(page).toHaveURL(/metric=el_pct_ead/);
  const after = await tile.evaluate((el) => getComputedStyle(el).backgroundColor);
  expect(after).not.toBe(before);
  const firstRow = page.locator("#rank-list .rank-row").first();
  const topId = await firstRow.getAttribute("data-id");
  const topItem = PE.dimensions.purpose.find((x) => x.id === topId);
  await expect(firstRow.locator(".rank-value")).toHaveText(pct(topItem.el_pct_ead));
  await expectNoErrors(errors);
});

test("state map hover shows a tooltip and clicking updates the linked details", async ({ page }) => {
  const errors = await open(page);
  const state = page.locator('#tile-map .us-map-svg .st[data-state="CA"]');
  const label = await state.getAttribute("aria-label");
  expect(label).toBe("CA");
  await state.hover();
  await expect(page.locator(".geo-tooltip")).toBeVisible();
  await expect(page.locator(".geo-tooltip")).toHaveText("CA");
  await state.click();
  await expect(page.locator("#summary-title")).toHaveText("CA");
  await expect(page.locator('[data-testid="summary-default-rate"]')).toHaveText(pct(PE.dimensions.state.find((x) => x.id === "CA").default_rate));
  await expect(page).toHaveURL(/sel=CA/);
  await expectNoErrors(errors);
});

test("explorer visuals expose only the segment name on hover and focus", async ({ page }) => {
  const errors = await open(page);
  const state = page.locator('#tile-map .us-map-svg .st[data-state="CA"]');
  await state.hover();
  await expect(page.locator(".geo-tooltip")).toHaveText("CA");
  await state.focus();
  await expect(page.locator(".geo-tooltip")).toHaveText("CA");
  await expect(state).toHaveAttribute("aria-label", "CA");

  await page.locator("#dimension-tabs button[data-dim='grade']").click();
  const grade = page.locator('.grade-step[data-id="B"]');
  await grade.hover();
  await expect(page.locator(".geo-tooltip")).toHaveText("Grade B");
  await grade.focus();
  await expect(page.locator(".geo-tooltip")).toHaveText("Grade B");
  await expect(grade).toHaveAttribute("aria-label", "Grade B");

  const cases = [
    ["purpose", "moving"],
    ["vintage", "2014"],
    ["home_ownership", "RENT"],
  ];
  for (const [dim, id] of cases) {
    await page.locator(`#dimension-tabs button[data-dim='${dim}']`).click();
    const item = PE.dimensions[dim].find((entry) => entry.id === id);
    const tile = page.locator(`#tile-map .tile[data-id="${id}"]`);
    await expect(tile).toHaveAttribute("title", item.name);
    await expect(tile).toHaveAttribute("aria-label", item.name);
    await tile.hover();
    expect(await tile.getAttribute("title")).toBe(item.name);
  }

  await page.locator("#dimension-tabs button[data-dim='state']").click();
  const chip = page.locator('.state-small-strip .state-chip[data-id="RI"]');
  await expect(chip).toHaveAttribute("title", "RI");
  await expect(chip).toHaveAttribute("aria-label", "RI");
  const titles = await page.locator("#tile-map [title]").evaluateAll((nodes) => nodes.map((node) => node.getAttribute("title")));
  for (const title of titles) {
    expect(title).not.toMatch(/%|CI|n\s*=|loans|share/i);
  }
  await expectNoErrors(errors);
});

test("selection block shows one context row, three metric rows and a model check", async ({ page }) => {
  const errors = await open(page);
  const selection = page.locator(".selection-part-a");
  const item = PE.dimensions.state.find((entry) => entry.id === "AL");
  const total = PE.dimensions.total[0];
  await page.locator('#tile-map .us-map-svg .st[data-state="AL"]').click();
  await expect(page.locator("#summary-title")).toHaveText("AL");
  await expect(selection.locator("#summary-subtitle, #summary-gap, .eyebrow")).toHaveCount(0);
  await expect(selection.locator('[data-testid="summary-loans"]')).toHaveText(item.n.toLocaleString("en-US"));
  await expect(selection.locator('[data-testid="summary-ead-share"]')).toHaveText(pct(item.ead_share, 1));
  const contextText = await selection.locator(".selection-context").innerText();
  expect(contextText.match(/Loans/g)?.length).toBe(1);
  expect(contextText.match(/Share of EAD/g)?.length).toBe(1);

  const rows = selection.locator(".selection-metric-row");
  await expect(rows).toHaveCount(3);
  await expect(selection.locator(".selection-metric-label > span")).toHaveText(["Observed default", "Predicted PD", "Expected loss (% of EAD)"]);
  await expect(selection.locator('[data-testid="summary-default-rate"]')).toHaveText(pct(item.default_rate));
  await expect(selection.locator('[data-testid="summary-pred-pd"]')).toHaveText(pct(item.pred_pd));
  await expect(selection.locator('[data-testid="summary-el"]')).toHaveText(pct(item.el_pct_ead));
  await expect(selection.locator(".selection-ci")).toContainText(`95% CI ${pct(item.ci_low)}–${pct(item.ci_high)}`);

  const rank = (metric) => {
    const peers = PE.dimensions.state
      .filter((entry) => !entry.insufficient && Number.isFinite(entry[metric]))
      .sort((a, b) => b[metric] - a[metric]);
    return peers.findIndex((entry) => entry.id === item.id) + 1;
  };
  for (const metric of ["default_rate", "pred_pd", "el_pct_ead"]) {
    const row = selection.locator(`[data-row-metric="${metric}"]`);
    const diff = (item[metric] - total[metric]) * 100;
    await expect(row.locator(".selection-metric-diff")).toHaveText(`${diff >= 0 ? "+" : ""}${diff.toFixed(2)} pp vs portfolio`);
    await expect(row.locator(".selection-metric-rank")).toHaveText(`#${rank(metric)} of 47`);
  }
  await expect(selection.locator('[data-row-metric="default_rate"]')).toHaveClass(/is-active/);
  const before = await selection.locator("#summary-kpis").innerText();
  await page.locator("#metric-tabs button[data-metric='pred_pd']").click();
  await expect(selection.locator('[data-row-metric="pred_pd"]')).toHaveClass(/is-active/);
  await expect(selection.locator('[data-row-metric="default_rate"]')).not.toHaveClass(/is-active/);
  expect(await selection.locator("#summary-kpis").innerText()).toBe(before);

  const expectedCheck = (() => {
    const observed = pct(item.default_rate);
    const predicted = pct(item.pred_pd);
    if (item.pred_pd < item.ci_low) {
      return `Model check: observed ${observed} vs predicted ${predicted}: the model underestimates risk by ${(Math.abs(item.default_rate - item.pred_pd) * 100).toFixed(2)} pp (predicted PD is outside the 95% CI).`;
    }
    if (item.pred_pd > item.ci_high) {
      return `Model check: observed ${observed} vs predicted ${predicted}: the model overestimates risk by ${(Math.abs(item.default_rate - item.pred_pd) * 100).toFixed(2)} pp (predicted PD is outside the 95% CI).`;
    }
    return `Model check: observed ${observed} vs predicted ${predicted}: predicted PD is within the 95% CI.`;
  })();
  await expect(selection.locator('[data-testid="summary-model-check"]')).toHaveText(expectedCheck);
  await expect(selection.locator('[data-info="model-check"]')).toBeVisible();

  await page.locator("#dimension-tabs button[data-dim='total']").click();
  await expect(page.locator("#summary-title")).toHaveText("Total portfolio");
  await expect(selection.locator(".selection-metric-row")).toHaveCount(3);
  await expect(selection.locator(".selection-metric-diff, .selection-metric-rank")).toHaveCount(0);
  await expect(selection.locator('[data-testid="summary-model-check"]')).toBeVisible();
  await expect(selection.locator('[data-row-metric="pred_pd"]')).toHaveClass(/is-active/);
  await expectNoErrors(errors);
});

test("model check reports an inside-CI segment", async ({ page }) => {
  const errors = await open(page);
  const item = PE.dimensions.state.find((entry) => entry.id === "HI");
  await page.goto(`/#view=explorer&dim=state&sel=${item.id}`);
  await page.waitForTimeout(500);
  await expect(page.locator("#summary-title")).toHaveText("HI");
  await expect(page.locator('[data-testid="summary-model-check"]')).toHaveText(
    `Model check: observed ${pct(item.default_rate)} vs predicted ${pct(item.pred_pd)}: predicted PD is within the 95% CI.`
  );
  await expectNoErrors(errors);
});

test("selection block keeps one size across every tab at 1360, 1100 and 390", async ({ page }) => {
  const errors = await open(page);
  const cases = [
    ["total", null],
    ["grade", "B"],
    ["purpose", "moving"],
    ["vintage", "2014"],
    ["home_ownership", "MORTGAGE"],
    ["state", "AL"],
  ];
  const measure = () => page.evaluate(() => {
    const selection = document.querySelector(".selection-part-a");
    const box = selection.getBoundingClientRect();
    return {
      width: box.width,
      height: box.height,
      scrollHeight: selection.scrollHeight,
      clientHeight: selection.clientHeight,
    };
  });
  for (const width of [1360, 1100, 390]) {
    let first = null;
    for (const [dim, sel] of cases) {
      await page.setViewportSize({ width, height: 900 });
      await page.goto(`/#view=explorer&dim=${dim}${sel ? `&sel=${sel}` : ""}`);
      await page.waitForTimeout(500);
      const next = await measure();
      first ||= next;
      expect(Math.abs(next.width - first.width), `${width} ${dim} width`).toBeLessThanOrEqual(2);
      expect(Math.abs(next.height - first.height), `${width} ${dim} height`).toBeLessThanOrEqual(2);
      expect(next.scrollHeight, `${width} ${dim} content`).toBeLessThanOrEqual(next.clientHeight + 1);
    }
    if (width >= 1100) {
      const alignment = await page.evaluate(() => {
        const box = (selector) => document.querySelector(selector).getBoundingClientRect();
        return {
          mapBottom: box(".map-panel").bottom,
          listBottom: box(".list-panel").bottom,
          howBottom: box(".how-built-panel").bottom,
          selectionBottom: box(".selection-part-a").bottom,
        };
      });
      expect(Math.abs(alignment.mapBottom - alignment.listBottom), `${width} top-row bottoms`).toBeLessThanOrEqual(2);
      expect(Math.abs(alignment.howBottom - alignment.selectionBottom), `${width} column bottoms`).toBeLessThanOrEqual(2);
    }
  }
  await expectNoErrors(errors);
});

test("explorer removed the score distribution and enlarges Selection type", async ({ page }) => {
  const errors = await open(page);
  await page.setViewportSize({ width: 1360, height: 900 });
  await page.goto("/#view=explorer&dim=state&sel=AL");
  await page.waitForTimeout(600);

  await expect(page.locator("#view-explorer #score-hist, #view-explorer [id*='score-distribution'], #view-explorer .selection-part-b")).toHaveCount(0);
  await expect(page.locator("#view-explorer")).not.toContainText(/score distribution/i);

  const type = await page.evaluate(() => {
    const selection = document.querySelector(".selection-part-a");
    const size = (selector, root = selection) => parseFloat(getComputedStyle(root.querySelector(selector)).fontSize);
    return {
      name: size("#summary-title"),
      activeValue: size(".selection-metric-row.is-active .selection-metric-value"),
      otherValues: [...selection.querySelectorAll(".selection-metric-row:not(.is-active) .selection-metric-value")]
        .map((el) => parseFloat(getComputedStyle(el).fontSize)),
      labels: [...selection.querySelectorAll(".selection-metric-label")].map((el) => parseFloat(getComputedStyle(el).fontSize)),
      context: parseFloat(getComputedStyle(selection.querySelector(".selection-context")).fontSize),
      compare: size(".selection-metric-compare"),
      rank: size(".selection-metric-rank"),
      ci: size(".selection-ci"),
      model: size("#summary-model-check"),
    };
  });
  expect(type.name).toBeGreaterThanOrEqual(28);
  expect(type.activeValue).toBeGreaterThanOrEqual(28);
  expect(type.otherValues).toHaveLength(2);
  for (const value of type.otherValues) expect(value).toBeGreaterThanOrEqual(24);
  for (const label of type.labels) expect(label).toBeGreaterThanOrEqual(14);
  expect(type.context).toBeGreaterThanOrEqual(14);
  expect(type.compare).toBeGreaterThanOrEqual(13);
  expect(type.rank).toBeGreaterThanOrEqual(13);
  expect(type.ci).toBeGreaterThanOrEqual(13);
  expect(type.model).toBeGreaterThanOrEqual(14);

  for (const width of [1360, 1100, 390]) {
    await page.setViewportSize({ width, height: 900 });
    await page.goto("/#view=explorer&dim=state&sel=AL");
    await page.waitForTimeout(400);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - innerWidth);
    expect(overflow, `${width}px horizontal overflow`).toBeLessThanOrEqual(1);
  }
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
  await expect(page.locator("#how-built-panel")).toBeVisible();
  await page.locator('#tile-map .us-map-svg .st[data-state="CA"]').click();
  await expect(page.locator("#summary-title")).toHaveText("CA");
  await expectNoErrors(errors);
});

test("hero KPI cards use ribbon pastels with centered ink text", async ({ page }) => {
  await page.emulateMedia({ colorScheme: "light" });
  const errors = await open(page);
  const cards = page.locator(".kpi-strip .kpi-card");
  await expect(cards).toHaveCount(4);
  const expectedBackgrounds = ["rgb(214, 205, 245)", "rgb(195, 207, 246)", "rgb(182, 220, 243)", "rgb(172, 223, 214)"];
  const results = await cards.evaluateAll((elements) => {
    const parse = (value) => {
      const match = value.match(/rgba?\(([\d.]+),\s*([\d.]+),\s*([\d.]+)/);
      return match ? [Number(match[1]), Number(match[2]), Number(match[3])] : [0, 0, 0];
    };
    const channel = (value) => {
      const v = value / 255;
      return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
    };
    const luminance = ([r, g, b]) => 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b);
    return elements.map((card) => {
      const style = getComputedStyle(card);
      const value = getComputedStyle(card.querySelector(".kpi-value"));
      const label = getComputedStyle(card.querySelector(".mini-label"));
      const sub = getComputedStyle(card.querySelector(".kpi-sub"));
      const fg = luminance(parse(value.color));
      const bg = luminance(parse(style.backgroundColor));
      return {
        background: style.backgroundColor,
        valueColor: value.color,
        valueAlign: value.textAlign,
        labelAlign: label.textAlign,
        subAlign: sub.textAlign,
        tintDisplay: getComputedStyle(card, "::after").display,
        ratio: (Math.max(fg, bg) + 0.05) / (Math.min(fg, bg) + 0.05),
      };
    });
  });
  results.forEach((result, index) => {
    expect(result.background).toBe(expectedBackgrounds[index]);
    expect(result.valueColor).toBe("rgb(31, 41, 55)");
    expect(result.valueAlign).toBe("center");
    expect(result.labelAlign).toBe("center");
    expect(result.subAlign).toBe("center");
    expect(result.tintDisplay).toBe("none");
    expect(result.ratio).toBeGreaterThanOrEqual(4.5);
  });
  await expectNoErrors(errors);
});

test("hero KPI cards use resolved, non-redundant texts", async ({ page }) => {
  await page.emulateMedia({ colorScheme: "light" });
  const errors = await open(page);
  const cards = page.locator(".kpi-strip .kpi-card");
  const meta = PE.meta;
  await expect(cards.nth(0).locator(".mini-label")).toHaveText("Loans analyzed");
  await expect(cards.nth(0)).toContainText(`Resolved Lending Club loans, ${DATA.display.data_years} · ${pct(meta.default_rate_total, 1)} ended in default`);
  await expect(cards.nth(1).locator(".mini-label")).toHaveText("Model accuracy (AUC)");
  await expect(cards.nth(1)).toContainText(`about ${Math.round(DATA.pd.champion_auc * 100)}% of the time`);
  await expect(cards.nth(1)).toContainText(`tested on ${DATA.display.oot_years} loans`);
  await expect(cards.nth(2).locator(".mini-label")).toHaveText("Expected loss");
  await expect(cards.nth(2)).toContainText(`of exposure: about ${DATA.display.el_amount} on average`);
  await expect(cards.nth(3).locator(".mini-label")).toHaveText("Bad-year loss (VaR 99.9%)");
  await expect(cards.nth(3)).toContainText("Exceeded once in 1,000 simulated years");
  await expect(cards.nth(3).locator(".kpi-sub")).toContainText("$");
  const heroText = await page.locator(".kpi-strip").innerText();
  expect(heroText).not.toMatch(/\{[a-zA-Z0-9_]+\}/);
  expect(heroText).not.toContain("Gini");
  expect(heroText).not.toMatch(/rho|ρ/i);
  expect(heroText).not.toContain("of EAD");
  for (const [width, height] of [[1360, 900], [1100, 900], [390, 844]]) {
    await page.setViewportSize({ width, height });
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - innerWidth);
    expect(overflow, `${width}px`).toBeLessThanOrEqual(1);
  }
  await expectNoErrors(errors);
});

test("back and forward restore explorer state", async ({ page }) => {
  const errors = await open(page);
  await page.locator('#tile-map .us-map-svg .st[data-state="CA"]').click();
  await expect(page).toHaveURL(/sel=CA/);
  await page.locator('#tile-map .us-map-svg .st[data-state="NY"]').click();
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
  await page.keyboard.press("Escape");
  await expect(page.locator("#welcome-overlay")).toBeHidden();
  await page.locator("#view-tab-explorer").click();
  await expect(page.locator("#view-explorer")).toBeVisible();
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

test("Evidence panels use two independent columns with natural heights", async ({ page }) => {
  const errors = await open(page);
  const expectedOrder = ["Model comparison", "Calibration and LGD", "Censoring and the test window", "What did not work, and limitations"];
  for (const [width, height] of [[1360, 900], [1100, 900]]) {
    await page.setViewportSize({ width, height });
    await page.goto("/#view=evidence");
    await expect(page.locator("#view-evidence")).toBeVisible();
    const layout = await page.evaluate(() => {
      const columns = [...document.querySelectorAll(".evidence-column")].map((column) => column.getBoundingClientRect());
      const panels = [...document.querySelectorAll(".evidence-column .panel")];
      const order = panels.map((panel) => panel.querySelector("h3").textContent);
      const blanks = panels.map((panel) => {
        const style = getComputedStyle(panel);
        const rect = panel.getBoundingClientRect();
        const kids = [...panel.children];
        const last = kids[kids.length - 1].getBoundingClientRect();
        return { title: panel.querySelector("h3").textContent, blank: Math.round(rect.bottom - last.bottom - parseFloat(style.paddingBottom || "0")) };
      });
      return {
        order,
        columnBottoms: columns.map((rect) => Math.round(rect.bottom)),
        blanks,
        columns: getComputedStyle(document.querySelector(".evidence-columns")).gridTemplateColumns.split(" ").length,
        overflow: document.documentElement.scrollWidth - window.innerWidth,
      };
    });
    expect(layout.order, `${width}px panel order`).toEqual(expectedOrder);
    expect(layout.columns, `${width}px column count`).toBe(2);
    expect(Math.abs(layout.columnBottoms[0] - layout.columnBottoms[1]), `${width}px column bottom difference`).toBeLessThan(2);
    const naturalBlanks = layout.blanks.filter((entry) => entry.title !== "Calibration and LGD").map((entry) => entry.blank);
    expect(Math.max(...naturalBlanks), `${width}px internal blank`).toBeLessThanOrEqual(40);
    expect(layout.overflow, `${width}px overflow`).toBeLessThanOrEqual(1);
  }
  await page.setViewportSize({ width: 900, height: 900 });
  await page.goto("/#view=evidence");
  const stacked = await page.evaluate(() => ({
    columns: getComputedStyle(document.querySelector(".evidence-columns")).gridTemplateColumns.split(" ").length,
    order: [...document.querySelectorAll(".evidence-column .panel")].map((panel) => panel.querySelector("h3").textContent),
    overflow: document.documentElement.scrollWidth - window.innerWidth,
  }));
  expect(stacked.columns).toBe(1);
  expect(stacked.order).toEqual(expectedOrder);
  expect(stacked.overflow).toBeLessThanOrEqual(1);
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

test("clicking the surface does not move the linked sliders", async ({ page }) => {
  const errors = await open(page);
  await page.locator("#view-tab-risk").click();
  const surface = page.locator("#surface-chart");
  await expect(surface.locator("canvas")).toBeVisible({ timeout: 20_000 });
  await surface.scrollIntoViewIfNeeded();
  await page.waitForTimeout(300);
  const readState = () => page.evaluate(() => ({
    rho: document.querySelector("#risk-rho").value,
    pd: document.querySelector("#risk-pd").value,
    lgd: document.querySelector("#risk-lgd").value,
    hash: location.hash,
    badge: document.querySelector("#scenario-badge").textContent,
  }));
  const before = await readState();
  const box = await surface.boundingBox();
  const cx = box.x + box.width * 0.52;
  const cy = box.y + box.height * 0.48;
  await page.mouse.click(cx, cy);
  await page.waitForTimeout(400);
  expect(await readState()).toEqual(before);

  const cameraBefore = await surface.evaluate((el) => JSON.stringify(el._fullLayout?.scene?.camera?.eye));
  await page.mouse.move(cx, cy);
  await page.mouse.down();
  await page.mouse.move(cx + 70, cy + 12, { steps: 10 });
  await page.mouse.up();
  await page.waitForTimeout(500);
  const cameraAfter = await surface.evaluate((el) => JSON.stringify(el._fullLayout?.scene?.camera?.eye));
  expect(cameraAfter, "dragging should still rotate the surface").not.toBe(cameraBefore);

  await page.mouse.move(cx, cy);
  await page.waitForTimeout(300);
  expect(await page.locator("#surface-chart .hoverlayer").count()).toBeGreaterThan(0);
  expect(await readState()).toEqual(before);
  await expectNoErrors(errors);
});

test("surface copy says it does not change the sliders", async ({ page }) => {
  const errors = await open(page);
  await page.locator("#view-tab-risk").click();
  await expect(page.locator("#surface-sub")).toContainText("Rotate/zoom");
  await expect(page.locator("#surface-sub")).toContainText("sliders stay");
  await page.locator('.info-btn[data-info="surface"]').click();
  await expect(page.locator("#info-popover")).toContainText("does not change the sliders");
  await page.keyboard.press("Escape");
  await page.locator("#view-tab-learn").click();
  const learnText = await page.locator("#view-learn").innerText();
  expect(learnText).not.toMatch(/click the surface|surface.{0,80}(move|moves|set|sets|change|changes) the sliders/i);
  const pageText = await page.locator("body").innerText();
  expect(pageText).not.toMatch(/click the surface|click a point to move the sliders/i);
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
  await expect(scenario.locator("#risk-kpis .summary-kpi")).toHaveCount(5);
  await expect(scenario.locator("#risk-kpis .mini-label")).toHaveText(["EL", "UL", "VaR 99%", "ES 99%", "ES 99.9%"]);
  await expect(scenario.locator("#risk-kpis")).not.toContainText("VaR 99.9%");
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
  const expectedGauge = async () => {
    const key = (await page.locator("#scenario-badge").innerText()).replace(/ · /g, "_");
    const exact = DATA.interactive.scenario_explorer.histograms[key].exact;
    return `${(exact["var_0.999_pct_ead"] * 100).toFixed(1)}%`;
  };
  let before = await gaugeValue.textContent();
  await page.locator("#risk-rho").fill("5");
  await expect.poll(() => gaugeValue.textContent()).not.toBe(before);
  await expect(gaugeValue).toHaveText(await expectedGauge());
  before = await gaugeValue.textContent();
  await page.locator("#risk-pd").fill("2");
  await expect.poll(() => gaugeValue.textContent()).not.toBe(before);
  await expect(gaugeValue).toHaveText(await expectedGauge());
  before = await gaugeValue.textContent();
  await page.locator("#risk-lgd").fill("1");
  await expect.poll(() => gaugeValue.textContent()).not.toBe(before);
  await expect(gaugeValue).toHaveText(await expectedGauge());
  await expectNoErrors(errors);
});

test("tail statistics use one decimal and the hero matches the Risk Lab base case", async ({ page }) => {
  const errors = await open(page);
  const hero = await page.locator("#kpi-var").innerText();
  await page.locator("#view-tab-risk").click();
  await page.waitForTimeout(800);
  const gauge = await page.locator("#risk-gauge text:not(.scale)").textContent();
  expect(gauge).toBe(hero);
  expect(hero).toMatch(/^\d+\.\d%$/);
  const tails = await page.locator("#risk-kpis [data-testid^='risk-var'], #risk-kpis [data-testid^='risk-es'], #risk-kpis [data-testid='risk-expected-loss'], #risk-kpis [data-testid='risk-unexpected-loss']").allInnerTexts();
  expect(tails.length).toBe(5);
  for (const value of tails) expect(value).toMatch(/^\d+\.\d%$/);
  const legendValues = await page.locator("#loss-legend .legend-value").allInnerTexts();
  expect(legendValues.length).toBe(5);
  for (const value of legendValues) expect(value).toMatch(/^\d+\.\d%$/);
  await expectNoErrors(errors);
});

test("Scenario KPI row has five cards with no empty cell at 1360, 1000 and 390", async ({ page }) => {
  const errors = await open(page);
  for (const [width, height] of [[1360, 900], [1000, 900], [390, 844]]) {
    await page.setViewportSize({ width, height });
    await page.goto("/#view=risk");
    await expect(page.locator("#loss-chart .plot-container")).toBeVisible({ timeout: 20_000 });
    const layout = await page.evaluate(() => {
      const grid = document.querySelector("#risk-kpis");
      const style = getComputedStyle(grid);
      const contentWidth = grid.getBoundingClientRect().width - parseFloat(style.paddingLeft || "0") - parseFloat(style.paddingRight || "0");
      const gap = parseFloat(style.columnGap || "0");
      const cards = [...grid.querySelectorAll(".summary-kpi")];
      const rows = new Map();
      cards.forEach((card) => {
        const rect = card.getBoundingClientRect();
        const key = Math.round(rect.top);
        if (!rows.has(key)) rows.set(key, []);
        rows.get(key).push(rect.width);
      });
      return {
        count: cards.length,
        columns: style.gridTemplateColumns.split(" ").length,
        lastColumnSpan: getComputedStyle(cards[cards.length - 1]).gridColumnStart,
        rowOccupancy: [...rows.values()].map((widths) => widths.reduce((sum, value) => sum + value, 0) + gap * (widths.length - 1)),
        contentWidth,
        overflow: document.documentElement.scrollWidth - window.innerWidth,
      };
    });
    expect(layout.count, `${width}px card count`).toBe(5);
    expect(layout.columns, `${width}px grid columns`).toBe(width >= 700 ? 5 : 2);
    for (const [index, occupancy] of layout.rowOccupancy.entries()) {
      expect(Math.abs(occupancy - layout.contentWidth), `${width}px row ${index} fills the grid`).toBeLessThanOrEqual(2);
    }
    if (width < 700) expect(layout.lastColumnSpan).toBe("span 2");
    expect(layout.overflow, `${width}px overflow`).toBeLessThanOrEqual(1);
  }
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
  await expect.poll(() => page.locator("#loss-chart").evaluate((el) => el.layout?.font?.color)).toBe("#e5e7eb");
  await expect.poll(() => page.locator("#surface-chart").evaluate((el) => el.layout?.font?.color)).toBe("#e5e7eb");

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

test("welcome dismissal opens Learn while the backdrop keeps the current tab", async ({ page }) => {
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
  await expectNoErrors(errors);
});

test("provenance drawer is removed and its key facts remain", async ({ page }) => {
  const errors = await open(page);
  await expect(page.locator("#provenance-btn, #provenance-drawer, #drawer-backdrop, #evidence-provenance")).toHaveCount(0);
  await page.locator("#view-tab-learn").click();
  await page.locator("#term-el > summary").click();
  await expect(page.locator("#term-el")).toContainText("Expected loss uses the exposure-weighted PD times the portfolio-level LGD, reconciled to the headline expected-loss rate.");
  const footer = page.locator(".site-footer");
  await expect(footer).toContainText("Reproduce:");
  await expect(footer.locator("code").filter({ hasText: "python -m src.pipeline all" })).toHaveCount(1);
  const bodyText = await page.locator("body").innerText();
  expect(bodyText).not.toMatch(/How do we know\?|Open provenance drawer|provenance drawer/i);
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

test("keyboard navigation reaches state shapes, grade steps and rows", async ({ page }) => {
  const errors = await open(page);
  const firstShape = page.locator('#tile-map .us-map-svg .st[data-state="AK"]');
  await firstShape.focus();
  await page.keyboard.press("Enter");
  await expect(page.locator("#summary-title")).toHaveText("AK");
  await page.locator('#dimension-tabs button[data-dim="grade"]').click();
  const stepA = page.locator('.grade-step[data-id="A"]');
  await stepA.focus();
  await page.keyboard.press("ArrowRight");
  await expect(page.locator('.grade-step[data-id="B"]')).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page.locator("#summary-title")).toHaveText("B");
  await page.locator("#rank-list .rank-row").first().focus();
  await page.keyboard.press("ArrowDown");
  const rowFocus = await page.evaluate(() => document.activeElement?.getAttribute("data-id"));
  expect(rowFocus).toBeTruthy();
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
  const required = ["loan", "default", "resolved-loan", "grade", "dti", "revolving-utilization", "annual-income", "home-ownership", "loan-purpose", "term", "vintage", "pd", "lgd", "ead", "el", "ul", "approval-rate", "bad-rate", "cutoff", "percentile", "scorecard", "score", "points", "pdo", "woe", "iv", "logistic-regression", "lightgbm", "champion", "features", "leakage", "calibration", "brier", "citl", "auc", "gini", "ks", "psi", "roc", "confusion-matrix", "out-of-time", "primary-window", "right-censoring", "small-segment-masking", "wilson-ci", "selection-bias", "monte-carlo", "vasicek", "rho", "systematic-factor", "stress-test", "var", "es", "loss-distribution", "importance-sampling", "basel-correlation", ];
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

test("Learn guide cards use ribbon pastels with ink text and white buttons", async ({ page }) => {
  await page.emulateMedia({ colorScheme: "light" });
  const errors = await open(page);
  await page.locator("#view-tab-learn").click();
  const cards = page.locator("#learn-guide .guide-card");
  await expect(cards).toHaveCount(5);
  const expectedBackgrounds = ["rgb(214, 205, 245)", "rgb(195, 207, 246)", "rgb(182, 220, 243)", "rgb(172, 223, 214)", "rgb(201, 232, 180)"];
  const results = await cards.evaluateAll((elements) => elements.map((card) => {
    const style = getComputedStyle(card);
    const title = getComputedStyle(card.querySelector("h3"));
    const body = getComputedStyle(card.querySelector("p"));
    const button = getComputedStyle(card.querySelector(".btn"));
    return {
      background: style.backgroundColor,
      titleColor: title.color,
      bodyColor: body.color,
      buttonColor: button.color,
      buttonBackground: button.backgroundColor,
      buttonBorderWidth: button.borderTopWidth,
      buttonBorderStyle: button.borderTopStyle,
    };
  }));
  results.forEach((result, index) => {
    expect(result.background).toBe(expectedBackgrounds[index]);
    expect(result.titleColor).toBe("rgb(31, 41, 55)");
    expect(result.bodyColor).toBe("rgb(31, 41, 55)");
    expect(result.buttonColor).toBe("rgb(31, 41, 55)");
    expect(result.buttonBackground).toBe("rgb(255, 255, 255)");
    expect(result.buttonBorderWidth).toBe("1px");
    expect(result.buttonBorderStyle).toBe("solid");
  });
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
  await expect(page.locator("#loss-legend .legend-item")).toHaveCount(5);
  const legend = await page.locator("#loss-legend").innerText();
  for (const label of ["EL", "VaR 99%", "VaR 99.9%", "ES 99%", "ES 99.9%"]) expect(legend).toContain(label);
  const legendValues = await page.locator("#loss-legend .legend-value").allInnerTexts();
  expect(legendValues.length).toBe(5);
  for (const value of legendValues) expect(value).toMatch(/^\d+\.\d%$/);
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

test("no unresolved placeholders appear in popovers, glossary or explorer panels", async ({ page }) => {
  const errors = await open(page);
  const placeholder = /\{[a-zA-Z0-9_]+\}/;
  const forbidden = ["undefined", "NaN"];
  const checkText = (text, where) => {
    expect(text, where).not.toMatch(placeholder);
    for (const word of forbidden) expect(text, where).not.toContain(word);
  };
  for (const view of ["explorer", "risk", "evidence", "learn"]) {
    await page.locator(`#view-tab-${view}`).click();
    await page.waitForTimeout(300);
    if (view === "explorer") {
      for (let step = 1; step <= 5; step += 1) {
        await page.locator(`.how-built-node-${step}`).click();
        await page.waitForTimeout(100);
        checkText(await page.locator("#how-built-reading").innerText(), `how-built step ${step}`);
      }
    }
    const ids = await page.evaluate(() => Array.from(new Set(Array.from(document.querySelectorAll("[data-info]")).map((el) => el.dataset.info))));
    for (const id of ids) {
      const button = page.locator(`[data-info="${id}"]`).first();
      if (!(await button.isVisible())) continue;
      await button.click();
      await page.waitForTimeout(100);
      checkText(await page.locator("#info-popover").innerText(), `info button ${id} on ${view}`);
      await page.keyboard.press("Escape");
    }
  }
  await page.locator("#view-tab-learn").click();
  await page.waitForTimeout(300);
  const glossaryText = await page.locator("#glossary-list").innerText();
  checkText(glossaryText, "glossary");
  checkText(await page.locator("#app-content").innerText(), "page");
  await expectNoErrors(errors);
});

test("explorer panels keep the State tab box geometry at 1360 and 1100", async ({ page }) => {
  const errors = await open(page);
  const selectors = [".how-built-panel", ".map-panel", ".selection-part-a", ".list-panel"];
  const readBoxes = () => page.evaluate((sels) => sels.map((selector) => {
    const rect = document.querySelector(selector).getBoundingClientRect();
    return { x: rect.x, y: rect.y, width: rect.width, height: rect.height };
  }), selectors);
  for (const width of [1360, 1100]) {
    await page.setViewportSize({ width, height: 900 });
    await page.goto("/#view=explorer");
    await page.waitForTimeout(800);
    await page.locator("#dimension-tabs button").filter({ hasText: "State" }).click();
    await page.waitForTimeout(500);
    const state = await readBoxes();
    for (const dim of ["Total", "Grade", "Purpose", "Vintage", "Home ownership"]) {
      await page.locator("#dimension-tabs button").filter({ hasText: dim }).click();
      await page.waitForTimeout(300);
      const boxes = await readBoxes();
      boxes.forEach((box, index) => {
        expect(box.x, `${width} ${dim} box ${index} x`).toBeCloseTo(state[index].x, 0);
        expect(box.y, `${width} ${dim} box ${index} y`).toBeCloseTo(state[index].y, 0);
        expect(box.width, `${width} ${dim} box ${index} width`).toBeCloseTo(state[index].width, 0);
        expect(box.height, `${width} ${dim} box ${index} height`).toBeCloseTo(state[index].height, 0);
      });
    }
    await page.locator("#dimension-tabs button").filter({ hasText: "State" }).click();
    await page.waitForTimeout(300);
    await page.locator('#tile-map .us-map-svg .st[data-state="CA"]').click();
    await page.waitForTimeout(300);
    const boxes = await readBoxes();
    boxes.forEach((box, index) => expect(box.height).toBeCloseTo(state[index].height, 0));
  }
  await expectNoErrors(errors);
});

test("resized explorer panels contain their content without clipping or horizontal scroll", async ({ page }) => {
  const errors = await open(page);
  const measure = () => page.evaluate(() => {
    const selection = document.querySelector(".selection-part-a");
    const list = document.querySelector(".rank-list");
    return {
      selectionClient: selection.clientHeight,
      selectionScroll: selection.scrollHeight,
      listClient: list.clientWidth,
      listScroll: list.scrollWidth,
      names: [...document.querySelectorAll(".rank-name")]
        .filter((name) => name.scrollWidth > name.clientWidth + 1)
        .map((name) => name.textContent.trim().slice(0, 40)),
    };
  });
  const check = async (label) => {
    const result = await measure();
    expect(result.selectionScroll, `${label} selection height`).toBeLessThanOrEqual(result.selectionClient + 1);
    expect(result.listScroll, `${label} ranked width`).toBeLessThanOrEqual(result.listClient + 1);
    expect(result.names, `${label} truncated names`).toEqual([]);
  };
  for (const width of [1360, 1100]) {
    await page.setViewportSize({ width, height: 900 });
    await page.goto("/#view=explorer");
    await page.waitForTimeout(700);
    for (const dim of ["Total", "State", "Grade", "Purpose", "Vintage", "Home ownership"]) {
      await page.locator("#dimension-tabs button").filter({ hasText: dim }).click();
      await page.waitForTimeout(200);
      await check(`${width} ${dim}`);
    }
    await page.locator("#dimension-tabs button").filter({ hasText: "State" }).click();
    await page.waitForTimeout(200);
    await page.locator('#tile-map .us-map-svg .st[data-state="CA"]').click();
    await page.waitForTimeout(200);
    await check(`${width} State CA`);
    await page.locator("#dimension-tabs button").filter({ hasText: "Purpose" }).click();
    await page.waitForTimeout(200);
    await page.locator("#rank-list .rank-row").first().click();
    await page.waitForTimeout(200);
    await check(`${width} Purpose selected row`);
  }
  await expectNoErrors(errors);
});

test("all visible text keeps a readable contrast in light and dark", async ({ page }) => {
  const contrastViolations = async () => page.evaluate(() => {
    const parse = (value) => {
      const match = value.match(/rgba?\(([\d.]+),\s*([\d.]+),\s*([\d.]+)(?:,\s*([\d.]+))?\)/);
      if (!match) return null;
      return { r: Number(match[1]), g: Number(match[2]), b: Number(match[3]), a: match[4] === undefined ? 1 : Number(match[4]) };
    };
    const channel = (value) => {
      const v = value / 255;
      return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
    };
    const luminance = (color) => 0.2126 * channel(color.r) + 0.7152 * channel(color.g) + 0.0722 * channel(color.b);
    const background = (element) => {
      let node = element;
      while (node) {
        const color = parse(getComputedStyle(node).backgroundColor);
        if (color && color.a > 0.05) return color;
        node = node.parentElement;
      }
      return { r: 255, g: 255, b: 255, a: 1 };
    };
    const out = [];
    for (const element of document.querySelectorAll("body *")) {
      if (element instanceof SVGElement || element.children.length) continue;
      const text = (element.textContent || "").trim();
      if (!text || element.closest("[aria-hidden='true']")) continue;
      const style = getComputedStyle(element);
      if (style.visibility === "hidden" || style.display === "none" || Number(style.opacity) === 0) continue;
      const rect = element.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) continue;
      const fg = parse(style.color);
      if (!fg || fg.a === 0) continue;
      const bg = background(element);
      const l1 = luminance(fg); const l2 = luminance(bg);
      const ratio = (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05);
      const size = parseFloat(style.fontSize);
      const bold = parseInt(style.fontWeight, 10) >= 700;
      const threshold = (size >= 24 || (size >= 18.66 && bold)) ? 3 : 4.5;
      if (ratio < threshold) out.push({ text: text.slice(0, 40), ratio: Number(ratio.toFixed(2)), threshold });
    }
    return out;
  });
  const errors = await open(page);
  for (const scheme of ["light", "dark"]) {
    await page.emulateMedia({ colorScheme: scheme, reducedMotion: "reduce" });
    await page.reload();
    await page.waitForTimeout(800);
    for (const view of ["learn", "explorer", "risk", "evidence"]) {
      await page.locator(`#view-tab-${view}`).click();
      await page.waitForTimeout(300);
      const violations = await contrastViolations();
      expect(violations, `${scheme} ${view}`).toEqual([]);
    }
    await page.locator("#view-tab-explorer").click();
    for (const dim of ["Total", "State", "Grade", "Purpose"]) {
      await page.locator("#dimension-tabs button").filter({ hasText: dim }).click();
      await page.waitForTimeout(300);
      const violations = await contrastViolations();
      expect(violations, `${scheme} ${dim}`).toEqual([]);
    }
  }
  await expectNoErrors(errors);
});

test("site source uses only the approved palette and neutral colors", async () => {
  const allowed = new Set([
    "#d6cdf5", "#a99be0", "#4b3d8a", "#c3cff6", "#8fa6ea", "#b6dcf3", "#7fbbe6",
    "#acdfd6", "#6fc6b9", "#c9e8b4", "#93d07f", "#1f2937", "#1d4ed8",
    "#f7f8fc", "#eef2ff", "#ffffff", "#fff", "#000", "#000000", "#e2e8f0", "#e5e7eb",
    "#0b1120", "#111827", "#172033", "#c4b5fd", "#93c5fd", "#0f172a",
    "#4b5563", "#5b6472", "#94a3b8", "#cbd5e1", "#d9dbe3",
    "#6a5aa8", "#566fb8", "#3d559c", "#4a86b3", "#356a96", "#3f9a8e", "#2c7c71", "#6aa14f", "#4f8638"
  ]);
  const files = [
    "docs/assets/site.css", "docs/assets/site.js", "docs/index.html",
    ...fs.readdirSync(path.join(ROOT, "docs/content")).filter((name) => name.endsWith(".json")).map((name) => `docs/content/${name}`)
  ];
  const found = [];
  for (const file of files) {
    const text = fs.readFileSync(path.join(ROOT, file), "utf8");
    for (const match of text.matchAll(/#[0-9a-fA-F]{3,8}/g)) {
      const before = text.slice(Math.max(0, match.index - 4), match.index);
      if (before === "url(") continue;
      const color = match[0].toLowerCase();
      if (!allowed.has(color)) found.push(`${file}: ${color}`);
    }
  }
  expect(found).toEqual([]);
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
  const duration = await page.evaluate(() => getComputedStyle(document.querySelector(".us-map-svg .st")).transitionDuration);
  expect(parseFloat(duration)).toBeLessThan(0.01);
  await expectNoErrors(errors);
});






