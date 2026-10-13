import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { chromium, firefox, webkit } from "@playwright/test";

const host = "127.0.0.1";
const port = "4322";
const baseUrl = process.env.SITE_BASE_URL ?? `http://${host}:${port}/kernos/`;
const npmCommand = process.platform === "win32" ? "npm.cmd" : "npm";
const browserName = process.env.BROWSER ?? "chromium";
const launchers = { chromium, firefox, webkit };
if (!(browserName in launchers)) {
  throw new Error(`Unsupported BROWSER=${browserName}; choose chromium, firefox, or webkit`);
}
const server = spawn(npmCommand, ["run", "preview", "--", "--host", host, "--port", port], {
  stdio: "ignore",
});

async function waitForPreview() {
  for (let attempt = 0; attempt < 40; attempt += 1) {
    try {
      const response = await fetch(baseUrl);
      if (response.ok) return;
    } catch {
      // Astro is still starting.
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error(`preview server did not start at ${baseUrl}`);
}

async function inspectLayoutOverflow(page) {
  return page.evaluate(() => {
    const offenders = [...document.querySelectorAll("*")]
      .filter((element) => {
        const rect = element.getBoundingClientRect();
        if (rect.right <= window.innerWidth + 1 && rect.left >= -1) return false;
        const style = getComputedStyle(element);
        if (style.position === "absolute" || style.position === "fixed") return false;
        for (let parent = element.parentElement; parent; parent = parent.parentElement) {
          const overflowX = getComputedStyle(parent).overflowX;
          if (["auto", "scroll", "hidden"].includes(overflowX)) return false;
        }
        return true;
      })
      .slice(0, 5)
      .map((element) => {
        const rect = element.getBoundingClientRect();
        return {
          tag: element.tagName,
          id: element.id,
          className: typeof element.className === "string" ? element.className.slice(0, 80) : "",
          right: Math.round(rect.right),
          width: Math.round(rect.width),
        };
      });
    return { ok: offenders.length === 0, offenders };
  });
}

try {
  await waitForPreview();
  const browser = await launchers[browserName].launch({ headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
    await page.goto(baseUrl, { waitUntil: "networkidle" });
    await assert.doesNotReject(() => page.locator("h1").waitFor({ state: "visible" }));
    assert.equal(await page.locator("nav.hidden").isVisible(), true);
    assert.equal(await page.locator("details").isVisible(), false);
    const desktopOverflow = await inspectLayoutOverflow(page);
    assert.equal(desktopOverflow.ok, true, `desktop layout overflow: ${JSON.stringify(desktopOverflow)}`);
    await page.locator(".skip-link").focus();
    await page.keyboard.press("Enter");
    assert.equal(new URL(page.url()).hash, "#main-content");

    await page.setViewportSize({ width: 320, height: 844 });
    const menu = page.locator("details");
    assert.equal(await menu.isVisible(), true);
    await menu.locator("summary").click();
    assert.equal(await page.locator('nav[aria-label="Mobile navigation"]').isVisible(), true);
    assert.equal(await page.locator('nav[aria-label="Mobile navigation"] a').count(), 5);
    const mobileOverflow = await inspectLayoutOverflow(page);
    assert.equal(mobileOverflow.ok, true, `small-phone layout overflow: ${JSON.stringify(mobileOverflow)}`);
    await menu.locator("summary").click();
    assert.equal(await page.locator('nav[aria-label="Mobile navigation"]').isVisible(), false);

    await page.locator("#api").scrollIntoViewIfNeeded();
    const gridTab = page.locator("#tab-grid");
    await gridTab.click();
    assert.equal(await gridTab.getAttribute("aria-selected"), "true");
    assert.equal(await page.locator("#panel-grid").isVisible(), true);
    assert.equal(await page.locator("#panel-fit").isVisible(), false);
    await gridTab.focus();
    await page.keyboard.press("ArrowRight");
    assert.equal(await page.locator("#tab-stream").getAttribute("aria-selected"), "true");
    assert.equal(await page.locator("#panel-stream").isVisible(), true);
    await page.keyboard.press("ArrowLeft");
    assert.equal(await page.locator("#tab-grid").getAttribute("aria-selected"), "true");

    const reduced = await browser.newPage({
      reducedMotion: "reduce",
      viewport: { width: 390, height: 844 },
    });
    await reduced.goto(baseUrl, { waitUntil: "networkidle" });
    assert.equal(await reduced.evaluate(() => matchMedia("(prefers-reduced-motion: reduce)").matches), true);
    await reduced.close();
  } finally {
    await browser.close();
  }
  console.log("browser smoke test passed");
} finally {
  server.kill();
}
