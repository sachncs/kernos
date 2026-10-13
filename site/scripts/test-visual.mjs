import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import { spawn } from "node:child_process";
import { chromium } from "@playwright/test";
import pixelmatch from "pixelmatch";
import { PNG } from "pngjs";

const host = "127.0.0.1";
const port = process.env.VISUAL_PORT ?? "4322";
const baseUrl = process.env.SITE_BASE_URL ?? `http://${host}:${port}/kernos/`;
const baselineDir = path.resolve("visual-baselines");
const npmCommand = process.platform === "win32" ? "npm.cmd" : "npm";
const cases = [
  { name: "home-desktop", width: 1280, height: 720 },
  { name: "home-phone", width: 390, height: 844 },
];
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

async function readPng(file) {
  return PNG.sync.read(await fs.readFile(file));
}

async function main() {
  await fs.mkdir(baselineDir, { recursive: true });
  await waitForPreview();
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage({ deviceScaleFactor: 1 });
    for (const testCase of cases) {
      await page.setViewportSize({ width: testCase.width, height: testCase.height });
      await page.goto(baseUrl, { waitUntil: "networkidle" });
      const baseline = path.join(baselineDir, `${testCase.name}.png`);
      const actual = await page.screenshot({ animations: "disabled" });
      if (process.env.UPDATE_SCREENSHOTS === "1") {
        await fs.writeFile(baseline, actual);
        continue;
      }
      const expected = await readPng(baseline);
      const received = PNG.sync.read(actual);
      assert.equal(received.width, expected.width, `${testCase.name}: width changed`);
      assert.equal(received.height, expected.height, `${testCase.name}: height changed`);
      const diff = new PNG({ width: expected.width, height: expected.height });
      const differingPixels = pixelmatch(
        expected.data,
        received.data,
        diff.data,
        expected.width,
        expected.height,
        { threshold: 0.1 },
      );
      if (differingPixels > 0) {
        const diffPath = path.join(baselineDir, `${testCase.name}.diff.png`);
        await fs.writeFile(diffPath, PNG.sync.write(diff));
      }
      assert.equal(differingPixels, 0, `${testCase.name}: ${differingPixels} pixels differ`);
    }
  } finally {
    await browser.close();
    server.kill();
  }
}

try {
  await main();
  console.log(process.env.UPDATE_SCREENSHOTS === "1" ? "visual baselines updated" : "visual baseline test passed");
} finally {
  server.kill();
}
