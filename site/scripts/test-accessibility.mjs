import assert from "node:assert/strict";
import { existsSync, readdirSync, readFileSync } from "node:fs";

const html = readFileSync(new URL("../dist/index.html", import.meta.url), "utf8");
const notFound = readFileSync(new URL("../dist/404.html", import.meta.url), "utf8");
const css = readdirSync(new URL("../dist/_astro", import.meta.url))
  .filter((file) => file.endsWith(".css"))
  .map((file) => readFileSync(new URL(`../dist/_astro/${file}`, import.meta.url), "utf8"))
  .join("\n");

const luminance = (hex) => {
  const channels = [1, 3, 5].map((offset) => parseInt(hex.slice(offset, offset + 2), 16) / 255);
  const linear = channels.map((channel) =>
    channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4
  );
  return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2];
};

const contrast = (foreground, background) => {
  const light = luminance(foreground);
  const dark = luminance(background);
  return (Math.max(light, dark) + 0.05) / (Math.min(light, dark) + 0.05);
};

assert.match(html, /<html lang="en"/);
assert.match(html, /class="skip-link" href="#main-content"/);
assert.match(html, /<main id="main-content"/);
assert.ok(existsSync(new URL("../dist/404.html", import.meta.url)), "Astro must emit a 404 page");
const ids = new Set([...html.matchAll(/\bid="([^\"]+)"/g)].map((match) => match[1]));
const hashLinks = [...html.matchAll(/href="#([^\"]+)"/g)].map((match) => match[1]);
assert.ok(hashLinks.length > 0, "the page must expose internal navigation links");
for (const target of hashLinks) assert.ok(ids.has(target), `missing internal link target: #${target}`);
assert.match(notFound, /Nothing here yet/);
assert.equal((html.match(/<h1\b/g) ?? []).length, 1);
assert.match(html, /<details[^>]*>/);
assert.match(html, /aria-label="Open navigation menu"/);
assert.match(html, /role="tablist"/);
assert.match(html, /aria-selected="true"/);
assert.match(html, /aria-controls="panel-fit"/);
assert.match(html, /role="tabpanel"[^>]*aria-labelledby="tab-fit"/);
assert.doesNotMatch(html, /href="LICENSE"/);
assert.doesNotMatch(html, /class=class/);
assert.doesNotMatch(html, /text-ink-500/);
assert.doesNotMatch(html, /0\.8923/);
assert.doesNotMatch(html, /exactly like an sklearn estimator/);
assert.doesNotMatch(html, /color:\s*"blue"/);
assert.match(html, /dataset, seed and training budget/);
assert.match(html, /docs\/evaluation-protocol\.md/);
assert.match(html, /docs\/api-reference\.md/);
assert.match(html, /70 \/ 15 \/ 15/);
assert.match(html, /five controlled CPU seeds/);
assert.match(html, /Small budget tier, five controlled CPU seeds/);
assert.match(html, /hardware and dataset changes can alter the outcome/);
assert.match(html, /peak Python allocation/);
assert.match(html, /Native RSS is not measured/);
assert.match(html, /Refresh is a measured trade-off, not a universal win/);
assert.match(html, /disabling refresh improves RMSE for some datasets/);
assert.match(html, /CI matrix: Python 3\.10–3\.14/);
assert.doesNotMatch(html, /Linux · macOS · Windows/);
assert.match(html, /Capabilities/);
assert.match(html, /System map/);
assert.match(html, /Evidence/);
assert.doesNotMatch(html, /fonts\.googleapis\.com/);
assert.match(html, /large batches or streaming data/);
assert.match(html, /refreshes landmarks only when they drift/);
assert.doesNotMatch(css, /#7c5cff|#06b6d4|#22d3ee|blue-500|cyan-500/);
assert.match(css, /#2dd4bf/);
assert.match(css, /#f97316/);
assert.ok(contrast("#8d9c98", "#0b1115") >= 4.5, "muted text must meet WCAG AA contrast");
assert.match(css, /prefers-reduced-motion/);
assert.match(css, /:focus-visible/);
console.log("built accessibility smoke test passed");
