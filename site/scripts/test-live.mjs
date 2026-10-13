import assert from "node:assert/strict";

const baseUrl = process.env.SITE_BASE_URL;
if (!baseUrl) {
  throw new Error("SITE_BASE_URL is required, for example https://example.test/kernos/");
}

const response = await fetch(baseUrl, { redirect: "follow" });
assert.equal(response.ok, true, `site returned HTTP ${response.status}`);
const html = await response.text();

assert.match(html, /Capabilities/, "deployed site is missing the capability section");
assert.match(html, /System map/, "deployed site is missing the architecture section");
assert.match(html, /Evidence/, "deployed site is missing the benchmark section");
assert.match(html, /CI matrix: Python 3\.10–3\.14/, "deployed site is missing the current release marker");
assert.doesNotMatch(html, /fonts\.googleapis\.com/, "deployed site still loads Google Fonts");
assert.doesNotMatch(html, /0\.8923/, "deployed site contains the retired hard-coded score");
assert.doesNotMatch(html, /Linux · macOS · Windows/, "deployed site contains the retired platform claim");

console.log(`live site contract passed: ${response.url}`);
