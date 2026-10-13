import assert from "node:assert/strict";

import { highlight } from "../src/lib/highlight.js";

const rendered = highlight('# comment\nprint("ok")\nmodel.fit(X, y)');

assert.match(rendered, /<span class="hl-c"># comment<\/span>/);
assert.match(rendered, /<span class="hl-s">"ok"<\/span>/);
assert.doesNotMatch(rendered, /class=class|<span class="hl-[^"]+"><span/);
console.log("syntax highlighter smoke test passed");
