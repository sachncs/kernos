import source from "../../../examples/benchmarks/site_snippets.py?raw";

const snippets = Object.fromEntries(
  [...source.matchAll(/# <snippet name="([^"]+)">\n([\s\S]*?)# <\/snippet>/g)].map((match) => [
    match[1],
    match[2].trim(),
  ])
);

export { snippets };
