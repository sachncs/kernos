import summary from "../../../examples/evaluation/results/summary.md?raw";

const matchRequired = (pattern, label) => {
  const match = summary.match(pattern);
  if (!match) throw new Error(`Missing benchmark summary field: ${label}`);
  return match;
};

const datasetCount = matchRequired(/Sweep over (\d+) dataset/, "dataset count")[1];
const delta = matchRequired(/Mean ΔRMSE vs best baseline \| ([^|]+) \|/, "mean delta")[1];
const nystromWins = matchRequired(/Win rate vs Nystrom \| ([^|]+) \|/, "Nyström win rate")[1];
const rffWins = matchRequired(/Win rate vs RFF \| ([^|]+) \|/, "RFF win rate")[1];
const rankingSection = summary.match(/## Per-dataset ranking[\s\S]*?(?=## Ablation impact)/)?.[0] ?? "";
const rankings = [...rankingSection.matchAll(/^\| ([^|]+) \| (\d+) \/ (\d+) \|/gm)].map((match) => ({
  name: match[1].trim(),
  rank: Number(match[2]),
  total: Number(match[3]),
}));

if (rankings.length !== Number(datasetCount)) {
  throw new Error(`Benchmark summary has ${rankings.length} rankings; expected ${datasetCount}`);
}

export const benchmarkData = {
  datasetCount,
  delta,
  nystromWins,
  rffWins,
  rankings,
  summary,
};
