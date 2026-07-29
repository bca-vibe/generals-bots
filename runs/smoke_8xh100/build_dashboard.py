#!/usr/bin/env python3
"""Generate the live training dashboard HTML from metrics.jsonl."""
import json, sys, html, datetime

metrics_path = sys.argv[1]
out_path = sys.argv[2]
status = sys.argv[3] if len(sys.argv) > 3 else "Running"

train_rows, eval_rows = [], []
with open(metrics_path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "loss" in rec and "iteration" in rec:
            train_rows.append(rec)
        else:
            eval_rows.append(rec)

now = datetime.datetime.now().strftime("%H:%M %Z").strip()
today = datetime.date.today().isoformat()

payload = json.dumps({"train": train_rows, "evals": eval_rows,
                      "updated": f"{today} {now}", "status": status},
                     separators=(",", ":"))

page = """<title>smoke_8xh100 live training</title>
<style>
  :root {
    color-scheme: light;
    --page: #f9f9f7; --surface: #fcfcfb;
    --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
    --grid: #e1e0d9; --baseline: #c3c2b7; --ring: rgba(11,11,11,0.10);
    --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a;
    --good: #0ca30c; --good-text: #006300;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --page: #0d0d0d; --surface: #1a1a19;
      --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
      --grid: #2c2c2a; --baseline: #383835; --ring: rgba(255,255,255,0.10);
      --s1: #3987e5; --s2: #d95926; --s3: #199e70;
      --good: #0ca30c; --good-text: #0ca30c;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19;
    --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --baseline: #383835; --ring: rgba(255,255,255,0.10);
    --s1: #3987e5; --s2: #d95926; --s3: #199e70;
    --good: #0ca30c; --good-text: #0ca30c;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--page); color: var(--ink);
    font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  .wrap { max-width: 1080px; margin: 0 auto; padding: 28px 20px 48px; }
  header { display: flex; flex-wrap: wrap; align-items: baseline; gap: 10px 14px; margin-bottom: 6px; }
  h1 { font-size: 21px; font-weight: 650; margin: 0; letter-spacing: -0.01em; }
  .chip {
    font-size: 12px; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase;
    padding: 2px 10px; border-radius: 999px; border: 1px solid var(--ring);
  }
  .chip.running { color: var(--good-text); border-color: currentColor; }
  .chip.running::before { content: "\\25CF "; margin-right: 4px; animation: pulse 2s ease-in-out infinite; }
  .chip.done { color: var(--ink-2); }
  @keyframes pulse { 50% { opacity: 0.35; } }
  @media (prefers-reduced-motion: reduce) { .chip.running::before { animation: none; } }
  .updated { color: var(--muted); font-size: 13px; margin-left: auto; }
  .sub { color: var(--ink-2); font-size: 13.5px; margin: 0 0 20px; }
  .progress { height: 4px; border-radius: 2px; background: var(--grid); margin: 0 0 24px; overflow: hidden; }
  .progress > div { height: 100%; background: var(--s1); border-radius: 2px; }
  .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin-bottom: 26px; }
  .tile { background: var(--surface); border: 1px solid var(--ring); border-radius: 8px; padding: 12px 14px; }
  .tile .label { font-size: 11.5px; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; color: var(--muted); }
  .tile .value { font-size: 25px; font-weight: 650; letter-spacing: -0.01em; margin-top: 2px; }
  .tile .delta { font-size: 12.5px; color: var(--ink-2); }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 14px; }
  .card { background: var(--surface); border: 1px solid var(--ring); border-radius: 8px; padding: 14px 16px 10px; }
  .card h2 { font-size: 13.5px; font-weight: 640; margin: 0; }
  .card .desc { font-size: 12px; color: var(--muted); margin: 1px 0 6px; }
  .legend { display: flex; gap: 14px; font-size: 12px; color: var(--ink-2); margin: 2px 0 4px; }
  .legend .key { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 5px; vertical-align: -1px; }
  .chart { position: relative; }
  .chart svg { display: block; width: 100%; height: auto; }
  .tip {
    position: absolute; pointer-events: none; display: none; z-index: 3;
    background: var(--surface); border: 1px solid var(--ring); border-radius: 6px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.12); padding: 6px 9px; font-size: 12px; white-space: nowrap;
  }
  .tip .t-it { color: var(--muted); }
  .tip .t-v { font-weight: 620; font-variant-numeric: tabular-nums; }
  details { margin-top: 26px; }
  summary { cursor: pointer; font-size: 13.5px; font-weight: 600; color: var(--ink-2); }
  .tblwrap { overflow-x: auto; margin-top: 10px; background: var(--surface); border: 1px solid var(--ring); border-radius: 8px; }
  table { border-collapse: collapse; width: 100%; font-size: 12.5px; font-variant-numeric: tabular-nums; }
  th, td { text-align: right; padding: 6px 12px; border-bottom: 1px solid var(--grid); white-space: nowrap; }
  th { color: var(--muted); font-weight: 600; font-size: 11.5px; letter-spacing: 0.04em; text-transform: uppercase; }
  tr:last-child td { border-bottom: none; }
  th:first-child, td:first-child { text-align: left; }
  .evalcard { margin-top: 26px; background: var(--surface); border: 1px solid var(--ring); border-radius: 8px; padding: 14px 16px; }
  .evalcard h2 { font-size: 13.5px; font-weight: 640; margin: 0 0 6px; }
  .evalcard pre { margin: 0; font-size: 12px; overflow-x: auto; color: var(--ink-2); }
  a:focus-visible, summary:focus-visible { outline: 2px solid var(--s1); outline-offset: 2px; }
</style>
<div class="wrap">
  <header>
    <h1>smoke_8xh100</h1>
    <span id="chip" class="chip running">Running</span>
    <span class="updated" id="updated"></span>
  </header>
  <p class="sub">PPO self-play test run &mdash; 8&times;H100, 2-hour budget &middot; generals-bots competition_l7 recipe &middot; 15.3M params</p>
  <div class="progress" aria-hidden="true"><div id="pbar" style="width:0%"></div></div>
  <div class="tiles" id="tiles"></div>
  <div class="grid" id="charts"></div>
  <div class="evalcard" id="evalcard" hidden>
    <h2>Eval records</h2>
    <pre id="evalpre"></pre>
  </div>
  <details>
    <summary>Data table &mdash; last 30 iterations</summary>
    <div class="tblwrap"><table id="tbl"></table></div>
  </details>
</div>
<script>
const DATA = __PAYLOAD__;
const T = DATA.train;
const fmt = (v, d=3) => v == null ? "\u2013" : (Math.abs(v) >= 1000 ? Math.round(v).toLocaleString("en-US") : v.toFixed(d));
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

document.getElementById("updated").textContent = "updated " + DATA.updated;
const chip = document.getElementById("chip");
chip.textContent = DATA.status;
chip.className = "chip " + (DATA.status === "Running" ? "running" : "done");

const last = T[T.length - 1] || {};
const elapsed = last.wall_seconds || 0;
document.getElementById("pbar").style.width = Math.min(100, elapsed / 7200 * 100).toFixed(1) + "%";

const hours = Math.floor(elapsed / 3600), mins = Math.round((elapsed % 3600) / 60);
const tiles = [
  ["Iteration", String(last.iteration ?? "\u2013"), "of ~" + Math.round(7200 / 10.6) + " expected"],
  ["Samples / sec", fmt(last.samples_per_second, 0), "across 8 GPUs"],
  ["Explained variance", fmt(last.explained_variance, 3), "value-net quality"],
  ["Entropy", fmt(last.entropy, 3), "coef " + fmt(last.entropy_coefficient, 4)],
  ["Elapsed", hours + "h " + String(mins).padStart(2, "0") + "m", "of 2h budget \u00b7 stage " + (last.stage ?? 0)],
];
document.getElementById("tiles").innerHTML = tiles.map(([l, v, d]) =>
  `<div class="tile"><div class="label">${l}</div><div class="value">${v}</div><div class="delta">${d}</div></div>`).join("");

const S = { s1: css("--s1"), s2: css("--s2"), s3: css("--s3") };
const CHARTS = [
  { title: "Total loss", desc: "policy + value + entropy terms", series: [{ k: r => r.loss, c: "s1", name: "loss" }] },
  { title: "Explained variance", desc: "1 = value net fully explains returns", series: [{ k: r => r.explained_variance, c: "s1", name: "explained variance" }], ymin: 0, ymax: 1 },
  { title: "Entropy", desc: "policy exploration (nats)", series: [{ k: r => r.entropy, c: "s1", name: "entropy" }] },
  { title: "Approx KL", desc: "per-update policy shift · target 0.02", series: [{ k: r => r.approximate_kl, c: "s1", name: "KL" }], refline: 0.02 },
  { title: "Value loss", desc: "HL-Gauss cross-entropy", series: [{ k: r => r.value_loss, c: "s1", name: "value loss" }] },
  { title: "Clip fraction", desc: "share of clipped policy updates", series: [{ k: r => r.clip_fraction, c: "s1", name: "clip fraction" }] },
  { title: "Episode outcomes", desc: "share of episodes per iteration", legend: true, series: [
      { k: r => r.wins / Math.max(1, r.episodes), c: "s1", name: "Wins" },
      { k: r => r.losses / Math.max(1, r.episodes), c: "s2", name: "Losses" },
      { k: r => r.draws / Math.max(1, r.episodes), c: "s3", name: "Draws" }], ymin: 0, ymax: 1 },
  { title: "Throughput", desc: "environment samples per second", series: [{ k: r => r.samples_per_second, c: "s1", name: "samples/s" }], fmtd: 0 },
];

const W = 340, H = 150, M = { t: 8, r: 10, b: 20, l: 46 };
function niceTicks(lo, hi, n) {
  const span = hi - lo || 1, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 5, 10].map(m => m * mag).find(s => span / s <= n + 0.5) || mag * 10;
  const ticks = []; for (let v = Math.ceil(lo / step) * step; v <= hi + step / 1e6; v += step) ticks.push(v);
  return ticks;
}
const chartsEl = document.getElementById("charts");
for (const cfg of CHARTS) {
  const xs = T.map(r => r.iteration);
  const all = cfg.series.flatMap(s => T.map(s.k)).filter(v => v != null && isFinite(v));
  if (!all.length) continue;
  let lo = cfg.ymin ?? Math.min(...all), hi = cfg.ymax ?? Math.max(...all);
  if (cfg.refline != null) { lo = Math.min(lo, cfg.refline); hi = Math.max(hi, cfg.refline); }
  if (lo === hi) { lo -= 0.5; hi += 0.5; }
  const pad = (hi - lo) * 0.06; if (cfg.ymin == null) lo -= pad; if (cfg.ymax == null) hi += pad;
  const x = v => M.l + (xs.length < 2 ? 0.5 : (v - xs[0]) / (xs[xs.length - 1] - xs[0])) * (W - M.l - M.r);
  const y = v => M.t + (1 - (v - lo) / (hi - lo)) * (H - M.t - M.b);
  const yticks = niceTicks(lo, hi, 4);
  const xticks = niceTicks(xs[0], xs[xs.length - 1], 5).filter(v => Number.isInteger(v));
  let g = "";
  for (const tv of yticks) g += `<line x1="${M.l}" x2="${W - M.r}" y1="${y(tv)}" y2="${y(tv)}" stroke="var(--grid)" stroke-width="1"/>` +
    `<text x="${M.l - 6}" y="${y(tv) + 3.5}" text-anchor="end" font-size="10" fill="var(--muted)" style="font-variant-numeric:tabular-nums">${fmt(tv, Math.abs(hi) < 0.2 ? 3 : Math.abs(hi) < 10 ? 2 : 0)}</text>`;
  for (const tv of xticks) g += `<text x="${x(tv)}" y="${H - 5}" text-anchor="middle" font-size="10" fill="var(--muted)" style="font-variant-numeric:tabular-nums">${tv}</text>`;
  g += `<line x1="${M.l}" x2="${W - M.r}" y1="${y(lo)}" y2="${y(lo)}" stroke="var(--baseline)" stroke-width="1"/>`;
  if (cfg.refline != null) g += `<line x1="${M.l}" x2="${W - M.r}" y1="${y(cfg.refline)}" y2="${y(cfg.refline)}" stroke="var(--muted)" stroke-width="1" stroke-dasharray="3 3"/>`;
  for (const s of cfg.series) {
    const pts = T.filter(r => { const v = s.k(r); return v != null && isFinite(v); })
                 .map(r => [x(r.iteration), y(s.k(r))]);
    if (!pts.length) continue;
    const path = pts.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1)).join("");
    g += `<path d="${path}" fill="none" stroke="${S[s.c]}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
    const lp = pts[pts.length - 1];
    g += `<circle cx="${lp[0]}" cy="${lp[1]}" r="3" fill="${S[s.c]}" stroke="var(--surface)" stroke-width="2"/>`;
  }
  const card = document.createElement("div");
  card.className = "card";
  card.innerHTML = `<h2>${cfg.title}</h2><div class="desc">${cfg.desc}</div>` +
    (cfg.legend ? `<div class="legend">` + cfg.series.map(s => `<span><span class="key" style="background:${S[s.c]}"></span>${s.name}</span>`).join("") + `</div>` : "") +
    `<div class="chart"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${cfg.title}">${g}` +
    `<line class="xhair" y1="${M.t}" y2="${H - M.b}" stroke="var(--baseline)" stroke-width="1" visibility="hidden"/></svg><div class="tip"></div></div>`;
  chartsEl.appendChild(card);
  const svg = card.querySelector("svg"), tip = card.querySelector(".tip"), xhair = card.querySelector(".xhair");
  svg.addEventListener("pointermove", e => {
    const rect = svg.getBoundingClientRect();
    const px = (e.clientX - rect.left) / rect.width * W;
    let best = 0, bd = Infinity;
    T.forEach((r, i) => { const d = Math.abs(x(r.iteration) - px); if (d < bd) { bd = d; best = i; } });
    const r = T[best], cx = x(r.iteration);
    xhair.setAttribute("x1", cx); xhair.setAttribute("x2", cx); xhair.setAttribute("visibility", "visible");
    tip.style.display = "block";
    tip.innerHTML = `<span class="t-it">iter ${r.iteration}</span><br>` +
      cfg.series.map(s => `<span class="t-v" style="color:${S[s.c]}">${fmt(s.k(r), cfg.fmtd ?? 3)}</span> <span class="t-it">${s.name}</span>`).join("<br>");
    const cr = card.querySelector(".chart").getBoundingClientRect();
    const tx = (cx / W) * cr.width;
    tip.style.left = Math.min(cr.width - tip.offsetWidth - 4, Math.max(0, tx + 10)) + "px";
    tip.style.top = "6px";
  });
  svg.addEventListener("pointerleave", () => { tip.style.display = "none"; xhair.setAttribute("visibility", "hidden"); });
}

if (DATA.evals.length) {
  document.getElementById("evalcard").hidden = false;
  document.getElementById("evalpre").textContent = DATA.evals.map(e => JSON.stringify(e)).join("\\n");
}

const cols = [["iter", r => r.iteration, 0], ["loss", r => r.loss, 4], ["value", r => r.value_loss, 4],
  ["entropy", r => r.entropy, 3], ["KL", r => r.approximate_kl, 4], ["clip", r => r.clip_fraction, 3],
  ["expl var", r => r.explained_variance, 3], ["W", r => r.wins, 0], ["L", r => r.losses, 0], ["D", r => r.draws, 0],
  ["samp/s", r => r.samples_per_second, 0]];
const rows = T.slice(-30).reverse();
document.getElementById("tbl").innerHTML =
  "<tr>" + cols.map(c => `<th>${c[0]}</th>`).join("") + "</tr>" +
  rows.map(r => "<tr>" + cols.map(c => `<td>${fmt(c[1](r), c[2])}</td>`).join("") + "</tr>").join("");
</script>
"""

with open(out_path, "w") as f:
    f.write(page.replace("__PAYLOAD__", payload))
print(f"wrote {out_path}: {len(train_rows)} train rows, {len(eval_rows)} eval rows")
