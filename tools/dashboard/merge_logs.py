#!/usr/bin/env python3
"""Merge stdout log lines (from query_logs JSON) into metrics.jsonl to cover
iterations lost after the node parked. Synthesized rows carry only the fields
present in stdout; wall_seconds is extrapolated at the observed sec/iter."""
import json, re, sys

logs_json, metrics_in, metrics_out = sys.argv[1:4]

real_train, real_evals = [], []
for line in open(metrics_in):
    r = json.loads(line)
    (real_train if "loss" in r else real_evals).append(r)
last_real = real_train[-1]
known_eval_iters = {int(e["iteration"]) for e in real_evals}

# observed pace from the real tail
a, b = real_train[-60], real_train[-1]
sec_per_iter = (b["wall_seconds"] - a["wall_seconds"]) / (b["iteration"] - a["iteration"])

iter_re = re.compile(
    r"iter\s+(\d+) \| loss ([\d.]+) \| entropy ([\d.]+) \| KL ([\d.]+) \| "
    r"episodes (\d+) W/L/D (\d+)/(\d+)/(\d+) \| EV ([\d.-]+) \| ([\d,]+) samples/s")
eval_re = re.compile(r"eval EMA: (\d+)W/(\d+)L/(\d+)D, score=([\d.]+)")

synth_train, synth_evals = {}, {}
last_iter_seen = None
for entry in json.load(open(logs_json))["lines"]:
    text = entry["line"]
    m = iter_re.search(text)
    if m:
        it = int(m.group(1))
        last_iter_seen = it
        if it > last_real["iteration"]:
            synth_train[it] = {
                "iteration": it,
                "loss": float(m.group(2)),
                "entropy": float(m.group(3)),
                "approximate_kl": float(m.group(4)),
                "episodes": int(m.group(5)),
                "wins": int(m.group(6)),
                "losses": int(m.group(7)),
                "draws": int(m.group(8)),
                "explained_variance": float(m.group(9)),
                "samples_per_second": float(m.group(10).replace(",", "")),
                "stage": last_real["stage"],
                "wall_seconds": last_real["wall_seconds"] + (it - last_real["iteration"]) * sec_per_iter,
                "synthesized_from_stdout": True,
            }
        continue
    m = eval_re.search(text)
    if m and last_iter_seen and last_iter_seen not in known_eval_iters:
        synth_evals[last_iter_seen] = {
            "iteration": last_iter_seen,
            "evaluation/wins": float(m.group(1)),
            "evaluation/losses": float(m.group(2)),
            "evaluation/draws": float(m.group(3)),
            "evaluation/score": float(m.group(4)),
            "synthesized_from_stdout": True,
        }

rows = real_train + [synth_train[k] for k in sorted(synth_train)]
evs = real_evals + [synth_evals[k] for k in sorted(synth_evals)]
with open(metrics_out, "w") as f:
    for r in sorted(rows, key=lambda r: r["iteration"]):
        f.write(json.dumps(r) + "\n")
    for e in sorted(evs, key=lambda e: e["iteration"]):
        f.write(json.dumps(e) + "\n")
print(f"real train rows: {len(real_train)} (to iter {last_real['iteration']}), "
      f"synthesized: {len(synth_train)} (to iter {max(synth_train) if synth_train else '-'}), "
      f"evals: {len(real_evals)} real + {len(synth_evals)} synthesized, "
      f"pace {sec_per_iter:.2f}s/iter")
