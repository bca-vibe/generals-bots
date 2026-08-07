"""Blog figure: the critic penalizes castle builds regardless of their true effect.

Reads atlas_with_values/paired_rollouts.npz (checkpoint-3000 castle value probe)
and produces critic_castle_scatter.html — a Plotly scatter of, per build
opportunity, the actual (counterfactual) effect of building on final game score
vs. the critic's predicted successor-value change.

Style: vintage 1950s comic book — aged newsprint, CMYK-ish red/teal poles,
ink panel frame, halftone backdrop. Palette poles validated with the dataviz
six-check validator on the cream surface (all pass; neutral gray is a diverging
midpoint, de-emphasized and relieved by legend + tooltips).
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

HERE = Path(__file__).resolve().parent
NPZ = HERE.parent / "atlas_with_values" / "paired_rollouts.npz"
OUT_HTML = HERE / "critic_castle_scatter.html"
OUT_CSV = HERE / "per_state.csv"

# ---------------------------------------------------------------- palette ----
INK = "#2a241d"          # warm ink black
INK_SOFT = "#5c5344"     # secondary text
PAPER = "#f2e8d5"        # aged newsprint
PANEL = "#f6eeda"        # plot panel, one step lighter
GRID = "#e2d5b8"         # hairline grid, one step off panel
RED = "#c53a2a"          # harmful builds (validated pole)
TEAL = "#086f94"         # uplifting builds (validated pole)
GRAY = "#a39a8c"         # no measurable effect (neutral midpoint, de-emphasized)

# ------------------------------------------------------------------- data ----
d = np.load(NPZ)
out = d["result__outcome"].astype(np.float64)            # (states, reps, branch)
val = d["result__post_actor_value"].astype(np.float64)
fin = d["result__finish_relative_turn"]
turn = d["feature__turn"]
B, C = 1, 0  # branch 1 = forced build, branch 0 = forced control

pair_ok = ~((fin[:, :, B] == 1) | (fin[:, :, C] == 1))   # pair survived the action
state_ok = pair_ok.sum(axis=1) > 0                       # 1,999 analyzable states

causal = (out[:, :, B] - out[:, :, C]).mean(axis=1)      # actual score effect
vdelta = np.where(pair_ok, val[:, :, B] - val[:, :, C], np.nan)
with np.errstate(invalid="ignore"):
    vdelta = np.nanmean(vdelta, axis=1)                  # critic's verdict

x = causal[state_ok] * 100.0                             # score points
y = vdelta[state_ok]
t = turn[state_ok]

pos = x > 0
neg = x < 0
zer = x == 0

np.savetxt(
    OUT_CSV,
    np.column_stack([t, x, y]),
    delimiter=",",
    header="turn,causal_score_effect_pts,critic_value_delta",
    comments="",
    fmt=["%d", "%.6f", "%.6f"],
)

# ------------------------------------------------------------------ figure ----
F_DISPLAY = "Bangers, 'Arial Black', sans-serif"
F_CAPS = "Oswald, 'Arial Narrow', sans-serif"
F_BODY = "Archivo, Georgia, sans-serif"

fig = go.Figure()

XMIN, XMAX = -105, 105
YMIN, YMAX = -1.78, 0.62

# Quadrant tint: builds that actually helped but the critic penalized.
fig.add_shape(
    type="rect", x0=0, x1=XMAX, y0=YMIN, y1=0,
    fillcolor="rgba(8,111,148,0.06)", line_width=0, layer="below",
)

# Zero lines carry the story; grid stays recessive.
fig.add_hline(y=0, line=dict(color=INK, width=1.6))
fig.add_vline(x=0, line=dict(color=INK, width=1.6))

groups = [
    ("Build helped", pos, TEAL, 7.5, 0.72),
    ("Build hurt", neg, RED, 7.5, 0.55),
    ("No effect", zer, GRAY, 5.5, 0.45),
]
for name, mask, color, size, alpha in groups:
    fig.add_trace(go.Scattergl(
        x=x[mask], y=y[mask], mode="markers", name=f"{name}  ({mask.sum()})",
        marker=dict(
            size=size, color=color, opacity=alpha,
            line=dict(color=INK, width=0.6),
        ),
        customdata=np.column_stack([t[mask]]),
        hovertemplate=(
            "turn %{customdata[0]}<br>"
            "actual effect on score: %{x:+.1f} pts<br>"
            "critic’s predicted ΔV: %{y:+.3f}<extra>" + name + "</extra>"
        ),
    ))

# Group means (full-sample causal groups), diamond with ink outline.
for mask, color in [(pos, TEAL), (neg, RED)]:
    fig.add_trace(go.Scatter(
        x=[x[mask].mean()], y=[y[mask].mean()], mode="markers",
        marker=dict(symbol="diamond", size=17, color=color,
                    line=dict(color=INK, width=2)),
        showlegend=False, hovertemplate=(
            "group mean<br>actual: %{x:+.1f} pts<br>critic: %{y:+.3f}<extra></extra>"),
    ))

# Comic narration boxes — a stacked pair straddling the zero line, top-right.
fig.add_annotation(
    x=68, y=0.36, xanchor="center", yanchor="middle",
    text=("<b>BUILDS THAT WON GAMES…</b><br>"
          "castles here raised final score<br>"
          "by +25 pts on average"),
    font=dict(family=F_BODY, size=13, color=INK),
    align="left", bgcolor=PANEL, bordercolor=INK, borderwidth=2,
    borderpad=8, showarrow=False,
)
fig.add_annotation(
    x=68, y=-1.28, xanchor="center", yanchor="middle",
    text=("<b>…STILL GOT MARKED DOWN!</b><br>"
          "mean predicted ΔV: <b>−0.46</b><br>"
          "positive verdicts: <b>0.3%</b>"),
    font=dict(family=F_BODY, size=13, color=INK),
    align="left", bgcolor=PANEL, bordercolor=RED, borderwidth=2,
    borderpad=8, showarrow=False,
)

# Right-edge zone labels for the y polarity.
for yy, txt, anch in [(0.3, "CRITIC: BUILD HELPS", "bottom"),
                      (-0.06, "CRITIC: BUILD HURTS", "top")]:
    fig.add_annotation(
        x=XMIN + 3, y=yy, xanchor="left", yanchor=anch,
        text=txt, showarrow=False,
        font=dict(family=F_CAPS, size=12, color=INK_SOFT),
    )

fig.update_layout(
    paper_bgcolor=PANEL, plot_bgcolor=PANEL,
    width=980, height=640,
    margin=dict(l=78, r=30, t=30, b=64),
    font=dict(family=F_BODY, color=INK),
    legend=dict(
        orientation="h", x=0, y=1.06, xanchor="left",
        font=dict(family=F_CAPS, size=14, color=INK),
        bgcolor="rgba(0,0,0,0)",
    ),
    xaxis=dict(
        title=dict(text="ACTUAL EFFECT OF BUILDING ON FINAL SCORE  (points, build − control)",
                   font=dict(family=F_CAPS, size=13, color=INK)),
        range=[XMIN, XMAX], gridcolor=GRID, gridwidth=1, zeroline=False,
        tickfont=dict(family=F_CAPS, size=12, color=INK_SOFT),
        ticksuffix="", dtick=25,
    ),
    yaxis=dict(
        title=dict(text="CRITIC’S PREDICTED VALUE CHANGE  (ΔV, build − control)",
                   font=dict(family=F_CAPS, size=13, color=INK)),
        range=[YMIN, YMAX], gridcolor=GRID, gridwidth=1, zeroline=False,
        tickfont=dict(family=F_CAPS, size=12, color=INK_SOFT),
    ),
    hoverlabel=dict(bgcolor=PAPER, bordercolor=INK,
                    font=dict(family=F_BODY, size=12, color=INK)),
)

# -------------------------------------------------------------------- page ----
fragment = pio.to_html(fig, full_html=False, include_plotlyjs="cdn",
                       config={"displayModeBar": False, "responsive": False})

page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>The Critic That Hated Castles</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Bangers&family=Oswald:wght@400;500&family=Archivo:ital,wght@0,400;0,600;1,400&display=swap" rel="stylesheet">
<style>
  body {{
    margin: 0; padding: 48px 16px 64px;
    background-color: {PAPER};
    /* halftone "dot photo" backdrop */
    background-image: radial-gradient(circle, rgba(42,36,29,0.055) 1.15px, transparent 1.3px);
    background-size: 9px 9px;
    font-family: Archivo, Georgia, sans-serif; color: {INK};
  }}
  .comic {{ max-width: 1030px; margin: 0 auto; }}
  h1 {{
    font-family: Bangers, 'Arial Black', sans-serif;
    font-size: 54px; letter-spacing: 2px; line-height: 1;
    margin: 0 0 6px; color: {INK};
    text-shadow: 3px 3px 0 rgba(197,58,42,0.35);
  }}
  h1 .accent {{ color: {RED}; }}
  .kicker {{
    font-family: Oswald, 'Arial Narrow', sans-serif; font-size: 14px;
    letter-spacing: 3px; text-transform: uppercase; color: {INK_SOFT};
    margin: 0 0 10px;
  }}
  .dek {{
    font-size: 16px; line-height: 1.45; max-width: 860px; margin: 0 0 22px;
  }}
  .panel {{
    display: inline-block; background: {PANEL};
    border: 3px solid {INK}; box-shadow: 7px 7px 0 rgba(42,36,29,0.9);
  }}
  .footer {{
    margin-top: 26px; max-width: 860px;
    font-size: 12.5px; line-height: 1.5; color: {INK_SOFT};
    border-top: 1px solid {GRID}; padding-top: 10px;
  }}
</style>
</head>
<body>
<div class="comic">
  <p class="kicker">Generals.io RL agent &nbsp;·&nbsp; castle value probe &nbsp;·&nbsp; checkpoint 3000</p>
  <h1>THE CRITIC THAT <span class="accent">HATED CASTLES!</span></h1>
  <p class="dek">Each dot is one castle-build opportunity from stochastic self-play
  (n&nbsp;=&nbsp;1,999). Horizontally: what actually happened to the final game score when we
  forced the build, versus a paired no-build control (16 matched rollouts each).
  Vertically: how the value head <i>predicted</i> the build would change its evaluation.
  A well-calibrated critic would put helpful builds above the line. It put
  99.4% of everything below it.</p>
  <div class="panel">{fragment}</div>
  <p class="footer">Paired stochastic continuations share opponent actions and future
  random draws. Score effects on [0,&thinsp;100] points; value on the network’s [−1,&thinsp;1]
  expectation scale. Group means shown as diamonds; 291 immediately-terminal pairs and
  17 states with no surviving pair excluded. Full analysis: castle_value_probe_iter3000.</p>
</div>
</body>
</html>
"""
OUT_HTML.write_text(page)
print(f"wrote {OUT_HTML}")
print(f"wrote {OUT_CSV}")
print(f"groups: helped={pos.sum()} hurt={neg.sum()} none={zer.sum()}  "
      f"helped mean ({x[pos].mean():+.1f} pts, {y[pos].mean():+.3f} dV)  "
      f"hurt mean ({x[neg].mean():+.1f} pts, {y[neg].mean():+.3f} dV)")
