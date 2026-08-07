"""Blog figure, style option 2: deep-space mission-control theme.

Same data and chart as make_critic_scatter.py (checkpoint-3000 castle value
probe: actual causal effect of building vs. the critic's predicted value
change), restyled — starfield backdrop, glowing star markers, HUD/telemetry
captions, retro-futurist type. Writes critic_castle_scatter_space.html.

Palette validated dark-mode on the panel surface (#0e1430): coral/cyan poles
pass all six checks; slate neutral is a diverging midpoint (chroma-low by
design), de-emphasized and relieved by legend + tooltips.
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

HERE = Path(__file__).resolve().parent
NPZ = HERE.parent / "atlas_with_values" / "paired_rollouts.npz"
OUT_HTML = HERE / "critic_castle_scatter_space.html"

# ---------------------------------------------------------------- palette ----
SPACE = "#070b1d"        # page: deep space
PANEL = "#0e1430"        # plot panel
GRID = "#1b2450"         # hairline grid
FRAME = "#2c3a6e"        # panel border
TEXT = "#dfe6ff"         # primary text
MUTED = "#8b97c9"        # secondary text
ZERO = "#93a1d6"         # zero lines
CORAL = "#e05545"        # harmful builds (validated pole)
CYAN = "#339ac0"         # uplifting builds (validated pole)
SLATE = "#5c6478"        # no measurable effect (neutral midpoint)

# ------------------------------------------------------------------- data ----
d = np.load(NPZ)
out = d["result__outcome"].astype(np.float64)
val = d["result__post_actor_value"].astype(np.float64)
fin = d["result__finish_relative_turn"]
turn = d["feature__turn"]
B, C = 1, 0

pair_ok = ~((fin[:, :, B] == 1) | (fin[:, :, C] == 1))
state_ok = pair_ok.sum(axis=1) > 0

causal = (out[:, :, B] - out[:, :, C]).mean(axis=1)
vdelta = np.where(pair_ok, val[:, :, B] - val[:, :, C], np.nan)
with np.errstate(invalid="ignore"):
    vdelta = np.nanmean(vdelta, axis=1)

x = causal[state_ok] * 100.0
y = vdelta[state_ok]
t = turn[state_ok]
pos, neg, zer = x > 0, x < 0, x == 0

# ------------------------------------------------------------------ figure ----
F_TITLE = "Orbitron, 'Arial Black', sans-serif"
F_BODY = "'Space Grotesk', Arial, sans-serif"
F_MONO = "'Space Mono', Menlo, monospace"

fig = go.Figure()
XMIN, XMAX = -105, 105
YMIN, YMAX = -1.78, 0.62

fig.add_shape(
    type="rect", x0=0, x1=XMAX, y0=YMIN, y1=0,
    fillcolor="rgba(51,154,192,0.05)", line_width=0, layer="below",
)
fig.add_hline(y=0, line=dict(color=ZERO, width=1.4))
fig.add_vline(x=0, line=dict(color=ZERO, width=1.4))

groups = [
    ("Build helped", pos, CYAN, 0.9),
    ("Build hurt", neg, CORAL, 0.75),
    ("No effect", zer, SLATE, 0.55),
]
# Glow halos first (underneath), then star cores.
for name, mask, color, _ in groups:
    fig.add_trace(go.Scattergl(
        x=x[mask], y=y[mask], mode="markers", showlegend=False,
        marker=dict(size=15, color=color, opacity=0.10, line_width=0),
        hoverinfo="skip",
    ))
for name, mask, color, alpha in groups:
    size = 6.5 if name != "No effect" else 4.5
    fig.add_trace(go.Scattergl(
        x=x[mask], y=y[mask], mode="markers", name=f"{name}  ({mask.sum()})",
        marker=dict(size=size, color=color, opacity=alpha,
                    line=dict(color=PANEL, width=0.8)),
        customdata=np.column_stack([t[mask]]),
        hovertemplate=(
            "turn %{customdata[0]}<br>"
            "actual effect on score: %{x:+.1f} pts<br>"
            "critic’s predicted ΔV: %{y:+.3f}<extra>" + name + "</extra>"
        ),
    ))

# Group means as stars.
for mask, color in [(pos, CYAN), (neg, CORAL)]:
    fig.add_trace(go.Scatter(
        x=[x[mask].mean()], y=[y[mask].mean()], mode="markers",
        marker=dict(symbol="star", size=20, color=color,
                    line=dict(color=TEXT, width=1.4)),
        showlegend=False, hovertemplate=(
            "group mean<br>actual: %{x:+.1f} pts<br>critic: %{y:+.3f}<extra></extra>"),
    ))

# HUD telemetry boxes.
fig.add_annotation(
    x=68, y=0.36, xanchor="center", yanchor="middle",
    text=("<b>▚ TELEMETRY — BUILDS THAT WON GAMES</b><br>"
          "avg effect on final score: <b>+25 pts</b>"),
    font=dict(family=F_MONO, size=12, color=TEXT),
    align="left", bgcolor="rgba(14,20,48,0.88)", bordercolor=CYAN,
    borderwidth=1, borderpad=8, showarrow=False,
)
fig.add_annotation(
    x=68, y=-1.28, xanchor="center", yanchor="middle",
    text=("<b>▚ CRITIC VERDICT — MARKED DOWN</b><br>"
          "mean predicted ΔV: <b>−0.46</b><br>"
          "positive verdicts: <b>0.3%</b>"),
    font=dict(family=F_MONO, size=12, color=TEXT),
    align="left", bgcolor="rgba(14,20,48,0.88)", bordercolor=CORAL,
    borderwidth=1, borderpad=8, showarrow=False,
)

for yy, txt, anch in [(0.3, "CRITIC: BUILD HELPS", "bottom"),
                      (-0.06, "CRITIC: BUILD HURTS", "top")]:
    fig.add_annotation(
        x=XMIN + 3, y=yy, xanchor="left", yanchor=anch,
        text=txt, showarrow=False,
        font=dict(family=F_MONO, size=11, color=MUTED),
    )

fig.update_layout(
    paper_bgcolor=PANEL, plot_bgcolor=PANEL,
    width=980, height=640,
    margin=dict(l=78, r=30, t=30, b=64),
    font=dict(family=F_BODY, color=TEXT),
    legend=dict(
        orientation="h", x=0, y=1.06, xanchor="left",
        font=dict(family=F_MONO, size=12.5, color=TEXT),
        bgcolor="rgba(0,0,0,0)",
    ),
    xaxis=dict(
        title=dict(text="ACTUAL EFFECT OF BUILDING ON FINAL SCORE  (points, build − control)",
                   font=dict(family=F_BODY, size=13, color=MUTED)),
        range=[XMIN, XMAX], gridcolor=GRID, gridwidth=1, zeroline=False,
        tickfont=dict(family=F_MONO, size=11, color=MUTED), dtick=25,
    ),
    yaxis=dict(
        title=dict(text="CRITIC’S PREDICTED VALUE CHANGE  (ΔV, build − control)",
                   font=dict(family=F_BODY, size=13, color=MUTED)),
        range=[YMIN, YMAX], gridcolor=GRID, gridwidth=1, zeroline=False,
        tickfont=dict(family=F_MONO, size=11, color=MUTED),
    ),
    hoverlabel=dict(bgcolor=SPACE, bordercolor=ZERO,
                    font=dict(family=F_MONO, size=12, color=TEXT)),
)

# -------------------------------------------------------------------- page ----
fragment = pio.to_html(fig, full_html=False, include_plotlyjs="cdn",
                       config={"displayModeBar": False, "responsive": False})

page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>No Lift-Off — the critic grounds every castle</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@600;800&family=Space+Grotesk:wght@400;500&family=Space+Mono:ital,wght@0,400;0,700;1,400&display=swap" rel="stylesheet">
<style>
  body {{
    margin: 0; padding: 48px 16px 64px;
    background-color: {SPACE};
    /* layered starfield */
    background-image:
      radial-gradient(circle, rgba(223,230,255,0.9) 0.7px, transparent 1.1px),
      radial-gradient(circle, rgba(160,180,255,0.55) 0.9px, transparent 1.4px),
      radial-gradient(circle, rgba(223,230,255,0.35) 1.2px, transparent 1.8px);
    background-size: 190px 190px, 310px 310px, 460px 460px;
    background-position: 0 0, 70px 120px, 160px 40px;
    font-family: 'Space Grotesk', Arial, sans-serif; color: {TEXT};
  }}
  .mission {{ max-width: 1030px; margin: 0 auto; }}
  h1 {{
    font-family: Orbitron, 'Arial Black', sans-serif; font-weight: 800;
    font-size: 44px; letter-spacing: 4px; line-height: 1.08;
    margin: 0 0 8px; color: {TEXT};
    text-shadow: 0 0 18px rgba(51,154,192,0.55);
  }}
  h1 .accent {{ color: {CORAL}; text-shadow: 0 0 18px rgba(224,85,69,0.5); }}
  .kicker {{
    font-family: 'Space Mono', Menlo, monospace; font-size: 12.5px;
    letter-spacing: 4px; text-transform: uppercase; color: {MUTED};
    margin: 0 0 12px;
  }}
  .dek {{ font-size: 16px; line-height: 1.5; max-width: 860px; margin: 0 0 22px; }}
  .panel {{
    display: inline-block; background: {PANEL};
    border: 1px solid {FRAME}; border-radius: 4px;
    box-shadow: 0 0 34px rgba(51,154,192,0.22), 0 0 4px rgba(147,161,214,0.35);
  }}
  .footer {{
    margin-top: 26px; max-width: 900px;
    font-family: 'Space Mono', Menlo, monospace;
    font-size: 11.5px; line-height: 1.6; color: {MUTED};
    border-top: 1px solid {GRID}; padding-top: 10px;
  }}
</style>
</head>
<body>
<div class="mission">
  <p class="kicker">Mission report &nbsp;·&nbsp; generals.io RL agent &nbsp;·&nbsp; castle value probe &nbsp;·&nbsp; checkpoint 3000</p>
  <h1>NO LIFT-OFF: THE CRITIC <span class="accent">GROUNDS EVERY CASTLE</span></h1>
  <p class="dek">Each point is one castle-build opportunity from stochastic self-play
  (n&nbsp;=&nbsp;1,999). Horizontally: what actually happened to the final game score when we
  forced the build, versus a paired no-build control (16 matched rollouts each).
  Vertically: how the value head <i>predicted</i> the build would change its evaluation.
  A calibrated critic would lift helpful builds above the line. Instead, 99.4% of
  everything stays below it — the upper half is empty space.</p>
  <div class="panel">{fragment}</div>
  <p class="footer">TELEMETRY — paired stochastic continuations share opponent actions and
  future random draws · score effects on [0,&thinsp;100] points · value on the network’s
  [−1,&thinsp;1] expectation scale · group means shown as stars · 291 immediately-terminal
  pairs and 17 states with no surviving pair excluded · full analysis:
  castle_value_probe_iter3000</p>
</div>
</body>
</html>
"""
OUT_HTML.write_text(page)
print(f"wrote {OUT_HTML}")
