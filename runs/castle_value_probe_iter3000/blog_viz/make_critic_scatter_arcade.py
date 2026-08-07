"""Blog figure, style option 3: retro arcade cabinet (space theme, arcade-ified).

Same data and chart as the other options (checkpoint-3000 castle value probe:
actual causal effect of building vs. the critic's predicted value change),
restyled as a CRT arcade screen — scanlines, phosphor-glow pixel markers,
HUD score strip, pixel fonts. Writes critic_castle_scatter_arcade.html.

Palette validated dark-mode on the screen surface (#08081a): magenta/cyan
poles pass all six checks (worst CVD ΔE 11.1); the "no effect" gray is pushed
below the identity lightness band on purpose — dim dead-pixel de-emphasis,
CVD-separated from both poles by lightness, relieved by legend + tooltips and
by position (those states sit exactly on x = 0).
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

HERE = Path(__file__).resolve().parent
NPZ = HERE.parent / "atlas_with_values" / "paired_rollouts.npz"
OUT_HTML = HERE / "critic_castle_scatter_arcade.html"

# ---------------------------------------------------------------- palette ----
ROOM = "#050510"         # page: dark arcade room
SCREEN = "#08081a"       # CRT screen
BEZEL = "#15152a"        # cabinet bezel
GRID = "#131a33"         # hairline grid
TEXT = "#d7e0ff"         # primary text
MUTED = "#7a86b8"        # secondary text
ZERO = "#c3cdf5"         # zero lines
MAGENTA = "#e0457a"      # harmful builds (validated pole)
CYAN = "#0095b3"         # uplifting builds (validated pole)
DEAD = "#45464e"         # no measurable effect (dead-pixel neutral)

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
F_PIXEL = "'Press Start 2P', monospace"
F_TERM = "VT323, Menlo, monospace"

fig = go.Figure()
XMIN, XMAX = -105, 105
YMIN, YMAX = -1.78, 0.62

fig.add_shape(
    type="rect", x0=0, x1=XMAX, y0=YMIN, y1=0,
    fillcolor="rgba(0,149,179,0.05)", line_width=0, layer="below",
)
fig.add_hline(y=0, line=dict(color=ZERO, width=1.4))
fig.add_vline(x=0, line=dict(color=ZERO, width=1.4))

groups = [
    ("Build helped", pos, CYAN, 0.9),
    ("Build hurt", neg, MAGENTA, 0.75),
    ("No effect", zer, DEAD, 0.6),
]
# Phosphor bloom underneath, pixel cores on top.
for name, mask, color, _ in groups:
    fig.add_trace(go.Scattergl(
        x=x[mask], y=y[mask], mode="markers", showlegend=False,
        marker=dict(symbol="square", size=13, color=color, opacity=0.10,
                    line_width=0),
        hoverinfo="skip",
    ))
for name, mask, color, alpha in groups:
    size = 6 if name != "No effect" else 4.5
    fig.add_trace(go.Scattergl(
        x=x[mask], y=y[mask], mode="markers", name=f"{name} ({mask.sum()})",
        marker=dict(symbol="square", size=size, color=color, opacity=alpha,
                    line=dict(color=SCREEN, width=0.8)),
        customdata=np.column_stack([t[mask]]),
        hovertemplate=(
            "turn %{customdata[0]}<br>"
            "actual effect on score: %{x:+.1f} pts<br>"
            "critic’s predicted ΔV: %{y:+.3f}<extra>" + name + "</extra>"
        ),
    ))

# Group means as big pixel blocks.
for mask, color in [(pos, CYAN), (neg, MAGENTA)]:
    fig.add_trace(go.Scatter(
        x=[x[mask].mean()], y=[y[mask].mean()], mode="markers",
        marker=dict(symbol="square", size=15, color=color,
                    line=dict(color=TEXT, width=1.5)),
        showlegend=False, hovertemplate=(
            "group mean<br>actual: %{x:+.1f} pts<br>critic: %{y:+.3f}<extra></extra>"),
    ))

# Arcade message boxes.
fig.add_annotation(
    x=68, y=0.36, xanchor="center", yanchor="middle",
    text=("<b>WAVE 1 — BUILDS THAT WON GAMES</b><br>"
          "+25 PTS AVG FINAL SCORE"),
    font=dict(family=F_TERM, size=17, color=TEXT),
    align="left", bgcolor="rgba(8,8,26,0.92)", bordercolor=CYAN,
    borderwidth=2, borderpad=8, showarrow=False,
)
fig.add_annotation(
    x=68, y=-1.28, xanchor="center", yanchor="middle",
    text=("<b>CRITIC SAYS: GAME OVER</b><br>"
          "MEAN ΔV −0.46 · POSITIVE 0.3%"),
    font=dict(family=F_TERM, size=17, color=TEXT),
    align="left", bgcolor="rgba(8,8,26,0.92)", bordercolor=MAGENTA,
    borderwidth=2, borderpad=8, showarrow=False,
)

for yy, txt, anch in [(0.3, "CRITIC: BUILD HELPS", "bottom"),
                      (-0.06, "CRITIC: BUILD HURTS", "top")]:
    fig.add_annotation(
        x=XMIN + 3, y=yy, xanchor="left", yanchor=anch,
        text=txt, showarrow=False,
        font=dict(family=F_TERM, size=15, color=MUTED),
    )

fig.update_layout(
    paper_bgcolor=SCREEN, plot_bgcolor=SCREEN,
    width=980, height=640,
    margin=dict(l=84, r=30, t=34, b=70),
    font=dict(family=F_TERM, color=TEXT),
    legend=dict(
        orientation="h", x=0, y=1.07, xanchor="left",
        font=dict(family=F_PIXEL, size=9, color=TEXT),
        bgcolor="rgba(0,0,0,0)", entrywidth=250,
    ),
    xaxis=dict(
        title=dict(text="ACTUAL EFFECT OF BUILDING ON FINAL SCORE (PTS, BUILD − CONTROL)",
                   font=dict(family=F_PIXEL, size=8.5, color=MUTED)),
        range=[XMIN, XMAX], gridcolor=GRID, gridwidth=1, zeroline=False,
        tickfont=dict(family=F_TERM, size=14, color=MUTED), dtick=25,
    ),
    yaxis=dict(
        title=dict(text="CRITIC’S PREDICTED VALUE CHANGE (ΔV)",
                   font=dict(family=F_PIXEL, size=8.5, color=MUTED)),
        range=[YMIN, YMAX], gridcolor=GRID, gridwidth=1, zeroline=False,
        tickfont=dict(family=F_TERM, size=14, color=MUTED),
    ),
    hoverlabel=dict(bgcolor=ROOM, bordercolor=ZERO,
                    font=dict(family=F_TERM, size=15, color=TEXT)),
)

# -------------------------------------------------------------------- page ----
fragment = pio.to_html(fig, full_html=False, include_plotlyjs="cdn",
                       config={"displayModeBar": False, "responsive": False})

page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Game Over, Castles! — arcade edition</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323&display=swap" rel="stylesheet">
<style>
  body {{
    margin: 0; padding: 40px 16px 64px;
    background-color: {ROOM};
    background-image: radial-gradient(ellipse at 50% 0%, #0c0c22 0%, {ROOM} 62%);
    font-family: VT323, Menlo, monospace; color: {TEXT};
  }}
  /* CRT scanlines over everything */
  body::after {{
    content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 99;
    background: repeating-linear-gradient(
      0deg, rgba(0,0,0,0.20) 0px, rgba(0,0,0,0.20) 1px, transparent 1px, transparent 3px);
  }}
  .cab {{ max-width: 1030px; margin: 0 auto; }}
  .hud {{
    display: flex; justify-content: space-between; max-width: 1000px;
    font-family: 'Press Start 2P', monospace; font-size: 10px;
    color: {MUTED}; margin: 0 0 18px; letter-spacing: 1px;
  }}
  .hud .score {{ color: {CYAN}; }}
  h1 {{
    font-family: 'Press Start 2P', monospace; font-size: 30px; line-height: 1.35;
    margin: 0 0 14px; color: {TEXT};
    text-shadow: 3px 3px 0 {MAGENTA}, 0 0 22px rgba(0,149,179,0.6);
  }}
  h1 .accent {{ color: {MAGENTA}; text-shadow: 3px 3px 0 #4d0f27, 0 0 22px rgba(224,69,122,0.55); }}
  .dek {{ font-size: 19px; line-height: 1.35; max-width: 880px; margin: 0 0 22px; }}
  .panel {{
    display: inline-block; background: {SCREEN};
    border: 8px solid {BEZEL}; border-radius: 10px;
    outline: 2px solid {CYAN};
    box-shadow: 0 0 30px rgba(0,149,179,0.35), inset 0 0 60px rgba(0,0,0,0.55);
  }}
  .coin {{
    font-family: 'Press Start 2P', monospace; font-size: 11px; color: {TEXT};
    margin-top: 22px; letter-spacing: 2px;
    animation: blink 1.2s steps(1) infinite;
  }}
  @keyframes blink {{ 50% {{ opacity: 0; }} }}
  @media (prefers-reduced-motion: reduce) {{ .coin {{ animation: none; }} }}
  .footer {{
    margin-top: 12px; max-width: 900px;
    font-size: 16px; line-height: 1.4; color: {MUTED};
    border-top: 1px solid {GRID}; padding-top: 10px;
  }}
</style>
</head>
<body>
<div class="cab">
  <div class="hud">
    <span>1UP&nbsp;&nbsp;1,999 STATES</span>
    <span class="score">HI-SCORE&nbsp;&nbsp;+93.7 PTS</span>
    <span>CREDIT&nbsp;&nbsp;3000</span>
  </div>
  <h1>GAME OVER, <span class="accent">CASTLES!</span></h1>
  <p class="dek">Each pixel is one castle-build opportunity from stochastic self-play
  (n&nbsp;=&nbsp;1,999). Horizontally: what actually happened to the final game score when we
  forced the build, versus a paired no-build control (16 matched rollouts each).
  Vertically: how the value head <i>predicted</i> the build would change its evaluation.
  A calibrated critic would lift helpful builds above the line. Instead, 99.4% of
  everything lands below it. No extra lives.</p>
  <div class="panel">{fragment}</div>
  <p class="coin">&gt; INSERT COIN TO CONTINUE TRAINING _</p>
  <p class="footer">Paired stochastic continuations share opponent actions and future
  random draws · score effects on [0,&thinsp;100] points · value on the network’s
  [−1,&thinsp;1] expectation scale · group means shown as large blocks · HI-SCORE is the
  largest observed causal uplift · 291 immediately-terminal pairs and 17 states with no
  surviving pair excluded · full analysis: castle_value_probe_iter3000</p>
</div>
</body>
</html>
"""
OUT_HTML.write_text(page)
print(f"wrote {OUT_HTML}")
print("max causal uplift:", x.max())
