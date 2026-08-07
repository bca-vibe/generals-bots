"""Blog figure, style option 4: generals.io fog-of-war map.

Same data and chart as the other options (checkpoint-3000 castle value probe:
actual causal effect of building vs. the critic's predicted value change),
styled as the game itself — the plot is a board, the "critic approves" half is
permanently fogged territory, points are army tiles in the repo's real match
colors, group means are generals, the legend is the in-game leaderboard, and
the takeaway reads as game chat. Writes critic_castle_scatter_fog.html.

Colors come from the project's own renderer (generals/gui/rendering.py and
examples/hunter_vs_expander.py): fog (70,73,76), visible path (200,200,200),
neutral castle (128,128,128), player blue (50,80,200), player red (200,50,50).
The red/blue poles pass all six palette checks on the board surface; the
neutral castle gray is zero-chroma by design, de-emphasized, and relieved by
the leaderboard table + tooltips.
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

HERE = Path(__file__).resolve().parent
NPZ = HERE.parent / "atlas_with_values" / "paired_rollouts.npz"
OUT_HTML = HERE / "critic_castle_scatter_fog.html"

# ------------------------------------------------- palette (from the game) ----
PAGE = "#222222"         # game site background
LAND = "#c8c8c8"         # VISIBLE_PATH — revealed empty tiles
FOG = "#46494c"          # FOG_OF_WAR
FOG_LINE = "#54585b"     # fog tile borders
MOUNTAIN = "#5d6165"     # fog obstacle silhouettes
GRID_MAJOR = "#adadad"   # revealed tile borders
GRID_MINOR = "#b9b9b9"
BLUE = "#3250c8"         # (50, 80, 200) — builds that helped
BLUE_EDGE = "#253c96"
RED = "#c83232"          # (200, 50, 50) — builds that hurt
RED_EDGE = "#962525"
CASTLE = "#808080"       # NEUTRAL_CASTLE — no measurable effect
TEXT = "#e6e6e6"         # WHITE (230,230,230)
MUTED = "#aaaeb2"

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
F_GAME = "Quicksand, 'Trebuchet MS', sans-serif"

fig = go.Figure()
XMIN, XMAX = -105, 105
YMIN, YMAX = -1.78, 0.62

# Fogged territory: everywhere the critic would approve a build.
fig.add_shape(type="rect", x0=XMIN, x1=XMAX, y0=0, y1=YMAX,
              fillcolor=FOG, line_width=0, layer="below")
# Fog keeps its own faint tile grid.
for gx in np.arange(-87.5, 105, 12.5):
    fig.add_shape(type="line", x0=gx, x1=gx, y0=0, y1=YMAX,
                  line=dict(color=FOG_LINE, width=1), layer="below")
for gy in (0.2, 0.4, 0.6):
    fig.add_shape(type="line", x0=XMIN, x1=XMAX, y0=gy, y1=gy,
                  line=dict(color=FOG_LINE, width=1), layer="below")
# Obstacle silhouettes in the fog (deterministic scatter, clear of the label).
rng = np.random.default_rng(7)
mx, my = [], []
while len(mx) < 26:
    cx = rng.uniform(-100, 100)
    cy = rng.uniform(0.05, 0.57)
    if abs(cx) < 46 and 0.22 < cy < 0.45:
        continue  # keep the fog label readable
    mx.append(cx)
    my.append(cy)
fig.add_trace(go.Scatter(
    x=mx, y=my, mode="text", text="▲", textfont=dict(size=12, color=MOUNTAIN),
    hoverinfo="skip", showlegend=False,
))
fig.add_annotation(
    x=0, y=0.34, xanchor="center", yanchor="middle",
    text=("<b>UNEXPLORED TERRITORY</b><br>"
          "the critic has never valued a castle build here"),
    font=dict(family=F_GAME, size=14, color="#9ba0a4"),
    align="center", showarrow=False,
)

# Fog boundary and the win/loss meridian (revealed half only).
fig.add_hline(y=0, line=dict(color="#33363a", width=2))
fig.add_shape(type="line", x0=0, x1=0, y0=YMIN, y1=0,
              line=dict(color="#8a8a8a", width=1.4))

groups = [
    ("Build helped", pos, BLUE, BLUE_EDGE, 6.5, 0.9),
    ("Build hurt", neg, RED, RED_EDGE, 6.5, 0.85),
    ("No effect", zer, CASTLE, "#6a6a6a", 4.5, 0.7),
]
for name, mask, color, edge, size, alpha in groups:
    fig.add_trace(go.Scattergl(
        x=x[mask], y=y[mask], mode="markers", showlegend=False,
        marker=dict(symbol="square", size=size, color=color, opacity=alpha,
                    line=dict(color=edge, width=0.8)),
        customdata=np.column_stack([t[mask]]),
        hovertemplate=(
            "turn %{customdata[0]}<br>"
            "actual effect on score: %{x:+.1f} pts<br>"
            "critic’s predicted ΔV: %{y:+.3f}<extra>" + name + "</extra>"
        ),
    ))

# Group means as generals (crown on an owner-colored tile).
for mask, color, edge in [(pos, BLUE, BLUE_EDGE), (neg, RED, RED_EDGE)]:
    fig.add_trace(go.Scatter(
        x=[x[mask].mean()], y=[y[mask].mean()], mode="markers+text",
        marker=dict(symbol="square", size=22, color=color,
                    line=dict(color=edge, width=2)),
        text="♛", textfont=dict(size=13, color="#ffffff"),
        textposition="middle center", showlegend=False,
        hovertemplate=("group mean (general)<br>actual: %{x:+.1f} pts<br>"
                       "critic: %{y:+.3f}<extra></extra>"),
    ))

fig.update_layout(
    paper_bgcolor=PAGE, plot_bgcolor=LAND,
    width=980, height=640,
    margin=dict(l=78, r=30, t=18, b=64),
    font=dict(family=F_GAME, color=TEXT),
    xaxis=dict(
        title=dict(text="ACTUAL EFFECT OF BUILDING ON FINAL SCORE  (points, build − control)",
                   font=dict(family=F_GAME, size=12.5, color=MUTED)),
        range=[XMIN, XMAX], gridcolor=GRID_MAJOR, gridwidth=1, zeroline=False,
        minor=dict(dtick=12.5, gridcolor=GRID_MINOR, gridwidth=1),
        tickfont=dict(family=F_GAME, size=12, color=MUTED), dtick=25,
    ),
    yaxis=dict(
        title=dict(text="CRITIC’S PREDICTED VALUE CHANGE  (ΔV, build − control)",
                   font=dict(family=F_GAME, size=12.5, color=MUTED)),
        range=[YMIN, YMAX], gridcolor=GRID_MAJOR, gridwidth=1, zeroline=False,
        minor=dict(dtick=0.2, gridcolor=GRID_MINOR, gridwidth=1),
        tickfont=dict(family=F_GAME, size=12, color=MUTED), dtick=0.5,
    ),
    hoverlabel=dict(bgcolor=PAGE, bordercolor="#555",
                    font=dict(family=F_GAME, size=13, color=TEXT)),
)

# -------------------------------------------------------------------- page ----
fragment = pio.to_html(fig, full_html=False, include_plotlyjs="cdn",
                       config={"displayModeBar": False, "responsive": False})

page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>The Critic's Fog of War</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Quicksand:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  body {{
    margin: 0; padding: 36px 16px 64px;
    background: {PAGE};
    font-family: Quicksand, 'Trebuchet MS', sans-serif; color: {TEXT};
  }}
  .game {{ max-width: 1030px; margin: 0 auto; }}
  .topbar {{
    display: flex; justify-content: space-between; align-items: flex-start;
    gap: 24px; margin-bottom: 14px;
  }}
  .turn {{ font-size: 26px; font-weight: 700; letter-spacing: 0.5px; }}
  .turn small {{ display: block; font-size: 12px; font-weight: 500; color: {MUTED};
                letter-spacing: 2px; text-transform: uppercase; }}
  table.leaderboard {{
    border-collapse: collapse; font-size: 13.5px; font-weight: 600;
    font-variant-numeric: tabular-nums;
  }}
  .leaderboard th {{
    font-weight: 500; color: {MUTED}; text-align: right; padding: 3px 10px;
    font-size: 11.5px; letter-spacing: 1px; text-transform: uppercase;
  }}
  .leaderboard th:first-child {{ text-align: left; }}
  .leaderboard td {{ padding: 4px 10px; text-align: right; }}
  .leaderboard td.player {{ text-align: left; color: #fff; min-width: 150px; }}
  .leaderboard tr.p-blue td.player {{ background: {BLUE}; }}
  .leaderboard tr.p-red td.player {{ background: {RED}; }}
  .leaderboard tr.p-gray td.player {{ background: {CASTLE}; }}
  .leaderboard tbody td {{ background: #2e2e2e; }}
  h1 {{ font-size: 40px; font-weight: 700; margin: 6px 0 8px; letter-spacing: 0.5px; }}
  .dek {{ font-size: 15.5px; font-weight: 500; line-height: 1.5; max-width: 880px;
         color: #cfcfcf; margin: 0 0 20px; }}
  .board {{ display: inline-block; border: 2px solid #111; }}
  .chat {{
    max-width: 620px; margin-top: 18px; background: #1b1b1b;
    border: 1px solid #333; border-radius: 4px; padding: 10px 14px;
    font-size: 14.5px; font-weight: 500; line-height: 1.65;
  }}
  .chat .sys {{ color: {MUTED}; font-style: italic; }}
  .chat .blue {{ color: #7f95e8; font-weight: 700; }}
  .chat .red {{ color: #e88b7f; font-weight: 700; }}
  .footer {{
    margin-top: 20px; max-width: 900px; font-size: 12.5px; font-weight: 500;
    line-height: 1.55; color: {MUTED};
    border-top: 1px solid #3a3a3a; padding-top: 10px;
  }}
</style>
</head>
<body>
<div class="game">
  <div class="topbar">
    <div class="turn"><small>Checkpoint</small>Turn 3,000</div>
    <table class="leaderboard">
      <thead><tr><th>Player</th><th>States</th><th>Δ score</th><th>Critic ΔV</th></tr></thead>
      <tbody>
        <tr class="p-blue"><td class="player">Builds that helped</td><td>624</td><td>+24.9</td><td>−0.46</td></tr>
        <tr class="p-red"><td class="player">Builds that hurt</td><td>899</td><td>−30.1</td><td>−0.63</td></tr>
        <tr class="p-gray"><td class="player">No effect</td><td>476</td><td>0.0</td><td>−0.19</td></tr>
      </tbody>
    </table>
  </div>
  <h1>The Critic’s Fog of War</h1>
  <p class="dek">The map below is every castle-build opportunity from stochastic
  self-play (n&nbsp;=&nbsp;1,999), placed by what the build actually did to the final game
  score (16 paired rollouts, build vs. no-build control) and by how the value head
  <i>predicted</i> its evaluation would change. Crowned tiles are group means. A
  calibrated critic would move the helpful blue armies up into open ground — but for
  this critic, everything above the line is permanent fog.</p>
  <div class="board">{fragment}</div>
  <div class="chat">
    <div class="sys">[system] forced 1,999 castle builds against paired no-build controls</div>
    <div><span class="blue">[blue]</span> our good builds raised final score by +25 points on average</div>
    <div><span class="red">[critic]</span> denied. all builds marked down — mean ΔV −0.46, only 0.3% approved</div>
    <div class="sys">[system] blue’s castles won games. the critic never saw it.</div>
  </div>
  <p class="footer">[replay notes] paired stochastic continuations share opponent
  actions and future random draws · score effects on [0,&thinsp;100] points · value on the
  network’s [−1,&thinsp;1] expectation scale · colors are the project’s own match palette;
  fog, land, and neutral-castle grays are the renderer’s exact tile colors · 291
  immediately-terminal pairs and 17 states with no surviving pair excluded · full
  analysis: castle_value_probe_iter3000</p>
</div>
</body>
</html>
"""
OUT_HTML.write_text(page)
print(f"wrote {OUT_HTML}")
