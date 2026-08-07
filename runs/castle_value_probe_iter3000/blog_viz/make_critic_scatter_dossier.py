"""Blog figure, style option 5: war-room / wargame dossier.

Same data and chart as the other options (checkpoint-3000 castle value probe:
actual causal effect of building vs. the critic's predicted value change),
styled as a declassified after-action report — manila file paper, stencil and
typewriter type, a field map with faint topo contours, NATO-flavored unit
markers (friendly blue squares vs. hostile red diamonds — shape doubles as a
secondary encoding), a grease-pencil circle around the anomalous cluster, and
a DECLASSIFIED stamp. Writes critic_castle_scatter_dossier.html.

Palette validated on the map surface (#ddd3b8): red/blue poles pass all six
checks (worst CVD ΔE 9.6); the olive neutral is a diverging midpoint (low
chroma by design), de-emphasized, shape-coded, and relieved by legend +
tooltips.
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

HERE = Path(__file__).resolve().parent
NPZ = HERE.parent / "atlas_with_values" / "paired_rollouts.npz"
OUT_HTML = HERE / "critic_castle_scatter_dossier.html"

# ---------------------------------------------------------------- palette ----
FOLDER = "#d8cfb4"       # manila file paper
MAP = "#ddd3b8"          # field-map paper (validated surface)
MEMO = "#e6dcc2"         # memo slips
INK = "#35322a"          # typewriter ink
MUTED = "#6b6552"        # faded ink
GRID = "#c6bb9c"         # map grid hairlines
CONTOUR = "rgba(139,132,104,0.40)"
BLUE = "#1e639e"         # friendly: builds that helped (validated pole)
RED = "#b03a30"          # hostile: builds that hurt (validated pole)
OLIVE = "#8b8468"        # no measurable effect (neutral midpoint)
GREASE = "#c0392b"       # grease pencil

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
F_STENCIL = "'Allerta Stencil', 'Arial Black', sans-serif"
F_TYPE = "'Special Elite', 'Courier New', monospace"
F_MONO = "'Courier Prime', 'Courier New', monospace"
F_PENCIL = "Caveat, 'Comic Sans MS', cursive"

fig = go.Figure()
XMIN, XMAX = -105, 105
YMIN, YMAX = -1.78, 0.62


def blob_path(cx, cy, rx, ry, phases, scale=1.0, n=60):
    """Closed wobbly loop (topo-contour flavored) as an SVG path in data coords."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 1 + 0.09 * np.sin(3 * th + phases[0]) + 0.06 * np.sin(5 * th + phases[1])
    px = cx + scale * rx * r * np.cos(th)
    py = cy + scale * ry * r * np.sin(th)
    return ("M " + " L ".join(f"{a:.2f},{b:.4f}" for a, b in zip(px, py)) + " Z")


# Faint topo contours in the quiet corners of the map.
for cx, cy, rx, ry, ph in [(-58, 0.30, 34, 0.20, (0.7, 2.1)),
                           (86, -1.52, 26, 0.20, (2.9, 1.3))]:
    for s in (1.0, 0.72, 0.45):
        fig.add_shape(type="path", path=blob_path(cx, cy, rx, ry, ph, s),
                      line=dict(color=CONTOUR, width=1), layer="below")

# Boundary lines.
fig.add_hline(y=0, line=dict(color=INK, width=1.6))
fig.add_shape(type="line", x0=0, x1=0, y0=YMIN, y1=YMAX,
              line=dict(color=MUTED, width=1.2))
fig.add_annotation(
    x=XMAX - 2, y=0.035, xanchor="right", yanchor="bottom",
    text="PHASE LINE ZERO — CRITIC APPROVAL BOUNDARY", showarrow=False,
    font=dict(family=F_STENCIL, size=10, color=MUTED),
)

groups = [
    ("Build helped", pos, BLUE, "square", 6.5, 0.85),
    ("Build hurt", neg, RED, "diamond", 6.5, 0.8),
    ("No effect", zer, OLIVE, "circle", 4.5, 0.6),
]
for name, mask, color, symbol, size, alpha in groups:
    fig.add_trace(go.Scattergl(
        x=x[mask], y=y[mask], mode="markers", name=f"{name} ({mask.sum()})",
        marker=dict(symbol=symbol, size=size, color=color, opacity=alpha,
                    line=dict(color=INK, width=0.6)),
        customdata=np.column_stack([t[mask]]),
        hovertemplate=(
            "turn %{customdata[0]}<br>"
            "actual effect on score: %{x:+.1f} pts<br>"
            "critic’s predicted ΔV: %{y:+.3f}<extra>" + name + "</extra>"
        ),
    ))

# Group means: HQ-sized unit symbols.
for mask, color, symbol in [(pos, BLUE, "square"), (neg, RED, "diamond")]:
    fig.add_trace(go.Scatter(
        x=[x[mask].mean()], y=[y[mask].mean()], mode="markers",
        marker=dict(symbol=symbol, size=17, color=color,
                    line=dict(color=INK, width=2)),
        showlegend=False, hovertemplate=(
            "group mean (HQ)<br>actual: %{x:+.1f} pts<br>"
            "critic: %{y:+.3f}<extra></extra>"),
    ))

# Grease-pencil ring around the friendly cluster the critic graded down.
ring = blob_path(24.9, -0.46, 26, 0.34, (1.9, 4.4), n=80)
fig.add_shape(type="path", path=ring, line=dict(color=GREASE, width=3), opacity=0.85)
fig.add_shape(type="path", path=blob_path(24.9, -0.462, 27.2, 0.352, (2.2, 4.1), n=80),
              line=dict(color=GREASE, width=1.5), opacity=0.55)
fig.add_annotation(
    x=42, y=-0.68, ax=64, ay=-1.02, axref="x", ayref="y",
    xanchor="center", yanchor="top",
    text="castles WON here —<br>still graded ✗", showarrow=True,
    arrowhead=1, arrowwidth=1.6, arrowcolor=GREASE,
    font=dict(family=F_PENCIL, size=20, color=GREASE), align="left",
)

# Typewritten memo slips.
fig.add_annotation(
    x=66, y=0.36, xanchor="center", yanchor="middle",
    text=("<b>INTEL SUMMARY 01</b><br>"
          "friendly build ops raised final<br>"
          "score by +25 pts on average"),
    font=dict(family=F_TYPE, size=12, color=INK),
    align="left", bgcolor=MEMO, bordercolor=INK, borderwidth=1.5,
    borderpad=8, showarrow=False,
)
fig.add_annotation(
    x=66, y=-1.42, xanchor="center", yanchor="middle",
    text=("<b>HQ ASSESSMENT</b><br>"
          "all build ops denied — mean ΔV −0.46<br>"
          "approved: 0.3%"),
    font=dict(family=F_TYPE, size=12, color=INK),
    align="left", bgcolor=MEMO, bordercolor=RED, borderwidth=1.5,
    borderpad=8, showarrow=False,
)

for yy, txt, anch in [(0.3, "CRITIC: BUILD HELPS", "bottom"),
                      (-0.07, "CRITIC: BUILD HURTS", "top")]:
    fig.add_annotation(
        x=XMIN + 3, y=yy, xanchor="left", yanchor=anch,
        text=txt, showarrow=False,
        font=dict(family=F_STENCIL, size=10.5, color=MUTED),
    )

fig.update_layout(
    paper_bgcolor=MAP, plot_bgcolor=MAP,
    width=980, height=640,
    margin=dict(l=78, r=30, t=30, b=64),
    font=dict(family=F_TYPE, color=INK),
    legend=dict(
        orientation="h", x=0, y=1.06, xanchor="left",
        font=dict(family=F_TYPE, size=13, color=INK),
        bgcolor="rgba(0,0,0,0)",
    ),
    xaxis=dict(
        title=dict(text="ACTUAL EFFECT OF BUILDING ON FINAL SCORE  (points, build − control)",
                   font=dict(family=F_STENCIL, size=11, color=MUTED)),
        range=[XMIN, XMAX], gridcolor=GRID, gridwidth=1, zeroline=False,
        tickfont=dict(family=F_MONO, size=11.5, color=MUTED), dtick=25,
    ),
    yaxis=dict(
        title=dict(text="CRITIC’S PREDICTED VALUE CHANGE  (ΔV, build − control)",
                   font=dict(family=F_STENCIL, size=11, color=MUTED)),
        range=[YMIN, YMAX], gridcolor=GRID, gridwidth=1, zeroline=False,
        tickfont=dict(family=F_MONO, size=11.5, color=MUTED), dtick=0.5,
    ),
    hoverlabel=dict(bgcolor=MEMO, bordercolor=INK,
                    font=dict(family=F_TYPE, size=12, color=INK)),
)

# -------------------------------------------------------------------- page ----
fragment = pio.to_html(fig, full_html=False, include_plotlyjs="cdn",
                       config={"displayModeBar": False, "responsive": False})

page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Operation Castle Gambit — after-action assessment</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Allerta+Stencil&family=Special+Elite&family=Courier+Prime&family=Caveat:wght@600&display=swap" rel="stylesheet">
<style>
  body {{
    margin: 0; padding: 34px 16px 64px;
    background-color: {FOLDER};
    background-image:
      radial-gradient(circle, rgba(53,50,42,0.035) 1px, transparent 1.4px),
      linear-gradient(180deg, transparent 57.9%, rgba(53,50,42,0.09) 58%, rgba(255,255,255,0.25) 58.15%, transparent 58.5%);
    background-size: 11px 11px, 100% 100%;
    font-family: 'Special Elite', 'Courier New', monospace; color: {INK};
  }}
  .dossier {{ max-width: 1030px; margin: 0 auto; position: relative; }}
  .banner {{
    text-align: center; font-size: 12px; letter-spacing: 4px;
    border-top: 2px solid {INK}; border-bottom: 2px solid {INK};
    padding: 5px 0; margin-bottom: 20px; color: {INK};
  }}
  h1 {{
    font-family: 'Allerta Stencil', 'Arial Black', sans-serif;
    font-size: 42px; letter-spacing: 3px; margin: 0 0 4px; color: {INK};
  }}
  .sub {{ font-size: 15px; letter-spacing: 1px; margin: 0 0 6px; color: {MUTED}; }}
  .meta {{ font-size: 12.5px; letter-spacing: 1px; color: {MUTED}; margin: 0 0 16px; }}
  .stamp {{
    position: absolute; top: 52px; right: 8px; transform: rotate(-12deg);
    border: 3px double {GREASE}; color: {GREASE}; opacity: 0.8;
    padding: 6px 14px; text-align: center;
    font-family: 'Allerta Stencil', sans-serif; font-size: 20px;
    letter-spacing: 3px; pointer-events: none;
  }}
  .stamp small {{ display: block; font-size: 9px; letter-spacing: 2px; }}
  .dek {{ font-size: 14.5px; line-height: 1.55; max-width: 880px; margin: 0 0 20px; }}
  .mapwrap {{ position: relative; display: inline-block; }}
  .mapwrap .frame {{ display: block; border: 1px solid {MUTED};
                    box-shadow: 3px 4px 0 rgba(53,50,42,0.25); }}
  .tape {{
    position: absolute; width: 92px; height: 24px;
    background: rgba(232,224,199,0.75); border: 1px solid rgba(53,50,42,0.12);
    top: -12px; pointer-events: none;
  }}
  .tape.l {{ left: 36px; transform: rotate(-5deg); }}
  .tape.r {{ right: 36px; transform: rotate(4deg); }}
  .footer {{
    margin-top: 24px; max-width: 900px; font-size: 12px; line-height: 1.6;
    color: {MUTED}; border-top: 1px solid rgba(53,50,42,0.3); padding-top: 10px;
  }}
</style>
</head>
<body>
<div class="dossier">
  <p class="banner">UNCLASSIFIED&nbsp;//&nbsp;APPROVED FOR PUBLIC RELEASE</p>
  <div class="stamp">DECLASSIFIED<small>BLOG RELEASE AUTHORIZED</small></div>
  <h1>OPERATION CASTLE GAMBIT</h1>
  <p class="sub">AFTER-ACTION ASSESSMENT — VALUE CRITIC · CHECKPOINT 3000</p>
  <p class="meta">FILE: castle_value_probe_iter3000 &nbsp;·&nbsp; CONTACTS: 1,999 &nbsp;·&nbsp; COPY 1 OF 1</p>
  <p class="dek">SUBJECT: castle construction, critic assessment thereof. Each unit
  marker is one castle-build opportunity from stochastic self-play, positioned by the
  build’s actual effect on final game score (16 paired rollouts, build vs. no-build
  control) and by the value head’s <i>predicted</i> change in its own evaluation.
  Friendly squares: builds that helped. Hostile diamonds: builds that hurt. A
  calibrated critic would place helpful operations above Phase Line Zero. Field
  reality: 99.4% of all operations were assessed below the line.</p>
  <div class="mapwrap">
    <span class="tape l"></span><span class="tape r"></span>
    <div class="frame">{fragment}</div>
  </div>
  <p class="footer">ANNEX A — METHOD: paired stochastic continuations share opponent
  actions and future random draws · score effects on [0,&thinsp;100] points · value on the
  network’s [−1,&thinsp;1] expectation scale · HQ symbols mark group means · 291
  immediately-terminal pairs and 17 states with no surviving pair excluded · full
  analysis: castle_value_probe_iter3000 · END OF ANNEX</p>
</div>
</body>
</html>
"""
OUT_HTML.write_text(page)
print(f"wrote {OUT_HTML}")
