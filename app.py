"""
LiquidJetSim - Capillary oscillations in a falling liquid jet
Team Samudramanthan | Ensemble Young Physicists' Tournament 2026

Interactive simulator of the Rayleigh-Plateau instability and the azimuthal
capillary modes of a falling liquid jet. Solves the per-mode damped-harmonic-
oscillator equations (n=0 growth, n=2,3,4 oscillation) with gravity thinning,
maps amplitude(t) to axial position z(t), and renders the cross-section
R(theta, z) as a heatmap. See PHYSICS.md for the full derivation.

Run:  streamlit run app.py
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import i0, i1
from scipy.optimize import brentq
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="LiquidJetSim", layout="wide",
                   initial_sidebar_state="expanded")

# ----------------------------------------------------------------------
# Theme  -  dark / futuristic design system
# ----------------------------------------------------------------------
BG        = "#05060a"     # near-black base
PANEL     = "#0d1018"     # panel surface
GRID      = "#1c2230"     # grid / hairlines
TEXT      = "#e6ebff"     # primary text
MUTE      = "#7a86a8"     # muted text
CYAN      = "#22d3ee"     # primary accent
VIOLET    = "#a855f7"     # secondary accent
LIME      = "#7dffb3"     # tertiary accent
COLORWAY  = [CYAN, VIOLET, LIME, "#ff6ad5"]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap');

.stApp {{
    background:
        radial-gradient(1100px 600px at 12% -8%, rgba(34,211,238,0.10), transparent 60%),
        radial-gradient(900px 700px at 108% 0%, rgba(168,85,247,0.12), transparent 55%),
        {BG};
    color: {TEXT};
    font-family: 'Space Grotesk', sans-serif;
}}
#MainMenu, footer, header {{ visibility: hidden; }}

section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0a0c14 0%, #06070c 100%);
    border-right: 1px solid {GRID};
}}
section[data-testid="stSidebar"] * {{ color: {TEXT}; }}

h1, h2, h3 {{ font-family: 'Space Grotesk', sans-serif; letter-spacing: .5px; }}

/* hero title */
.hero-title {{
    font-size: 2.5rem; font-weight: 700; line-height: 1.05; margin: 0;
    background: linear-gradient(92deg, {CYAN}, {VIOLET} 55%, {LIME});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}}
.hero-sub {{ color: {MUTE}; font-family: 'JetBrains Mono', monospace;
    font-size: .82rem; letter-spacing: 2px; text-transform: uppercase; }}
.hero-rule {{ height: 1px; margin: 1rem 0 1.4rem 0;
    background: linear-gradient(90deg, {CYAN}, transparent); }}

/* metric cards */
div[data-testid="stMetric"] {{
    background: linear-gradient(160deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
    border: 1px solid {GRID}; border-radius: 14px; padding: 14px 16px;
    box-shadow: 0 0 0 1px rgba(34,211,238,0.04), 0 10px 30px rgba(0,0,0,0.35);
    backdrop-filter: blur(6px);
}}
div[data-testid="stMetricValue"] {{ font-family: 'JetBrains Mono', monospace;
    color: {CYAN}; }}
div[data-testid="stMetricLabel"] {{ color: {MUTE}; }}

/* section labels */
.sec {{ font-family: 'JetBrains Mono', monospace; font-size: .78rem;
    letter-spacing: 3px; text-transform: uppercase; color: {VIOLET};
    margin: 1.6rem 0 .3rem 0; }}
.sec-title {{ font-size: 1.35rem; font-weight: 600; margin: 0 0 .6rem 0; }}

.chip {{ display:inline-block; font-family:'JetBrains Mono',monospace;
    font-size:.72rem; padding:2px 10px; border-radius:999px; margin-right:6px;
    border:1px solid {GRID}; color:{MUTE}; }}

.stPlotlyChart {{ border:1px solid {GRID}; border-radius:14px; overflow:hidden;
    background:{PANEL}; }}
</style>
""", unsafe_allow_html=True)


def dark_layout(fig, height=420, title=None):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(family="JetBrains Mono, monospace", color=TEXT, size=12),
        colorway=COLORWAY, height=height, title=title,
        margin=dict(l=12, r=12, t=40 if title else 14, b=12),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


FUTURE_SCALE = [[0.0, "#05060a"], [0.25, "#123047"], [0.5, "#0e7490"],
                [0.75, "#22d3ee"], [1.0, "#e0fbff"]]

# ----------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------
st.sidebar.markdown("<div class='hero-title' style='font-size:1.5rem'>LIQUIDJETSIM</div>",
                    unsafe_allow_html=True)
st.sidebar.markdown("<div class='hero-sub'>control deck</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='hero-rule'></div>", unsafe_allow_html=True)

st.sidebar.markdown("**Fluid**")
rho   = st.sidebar.number_input("Density  rho  [kg/m3]", 100.0, 2000.0, 1000.0, 10.0)
sigma = st.sidebar.number_input("Surface tension  sigma  [N/m]", 0.01, 0.2, 0.072,
                                0.001, format="%.3f")
g = 9.81

st.sidebar.markdown("**Jet**")
R0 = st.sidebar.slider("Initial radius  R0  [mm]", 1.0, 15.0, 10.0, 0.5) * 1e-3
v0 = st.sidebar.slider("Initial speed  v0  [m/s]", 0.2, 2.0, 0.6, 0.1)

st.sidebar.markdown("**Viscosity**")
regime = st.sidebar.radio("Regime", ["Underdamped", "Overdamped", "Custom"])
if regime == "Underdamped":
    mu = 0.05
elif regime == "Overdamped":
    mu = 2.0
else:
    mu = st.sidebar.slider("Viscosity  mu  [Pa s]", 0.001, 3.0, 0.05, 0.001)

modes = st.sidebar.multiselect("Modes", [0, 2, 3, 4], default=[0, 2, 3, 4])

st.sidebar.markdown("<div class='hero-rule'></div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='hero-sub'>team samudramanthan</div>",
                    unsafe_allow_html=True)

Oh = mu / np.sqrt(rho * sigma * R0)

# ----------------------------------------------------------------------
# Physics helpers
# ----------------------------------------------------------------------
def R_of_t(t):
    return R0 * np.sqrt(v0 / (v0 + g * t))

def dho_rhs(t, y, n):
    R = R_of_t(t)
    if n == 0:                                   # Rayleigh-Plateau growth
        s0    = 0.343 * np.sqrt(sigma / (rho * R**3))
        gamma = 3 * mu / (rho * R**2)            # Trouton factor 3
        stiff = -s0**2
    else:                                        # azimuthal capillary oscillation
        omega = np.sqrt(n * (n**2 - 1) * sigma / (rho * R**3))
        gamma = n * mu / (rho * R**2)
        stiff = omega**2
    d, ddot = y
    return [ddot, -2 * gamma * ddot - stiff * d]

def solve_mode(n, tspan):
    sol = solve_ivp(dho_rhs, (tspan[0], tspan[-1]), [1.5e-3, 0.0], t_eval=tspan,
                    args=(n,), rtol=1e-3, atol=1e-6, method="RK45")
    return sol.y[0]

def rp_growth_rate(x):
    x = np.asarray(x, dtype=float)
    return np.sqrt(np.clip(x * (i1(x) / i0(x)) * (1 - x**2), 0, None))

def kR_max():
    f = lambda x: (1 - (i1(x)/i0(x))**2) * (1 - x**2) - 2 * x * (i1(x)/i0(x))
    return brentq(f, 0.3, 0.99)

# ----------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------
st.markdown("<div class='hero-sub'>capillary oscillations // falling liquid jet</div>",
            unsafe_allow_html=True)
st.markdown("<div class='hero-title'>The pattern is real.</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-rule'></div>", unsafe_allow_html=True)
st.markdown(
    "<span style='color:#7a86a8'>The swellings on a poured stream are the linearly "
    "selected dominant mode of the <b style='color:#e6ebff'>Rayleigh-Plateau "
    "instability</b>, not an illusion. Full derivation in "
    "<code>PHYSICS.md</code>.</span>", unsafe_allow_html=True)

xmax   = kR_max()
lam_rp = 2 * np.pi * R0 / xmax
We     = rho * v0**2 * R0 / sigma
c1, c2, c3, c4 = st.columns(4)
c1.metric("Wavelength lambda", f"{1e3*lam_rp:.1f} mm", f"kR = {xmax:.3f}")
c2.metric("lambda / R", f"{lam_rp/R0:.2f}", "9.01 universal")
c3.metric("Weber", f"{We:.0f}", "convective" if We > 1 else "absolute")
c4.metric("Ohnesorge", f"{Oh:.3f}", "viscous" if Oh > 1 else "inviscid")

# ----------------------------------------------------------------------
# Cross-section heatmaps
# ----------------------------------------------------------------------
tspan = np.linspace(0, 0.5, 600)
z     = v0 * tspan + 0.5 * g * tspan**2
Rz    = R_of_t(tspan)
theta = np.linspace(0, 2 * np.pi, 200)

st.markdown("<div class='sec'>module 01</div>", unsafe_allow_html=True)
st.markdown(f"<div class='sec-title'>Cross-section radius  R(theta, z)  &mdash; "
            f"{regime.lower()} regime</div>", unsafe_allow_html=True)
st.markdown(f"<span class='chip'>Oh {Oh:.3f}</span>"
            f"<span class='chip'>R0 {1e3*R0:.1f} mm</span>"
            f"<span class='chip'>v0 {v0:.1f} m/s</span>", unsafe_allow_html=True)

if not modes:
    st.info("Select at least one mode in the control deck.")
else:
    cols = st.columns(len(modes))
    for col, n in zip(cols, modes):
        delta = solve_mode(n, tspan)
        surf  = Rz[:, None] + delta[:, None] * np.cos(n * theta[None, :])
        fig = go.Figure(go.Heatmap(x=theta, y=z, z=surf * 1e3,
                                   colorscale=FUTURE_SCALE,
                                   colorbar=dict(title="R [mm]", thickness=10)))
        dark_layout(fig, height=430, title=f"n = {n}")
        fig.update_xaxes(title="theta [rad]")
        fig.update_yaxes(title="z [m]")
        col.plotly_chart(fig, use_container_width=True)
        if n >= 2:
            mu_c = np.sqrt(rho * sigma * R0 * (n**2 - 1) / n)
            state = "overdamped" if mu > mu_c else "underdamped"
            col.markdown(f"<span class='chip'>mu_c {mu_c:.2f} Pa s &rarr; "
                         f"{state}</span>", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Growth-rate curve
# ----------------------------------------------------------------------
st.markdown("<div class='sec'>module 02</div>", unsafe_allow_html=True)
st.markdown("<div class='sec-title'>Rayleigh-Plateau growth rate &mdash; "
            "why lambda = 9.01 R</div>", unsafe_allow_html=True)

xg = np.linspace(1e-3, 1.4, 400)
sg = rp_growth_rate(xg)
figg = go.Figure()
figg.add_trace(go.Scatter(x=xg, y=sg, mode="lines",
                          line=dict(color=CYAN, width=3),
                          fill="tozeroy", fillcolor="rgba(34,211,238,0.10)"))
figg.add_vrect(x0=1, x1=1.4, fillcolor=VIOLET, opacity=0.08, line_width=0,
               annotation_text="stable  kR>1", annotation_font_color=MUTE)
figg.add_vline(x=xmax, line_dash="dash", line_color=LIME,
               annotation_text=f"kR = {xmax:.3f}", annotation_font_color=LIME)
dark_layout(figg, height=400)
figg.update_xaxes(title="kR")
figg.update_yaxes(title="growth rate  /  sqrt(sigma/rho R^3)")
st.plotly_chart(figg, use_container_width=True)

st.markdown(
    "<span style='color:#7a86a8'>"
    "kR &lt; 1: long waves unstable, the jet breaks into drops. &nbsp; "
    "kR &gt; 1: surface tension suppresses short waves. &nbsp; "
    "Peak at kR = 0.697 selects lambda = 9.01 R. &nbsp; "
    "Predictions: node spacing ~ 9R; double pour height doubles lambda; "
    "soap shortens lambda.</span>", unsafe_allow_html=True)
