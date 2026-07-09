"""
LiquidJetSim - Capillary oscillations in a falling liquid jet
Team Samudramanthan | Ensemble Young Physicists' Tournament 2026

Interactive simulator of the Rayleigh-Plateau instability and the azimuthal
capillary modes of a falling liquid jet, extended with surfactant (detergent),
electric-field and magnetic-field effects. Includes a live animated falling
jet, cross-section heatmaps, the Rayleigh-Plateau growth-rate curve, and
Physical-insights / Applications pages. See PHYSICS.md for the derivation.

Run:  streamlit run app.py
"""

import json
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import i0, i1
from scipy.optimize import brentq, minimize_scalar
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="LiquidJetSim", layout="wide",
                   initial_sidebar_state="expanded")

# ----------------------------------------------------------------------
# Design tokens
# ----------------------------------------------------------------------
BG, PANEL, GRID = "#05060a", "#0d1018", "#232a3a"
TEXT, MUTE      = "#dfe6f5", "#8b97b8"
CYAN, VIOLET, LIME = "#38d6ee", "#b070f0", "#8affc0"
COLORWAY = [CYAN, VIOLET, LIME, "#ff8ad8"]
EPS0, MU0 = 8.854e-12, 4 * np.pi * 1e-7

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600&family=JetBrains+Mono:wght@300;400;500&display=swap');

.stApp {{
    background:
        radial-gradient(1100px 600px at 10% -8%, rgba(56,214,238,0.08), transparent 60%),
        radial-gradient(900px 700px at 108% 0%, rgba(176,112,240,0.10), transparent 55%),
        {BG};
    color: {TEXT};
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 400;
}}
#MainMenu, footer {{ visibility: hidden; }}

/* keep the sidebar + its expand control visible */
section[data-testid="stSidebar"] {{
    background: {PANEL};
    border-right: 1px solid {GRID};
    min-width: 300px;
}}
section[data-testid="stSidebar"] * {{ color: {TEXT} !important; }}
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {{ visibility: visible !important; color: {CYAN} !important; }}
section[data-testid="stSidebar"] label p {{ color: {MUTE} !important; font-size: .82rem; }}

h1,h2,h3 {{ font-family:'Space Grotesk',sans-serif; font-weight:500; letter-spacing:.2px; }}

.hero-title {{ font-size:2.3rem; font-weight:600; line-height:1.06; margin:0;
    background:linear-gradient(92deg,{CYAN},{VIOLET} 55%,{LIME});
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
.hero-sub {{ color:{MUTE}; font-family:'JetBrains Mono',monospace; font-weight:400;
    font-size:.8rem; letter-spacing:2px; text-transform:uppercase; }}
.hero-rule {{ height:1px; margin:.9rem 0 1.2rem 0;
    background:linear-gradient(90deg,{CYAN},transparent); }}

div[data-testid="stMetric"] {{
    background:linear-gradient(160deg, rgba(255,255,255,0.035), rgba(255,255,255,0.01));
    border:1px solid {GRID}; border-radius:14px; padding:14px 16px;
    box-shadow:0 10px 30px rgba(0,0,0,0.30); }}
div[data-testid="stMetricValue"] {{ font-family:'JetBrains Mono',monospace;
    font-weight:500; color:{CYAN}; }}
div[data-testid="stMetricLabel"] {{ color:{MUTE}; }}

.sec {{ font-family:'JetBrains Mono',monospace; font-weight:400; font-size:.75rem;
    letter-spacing:3px; text-transform:uppercase; color:{VIOLET}; margin:1.6rem 0 .3rem 0; }}
.sec-title {{ font-size:1.3rem; font-weight:500; margin:0 0 .6rem 0; }}
.chip {{ display:inline-block; font-family:'JetBrains Mono',monospace; font-weight:400;
    font-size:.72rem; padding:2px 10px; border-radius:999px; margin:0 6px 6px 0;
    border:1px solid {GRID}; color:{MUTE}; }}
.card {{ background:rgba(255,255,255,0.02); border:1px solid {GRID}; border-radius:14px;
    padding:18px 20px; margin-bottom:14px; }}
.card h4 {{ margin:.1rem 0 .4rem 0; font-weight:500; color:{TEXT}; }}
.card p, .card li {{ color:{MUTE}; font-weight:400; }}
.stPlotlyChart {{ border:1px solid {GRID}; border-radius:14px; overflow:hidden; background:{PANEL}; }}
</style>
""", unsafe_allow_html=True)


def dark_layout(fig, height=420, title=None):
    fig.update_layout(template="plotly_dark", paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                      font=dict(family="JetBrains Mono, monospace", color=TEXT, size=12),
                      colorway=COLORWAY, height=height, title=title,
                      margin=dict(l=12, r=12, t=40 if title else 14, b=12))
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


FUTURE_SCALE = [[0.0, "#05060a"], [0.25, "#123047"], [0.5, "#0e7490"],
                [0.75, "#38d6ee"], [1.0, "#e0fbff"]]

# ----------------------------------------------------------------------
# Sidebar : navigation + parameters
# ----------------------------------------------------------------------
st.sidebar.markdown("<div class='hero-title' style='font-size:1.4rem'>LIQUIDJETSIM</div>",
                    unsafe_allow_html=True)
st.sidebar.markdown("<div class='hero-sub'>control deck</div>", unsafe_allow_html=True)
page = st.sidebar.radio("View", ["Simulator", "Physical insights", "Applications"])
st.sidebar.markdown("<div class='hero-rule'></div>", unsafe_allow_html=True)

st.sidebar.markdown("**Fluid**")
rho   = st.sidebar.number_input("Density  rho  [kg/m3]", 100.0, 2000.0, 1000.0, 10.0)
sigma0 = st.sidebar.number_input("Surface tension  sigma0  [N/m]", 0.01, 0.2, 0.072,
                                 0.001, format="%.3f")
g = 9.81

st.sidebar.markdown("**Jet**")
R0 = st.sidebar.slider("Initial radius  R0  [mm]", 1.0, 15.0, 10.0, 0.5) * 1e-3
v0 = st.sidebar.slider("Initial speed  v0  [m/s]", 0.2, 2.0, 0.6, 0.1)

st.sidebar.markdown("**Viscosity**")
regime = st.sidebar.radio("Regime", ["Underdamped", "Overdamped", "Custom"])
mu = 0.05 if regime == "Underdamped" else 2.0 if regime == "Overdamped" else \
    st.sidebar.slider("Viscosity  mu  [Pa s]", 0.001, 3.0, 0.05, 0.001)

st.sidebar.markdown("**Fields & additives**")
deterg = st.sidebar.slider("Detergent  [% saturation]", 0, 100, 0, 5) / 100.0
E_kv   = st.sidebar.slider("Electric field  E  [kV/cm]", 0.0, 20.0, 0.0, 0.5)
B_T    = st.sidebar.slider("Magnetic field  B  [T]  (ferrofluid)", 0.0, 0.5, 0.0, 0.01)
chi_m  = 1.0  # effective magnetic susceptibility of the ferrofluid model

modes = st.sidebar.multiselect("Azimuthal modes", [0, 2, 3, 4], default=[0, 2, 3, 4])
st.sidebar.markdown("<div class='hero-rule'></div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='hero-sub'>team samudramanthan</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Effective surface tension + field Bond numbers
# ----------------------------------------------------------------------
# Detergent lowers surface tension (up to ~60%).
sigma = sigma0 * (1.0 - 0.6 * deterg)
E = E_kv * 1e5          # kV/cm -> V/m
# Axial electric field on a conducting jet: stabilising (electric Bond number).
Gamma_E = EPS0 * E**2 * R0 / sigma
# Normal magnetic field on a ferrofluid: destabilising. Normalised magnetic Bond
# number - the coefficient absorbs mu0, susceptibility and demagnetising factor so
# the 0-0.5 T slider maps onto an O(1-3) coupling for the model.
Gamma_M = 12.0 * chi_m * (B_T ** 2)

Oh = mu / np.sqrt(rho * sigma * R0)
We = rho * v0**2 * R0 / sigma

# ----------------------------------------------------------------------
# Physics
# ----------------------------------------------------------------------
def R_of_t(t):
    return R0 * np.sqrt(v0 / (v0 + g * t))

def rp_bracket(x):
    """Modified n=0 dispersion bracket with electric + magnetic fields."""
    return (1 - x**2) * (1 + Gamma_M) - Gamma_E * x**2

def rp_growth(x):
    x = np.asarray(x, dtype=float)
    val = x * (i1(x) / i0(x)) * rp_bracket(x)
    return np.sqrt(np.clip(val, 0, None))          # units of sqrt(sigma/rho R^3)

def x_marginal():
    """Largest kR that is still unstable, given the fields."""
    num = 1 + Gamma_M
    xc2 = num / (num + Gamma_E)          # bracket = 0
    return np.sqrt(max(min(xc2, 0.999), 1e-4))

def x_fastest():
    xc = x_marginal()
    r = minimize_scalar(lambda x: -x * (i1(x)/i0(x)) * rp_bracket(x),
                        bounds=(1e-3, xc * 0.999), method="bounded")
    return float(r.x)

def dho_rhs(t, y, n):
    R = R_of_t(t)
    if n == 0:
        s0 = 0.343 * np.sqrt(sigma / (rho * R**3)) * np.sqrt(max(rp_bracket(x_fastest())
                                                                 / max(1 - x_fastest()**2, 1e-6), 0.0))
        gamma, stiff = 3 * mu / (rho * R**2), -s0**2
    else:
        omega = np.sqrt(n * (n**2 - 1) * sigma / (rho * R**3))
        gamma, stiff = n * mu / (rho * R**2), omega**2
    d, ddot = y
    return [ddot, -2 * gamma * ddot - stiff * d]

def solve_mode(n, tspan):
    sol = solve_ivp(dho_rhs, (tspan[0], tspan[-1]), [1.5e-3, 0.0], t_eval=tspan,
                    args=(n,), rtol=1e-3, atol=1e-6, method="RK45")
    return sol.y[0]

xmax   = x_fastest()
lam_rp = 2 * np.pi * R0 / xmax
s_dim  = rp_growth(xmax) * np.sqrt(sigma / (rho * R0**3))     # 1/s
Lb     = 2.9 * R0 * np.sqrt(max(We, 1e-6)) * 2.3              # eta0/R ~ 0.1

# ----------------------------------------------------------------------
# Live falling-jet animation (HTML5 canvas)
# ----------------------------------------------------------------------
_ANIM_TEMPLATE = r"""
<div id="wrap" style="width:100%;background:__PANEL__;border:1px solid __GRID__;
     border-radius:14px;overflow:hidden;">
  <canvas id="jet" style="display:block;width:100%;height:540px;"></canvas>
</div>
<script>
const P = __PARAMS__;
const cv = document.getElementById('jet');
const ctx = cv.getContext('2d');
function fit(){ const w = cv.clientWidth; cv.width = w; cv.height = 540; }
fit(); window.addEventListener('resize', fit);

const g = 9.81, A0 = 0.045;
const Zmax_mm = Math.min(Math.max(P.Lb_mm*1.3, 55), 400);

function Rz_mm(zm){ return P.R0mm * Math.sqrt(P.v0/(P.v0 + g*zm)); }
function amp(zm){ return Math.min(A0*Math.exp(P.s*zm/P.v0), 0.9); }

let t0 = null;
function frame(ts){
  if(t0===null) t0 = ts;
  const t = (ts - t0)/1000.0;              // seconds
  const W = cv.width, H = cv.height, cx = W*0.5;
  ctx.clearRect(0,0,W,H);

  const pxPerMm = H / Zmax_mm;
  const maxR = P.R0mm*1.9;
  const rScale = Math.min(4.0, (0.36*W)/(maxR));
  const k = 2*Math.PI/(P.lam_mm/1000.0);   // 1/m
  const zb_mm = Math.min(P.Lb_mm, Zmax_mm*0.82);

  // nozzle
  ctx.fillStyle = '#161c28';
  ctx.fillRect(cx - P.R0mm*rScale - 8, 0, 2*(P.R0mm*rScale+8), 10);

  // continuous jet up to break-up
  const grad = ctx.createLinearGradient(0,0,0,H);
  grad.addColorStop(0, P.cyan); grad.addColorStop(0.6,'#1a9fc0');
  grad.addColorStop(1, P.violet);
  ctx.beginPath();
  const ybMax = zb_mm*pxPerMm;
  for(let y=0; y<=ybMax; y+=2){
    const zm = y/pxPerMm, zM = zm/1000.0;
    const r = Rz_mm(zm)*(1 + amp(zm)*Math.cos(k*(zM - P.v0*t)));
    ctx.lineTo(cx - r*rScale, y);
  }
  for(let y=ybMax; y>=0; y-=2){
    const zm = y/pxPerMm, zM = zm/1000.0;
    const r = Rz_mm(zm)*(1 + amp(zm)*Math.cos(k*(zM - P.v0*t)));
    ctx.lineTo(cx + r*rScale, y);
  }
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.shadowColor = P.cyan; ctx.shadowBlur = 16; ctx.fill(); ctx.shadowBlur = 0;

  // detached drops below break-up
  const lam_mm = P.lam_mm;
  const dropR = P.R0mm*1.55*rScale;
  const off = ((P.v0*t*1000.0) % lam_mm);
  for(let i=0; i<40; i++){
    const zd = zb_mm + off + i*lam_mm;
    if(zd > Zmax_mm) break;
    const y = zd*pxPerMm;
    ctx.beginPath();
    ctx.ellipse(cx, y, dropR*0.85, dropR, 0, 0, 2*Math.PI);
    ctx.fillStyle = P.violet; ctx.globalAlpha = 0.92;
    ctx.shadowColor = P.violet; ctx.shadowBlur = 12; ctx.fill();
    ctx.shadowBlur = 0; ctx.globalAlpha = 1.0;
  }

  // break-up marker
  ctx.strokeStyle = P.mute; ctx.setLineDash([4,5]); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, zb_mm*pxPerMm); ctx.lineTo(W, zb_mm*pxPerMm);
  ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = P.mute; ctx.font = "11px monospace";
  ctx.fillText("break-up  Lb = " + (P.Lb_mm/10).toFixed(0) + " cm", 8, zb_mm*pxPerMm - 6);
  ctx.fillText("lambda = " + P.lam_mm.toFixed(1) + " mm", 8, 24);

  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
</script>
"""

# ======================================================================
# PAGE : SIMULATOR
# ======================================================================
if page == "Simulator":
    st.markdown("<div class='hero-sub'>capillary oscillations // falling liquid jet</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='hero-title'>The pattern is real.</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-rule'></div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Wavelength lambda", f"{1e3*lam_rp:.1f} mm", f"kR = {xmax:.3f}")
    c2.metric("Growth rate s", f"{s_dim:.1f} /s", f"Lb = {1e2*Lb:.0f} cm")
    c3.metric("Weber", f"{We:.0f}", "convective" if We > 1 else "absolute")
    c4.metric("Ohnesorge", f"{Oh:.3f}", "viscous" if Oh > 1 else "inviscid")

    st.markdown(f"<span class='chip'>sigma_eff {1e3*sigma:.1f} mN/m</span>"
                f"<span class='chip'>Bo_E {Gamma_E:.2f}</span>"
                f"<span class='chip'>Bo_M {Gamma_M:.2f}</span>"
                f"<span class='chip'>Oh {Oh:.3f}</span>", unsafe_allow_html=True)

    # -------- live animated falling jet --------
    st.markdown("<div class='sec'>module 00</div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-title'>Live falling jet</div>", unsafe_allow_html=True)

    params = dict(R0mm=1e3*R0, lam_mm=1e3*lam_rp, s=s_dim, v0=v0, Lb_mm=1e3*Lb,
                  cyan=CYAN, violet=VIOLET, panel=PANEL, grid=GRID, mute=MUTE)
    anim = (_ANIM_TEMPLATE
            .replace("__PARAMS__", json.dumps(params))
            .replace("__PANEL__", PANEL).replace("__GRID__", GRID))
    components.html(anim, height=560)

    # -------- cross-section heatmaps --------
    tspan = np.linspace(0, 0.5, 600)
    z  = v0 * tspan + 0.5 * g * tspan**2
    Rz = R_of_t(tspan)
    theta = np.linspace(0, 2 * np.pi, 200)

    st.markdown("<div class='sec'>module 01</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='sec-title'>Cross-section radius R(theta, z) &mdash; "
                f"{regime.lower()} regime</div>", unsafe_allow_html=True)
    if not modes:
        st.info("Select at least one mode in the control deck.")
    else:
        cols = st.columns(len(modes))
        for col, n in zip(cols, modes):
            delta = solve_mode(n, tspan)
            surf = Rz[:, None] + delta[:, None] * np.cos(n * theta[None, :])
            fig = go.Figure(go.Heatmap(x=theta, y=z, z=surf * 1e3, colorscale=FUTURE_SCALE,
                                       colorbar=dict(title="R [mm]", thickness=10)))
            dark_layout(fig, 420, f"n = {n}")
            fig.update_xaxes(title="theta [rad]"); fig.update_yaxes(title="z [m]")
            col.plotly_chart(fig, width='stretch')
            if n >= 2:
                mu_c = np.sqrt(rho * sigma * R0 * (n**2 - 1) / n)
                state = "overdamped" if mu > mu_c else "underdamped"
                col.markdown(f"<span class='chip'>mu_c {mu_c:.2f} Pa s &rarr; {state}</span>",
                             unsafe_allow_html=True)

    # -------- growth-rate curve --------
    st.markdown("<div class='sec'>module 02</div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-title'>Rayleigh-Plateau growth rate</div>",
                unsafe_allow_html=True)
    xg = np.linspace(1e-3, 1.4, 400)
    figg = go.Figure()
    figg.add_trace(go.Scatter(x=xg, y=rp_growth(xg), mode="lines",
                              line=dict(color=CYAN, width=2.5),
                              fill="tozeroy", fillcolor="rgba(56,214,238,0.10)"))
    figg.add_vline(x=xmax, line_dash="dash", line_color=LIME,
                   annotation_text=f"kR = {xmax:.3f}", annotation_font_color=LIME)
    figg.add_vrect(x0=x_marginal(), x1=1.4, fillcolor=VIOLET, opacity=0.08, line_width=0,
                   annotation_text="stable", annotation_font_color=MUTE)
    dark_layout(figg, 380)
    figg.update_xaxes(title="kR"); figg.update_yaxes(title="growth rate / sqrt(sigma/rho R^3)")
    st.plotly_chart(figg, width='stretch')
    st.markdown(f"<span style='color:{MUTE}'>Detergent lowers sigma (shorter lambda). "
                "An axial electric field shrinks the unstable band (stabilising); a normal "
                "magnetic field on a ferrofluid widens it (destabilising). Adjust the sliders "
                "and watch kR and the jet respond.</span>", unsafe_allow_html=True)

# ======================================================================
# PAGE : PHYSICAL INSIGHTS
# ======================================================================
elif page == "Physical insights":
    st.markdown("<div class='hero-sub'>the physics</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-title'>Physical insights</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-rule'></div>", unsafe_allow_html=True)
    st.markdown(f"""
<div class='card'><h4>Why a cylinder breaks up</h4>
<p>Surface tension minimises area. A column is unstable to any wavelength longer than
its circumference (lambda &gt; 2 pi R, i.e. kR &lt; 1) before any dynamics. The fastest
mode sits at kR = 0.697, giving the universal spacing lambda = 9.01 R.</p></div>

<div class='card'><h4>Same agent, opposite effects</h4>
<p>Surface tension destabilises the axisymmetric n=0 mode at long wavelengths yet
stabilises every azimuthal n &ge; 2 mode. The curvature term (n^2 - 1)/R^2 flips sign
between n=0 and higher modes.</p></div>

<div class='card'><h4>Each mode is a damped oscillator</h4>
<p>From Navier-Stokes plus the free-surface boundary conditions, every mode obeys
delta'' + 2 gamma_n delta' + omega_n^2 delta = 0, with omega_n = sqrt(n(n^2-1) sigma /
rho R^3) and gamma_n = n mu / rho R^2. Above mu_c,n the oscillation becomes overdamped
and the cross-section simply relaxes to a circle.</p></div>

<div class='card'><h4>Detergent</h4>
<p>Surfactant lowers sigma, so lambda proportional to sigma^-1/2 shortens and the jet
breaks sooner in wavelength count (Lb proportional to sqrt(We) also drops). This is the
soap test: add detergent and the bead spacing visibly contracts.</p></div>

<div class='card'><h4>Electric field (electrohydrodynamics)</h4>
<p>An axial field on a conducting jet adds a Maxwell tension along z that opposes
axial necking. In the simplified model the bracket becomes (1 - x^2) - Bo_E x^2, so the
marginal wavenumber drops to kR_c = 1/sqrt(1 + Bo_E): the field <i>suppresses</i>
break-up and lengthens the jet. Push it far enough and you reach the electrospinning /
cone-jet regime.</p></div>

<div class='card'><h4>Magnetic field (ferrohydrodynamics)</h4>
<p>For a magnetisable (ferro) fluid a normal field adds a destabilising magnetic
Bond number: bracket becomes (1 - x^2)(1 + Bo_M) - Bo_E x^2, widening the unstable band
and speeding break-up - the same mechanism behind the Rosensweig spike instability.
Plain water is non-magnetic, so B has no effect there.</p></div>

<div class='card'><h4>Convective vs absolute</h4>
<p>By Briggs-Bers the transition is at We approximately 1. Kinay's pour has We approx
110, so the pattern is convective - swept downstream while growing - yet lambda = 9.01 R
is selected either way. The pattern is real regardless of regime.</p></div>
""", unsafe_allow_html=True)
    st.info("Full derivations, dispersion relations and references are in PHYSICS.md.")

# ======================================================================
# PAGE : APPLICATIONS
# ======================================================================
else:
    st.markdown("<div class='hero-sub'>where this shows up</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-title'>Applications</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-rule'></div>", unsafe_allow_html=True)
    st.markdown(f"""
<div class='card'><h4>Inkjet printing</h4>
<p>Driving the orifice at f_opt = 0.697 U0 / 2 pi R (about 39 Hz for a 4 mm jet, far
higher for print heads) locks the break-up into uniform drops at lambda = 9.01 R -
one controlled droplet per cycle.</p></div>

<div class='card'><h4>Electrospraying & electrospinning</h4>
<p>The electric-field term is the working principle of the Taylor cone: raise Bo_E and
the jet thins into fine fibres or a monodisperse charged spray - used for nanofibres,
mass spectrometry and drug delivery.</p></div>

<div class='card'><h4>Ferrofluid actuation</h4>
<p>The magnetic term maps onto ferrofluid devices: field-tunable valves, seals and
the Rosensweig instability, where a normal field breaks a smooth surface into a lattice
of spikes.</p></div>

<div class='card'><h4>Spray, combustion & cooling</h4>
<p>Fuel injectors, agricultural nozzles and spray cooling all live or die by drop size;
d approx 3.78 R and Lb approx 2.9 R sqrt(We) ln(R/eta0) set the atomisation.</p></div>

<div class='card'><h4>Detergent & coating</h4>
<p>Surfactants tune sigma to control break-up length in curtain coating, fibre spinning
and microfluidic droplet generators.</p></div>

<div class='card'><h4>Measure it yourself</h4>
<p>Three ruler-and-phone tests: node spacing approx 9R; double the pour height and
lambda doubles (lambda proportional to v0); add soap and lambda shortens
(lambda proportional to sigma^-1/2).</p></div>
""", unsafe_allow_html=True)
