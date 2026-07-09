import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.special import iv
import math

# --- PAGE CONFIGURATION & PROFESSIONAL DARK THEME ---
st.set_page_config(layout="wide", page_title="Liquid Jet Instability Analysis")

# Custom CSS for Professional Dark Mode, Inter, Montserrat, and JetBrains Mono
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Montserrat:wght@600;700&family=JetBrains+Mono:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background-color: #0E1117;
        color: #E2E8F0;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Montserrat', sans-serif !important;
        color: #FFFFFF !important;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    /* Highlight color for the slider bar */
    .stSlider > div > div > div > div { background-color: #3B82F6; }
    
    /* WIDGET TEXT COLOR OVERRIDES */
    label, label p, label div, .stRadio p, .stCheckbox p, .stSlider p, .stSelectbox p {
        color: #F8FAFC !important;
        font-weight: 500 !important;
    }
    
    /* Slider numbers */
    div[data-testid="stThumbValue"] {
        color: #10B981 !important; 
        font-weight: 700 !important;
    }
    div[data-testid="stTickBarMin"], div[data-testid="stTickBarMax"] {
        color: #94A3B8 !important; 
    }
    
    /* Selectbox dropdown text */
    div[data-baseweb="select"] span, div[data-baseweb="select"] div {
        color: #FFFFFF !important;
    }
    
    div[data-testid="stMetricValue"] { color: #38BDF8; font-weight: 600; }
    
    /* Aggressively target all inner text elements of the metric label to force white */
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricLabel"] div { 
        color: #FFFFFF !important; 
        font-size: 1.1rem !important; 
        font-weight: 500 !important; 
    }
    
    .css-1d391kg { background-color: #1E293B; border-radius: 8px; padding: 20px; border: 1px solid #334155; }
    hr { border-color: #334155; }
    
    button[data-baseweb="tab"] {
        background-color: transparent;
        color: #94A3B8;
        font-family: 'Montserrat', sans-serif;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #38BDF8;
        border-bottom: 2px solid #38BDF8;
    }

    /* Custom class for the stability notes */
    .stability-note {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #94A3B8;
        background-color: #161B22;
        padding: 15px;
        border-left: 3px solid #10B981;
        border-radius: 4px;
        margin-top: 15px;
    }
    .stability-note b {
        color: #E2E8F0;
    }
    </style>
""", unsafe_allow_html=True)

# --- PHYSICS ENGINE & CONSTANTS ---
FLUIDS = {
    "Water": [1000, 0.072, 0.001, 5.0],
    "Glycerol": [1260, 0.064, 1.412, 0.01],
    "Ethanol": [789, 0.022, 0.0012, 0.1],
    "Mercury": [13500, 0.485, 0.0015, 1e6]
}

def calculate_physics(fluid, H_cm, R_mm, eps_pct, field_type, field_str):
    rho, sigma, mu, cond = FLUIDS[fluid]
    
    H = H_cm / 100.0
    R = R_mm / 1000.0
    eps = eps_pct / 100.0
    
    U0 = math.sqrt(2 * 9.81 * H)
    
    We = (rho * U0**2 * R) / sigma
    Oh = mu / math.sqrt(rho * sigma * R)
    
    lambda_max_m = 9.01 * R
    s_max_base = 0.343 * math.sqrt(sigma / (rho * R**3))
    
    s_max = s_max_base
    if field_type == "Electric (Radial)":
        e_effect = 0.02 * s_max_base * (field_str / 20.0)**2
        s_max = s_max_base + e_effect
    elif field_type == "Magnetic (Axial)":
        gamma_mhd = 1.5 * s_max_base * (field_str / 5.0)**2 * (cond / 5.0) 
        s_max = math.sqrt(s_max_base**2 + (gamma_mhd/2)**2) - gamma_mhd/2
    
    s_max = max(s_max, 1e-5) 
    
    freq_opt = U0 / lambda_max_m
    L_break = (U0 / s_max) * math.log(1 / max(eps, 1e-4)) 
    D_drop = 3.78 * R
    
    return {
        "rho": rho, "sigma": sigma, "mu": mu, "U0": U0, "R": R,
        "We": We, "Oh": Oh, "lambda_max": lambda_max_m, 
        "s_max": s_max, "freq_opt": freq_opt,
        "L_break": L_break, 
        "D_drop": D_drop
    }

def dispersion_relation(sigma, rho, R, s_max_actual, s_max_base):
    kR = np.linspace(0.01, 1.3, 100)
    s_vals = []
    
    scaling_factor = s_max_actual / s_max_base if s_max_base > 0 else 1.0

    for x in kR:
        if x < 1:
            ratio = iv(1, x) / iv(0, x)
            val = math.sqrt(x * ratio * (1 - x**2)) * scaling_factor
            s_vals.append(val)
        else:
            s_vals.append(0.0)
    return kR, np.array(s_vals)

# --- UI LAYOUT ---
st.title("Electrodynamic Liquid Jet Instability Analysis")
st.markdown("Computational simulation of capillary pinch-off, modal superposition, and electromagnetic fluid coupling.")

metric_row = st.columns(4)
st.markdown("---")

col_params, col_jet, col_cross, col_plots = st.columns([1, 1.2, 1.2, 1.2])

# --- COLUMN 1: PARAMETER CONTROLS ---
with col_params:
    st.subheader("System Parameters")
    
    st.markdown("**Fluid Properties**")
    fluid = st.radio("Material", ["Water", "Glycerol", "Ethanol", "Mercury"], label_visibility="collapsed")
    
    st.markdown("**Geometric Setup**")
    H_cm = st.slider("Hydraulic Head H (cm)", 1.0, 30.0, 10.0)
    R_mm = st.slider("Orifice Radius R (mm)", 1.0, 10.0, 4.0)
    eps_pct = st.slider("Initial Perturbation ε (%)", 0.1, 10.0, 2.0)
    
    st.markdown("**Azimuthal Mode (n)**")
    mode = st.radio("Mode", [0, 2, 3, 4], horizontal=True, label_visibility="collapsed")
    
    st.markdown("**External Fields**")
    field_type = st.selectbox("Apply Field", ["None", "Electric (Radial)", "Magnetic (Axial)"], label_visibility="collapsed")
    
    field_str = 0.0
    if field_type == "Electric (Radial)":
        field_str = st.slider("E-Field Strength (kV/m)", 0.0, 100.0, 50.0)
    elif field_type == "Magnetic (Axial)":
        field_str = st.slider("B-Field Strength (Tesla)", 0.0, 10.0, 2.0)

# Compute Physics
p = calculate_physics(fluid, H_cm, R_mm, eps_pct, field_type, field_str)

# --- UPDATE TOP METRICS ---
metric_row[0].metric("Dominant Wavelength (λ)", f"{p['lambda_max']*1000:.1f} mm")
metric_row[1].metric("Maximum Growth Rate (s)", f"{p['s_max']:.2f} s⁻¹")
metric_row[2].metric("Breakup Length (L)", f"{p['L_break']*100:.1f} cm")
metric_row[3].metric("Weber Number (We)", f"{int(p['We'])}")

# --- COLUMN 2: DYNAMIC JET PROFILE ---
with col_jet:
    st.subheader("Longitudinal Profile")
    
    z_max = p['L_break'] * 1.1
    z = np.linspace(0, z_max, 500)
    
    growth_factor = np.exp(p['s_max'] * z / p['U0'])
    wave_component = np.cos(2 * np.pi * z / p['lambda_max'])
    
    r_z = 1.0 - (eps_pct/100.0) * growth_factor * wave_component
    r_z = np.clip(r_z, 0, 2.5)
    
    fig_jet = go.Figure()
    
    fig_jet.add_trace(go.Scatter(
        x=r_z, y=-z, 
        fill='tonexty', 
        mode='lines', 
        line=dict(color='#38BDF8', width=2.5),
        fillcolor='rgba(56, 189, 248, 0.15)'
    ))
    fig_jet.add_trace(go.Scatter(
        x=-r_z, y=-z, 
        fill='tonexty', 
        mode='lines', 
        line=dict(color='#38BDF8', width=2.5),
        fillcolor='rgba(56, 189, 248, 0.15)'
    ))
    
    fig_jet.update_layout(
        template="plotly_dark",
        xaxis=dict(visible=False, range=[-2.5, 2.5]), 
        yaxis=dict(visible=False), 
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
        margin=dict(l=0,r=0,t=0,b=0), height=500, showlegend=False
    )
    # Updated to width="stretch" to fix Streamlit deprecation warning
    st.plotly_chart(fig_jet, width="stretch")

# --- COLUMN 3: DYNAMIC CROSS SECTIONS & STABILITY NOTES ---
with col_cross:
    st.subheader("Azimuthal Cross-Sections")
    theta = np.linspace(0, 2*np.pi, 100)
    fig_cross = go.Figure()
    
    z_slices = [p['L_break'] * 0.3, p['L_break'] * 0.6, p['L_break'] * 0.9]
    
    for i, z_val in enumerate(z_slices):
        dynamic_amp = (eps_pct/100.0) * math.exp(p['s_max'] * z_val / p['U0'])
        dynamic_amp = min(dynamic_amp, 0.95)
        
        if mode == 0:
            r = 1.0 - dynamic_amp * 0.2 
        else:
            r = 1.0 + dynamic_amp * np.cos(mode * theta)
            
        x = r * np.cos(theta)
        y = r * np.sin(theta) + (3 - i) * 3 
        
        fig_cross.add_trace(go.Scatter(
            x=x, y=y, 
            fill='toself', 
            mode='lines', 
            line=dict(color='#818CF8', width=2),
            fillcolor='rgba(129, 140, 248, 0.4)'
        ))
        fig_cross.add_annotation(
            x=0, y=(3-i)*3 - 1.5, 
            text=f"z = {z_val*100:.1f} cm", 
            showarrow=False, font=dict(color="#94A3B8")
        )

    fig_cross.update_layout(
        template="plotly_dark",
        xaxis=dict(visible=False, range=[-2, 2]), 
        yaxis=dict(visible=False), 
        showlegend=False, 
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
        margin=dict(l=0,r=0,t=0,b=0), height=380
    )
    # Updated to width="stretch" to fix Streamlit deprecation warning
    st.plotly_chart(fig_cross, width="stretch")

    # Dynamic Stability Note
    stability_html = f"<div class='stability-note'><b>Mode n={mode} Assessment:</b><br>"
    if mode == 0:
        stability_html += "Unstable (for kR < 1). Surface tension destabilizes long wavelengths, driving pinch-off."
    elif mode == 2:
        stability_html += "Strictly Stable. Surface tension acts as a restoring force, creating capillary oscillations."
    elif mode == 3:
        stability_html += "Strictly Stable. Higher-order restoring forces dominate."
    elif mode == 4:
        stability_html += "Strictly Stable. Rapidly damped harmonic oscillations."
    stability_html += "</div>"
    st.markdown(stability_html, unsafe_allow_html=True)

# --- COLUMN 4: DISPERSION & ANALYTICS ---
with col_plots:
    st.subheader("Dispersion Relation")
    s_max_base = 0.343 * math.sqrt(p['sigma'] / (p['rho'] * p['R']**3))
    kR, s_vals = dispersion_relation(p['sigma'], p['rho'], p['R'], p['s_max'], s_max_base)
    
    fig_disp = go.Figure()
    fig_disp.add_trace(go.Scatter(
        x=kR, y=s_vals, 
        fill='tozeroy', 
        mode='lines', 
        line=dict(color='#10B981', width=2.5),
        fillcolor='rgba(16, 185, 129, 0.2)'
    ))
    fig_disp.update_layout(
        template="plotly_dark",
        xaxis_title="Dimensionless Wavenumber (kR)", 
        yaxis_title="Growth Rate (s)", 
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0,r=0,t=0,b=0), height=250
    )
    # Updated to width="stretch" to fix Streamlit deprecation warning
    st.plotly_chart(fig_disp, width="stretch")
    
    st.subheader("Analytical Predictions")
    st.markdown(f"""
    <div style="background-color: #1E293B; padding: 15px; border-radius: 8px; border: 1px solid #334155;">
    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
        <span style="color: #94A3B8;">Droplet Diameter:</span> 
        <span style="color: #E2E8F0; font-weight: 500;">{p['D_drop']*1000:.2f} mm</span>
    </div>
    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
        <span style="color: #94A3B8;">Optimal Forcing:</span> 
        <span style="color: #E2E8F0; font-weight: 500;">{p['freq_opt']:.2f} Hz</span>
    </div>
    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
        <span style="color: #94A3B8;">λ / R Ratio:</span> 
        <span style="color: #E2E8F0; font-weight: 500;">{(p['lambda_max']/p['R']):.2f}</span>
    </div>
    <div style="display: flex; justify-content: space-between;">
        <span style="color: #94A3B8;">Ohnesorge (Oh):</span> 
        <span style="color: #E2E8F0; font-weight: 500;">{p['Oh']:.4f}</span>
    </div>
    </div>
    """, unsafe_allow_html=True)

# --- COMPREHENSIVE THEORETICAL FRAMEWORK ---
st.markdown("---")
st.header("Theoretical Framework and Governing Equations")

tab1, tab2, tab3, tab4 = st.tabs([
    "Fluid Dynamics & Navier-Stokes", 
    "Rayleigh-Plateau Dispersion", 
    "Azimuthal Oscillations", 
    "Electromagnetic Coupling"
])

# Fixed SyntaxWarnings by making all markdown blocks containing LaTeX raw strings (r"...")
with tab1:
    st.markdown(r"""
    ### The Convective Instability Regime
    The base state of the falling liquid jet is governed by the incompressible Navier-Stokes and continuity equations:
    """)
    st.latex(r"\rho \left( \frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} \right) = -\nabla p + \mu \nabla^2 \mathbf{u} + \rho \mathbf{g}")
    st.latex(r"\nabla \cdot \mathbf{u} = 0")
    st.markdown(r"""
    The flow regime is defined by the **Weber number** ($We$), which dictates the ratio of inertial forces to surface tension. For a jet poured from a standard height, $We \gg 1$. 
    """)
    st.latex(r"We = \frac{\rho U_0^2 R}{\sigma}")
    st.markdown(r"""
    Because $We > 1$, the instability is **convective**. The perturbations do not grow in place; they grow spatially as they are swept downstream at velocity $U_0$. The spatial evolution of the perturbation amplitude $\delta(z)$ follows:
    """)
    st.latex(r"\delta(z) = \epsilon R \exp\left( \frac{s_{max} z}{U_0} \right)")

with tab2:
    st.markdown(r"""
    ### Rayleigh-Plateau Dispersion Relation
    A cylindrical column of liquid is geometrically unstable because a cylinder possesses more surface area than a sphere of the same volume. The surface tension seeks to minimize this area, driving fluid from the "necks" into the "bulges".
    
    By applying the kinematic and dynamic boundary conditions at the free surface, the growth rate $s$ for the axisymmetric ($n=0$) mode as a function of the dimensionless wavenumber $kR$ is given by the dispersion relation:
    """)
    st.latex(r"s^2 = \frac{\sigma}{\rho R^3} (kR) \frac{I_1(kR)}{I_0(kR)} (1 - (kR)^2)")
    st.markdown(r"""
    Where $I_0$ and $I_1$ are modified Bessel functions of the first kind. The jet is only unstable when $kR < 1$. To find the wavelength that will naturally dominate the physical system, we maximize this growth rate ($\partial s / \partial (kR) = 0$), yielding the universal optimum:
    """)
    st.latex(r"k_{max}R \approx 0.697 \quad \implies \quad \lambda_{max} = \frac{2\pi R}{0.697} \approx 9.01 R")
    
    st.markdown(r"""
    ### Viscous Damping (Ohnesorge Number)
    Viscosity acts as a damping agent, analyzed via the Ohnesorge number ($Oh = \mu / \sqrt{\rho \sigma R}$). While viscosity drastically reduces the maximum growth rate $s_{max}$, it does *not* stabilize the $kR < 1$ regime; the jet will always eventually break.
    """)

with tab3:
    st.markdown(r"""
    ### Azimuthal Modes and Exotic Cross-Sections
    If the orifice is not perfectly circular (e.g., pouring over the lip of a glass), it introduces non-axisymmetric perturbations modeled by the azimuthal integer $n$:
    """)
    st.latex(r"r(\theta, z) = R_0 + \delta(z) \cos(n\theta)")
    st.markdown(r"""
    Linearizing the Navier-Stokes equations radially at the boundary yields a classical Damped Harmonic Oscillator equation for these $n \ge 2$ modes:
    """)
    st.latex(r"\ddot{\delta} + 2\gamma_n \dot{\delta} + \omega_n^2 \delta = 0")
    st.markdown(r"""
    Where the restoring frequency and viscous damping coefficients are:
    """)
    st.latex(r"\omega_n = \sqrt{\frac{n(n^2-1)\sigma}{\rho R_0^3}}, \quad \gamma_n = \frac{n\mu}{\rho R_0^2}")
    st.markdown(r"""
    Because water is highly underdamped ($\gamma_n \ll \omega_n$), the $n=2$ (elliptical) mode will stably oscillate, swapping its major and minor axes as it falls, producing the distinct "nodes and antinodes" often observed in pouring water.
    """)

with tab4:
    st.markdown(r"""
    ### Electrohydrodynamics (Radial Electric Field)
    Applying a radial electric field induces surface charges on a dielectric fluid. The interaction generates an outward **Maxwell Stress**, modifying the dynamic boundary condition at the interface:
    """)
    st.latex(r"P_{E} = \frac{1}{2} \epsilon_0 (\epsilon_r - 1) E_n^2")
    st.latex(r"\Delta P_{interface} = \sigma \kappa - P_{E}")
    st.markdown(r"""
    This electrostatic pressure opposes surface tension, physically pulling the fluid outward. This is strictly **destabilizing**, accelerating the pinch-off process and reducing the breakup length. If the electric field is strong enough, it triggers the whipping instability utilized in electrospinning.
    
    ### Magnetohydrodynamics (Axial Magnetic Field)
    When a conductive fluid (e.g., Mercury) falls through an axial magnetic field $\mathbf{B} = B_0 \hat{z}$, any radial deformation forces the fluid across the magnetic field lines. This induces an azimuthal eddy current $\mathbf{J}$:
    """)
    st.latex(r"\mathbf{J} = \sigma_e (\mathbf{u} \times \mathbf{B})")
    st.markdown(r"""
    This current interacts with the magnetic field to produce a volumetric **Lorentz Force** that opposes the motion that created it (Lenz's Law):
    """)
    st.latex(r"\mathbf{F}_L = \mathbf{J} \times \mathbf{B}")
    st.markdown(r"""
    This force acts as a massive damping term ($\gamma_{MHD} \propto \sigma_e B^2 / \rho$), heavily stabilizing the jet, reducing the growth rate, and stretching the droplets into long, spindle-like structures before pinch-off.
    """)