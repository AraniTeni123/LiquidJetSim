# LiquidJetSim

**Capillary oscillations in a falling liquid jet** — an interactive simulator of the
Rayleigh–Plateau instability and the higher azimuthal capillary modes that shape a
falling stream of water.

Team **Samudramanthan** · Ensemble Young Physicists'
Tournament 2026.

**Live app:** https://liquidjetsim-skknpbrhdpstmjxh4viqel.streamlit.app/

---

## The problem

Pour water from a height of ~10 cm and you see periodic swellings and necks running
down the jet. Are they real, or an optical illusion? They are **real**: the pattern
is the linearly selected dominant mode of the **Rayleigh–Plateau instability**, with
the universal wavelength

$$\lambda = 9.01\,R.$$

A non-circular (glass-lip) orifice additionally excites the azimuthal modes
$n = 2, 3, 4$, producing the morphing cross-sections (heart, bowtie, jagged) the app
can reproduce. The full theory — governing equations, boundary conditions, every
derivation — is in **[PHYSICS.md](PHYSICS.md)**.

---

## What's in this repo

| File | Description |
|------|-------------|
| [app.py](app.py) | Streamlit web app: interactive heatmaps of $R(\theta,z)$ and the Rayleigh–Plateau growth-rate curve. |
| [simulate_jet.m](simulate_jet.m) | Original MATLAB simulation (`ode45`) that produced the slide figures. |
| [PHYSICS.md](PHYSICS.md) | Complete physics with derivations (NS to 1D jet, DHO modes, dispersion relations, breakup scaling, exotic patterns). |
| [requirements.txt](requirements.txt) | Python dependencies for the Streamlit app. |

---

## Run the Streamlit app

```bash
git clone https://github.com/AraniTeni123/LiquidJetSim.git
cd LiquidJetSim
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501. Use the sidebar to change $\rho$, $\sigma$, $R_0$,
$v_0$, the viscosity regime (under/over-damped), and which modes to display.

## Run the MATLAB simulation

```matlab
simulate_jet                 % default: underdamped (Oh = 0.059)
simulate_jet('overdamped')   % Oh = 2.36, all azimuthal modes overdamped
```

Requires MATLAB with `ode45` (base install). Outputs the $R(\theta,z)$ heatmaps for
modes $n = 0,2,3,4$ and prints the $n=2$ wavelength check and the critical
viscosities.

---

## The model in one paragraph

Each mode reduces to a **damped harmonic oscillator** derived from incompressible
Navier–Stokes plus the kinematic and dynamic free-surface boundary conditions:

$$\ddot\delta + 2\gamma_n\,\dot\delta + \omega_n^2\,\delta = 0,\qquad
\omega_n=\sqrt{\frac{n(n^2-1)\sigma}{\rho R^3}},\qquad
\gamma_n=\frac{n\mu}{\rho R^2}.$$

The axisymmetric mode $n=0$ has negative stiffness (Rayleigh–Plateau **growth**,
$s_0 = 0.343\sqrt{\sigma/\rho R^3}$, Trouton damping $3\mu$); the azimuthal modes
$n\ge 2$ are **stable** oscillations. Gravity thinning
$R(t)=R_0\sqrt{v_0/(v_0+gt)}$ makes the coefficients time-dependent, and the
space–time map $z(t)=v_0 t+\tfrac12 g t^2$ turns amplitude-vs-time into the observed
cross-section-vs-height.

## Key numbers (validated)

| Quantity | Formula | Value |
|----------|---------|-------|
| Selected wavelength | $9.01R$ | 36 mm ($R=4$ mm) |
| Fastest-growing mode | $kR = 0.697$ | root of Bessel eqn |
| Max growth rate | $0.343\sqrt{\sigma/\rho R^3}$ | 12 s⁻¹ |
| $n=2$ wavelength (sim) | $2\pi v_0/\omega_2$ | 0.175 m vs 0.180 theory (**2.8%**) |
| Critical Ohnesorge | $\sqrt{(n^2-1)/n}$ | 1.22, 1.63, 1.94 ($n=2,3,4$) |
| Kinay's Weber number | $\rho v_0^2 R/\sigma$ | ≈110 (convective) |

## Physics regimes

- **Water** ($\mathrm{Oh}\approx 0.002$): inviscid Rayleigh theory is essentially
  exact; the jet always breaks into drops.
- **Convective vs absolute** (Briggs–Bers): Kinay's pour has $\mathrm{We}\approx 110$
  → convective, so the pattern is swept downstream while growing. $\lambda=9.01R$ is
  selected in **both** regimes.
- **Viscous fluids** ($\mathrm{Oh}>1$): azimuthal modes become overdamped (no
  oscillation), but $n=0$ still grows — viscosity delays but never prevents breakup.

---

## References

1. Lord Rayleigh, *On the instability of jets*, Proc. London Math. Soc. **s1-10**(1),
   4–13 (1879).
2. S. Chandrasekhar, *Hydrodynamic and Hydromagnetic Stability*, Oxford (1961).
3. P. Huerre & P. A. Monkewitz, Annu. Rev. Fluid Mech. **22**, 473–537 (1990).
4. J. Eggers, Rev. Mod. Phys. **69**(3), 865 (1997).
