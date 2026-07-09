# Physics of Capillary Oscillations in a Falling Liquid Jet

**Team Samudramanthan — Ensemble Young Physicists' Tournament 2026**

This document contains the complete theoretical development behind `LiquidJetSim`:
the governing equations, the reduction to a 1D slender-jet model, and the full
derivation of every result used in the simulation and the Streamlit app. Equations
render on GitHub.

---

## 1. Dimensionless Numbers and Regime

For a jet of radius $R_0$, speed $v_0$, density $\rho$, dynamic viscosity $\mu$
and surface tension $\sigma$:

| Number | Definition | Meaning |
|--------|------------|---------|
| Reynolds | $\mathrm{Re}=\rho v_0 R_0/\mu$ | inertia / viscosity |
| Weber | $\mathrm{We}=\rho v_0^2 R_0/\sigma$ | inertia / capillarity |
| Ohnesorge | $\mathrm{Oh}=\mu/\sqrt{\rho\sigma R_0}$ | viscous damping |

**Kinay's pour.** Free fall from $H$ gives $U_0=\sqrt{2gH}\approx 1.4\ \mathrm{m/s}$
for $H=10\ \mathrm{cm}$. With $Q\sim 80\ \mathrm{ml/s}$, mass conservation
$Q=\pi R^2 U_0$ gives $R=\sqrt{Q/\pi U_0}\approx 4\ \mathrm{mm}$, so

$$\mathrm{We}=\frac{\rho U_0^2 R}{\sigma}
=\frac{1000\times 1.96\times 0.004}{0.072}\approx 110 \gg 1.$$

Because $\mathrm{We}\gg 1$ the jet is **convectively unstable**. Absolute
instability requires $\mathrm{We}<1$ ($H<0.5\ \mathrm{mm}$: a drip). The selected
wavelength is $9.01R$ in both regimes, so the pattern is real either way. For water
$\mathrm{Oh}\approx 0.002 \ll 1$: inviscid Rayleigh theory is accurate.

---

## 2. Geometry Before Dynamics — Why a Cylinder Is Unstable

Surface tension minimizes area. A cylinder of radius $R$ and length $\lambda$ has a
larger surface than a sphere of equal volume whenever $\lambda>2\pi R$. So a liquid
column is unstable to breakup for any wavelength longer than its circumference,
before any dynamics. The stability boundary $\lambda=2\pi R$ ($kR=1$) is purely
geometric.

The mean curvature of $r_s=R_0+\hat\eta\cos(kz)\cos(n\theta)$ is

$$\delta\kappa=\hat\eta\left(k^2+\frac{n^2-1}{R_0^2}\right).$$

| Mode $n$ | Curvature factor | Unstable when | Result |
|:--:|:--:|:--:|:--|
| $0$ | $k^2-1/R_0^2$ | $kR_0<1$ | Unstable (Rayleigh–Plateau) |
| $1$ | $k^2$ | never | Neutral (translation) |
| $\ge 2$ | $k^2+(n^2-1)/R_0^2$ | never | Stable oscillation |

By Young–Laplace $\delta p=\sigma\,\delta\kappa$. For $n=0$, $kR_0<1$ the pressure at
a bulge is lower, so fluid is sucked in and the bulge grows. For $n\ge 2$ the
azimuthal term always restores. The same agent — surface tension — destabilises long
axisymmetric waves and stabilises higher azimuthal modes.

---

## 3. Governing Equations: Navier–Stokes → 1D Slender Jet

$$\rho\left(\partial_t \mathbf{u}+(\mathbf{u}\cdot\nabla)\mathbf{u}\right)
=-\nabla p+\mu\nabla^2\mathbf{u}+\rho g\,\hat z,\qquad \nabla\cdot\mathbf{u}=0.$$

With aspect ratio $\varepsilon=R_0/L\ll 1$ ($\lambda=9.01R\Rightarrow R/L\approx
0.11$), area-averaging over $A=\pi R^2$ gives

$$\partial_t A+\partial_z(Au_z)=0,\qquad
\rho(\partial_t u_z+u_z\partial_z u_z)=\rho g-\partial_z p+3\mu\,\partial_{zz}u_z.$$

The factor 3 is the **Trouton ratio** for extensional flow: $\tau_{zz}=2\mu\dot
\varepsilon$ and $\tau_{rr}=-\mu\dot\varepsilon$ both contribute axially.

Two families: $n=0$ (axisymmetric) → Rayleigh–Plateau, grows for $kR_0<1$; $n\ge 2$
(azimuthal) → stable oscillations, need a non-circular orifice. A glass lip makes an
effectively elliptical orifice, seeding both $n=0$ and $n=2$ simultaneously.

---

## 4. Free-Surface Boundary Conditions

Linearised onto $r=R_0$.

**Kinematic (inertia):** $u_r|_{R_0}=\partial_t r_s=\dot\delta\cos(n\theta)$. With
potential $\phi=C(t)r^n\cos(n\theta)$, $C(t)=\dot\delta/(nR_0^{n-1})$.

**Normal dynamic (restoring):** $[-p+\tau_{rr}]_{R_0}=\sigma\kappa$, perturbation
$p'|_{R_0}-\tau_{rr}|_{R_0}=\sigma\,\delta\kappa$.

**Tangential dynamic (damping):** $\tau_{r\theta}|_{R_0}=0$ gives
$\gamma_n=n\mu/\rho R_0^2$.

---

## 5. Curvature → Pressure → ODE (azimuthal $n\ge 2$)

Curvature $\kappa\approx\frac{1}{R_0}\left(1+\frac{(n^2-1)\delta\cos(n\theta)}{R_0}
\right)$. Interior pressure $p'=P_n(t)r^n\cos(n\theta)$ ($\nabla^2p'=0$), so
$\partial_r p'|_{R_0}=\frac{n(n^2-1)\sigma}{R_0^3}\delta\cos(n\theta)$. Radial
momentum at $r=R_0$:

$$\rho\ddot\delta=-\frac{n(n^2-1)\sigma}{R_0^3}\delta-\frac{2n\mu}{R_0^2}\dot\delta,$$

which is the damped harmonic oscillator

$$\ddot\delta+2\gamma_n\dot\delta+\omega_n^2\delta=0,\qquad
\omega_n=\sqrt{\frac{n(n^2-1)\sigma}{\rho R_0^3}},\quad
\gamma_n=\frac{n\mu}{\rho R_0^2}.$$

Underdamped/overdamped threshold at
$\mu_{c,n}=\sqrt{\rho\sigma R_0(n^2-1)/n}$, i.e. $\mathrm{Oh}_{c,n}=\sqrt{(n^2-1)/n}
=1.22,1.63,1.94$ for $n=2,3,4$.

---

## 6. Where $\lambda=9.01R$ Comes From ($n=0$)

Rayleigh's inviscid dispersion relation:

$$\Omega^2=\frac{\sigma}{\rho}k\,\frac{I_1(kR)}{I_0(kR)}\left(\frac{1}{R^2}-k^2\right).$$

Unstable iff $kR<1$. Growth rate $s(x)=\sqrt{\frac{\sigma}{\rho R^3}}\sqrt{x
\frac{I_1(x)}{I_0(x)}(1-x^2)}$, $x=kR$. Maximising with the Bessel recurrences gives

$$x_\text{max}=kR=0.697\;\Rightarrow\;\lambda_\text{max}=\frac{2\pi R}{0.697}=9.01R,
\qquad s_\text{max}\approx 0.343\sqrt{\frac{\sigma}{\rho R^3}}.$$

As $kR\to0$, $I_1/I_0\to0$ (no driving pressure); as $kR\to1$, $(1-k^2R^2)\to0$
(marginal). The peak at 0.697 is the optimal compromise. For water $R=4\ \mathrm{mm}$:
$\lambda\approx36\ \mathrm{mm}$, $s_\text{max}\approx12\ \mathrm{s^{-1}}$.

---

## 7. Effect of Viscosity — Chandrasekhar (1961)

The viscous dispersion relation leaves the boundary $kR=1$ unchanged for all Oh,
shifts the growth peak to longer $\lambda$, and reduces the growth rate everywhere.
Viscosity always damps, never destabilises. For water $\mathrm{Oh}\approx0.002$, so
inviscid theory is essentially exact.

---

## 8. Breakup Length and Drop Size

$$L_b\approx 2.9\,R\sqrt{\mathrm{We}}\,\ln\frac{R}{\eta_0}.$$

$L_b\propto v_0$: faster pours break further downstream but always break. Drop
diameter $d\approx3.78R$. Clean jet ($\eta_0/R\sim10^{-3}$) gives $L_b\approx50\
\mathrm{cm}$; the glass lip forces $\eta_0/R\sim0.1$, giving $L_b\approx12\
\mathrm{cm}$, matching observation.

---

## 9. Absolute vs Convective Instability — Briggs–Bers

Find the saddle $k^*$ where $d\omega/dk=0$ and evaluate $\omega_0=\omega(k^*)$:
$\mathrm{Im}(\omega_0)>0$ → absolute (grows in place); $<0$ → convective (swept
downstream). Transition at $\mathrm{We}\approx1$. Kinay's pour $\mathrm{We}\approx
110$ → convective, but $\lambda=9.01R$ is selected in both regimes.

---

## 10. Simulation Model (what the code integrates)

$$\ddot\delta+2\gamma_n(t)\dot\delta+\omega_n^2(t)\delta=0\ (n\ge2),\qquad
\ddot\delta+2\gamma_0(t)\dot\delta-s_0^2(t)\delta=0\ (n=0),$$

with gravity thinning $R(t)=R_0\sqrt{v_0/(v_0+gt)}$, $\omega_n(t)=\sqrt{n(n^2-1)
\sigma/\rho R^3}$, $\gamma_n(t)=n\mu/\rho R^2$, $\gamma_0=3\mu/\rho R^2$ (Trouton),
$s_0=0.343\sqrt{\sigma/\rho R^3}$. Space–time map $z(t)=v_0t+\tfrac12gt^2$; surface
$R(\theta,z)=R_z(z)+\delta(z)\cos(n\theta)$. Solver: MATLAB `ode45`, RelTol $10^{-3}$,
AbsTol $10^{-6}$, $t\in[0,0.5]\,\mathrm{s}$, 600 points, $\delta_0=1.5\ \mathrm{mm}$.

**Validation.** For $n=2$, $R_0=10\ \mathrm{mm}$, $v_0=0.6\ \mathrm{m/s}$:
$\omega_2=\sqrt{6\sigma/\rho R_0^3}=20.8\ \mathrm{rad/s}$, $\lambda_2=2\pi v_0/\omega_2
\approx0.18\ \mathrm{m}$; simulation gives $0.175\ \mathrm{m}$ — within 2.8%. The
overdamped run confirms $\mathrm{Oh}_{c,2},\mathrm{Oh}_{c,3},\mathrm{Oh}_{c,4}$.

---

## 11. Exotic Patterns

**Superposition:** $R_\text{tot}=R_\text{mean}+\delta_1\cos(n_1\theta)+\delta_2
\cos(n_2\theta+\varphi)$; modes drift in/out of phase → heart (2,3), bowtie (2,4),
jagged (3,4). **Helical swirl:** angular-momentum spin-up $\Omega\propto(R_0/R_z)^2$
corkscrews the pattern. **Orifice forcing (inkjet):** resonance at $\omega_d=\omega_n$,
optimal $f_\text{opt}=0.697U_0/2\pi R\approx39\ \mathrm{Hz}$. **Parametric (Mathieu):**
$g(t)=g_0+A_g\sin(\omega_g t)$, principal tongue $\omega_g=2\omega_n$ (Savart 1833).

---

## 12. Complete Mode Picture

| Mode | Nature | Wavelength |
|:--|:--|:--|
| $n=0,\ kR<1$ | unstable (Rayleigh–Plateau) | $9.01R$ |
| $n=0,\ kR>1$ | stable | — |
| $n=1$ | neutral | — |
| $n=2$ | stable oscillation | $2\pi v/\sqrt{6\sigma/\rho R^3}$ |
| $n=3$ | stable oscillation | $2\pi v/\sqrt{24\sigma/\rho R^3}$ |
| $n=4$ | stable oscillation | $2\pi v/\sqrt{60\sigma/\rho R^3}$ |

Three falsifiable predictions: node spacing $\approx9R$; double pour height →
double $\lambda$ ($\lambda\propto v_0$); add soap → shorter $\lambda$
($\lambda\propto\sigma^{-1/2}$). Butta is right only for $\mathrm{Oh}>1$; water has
$\mathrm{Oh}=0.002$, so the pattern is the linearly selected dominant Rayleigh–Plateau
mode — not an illusion.

---

## References

1. Lord Rayleigh, *On the instability of jets*, Proc. London Math. Soc. **s1-10**(1),
   4–13 (1879).
2. S. Chandrasekhar, *Hydrodynamic and Hydromagnetic Stability*, Oxford (1961).
3. P. Huerre & P. A. Monkewitz, Annu. Rev. Fluid Mech. **22**, 473–537 (1990).
4. J. Eggers, Rev. Mod. Phys. **69**(3), 865 (1997).
