"""
Partie 2 — Processus max-stables : génération des figures du rapport.

Produit 4 figures :
  1. Construction spectrale en 1D  (le champ = enveloppe supérieure de "tempêtes")
  2. Réalisation d'un champ de Smith en 2D
  3. Fonctions de corrélation isotropes (Tableau 1 de l'article)
  4. Coefficient extrémal Theta(h)  (pont vers la Partie 3)

Dépendances : numpy, scipy, matplotlib.
Sortie : fichiers PNG dans le dossier OUTDIR.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import gamma as gammafn, kv   # Bessel modifiée (Whittle-Matérn)

# --- adaptez ce chemin pour votre dépôt : par ex. "latex/figures" ---
OUTDIR = "projet-tve/latex/figures/mamadou_figures"
os.makedirs(OUTDIR, exist_ok=True)

# ------- charte graphique (identité bleue du projet) -------
NAVY, BLUE, AMBER, TEAL, PURPLE = "#17456E", "#2E77B5", "#E8663C", "#2E9C8E", "#7F77DD"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 12,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8A9AA8", "axes.labelcolor": "#22303C",
    "text.color": "#22303C", "xtick.color": "#5B6B7B", "ytick.color": "#5B6B7B",
})


# ============================================================
# Simulation d'un champ de Smith (algorithme spectral de Poisson)
#   Z(x) = max_i  zeta_i * phi_Sigma(x - S_i)
#   zeta_i = |D| / Gamma_i , Gamma_i = somme d'exponentielles (arrivées de Poisson)
#   S_i uniformes sur un domaine élargi D ; marges de Fréchet standard.
# ============================================================
def smith_field_1d(x, sigma=0.8, pad=3.0, seed=0, keep_bumps=True):
    rng = np.random.default_rng(seed)
    lo, hi = x.min() - pad, x.max() + pad
    areaD = hi - lo
    phimax = 1.0 / (sigma * np.sqrt(2 * np.pi))
    Z = np.zeros_like(x)
    G = 0.0
    bumps = []
    while True:
        G += rng.exponential()
        zeta = areaD / G
        if zeta * phimax < Z.min():        # règle d'arrêt exacte
            break
        s = rng.uniform(lo, hi)
        bump = zeta * norm.pdf(x, loc=s, scale=sigma)
        Z = np.maximum(Z, bump)
        if keep_bumps:
            bumps.append((s, bump))
    return Z, bumps


def smith_field_2d(gx, sigma=1.0, pad=5.0, seed=0):
    rng = np.random.default_rng(seed)
    X, Y = np.meshgrid(gx, gx)
    lo, hi = gx.min() - pad, gx.max() + pad
    areaD = (hi - lo) ** 2
    inv = 1.0 / (2 * np.pi * sigma ** 2)
    Z = np.zeros_like(X)
    G = 0.0
    while True:
        G += rng.exponential()
        zeta = areaD / G
        if zeta * inv < Z.min():
            break
        sx, sy = rng.uniform(lo, hi, 2)
        Z = np.maximum(Z, zeta * inv * np.exp(-((X - sx) ** 2 + (Y - sy) ** 2) / (2 * sigma ** 2)))
    return Z


# ============================================================
# FIGURE 1 — construction spectrale en 1D
# ============================================================
def figure_construction_1d():
    x = np.linspace(0, 10, 800)
    Z, bumps = smith_field_1d(x, sigma=0.75, seed=5)
    ymax = np.quantile(Z, 0.995) * 1.15    # cadrage robuste (queue lourde)

    fig, ax = plt.subplots(figsize=(8, 4.3))
    # tempêtes élémentaires (celles qui "affleurent" l'enveloppe)
    drawn = 0
    for s, b in bumps:
        if b.max() > 0.18 * Z.max():
            ax.plot(x, b, color=BLUE, lw=1, alpha=0.35)
            drawn += 1
    # enveloppe supérieure = réalisation du champ max-stable
    ax.plot(x, Z, color=NAVY, lw=2.6, label="champ $Z(x)$ = max des tempêtes")
    ax.plot([], [], color=BLUE, lw=1, alpha=0.6, label="tempêtes élémentaires $\\zeta_i\\,\\varphi(x-S_i)$")
    ax.set_ylim(0, ymax)
    ax.set_xlabel("position $x$")
    ax.set_ylabel("intensité")
    ax.set_title("Construction spectrale : le champ = maximum de tempêtes")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig1_construction_spectrale_1d.png"))
    plt.close(fig)


# ============================================================
# FIGURE 2 — champ de Smith en 2D
# ============================================================
def figure_champ_2d():
    gx = np.linspace(0, 10, 140)
    Z = smith_field_2d(gx, sigma=1.0, seed=7)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.pcolormesh(gx, gx, np.log(Z), cmap="magma", shading="auto")
    cs = ax.contour(gx, gx, np.log(Z), levels=6, colors="white", linewidths=0.5, alpha=0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_title("Réalisation d'un champ max-stable de Smith  ($\\log Z$)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="$\\log Z$")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig2_champ_smith_2d.png"))
    plt.close(fig)


# ============================================================
# FIGURE 3 — fonctions de corrélation isotropes (Tableau 1)
# ============================================================
def rho_whittle_matern(h, c1=1.0, c2=1.0):
    h = np.asarray(h, float)
    out = np.ones_like(h)
    m = h > 0
    z = h[m] / c1
    out[m] = (2 ** (1 - c2) / gammafn(c2)) * (z ** c2) * kv(c2, z)
    return out

def rho_cauchy(h, c1=1.0, c2=1.0):
    return (1 + (np.asarray(h, float) / c1) ** 2) ** (-c2)

def rho_powered_exp(h, c1=1.0, c2=1.0):
    return np.exp(-(np.asarray(h, float) / c1) ** c2)

def figure_correlations():
    h = np.linspace(0, 4, 400)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.plot(h, rho_whittle_matern(h, 1.0, 1.5), color=NAVY, lw=2.2, label="Whittle–Matérn ($c_2=1.5$)")
    ax.plot(h, rho_cauchy(h, 1.0, 1.0), color=AMBER, lw=2.2, label="Cauchy ($c_2=1$)")
    ax.plot(h, rho_powered_exp(h, 1.0, 1.0), color=TEAL, lw=2.2, label="Exponentielle puissance ($c_2=1$)")
    ax.plot(h, rho_powered_exp(h, 1.0, 0.5), color=PURPLE, lw=2.2, ls="--",
            label="Exp. puissance ($c_2=0.5$)")
    ax.set_xlabel("distance $h$"); ax.set_ylabel(r"corrélation $\rho(h)$")
    ax.set_ylim(0, 1.02)
    ax.set_title("Familles de fonctions de corrélation isotropes ($c_1=1$)")
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig3_fonctions_correlation.png"))
    plt.close(fig)


# ============================================================
# FIGURE 4 — coefficient extrémal Theta(h)   (pont vers Partie 3)
# ============================================================
def theta_smith(h):        # Sigma = I
    return 2 * norm.cdf(h / 2)

def theta_schlather(h):    # rho = exponentielle
    return 1 + np.sqrt((1 - np.exp(-h)) / 2)

def theta_tube(h, Rb=1.0):
    h = np.atleast_1d(h).astype(float)
    hb = 1 / (np.pi * Rb ** 2)
    out = np.full_like(h, 2.0)
    m = h <= 2 * Rb
    d = h[m]; r = np.sqrt(np.clip(4 * Rb ** 2 - d ** 2, 0, None))
    out[m] = 2 * (1 - hb * (Rb ** 2 * np.arcsin(r / (2 * Rb)) - d / 4 * r))
    return out

def figure_coef_extremal():
    h = np.linspace(0, 6, 400)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.plot(h, theta_smith(h), color=NAVY, lw=2.4, label="Smith")
    ax.plot(h, theta_tube(h), color=BLUE, lw=2.4, label="Tube ($R_b=1$)")
    ax.plot(h, theta_schlather(h), color=AMBER, lw=2.4, label="Schlather")
    ax.axhline(2, color="#9AA8B6", ls="--", lw=1)
    ax.axhline(1, color="#C4CDD6", ls=":", lw=1)
    ax.text(0.15, 1.03, "dépendance", color="#7B8A98", fontsize=10)
    ax.text(4.6, 1.93, "indépendance", color="#7B8A98", fontsize=10)
    ax.set_xlabel("distance $h$"); ax.set_ylabel(r"coefficient extrémal $\Theta(h)$")
    ax.set_ylim(0.95, 2.08)
    ax.set_title(r"Coefficient extrémal $\Theta \in [1,2]$ : dépendance des extrêmes")
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig4_coefficient_extremal.png"))
    plt.close(fig)


if __name__ == "__main__":
    figure_construction_1d()
    figure_champ_2d()
    figure_correlations()
    figure_coef_extremal()
    print("4 figures générées dans :", OUTDIR)