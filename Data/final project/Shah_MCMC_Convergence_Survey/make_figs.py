import numpy as np, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sim_ax import run_chains

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6, "lines.linewidth": 1.1, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
BLUE, ORANGE, GREEN, GREY = "#1f5aa6", "#d1631f", "#2e8b57", "#7a7a7a"

# ---------------- Figure 1: geometry of two-component Gibbs ----------------
def ds_gibbs_path(rho, x0, y0, n, rng):
    pts = [(x0, y0)]; x, y = x0, y0
    s = np.sqrt(1 - rho**2)
    for _ in range(n):
        y = rho*x + s*rng.standard_normal(); pts.append((x, y))   # update y | x
        x = rho*y + s*rng.standard_normal(); pts.append((x, y))   # update x | y
    return np.array(pts)

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.25), gridspec_kw={"width_ratios": [1, 1, 1.25]})
rng = np.random.default_rng(3)
g = np.linspace(-3.6, 3.6, 300); X, Y = np.meshgrid(g, g)
for ax, rho, n, lab in [(axes[0], 0.5, 6, "(a)"), (axes[1], 0.95, 25, "(b)")]:
    Q = (X**2 - 2*rho*X*Y + Y**2)/(1 - rho**2)
    ax.contour(X, Y, Q, levels=[1, 4, 9], colors=GREY, linewidths=0.5)
    pts = ds_gibbs_path(rho, -3.0, -3.0, n, rng)
    ax.plot(pts[:, 0], pts[:, 1], "-", color=BLUE, lw=0.8)
    ax.plot(pts[0, 0], pts[0, 1], "o", color=ORANGE, ms=3.5, zorder=5)
    ax.set_xlim(-3.6, 3.6); ax.set_ylim(-3.6, 3.6); ax.set_aspect("equal")
    ax.set_xticks([-3, 0, 3]); ax.set_yticks([-3, 0, 3])
    ax.set_title(f"{lab} $\\rho={rho}$: {n} sweeps")
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$", labelpad=1)

ax = axes[2]
gam = np.linspace(0.02, 0.995, 400)
for r, col, ls in [(0.5, BLUE, "-"), (0.3, GREEN, "--")]:
    rho_rg = (1 + np.sqrt(1 - 4*r*(1-r)*(1 - gam**2)))/2
    R = (-np.log(gam**2)/2) / (-np.log(rho_rg))       # RG draws per DG draw
    ax.plot(gam, R, ls, color=col, label=f"random scan, $r={r}$")
    ax.axhline(1/(2*r*(1-r)), color=col, lw=0.6, ls=":")
ax.set_ylim(1.5, 12); ax.set_xlim(0, 1)
ax.set_yticks([2, 4, 6, 8, 10, 12])
ax.set_xlabel("maximal correlation $\\bar\\gamma$")
ax.set_ylabel("draws needed: RG / DG", labelpad=1)
ax.set_title("(c) relative cost of random scan")
ax.legend(frameon=False, loc="upper right")
fig.tight_layout(w_pad=1.0)
fig.savefig("figs/fig_gibbs_geometry.pdf")
plt.close(fig)

# ---------------- Figure 3: almost-exchangeable Gibbs sampler ----------------
res = json.load(open("iat_results.json"))
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), gridspec_kw={"width_ratios": [1, 1.25, 1.15]})
# (a) path in the unit square, d=2, A=20, from the corner 0
rng = np.random.default_rng(11)
d, A = 2, 30
C = np.array([[0, .5], [.5, 0]]); c = C.sum(1); sig = np.sqrt(1/(2*A**2*c))
from scipy.special import ndtr, ndtri
p = np.zeros(2); path = [p.copy()]
for t in range(1500):
    i = rng.integers(2); m = p[1-i]; s = sig[i]
    a, b = ndtr(-m/s), ndtr((1-m)/s); u = a + (b-a)*rng.random()
    p[i] = min(max(m + s*ndtri(u), 0), 1); path.append(p.copy())
path = np.array(path)
ax = axes[0]
band = np.linspace(0, 1, 50)
ax.fill_between(band, np.clip(band-2/A, 0, 1), np.clip(band+2/A, 0, 1), color="#e8eef7", lw=0)
ax.plot([0, 1], [0, 1], color=GREY, lw=0.5, ls="--")
segs = np.stack([path[:-1], path[1:]], axis=1)
lc = LineCollection(segs, cmap="viridis", linewidths=0.6)
lc.set_array(np.arange(len(segs))); ax.add_collection(lc)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1])
ax.set_xlabel("$p_1$"); ax.set_ylabel("$p_2$", labelpad=1)
ax.set_title(f"(a) $d=2$, $A={A}$: 1500 updates")
# (b) barycenter traces for two values of A, started at the corner 0
ax = axes[1]
for A_, col, lw, al in [(10, BLUE, 0.5, 0.75), (50, ORANGE, 0.9, 1.0)]:
    out, _ = run_chains(2, A_, 10000, nchains=1, p0=np.full(2, 0.5), rng=np.random.default_rng(8), record_every=5)
    ax.plot(np.arange(1, out.shape[0]+1)*5/1000, out[:, 0], color=col, lw=lw, alpha=al, label=f"$A={A_}$")
ax.set_xlabel("single-site updates ($\\times 10^3$)"); ax.set_ylabel("barycenter $\\bar p$", labelpad=1)
ax.set_ylim(0, 1); ax.set_xlim(0, 10); ax.legend(frameon=True, framealpha=0.92, edgecolor="none", loc="lower right", ncol=2)
ax.set_title("(b) slow diffusion along the diagonal")
# (c) IAT vs A
ax = axes[2]
for d_, col, mk in [(2, BLUE, "o"), (4, GREEN, "s")]:
    As = sorted(int(k.split("_")[1]) for k in res if k.startswith(f"{d_}_"))
    taus = [res[f"{d_}_{a}"] for a in As]
    ax.loglog(As, taus, mk+"-", color=col, ms=3.5, lw=0.9, label=f"$d={d_}$")
Aref = np.array([4, 64]); ax.loglog(Aref, 0.25*2*Aref**2, ":", color=GREY, lw=0.9, label="slope 2")
ax.set_xlabel("$A$"); ax.set_ylabel("IAT of $\\bar p$ (updates)", labelpad=1)
ax.set_xticks([4, 8, 16, 32, 64]); ax.set_xticklabels(["4", "8", "16", "32", "64"])
from matplotlib.ticker import NullFormatter
ax.xaxis.set_minor_formatter(NullFormatter())
ax.legend(frameon=False, loc="upper left"); ax.set_title("(c) autocorrelation time $\\propto A^2$")
fig.tight_layout(w_pad=1.0)
fig.savefig("figs/fig_almost_exch.pdf")
plt.close(fig)
print("done")
