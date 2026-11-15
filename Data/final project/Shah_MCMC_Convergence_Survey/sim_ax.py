# Random-scan Gibbs sampler for de Finetti's almost-exchangeable prior
#   pi(p) ∝ exp(-A^2 sum_{i<j} c_ij (p_i-p_j)^2) on [0,1]^d
# Complete graph with uniform weights normalized so sum_{i!=j} c_ij = 1.
import numpy as np
from scipy.special import ndtr, ndtri

def run_chains(d, A, nsteps, nchains, p0=None, rng=None, record_every=1):
    rng = np.random.default_rng() if rng is None else rng
    C = np.full((d, d), 1.0/(d*(d-1))); np.fill_diagonal(C, 0.0)
    c = C.sum(axis=1)                      # c_i
    sig = np.sqrt(1.0/(2*A**2*c))          # conditional sd
    P = np.zeros((nchains, d)) if p0 is None else np.tile(p0, (nchains,1)).astype(float)
    w = c / c.sum()                        # barycenter weights (= c since sum c = 1)
    out = np.empty((nsteps//record_every, nchains))
    idx = np.arange(nchains)
    for t in range(nsteps):
        i = rng.integers(0, d, size=nchains)
        phat = (P * C[i]).sum(axis=1) / c[i]          # weighted neighbour average
        s = sig[i]
        a = ndtr((0.0 - phat)/s); b = ndtr((1.0 - phat)/s)
        u = a + (b - a)*rng.random(nchains)
        u = np.clip(u, 1e-300, 1-1e-16)
        P[idx, i] = np.clip(phat + s*ndtri(u), 0.0, 1.0)
        if (t+1) % record_every == 0:
            out[(t+1)//record_every - 1] = P @ w
    return out, P

def iat(x, c=5.0):
    """Integrated autocorrelation time with Sokal's automatic window (x: n x chains)."""
    x = x - x.mean(axis=0)
    n = x.shape[0]
    f = np.fft.rfft(x, n=2*n, axis=0)
    acf = np.fft.irfft(f*np.conj(f), axis=0)[:n].mean(axis=1)
    acf /= acf[0]
    taus = 2.0*np.cumsum(acf) - 1.0
    for M in range(1, n):
        if M >= c*taus[M]:
            return taus[M]
    return taus[-1]
