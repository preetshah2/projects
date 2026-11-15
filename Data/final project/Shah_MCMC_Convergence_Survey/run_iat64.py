import numpy as np, json, time
from sim_ax import run_chains, iat
rng = np.random.default_rng(777)
res = json.load(open("iat_results.json"))
d, A = 2, 64
t0 = time.time()
nsteps = int(400 * d * A**2) + 20000
burn = int(20 * d * A**2) + 2000
rec = max(1, d*A**2//200)
out, _ = run_chains(d, A, nsteps, nchains=64, p0=np.full(d, 0.5), rng=rng, record_every=rec)
tau = iat(out[burn//rec:]) * rec
res[f"{d}_{A}"] = tau
print(d, A, round(tau,1), round(tau/(d*A**2),3), f"{time.time()-t0:.1f}s")
json.dump(res, open("iat_results.json","w"), indent=1)
