import numpy as np, json, time
from sim_ax import run_chains, iat
rng = np.random.default_rng(12345)
res = {}
for d in [2, 4]:
    for A in [4, 8, 16, 32]:
        t0 = time.time()
        nsteps = int(400 * d * A**2) + 20000
        burn = int(20 * d * A**2) + 2000
        rec = max(1, d*A**2//200)
        out, _ = run_chains(d, A, nsteps, nchains=64, p0=np.full(d, 0.5), rng=rng, record_every=rec)
        x = out[burn//rec:]
        tau = iat(x) * rec
        res[f"{d}_{A}"] = tau
        print(d, A, nsteps, round(tau,1), round(tau/(d*A**2),3), f"{time.time()-t0:.1f}s", flush=True)
json.dump(res, open("iat_results.json","w"), indent=1)
