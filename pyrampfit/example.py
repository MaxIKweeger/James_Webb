"""Example / smoke test for the Rust-accelerated ramp fitter.

Builds a synthetic calibrated ramp cube with a known slope and an injected
cosmic-ray jump, runs `jwst_rampfit.fit_ramps`, and checks the recovered rate.

Run after `maturin develop --release` (from this folder):

    python example.py

How it plugs into the pipeline
------------------------------
`stcal.ramp_fitting` is called by the jwst `RampFitStep` with a RampData object
that already holds the calibrated `data`, the `groupdq` flags (saturation / jump /
do-not-use set by earlier steps), plus `gain`, `read_noise`, and `group_time`.
This function takes exactly those arrays and returns slope / var_poisson /
var_rnoise / err, so it can serve as an accelerated backend for that step.
"""

import time
import numpy as np
import jwst_rampfit as jr

# --- synthetic data -------------------------------------------------------
rng = np.random.default_rng(0)
nints, ngroups, nrows, ncols = 4, 60, 64, 64
group_time = 0.9
true_rate = 12.0  # DN/s
gain = np.full((nrows, ncols), 1.4, dtype=np.float32)
readnoise = np.full((nrows, ncols), 11.0, dtype=np.float32)

# Ramps: linear in time + read noise; groupdq all good.
t = (np.arange(ngroups) * group_time).reshape(1, ngroups, 1, 1)
data = (true_rate * t + rng.normal(0, 5, (nints, ngroups, nrows, ncols))).astype(np.float32)
groupdq = np.zeros((nints, ngroups, nrows, ncols), dtype=np.uint8)

# Inject a cosmic ray on one pixel: a jump at group 30, flagged JUMP_DET.
data[0, 30:, 10, 10] += 4000.0
groupdq[0, 30, 10, 10] |= jr.JUMP_DET

t0 = time.perf_counter()
out = jr.fit_ramps(data, groupdq, gain, readnoise, group_time)
dt = time.perf_counter() - t0

slope = out["slope"]
print(f"fit_ramps: {data.size/1e6:.1f} Msamples in {dt*1e3:.1f} ms")
print(f"median rate = {np.nanmedian(slope):.3f} DN/s  (expected ~{true_rate})")
print(f"CR pixel (10,10) rate = {slope[10,10]:.3f} DN/s  (jump handled, should stay ~{true_rate})")
print(f"err median = {np.nanmedian(out['err']):.4f}")

assert abs(np.nanmedian(slope) - true_rate) < 0.5, "slope off"
assert abs(slope[10, 10] - true_rate) < 1.5, "jump not handled"
print("OK")
