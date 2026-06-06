# James Webb Ramp-Fit

A from-scratch **JWST Stage-1 "ramp fitting" engine** written in Rust, with a **CPU** path
(`rayon`) and a **GPU** path (`wgpu` compute shaders). It reproduces the slope-fitting
step of the official STScI `jwst` calibration pipeline, validated **to the pixel** against
the archived `rate.fits` products — and runs it **~4.8× faster** than the CPU version.

> The folder name `Compare_JWT` originally hinted at JSON Web Tokens — it actually stands for
> the **James Webb Telescope**. This project has nothing to do with web auth.

---

## What problem this solves

JWST detectors don't read out a single image. During an exposure each pixel is sampled
**many times "up the ramp"** (non-destructive reads). The raw `*_uncal.fits` files are
therefore 4-D cubes (integration × group × row × column) of several gigabytes. Turning that
into a calibrated count-rate image (electrons-equivalent per second) requires fitting a
**slope** to each pixel's ramp, after a chain of detector corrections.

That step is per-pixel, embarrassingly parallel, numerically simple, and operates on huge
data — an excellent fit for Rust + SIMD + GPU. This repo implements the whole step and
checks every number against the official pipeline output.

## What it does (the full pipeline)

For every pixel, every integration, this engine applies, in order:

1. **Saturation rejection** — truncate the ramp at the first group above the per-pixel
   saturation threshold (CRDS `saturation` reference).
2. **Superbias subtraction** — remove the per-pixel pedestal (CRDS `superbias`).
3. **Linearity correction** — per-pixel polynomial in bias-subtracted DN (CRDS `linearity`,
   5 coefficients, Horner evaluation).
4. **Dark subtraction** — subtract the per-group dark ramp (CRDS `dark`).
5. **Optimal (Fixsen 2000) weighting** — SNR-binned power-law weights, with the regression
   constants precomputed per `(weight-power, segment-length)` so the hot loop has no `pow()`.
6. **Read-noise & Poisson variances** — `var_R` and `var_P` per segment, producing
   `VAR_RNOISE` and `VAR_POISSON` maps.
7. **Cosmic-ray (jump) detection** — two-point difference method with a robust MAD-based
   noise estimate; the ramp is split into segments at detected jumps.
8. **Inverse-variance combination** — segments → integration → exposure.
9. **`gain_scale`** — final `GAINFACT` multiply (matches the pipeline's gain-scale step).

A minimal pure-Rust **FITS reader** (`src/fits.rs`, `mmap`-backed) handles all I/O — no
`cfitsio`, no Python, no external C dependency.

## Validation

Validated against the official `jw01366003001_04101_00001-seg001_nrs1_rate.fits`
(WASP-39, NIRSpec NRS1, BOTS time series) for a 2048×32×70×155 cube:

| Output | Agreement vs official pipeline |
| --- | --- |
| `SCI` count rate, bright trace (≥20 DN/s) | ratio **1.0000** (exact) |
| `SCI` count rate, faint (1–20 DN/s) | 0.985 – 0.996 |
| `VAR_POISSON` | **0.997** (essentially perfect) |
| `VAR_RNOISE` | 1.064 (~6 %) |
| Jump rate | ~0.03 per pixel-integration (physical) |

The remaining few-percent residuals come from steps deliberately **not** implemented
(reference-pixel / 1/f correction, the DQ `mask`) and from `stcal`'s exact weighted
read-noise-variance treatment.

## Performance

Full phase-3 pipeline on the 1.3 GB test cube (Intel i9 10th gen, RTX 4070 Ti):

| Version | Time | Speedup |
| --- | --- | --- |
| CPU (`rayon`, 20 threads) | 5.4 s | 1× |
| GPU (`wgpu`, f32 upload) | 1.70 s | 3.2× |
| GPU + packed-i16 upload | 1.36 s | 4.0× |
| GPU + packed-i16 + double-buffering | **1.12 s** | **~4.8×** |

Things learned along the way (measured, not assumed):

- The bright-trace discrepancy was **not** linearity — it was the constant `GAINFACT = 1.429`
  `gain_scale` factor (a flux-independent ratio is a units/gain clue, not a non-linearity one).
- Jump detection initially over-flagged 7×/pixel because **1/f noise** (no refpix step)
  inflates the group-to-group scatter to ~2.8× the read-noise model; a **MAD-based** robust
  sigma fixes it.
- A GPU only wins when arithmetic intensity is high: the cheap OLS slope kernel is
  PCIe/memory-bound and shows **no** GPU speedup, while the weighting + variances + jump
  kernel does — the GPU hides the per-thread sorting behind thousands of threads.

## Repository layout

```
src/fits.rs   minimal pure-Rust FITS reader (mmap, HDU/card parsing)
src/main.rs   CPU pipeline + CLI + validation against rate.fits
src/gpu.rs    wgpu compute port (WGSL kernels, packed-i16 upload, double-buffering)
```

## Getting started

### Prerequisites

- **Rust** (stable, edition 2024 — i.e. a recent toolchain).
- For the GPU path: any Vulkan/DX12/Metal-capable GPU. **No CUDA toolkit required** — `wgpu`
  uses the regular graphics driver.

### Build

```bash
cargo build --release
```

### Get the test data

The FITS files are large and are **not** committed (see `.gitignore`). Download them into a
`data/` folder. The raw cube and the reference `rate.fits` come from
[MAST](https://mast.stsci.edu); the calibration references come from
[CRDS](https://jwst-crds.stsci.edu).

PowerShell (Windows):

```powershell
New-Item -ItemType Directory -Force data | Out-Null
$mast = "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:JWST/product"
$crds = "https://jwst-crds.stsci.edu/unchecked_get/references/jwst"
foreach ($f in @(
  "$mast/jw01366003001_04101_00001-seg001_nrs1_uncal.fits",   # raw ramp cube (1.3 GB)
  "$mast/jw01366003001_04101_00001-seg001_nrs1_rate.fits",    # official result (validation)
  "$crds/jwst_nirspec_saturation_0028.fits",
  "$crds/jwst_nirspec_superbias_0427.fits",
  "$crds/jwst_nirspec_linearity_0024.fits",
  "$crds/jwst_nirspec_dark_0438.fits",
  "$crds/jwst_nirspec_readnoise_0043.fits",
  "$crds/jwst_nirspec_gain_0025.fits")) {
  $name = ($f -split "/")[-1] -replace "\?.*",""
  Invoke-WebRequest -Uri $f -OutFile (Join-Path data $name)
}
```

(Use `curl -L -o data/<name> "<url>"` for the bash/macOS/Linux equivalent.)

### Run

CPU, full phase-3 pipeline with validation:

```bash
cargo run --release -- data/jw01366003001_04101_00001-seg001_nrs1_uncal.fits \
  --rate data/jw01366003001_04101_00001-seg001_nrs1_rate.fits \
  --sat  data/jwst_nirspec_saturation_0028.fits \
  --bias data/jwst_nirspec_superbias_0427.fits \
  --lin  data/jwst_nirspec_linearity_0024.fits \
  --dark data/jwst_nirspec_dark_0438.fits \
  --rn   data/jwst_nirspec_readnoise_0043.fits \
  --gain data/jwst_nirspec_gain_0025.fits \
  --gainfact 1.429
```

Add `--gpu` to run the same computation on the GPU.

### CLI flags

| Flag | Meaning |
| --- | --- |
| `<uncal.fits>` | input raw ramp cube (positional, required) |
| `--rate F` | official `rate.fits` to validate against (optional) |
| `--sat F` | CRDS saturation reference (enables saturation rejection) |
| `--bias F` | CRDS superbias reference |
| `--lin F` | CRDS linearity reference |
| `--dark F` | CRDS dark reference |
| `--rn F` | CRDS read-noise reference (enables Fixsen weights + variances) |
| `--gain F` | CRDS gain reference |
| `--gainfact X` | gain-scale factor (NIRSpec NRS1 = `1.429`) |
| `--jumpthresh X` | jump rejection threshold in sigma (default `4.0`, `0` disables) |
| `--gpu` | run on the GPU via `wgpu` |

Calibration steps are activated only when their reference file is supplied, so you can run
the pipeline incrementally (e.g. slope only, then + saturation, then + variances, …).

## Limitations / not implemented

- Reference-pixel / 1/f (`refpix`) and the DQ `mask` (`DO_NOT_USE`) steps are not applied,
  which accounts for the last ~1 % on faint pixels and a few un-masked dead pixels.
- Tested on NIRSpec; other instruments/subarrays should work but are unverified.
- The GPU kernels use `f32`; results match the `f64` CPU path to the displayed precision.
- The jump GPU kernel supports up to 80 groups per ramp.

## Data & credits

Data products courtesy of the **Mikulski Archive for Space Telescopes (MAST)** and the
**Calibration Reference Data System (CRDS)** at STScI. The ramp-fitting algorithm follows
the public `jwst` / `stcal` pipeline documentation and Fixsen et al. (2000).
This project is an independent reimplementation for learning and performance, not an
official STScI product.
