//! JWST ramp-fit engine.
//!
//! Phase 0 : open a `_uncal` FITS ramp cube, read SCI geometry.
//! Phase 1 : per-pixel OLS slope (DN/s) over the up-the-ramp groups.
//! Phase 2 : per-group calibration before the fit — saturation rejection,
//!   superbias, linearity, dark, then a final gain_scale multiply.
//! Phase 3a: Fixsen *optimal* weighting (SNR-binned power law), per-integration
//!   read-noise & Poisson variances, and inverse-variance combination of the
//!   integrations. Produces SCI + VAR_RNOISE + VAR_POISSON, validated against the
//!   official rate.fits extensions.
//!
//! TODO phase 3b: jump (cosmic-ray) detection + multi-segment ramps; then GPU.
//!
//! Usage:
//!   compare_jwt <uncal> [--rate F] [--sat F] [--bias F] [--lin F] [--dark F]
//!               [--rn F] [--gain F] [--gainfact X]

mod fits;
mod gpu;

use fits::{Fits, Hdu};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Fixsen power-law exponents and the SNR thresholds that select them.
const P_VALUES: [f64; 6] = [0.0, 0.4, 1.0, 3.0, 6.0, 10.0];

fn snr_pidx(data: f64, rn: f64, gain: f64) -> usize {
    if data <= 0.0 {
        return 0;
    }
    let s = data * gain / (rn * rn + data * gain).sqrt();
    match s {
        s if s < 5.0 => 0,
        s if s < 10.0 => 1,
        s if s < 20.0 => 2,
        s if s < 50.0 => 3,
        s if s < 100.0 => 4,
        _ => 5,
    }
}

/// Precomputed weights + regression constants for one (P, m) pair. Sw/Swx/Swxx
/// and the determinant depend only on the weights and the group positions, not on
/// the data, so they're computed once and reused across all pixels.
#[derive(Clone, Default)]
struct WTab {
    w: Vec<f64>,
    sw: f64,
    swx: f64,
    denom: f64,
}

/// Build `wtab[pidx][m]` for m in 0..=ngroup (entries with m<2 are unused).
fn build_weight_tables(ngroup: usize) -> Vec<Vec<WTab>> {
    let mut tabs = vec![vec![WTab::default(); ngroup + 1]; P_VALUES.len()];
    for (pidx, &p) in P_VALUES.iter().enumerate() {
        for m in 2..=ngroup {
            let mid = (m - 1) as f64 / 2.0;
            let w: Vec<f64> = (0..m)
                .map(|g| {
                    if mid == 0.0 { 1.0 } else { ((g as f64 - mid).abs() / mid).powf(p) }
                })
                .collect();
            let sw: f64 = w.iter().sum();
            let swx: f64 = w.iter().enumerate().map(|(g, wi)| wi * g as f64).sum();
            let swxx: f64 = w.iter().enumerate().map(|(g, wi)| wi * (g * g) as f64).sum();
            let denom = sw * swxx - swx * swx;
            tabs[pidx][m] = WTab { w, sw, swx, denom };
        }
    }
    tabs
}

/// Optimal-weight fit of one ramp segment. Returns (slope DN/s, var_R, var_P).
fn weighted_seg_fit(ramp: &[f64], rn: f64, gain: f64, tgroup: f64, wtab: &[Vec<WTab>]) -> Option<(f64, f64, f64)> {
    let m = ramp.len();
    if m < 2 {
        return None;
    }
    let data = ramp[m - 1] - ramp[0];
    let pidx = snr_pidx(data, rn, gain);
    let t = &wtab[pidx][m];
    let (mut swy, mut swxy) = (0.0, 0.0);
    for (g, &y) in ramp.iter().enumerate() {
        let wy = t.w[g] * y;
        swy += wy;
        swxy += g as f64 * wy;
    }
    let slope = (t.sw * swxy - t.swx * swy) / t.denom / tgroup;
    let m3 = (m * m * m - m) as f64;
    let var_r = 12.0 * (rn * rn / 2.0) / (m3 * tgroup * tgroup);
    let var_p = slope.max(0.0) / (gain * tgroup * (m as f64 - 1.0));
    Some((slope, var_r, var_p))
}

/// Median of a slice (O(n) via select_nth on a local copy).
fn robust_median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut tmp = v.to_vec();
    let k = tmp.len() / 2;
    tmp.select_nth_unstable_by(k, |a, b| a.partial_cmp(b).unwrap());
    tmp[k]
}

/// Two-point difference jump detection. Returns sorted group indices that each
/// begin a new segment (a cosmic-ray hit lands on that group).
fn detect_jumps(ramp: &[f64], rn: f64, gain: f64, thresh: f64) -> Vec<usize> {
    let m = ramp.len();
    if m < 3 || thresh <= 0.0 {
        return Vec::new();
    }
    let nd = m - 1;
    let diffs: Vec<f64> = (0..nd).map(|i| ramp[i + 1] - ramp[i]).collect();
    let mut excluded = vec![false; nd];
    let mut jumps = Vec::new();
    loop {
        let active: Vec<f64> = (0..nd).filter(|&i| !excluded[i]).map(|i| diffs[i]).collect();
        if active.len() < 2 {
            break;
        }
        let med = robust_median(&active);
        // Robust scatter from the data (MAD): the read-noise model underestimates
        // the real group-to-group scatter because 1/f noise (no refpix step here)
        // inflates it. Use MAD-derived sigma, with the read-noise model as a floor.
        let mut dev: Vec<f64> = active.iter().map(|x| (x - med).abs()).collect();
        let mad = robust_median(&dev);
        dev.clear();
        let model = (rn * rn + med.max(0.0) / gain).sqrt();
        let sigma = (1.4826 * mad).max(model);
        if !(sigma > 0.0) {
            break;
        }
        let mut best = (thresh, usize::MAX);
        for i in 0..nd {
            if !excluded[i] {
                let ratio = (diffs[i] - med) / sigma;
                if ratio > best.0 {
                    best = (ratio, i);
                }
            }
        }
        if best.1 == usize::MAX {
            break;
        }
        excluded[best.1] = true;
        jumps.push(best.1 + 1); // the higher-index group starts the new segment
    }
    jumps.sort_unstable();
    jumps
}

/// Combine per-segment (slope, var_R, var_P) by inverse variance into one
/// (slope, var_R, var_P) at the next level up.
fn combine_inv_var(segs: &[(f64, f64, f64)]) -> Option<(f64, f64, f64)> {
    let (mut sw, mut sws, mut s2r, mut s2p) = (0.0, 0.0, 0.0, 0.0);
    for &(s, vr, vp) in segs {
        let vc = vr + vp;
        if !(vc.is_finite() && vc > 0.0) {
            continue;
        }
        let w = 1.0 / vc;
        sw += w;
        sws += w * s;
        s2r += w * w * vr;
        s2p += w * w * vp;
    }
    if sw <= 0.0 {
        return None;
    }
    Some((sws / sw, s2r / (sw * sw), s2p / (sw * sw)))
}

fn main() {
    // --- flag parser -------------------------------------------------------
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let (mut path, mut rate, mut sat, mut bias, mut lin, mut dark, mut rn, mut gain) =
        (None, None, None, None, None, None, None, None);
    let mut gainfact = 1.0f64;
    let mut jthresh = 4.0f64; // jump rejection threshold (sigma); 0 disables
    let mut use_gpu = false;
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--gpu" => { use_gpu = true; i += 1; }
            "--rate" => { rate = argv.get(i + 1).cloned(); i += 2; }
            "--sat" => { sat = argv.get(i + 1).cloned(); i += 2; }
            "--bias" => { bias = argv.get(i + 1).cloned(); i += 2; }
            "--lin" => { lin = argv.get(i + 1).cloned(); i += 2; }
            "--dark" => { dark = argv.get(i + 1).cloned(); i += 2; }
            "--rn" => { rn = argv.get(i + 1).cloned(); i += 2; }
            "--gain" => { gain = argv.get(i + 1).cloned(); i += 2; }
            "--gainfact" => { gainfact = argv.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(1.0); i += 2; }
            "--jumpthresh" => { jthresh = argv.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(4.0); i += 2; }
            other => { if path.is_none() { path = Some(other.to_string()); } i += 1; }
        }
    }
    let path = path.unwrap_or_else(|| "data/jw01366003001_04101_00001-seg001_nrs1_uncal.fits".into());

    println!("Opening {path}");
    let f = Fits::open(&path).expect("failed to open FITS");

    // --- Phase 0: geometry -------------------------------------------------
    let sci = f.find("SCI").expect("no SCI extension found");
    let ncol = sci.int("NAXIS1").expect("NAXIS1") as usize;
    let nrow = sci.int("NAXIS2").expect("NAXIS2") as usize;
    let ngroup = sci.int("NAXIS3").expect("NAXIS3") as usize;
    let nint = sci.int("NAXIS4").expect("NAXIS4") as usize;
    assert_eq!(sci.int("BITPIX"), Some(16), "this phase only handles BITPIX=16 raw integers");
    let bzero = sci.float("BZERO").unwrap_or(0.0);
    let bscale = sci.float("BSCALE").unwrap_or(1.0);
    let tgroup = f.hdus[0].float("TGROUP").unwrap_or(1.0);
    let npix = nrow * ncol;
    println!("SCI cube: {ncol} col x {nrow} row x {ngroup} groups x {nint} integrations  ({} Msamples)", npix * ngroup * nint / 1_000_000);

    // --- Calibration references (all optional) -----------------------------
    let saturation = load_opt(&sat, "[2a] saturation", || load_2d(sat.as_ref().unwrap(), "SCI", &f, nrow, ncol, f64::INFINITY), || vec![f64::INFINITY; npix]);
    let superbias = load_opt(&bias, "[2b] superbias ", || load_2d(bias.as_ref().unwrap(), "SCI", &f, nrow, ncol, 0.0), || vec![0.0; npix]);
    let (coeffs, ncoeff) = match &lin {
        Some(p) => { let (c, n) = load_linearity(p, &f, nrow, ncol); println!("[2b] linearity : {p}  ({n} coeffs)"); (c, n) }
        None => { println!("[2b] linearity : off"); (Vec::new(), 0) }
    };
    let has_lin = ncoeff > 0;
    let dark_ramp = match &dark {
        Some(p) => { let d = load_dark(p, &f, nrow, ncol, ngroup); println!("[2c] dark      : {p}"); d }
        None => { println!("[2c] dark      : off"); Vec::new() }
    };
    let has_dark = !dark_ramp.is_empty();

    // Phase 3a needs read noise (DN) and gain (e-/DN) to weight and to form variances.
    let has_var = rn.is_some() && gain.is_some();
    let readnoise = load_opt(&rn, "[3a] readnoise ", || load_2d(rn.as_ref().unwrap(), "SCI", &f, nrow, ncol, f64::NAN), || vec![f64::NAN; npix]);
    let gainmap = load_opt(&gain, "[3a] gain      ", || load_2d(gain.as_ref().unwrap(), "SCI", &f, nrow, ncol, f64::NAN), || vec![f64::NAN; npix]);
    println!("[3a] weighting : {}", if has_var { "Fixsen optimal + inverse-variance combine" } else { "uniform / simple mean" });

    let wtab = build_weight_tables(ngroup);

    let bytes = &f.mmap[sci.data_offset..sci.data_offset + sci.data_len];
    let raw = |idx: usize| -> f64 {
        let b = idx * 2;
        bzero + bscale * (i16::from_be_bytes([bytes[b], bytes[b + 1]]) as f64)
    };

    // --- GPU path ----------------------------------------------------------
    if use_gpu {
        let samples = (npix * ngroup * nint) as f64;
        let (rate_map, vrn, vpo, gtime) = if has_var {
            // Phase-3a kernel: Fixsen weights + variances (no jumps on GPU yet).
            gpu::run_gpu_weighted(bytes, npix, ngroup, nint, bzero, bscale, tgroup, gainfact,
                &saturation, &superbias, &coeffs, ncoeff, &dark_ramp, &readnoise, &gainmap, jthresh)
        } else {
            // Phase-2 kernel: uniform-weight OLS, simple mean.
            let (r, t) = gpu::run_gpu(bytes, npix, nrow, ncol, ngroup, nint, bzero, bscale, tgroup,
                gainfact, &saturation, &superbias, &coeffs, ncoeff, &dark_ramp);
            (r, Vec::new(), Vec::new(), t)
        };
        let mut sorted: Vec<f64> = rate_map.iter().copied().filter(|x| x.is_finite()).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("\n[gpu] Rate: min={:.3} median={:.4} max={:.3} DN/s", sorted[0], sorted[sorted.len() / 2], sorted.last().unwrap());
        println!("[gpu] Computed in {:.2}s  ({:.0} Msamples/s){}", gtime, samples / gtime / 1e6,
            if gainfact != 1.0 { format!(", GAINFACT={gainfact}") } else { String::new() });
        if let Some(rp) = &rate {
            compare_rate(rp, &rate_map, nrow, ncol);
            if has_var {
                compare_plane(rp, "VAR_RNOISE", &vrn);
                compare_plane(rp, "VAR_POISSON", &vpo);
            }
        }
        return;
    }

    // --- Per-integration fit, combined by inverse variance -----------------
    // Accumulators per pixel: Σw, Σw·slope, Σw²·var_R, Σw²·var_P, n_used.
    let t0 = Instant::now();
    let njumps = AtomicU64::new(0);
    let zero = || (vec![0f64; npix], vec![0f64; npix], vec![0f64; npix], vec![0f64; npix], vec![0u32; npix]);
    let (sw, sws, sw2r, sw2p, cnt) = (0..nint)
        .into_par_iter()
        .map(|it| {
            // 1) Build the calibrated ramp for this integration, pixel-major
            //    (yb[p*ngroup + g]) so the per-pixel fit reads contiguously.
            let mut yb = vec![0f64; ngroup * npix];
            let mut m = vec![0u32; npix];
            let mut done = vec![false; npix];
            for g in 0..ngroup {
                let base = (it * ngroup + g) * npix;
                for p in 0..npix {
                    if done[p] {
                        continue;
                    }
                    let v = raw(base + p);
                    if v >= saturation[p] {
                        done[p] = true;
                        continue;
                    }
                    let x = v - superbias[p];
                    let mut y = if has_lin {
                        let off = p * ncoeff;
                        let mut acc = coeffs[off + ncoeff - 1];
                        for k in (0..ncoeff - 1).rev() {
                            acc = acc * x + coeffs[off + k];
                        }
                        acc
                    } else {
                        x
                    };
                    if has_dark {
                        y -= dark_ramp[g * npix + p];
                    }
                    let mc = m[p] as usize;
                    yb[p * ngroup + mc] = y; // store contiguously among usable groups
                    m[p] += 1;
                }
            }

            // 2) Per-pixel fit: jump detection -> segments -> inverse variance.
            let (mut a_sw, mut a_sws, mut a_2r, mut a_2p, mut a_cnt) = zero();
            let mut jcount = 0u64;
            for p in 0..npix {
                let mp = m[p] as usize;
                if mp < 2 {
                    continue;
                }
                let ramp = &yb[p * ngroup..p * ngroup + mp];

                let (slope, var_r, var_p, varc) = if has_var {
                    let (r, gn) = (readnoise[p], gainmap[p]);
                    if !(r.is_finite() && gn.is_finite() && r > 0.0 && gn > 0.0) {
                        continue; // bad calibration pixel -> skip this integration
                    }
                    let jumps = detect_jumps(ramp, r, gn, jthresh);
                    jcount += jumps.len() as u64;
                    let mut bounds = Vec::with_capacity(jumps.len() + 2);
                    bounds.push(0usize);
                    bounds.extend_from_slice(&jumps);
                    bounds.push(mp);
                    let mut segs = Vec::with_capacity(bounds.len());
                    for w in bounds.windows(2) {
                        if w[1] - w[0] >= 2 {
                            if let Some(s) = weighted_seg_fit(&ramp[w[0]..w[1]], r, gn, tgroup, &wtab) {
                                segs.push(s);
                            }
                        }
                    }
                    match combine_inv_var(&segs) {
                        Some((s, vr, vp)) => (s, vr, vp, vr + vp),
                        None => continue,
                    }
                } else {
                    // Uniform-weight single-segment OLS, simple mean across integrations.
                    let t = &wtab[0][mp];
                    let (mut swy, mut swxy) = (0.0, 0.0);
                    for (g, &y) in ramp.iter().enumerate() {
                        let wy = t.w[g] * y;
                        swy += wy;
                        swxy += g as f64 * wy;
                    }
                    (((t.sw * swxy - t.swx * swy) / t.denom / tgroup), 0.0, 0.0, 1.0)
                };
                if !(varc.is_finite() && varc > 0.0) {
                    continue;
                }
                let w = 1.0 / varc;
                a_sw[p] += w;
                a_sws[p] += w * slope;
                a_2r[p] += w * w * var_r;
                a_2p[p] += w * w * var_p;
                a_cnt[p] += 1;
            }
            njumps.fetch_add(jcount, Ordering::Relaxed);
            (a_sw, a_sws, a_2r, a_2p, a_cnt)
        })
        .reduce(zero, |mut a, b| {
            for p in 0..npix {
                a.0[p] += b.0[p];
                a.1[p] += b.1[p];
                a.2[p] += b.2[p];
                a.3[p] += b.3[p];
                a.4[p] += b.4[p];
            }
            a
        });

    // Final maps. gain_scale multiplies the rate by GAINFACT (variance by GAINFACT²).
    let gf2 = gainfact * gainfact;
    let mut rate_map = vec![f64::NAN; npix];
    let mut var_rn = vec![f64::NAN; npix];
    let mut var_po = vec![f64::NAN; npix];
    for p in 0..npix {
        if cnt[p] > 0 && sw[p] > 0.0 {
            rate_map[p] = gainfact * sws[p] / sw[p];
            if has_var {
                let sw2 = sw[p] * sw[p];
                var_rn[p] = gf2 * sw2r[p] / sw2;
                var_po[p] = gf2 * sw2p[p] / sw2;
            }
        }
    }
    let elapsed = t0.elapsed();

    let mut sorted: Vec<f64> = rate_map.iter().copied().filter(|x| x.is_finite()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = sorted[sorted.len() / 2];
    println!("\nRate: min={:.3} median={med:.4} max={:.3} DN/s  ({} px no ramp)", sorted[0], sorted.last().unwrap(), npix - sorted.len());
    let samples = (npix * ngroup * nint) as f64;
    println!("Computed in {:.2?}  ({:.0} Msamples/s over {} threads){}",
        elapsed, samples / elapsed.as_secs_f64() / 1e6, rayon::current_num_threads(),
        if gainfact != 1.0 { format!(", GAINFACT={gainfact}") } else { String::new() });
    if has_var {
        let nj = njumps.load(Ordering::Relaxed);
        println!("Jumps detected: {nj} (thresh={jthresh}sigma, ~{:.3} per integration-pixel)", nj as f64 / (nint * npix) as f64);
    }

    if let Some(rp) = rate {
        compare_rate(&rp, &rate_map, nrow, ncol);
        if has_var {
            compare_plane(&rp, "VAR_RNOISE", &var_rn);
            compare_plane(&rp, "VAR_POISSON", &var_po);
        }
    }
}

/// Helper to log + load an optional reference, or its default.
fn load_opt<F: FnOnce() -> Vec<f64>, G: FnOnce() -> Vec<f64>>(opt: &Option<String>, label: &str, load: F, default: G) -> Vec<f64> {
    match opt {
        Some(p) => { println!("{label}: {p}"); load() }
        None => { println!("{label}: off"); default() }
    }
}

fn kw_int(f: &Fits, key: &str) -> Option<i64> {
    f.hdus[0].int(key).or_else(|| f.find("SCI").and_then(|h| h.int(key)))
}

fn subarray_offset(reff: &Fits, sci_f: &Fits) -> (usize, usize) {
    let row = (kw_int(sci_f, "SUBSTRT2").unwrap_or(1) - kw_int(reff, "SUBSTRT2").unwrap_or(1)) as usize;
    let col = (kw_int(sci_f, "SUBSTRT1").unwrap_or(1) - kw_int(reff, "SUBSTRT1").unwrap_or(1)) as usize;
    (row, col)
}

fn load_2d(ref_path: &str, ext: &str, sci_f: &Fits, nrow: usize, ncol: usize, fill: f64) -> Vec<f64> {
    let rf = Fits::open(ref_path).expect("failed to open reference");
    let r = rf.find(ext).expect("missing extension in reference");
    assert_eq!(r.int("BITPIX"), Some(-32), "reference must be float32");
    let rncol = r.int("NAXIS1").unwrap() as usize;
    let (row_off, col_off) = subarray_offset(&rf, sci_f);
    let bytes = &rf.mmap[r.data_offset..r.data_offset + r.data_len];
    let mut out = vec![fill; nrow * ncol];
    for rr in 0..nrow {
        for cc in 0..ncol {
            let b = ((row_off + rr) * rncol + (col_off + cc)) * 4;
            let v = f32::from_be_bytes([bytes[b], bytes[b + 1], bytes[b + 2], bytes[b + 3]]) as f64;
            out[rr * ncol + cc] = if v.is_finite() { v } else { fill };
        }
    }
    out
}

fn load_linearity(ref_path: &str, sci_f: &Fits, nrow: usize, ncol: usize) -> (Vec<f64>, usize) {
    let rf = Fits::open(ref_path).expect("failed to open linearity reference");
    let r = rf.find("COEFFS").expect("no COEFFS extension");
    assert_eq!(r.int("BITPIX"), Some(-32), "linearity must be float32");
    let rncol = r.int("NAXIS1").unwrap() as usize;
    let rnrow = r.int("NAXIS2").unwrap() as usize;
    let ncoeff = r.int("NAXIS3").unwrap() as usize;
    let (row_off, col_off) = subarray_offset(&rf, sci_f);
    let bytes = &rf.mmap[r.data_offset..r.data_offset + r.data_len];
    let getv = |k: usize, rr: usize, cc: usize| -> f64 {
        let b = ((k * rnrow + (row_off + rr)) * rncol + (col_off + cc)) * 4;
        f32::from_be_bytes([bytes[b], bytes[b + 1], bytes[b + 2], bytes[b + 3]]) as f64
    };
    let mut out = vec![0f64; nrow * ncol * ncoeff];
    for rr in 0..nrow {
        for cc in 0..ncol {
            for k in 0..ncoeff {
                let v = getv(k, rr, cc);
                out[(rr * ncol + cc) * ncoeff + k] = if v.is_finite() { v } else { 0.0 };
            }
        }
    }
    (out, ncoeff)
}

fn load_dark(ref_path: &str, sci_f: &Fits, nrow: usize, ncol: usize, ngroup: usize) -> Vec<f64> {
    let rf = Fits::open(ref_path).expect("failed to open dark reference");
    let r = rf.find("SCI").expect("no SCI in dark reference");
    assert_eq!(r.int("BITPIX"), Some(-32), "dark must be float32");
    let rncol = r.int("NAXIS1").unwrap() as usize;
    let rnrow = r.int("NAXIS2").unwrap() as usize;
    let dark_ng = r.int("NAXIS3").unwrap() as usize;
    let (row_off, col_off) = subarray_offset(&rf, sci_f);
    let bytes = &rf.mmap[r.data_offset..r.data_offset + r.data_len];
    let getv = |g: usize, rr: usize, cc: usize| -> f64 {
        let b = ((g * rnrow + (row_off + rr)) * rncol + (col_off + cc)) * 4;
        f32::from_be_bytes([bytes[b], bytes[b + 1], bytes[b + 2], bytes[b + 3]]) as f64
    };
    let npix = nrow * ncol;
    let mut out = vec![0f64; ngroup * npix];
    for g in 0..ngroup {
        let gd = g.min(dark_ng - 1);
        for rr in 0..nrow {
            for cc in 0..ncol {
                let v = getv(gd, rr, cc);
                out[g * npix + rr * ncol + cc] = if v.is_finite() { v } else { 0.0 };
            }
        }
    }
    out
}

fn median(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if v.is_empty() { f64::NAN } else { v[v.len() / 2] }
}

/// Read a 2D float32 extension from a FITS file into a Vec.
fn read_plane(ref_path: &str, ext: &str) -> (Vec<f64>, usize) {
    let rf = Fits::open(ref_path).expect("failed to open reference FITS");
    let h: &Hdu = rf.find(ext).unwrap_or_else(|| panic!("no {ext} extension"));
    assert_eq!(h.int("BITPIX"), Some(-32));
    let n = (h.int("NAXIS1").unwrap() * h.int("NAXIS2").unwrap()) as usize;
    let bytes = &rf.mmap[h.data_offset..h.data_offset + h.data_len];
    let out = (0..n)
        .map(|p| {
            let b = p * 4;
            f32::from_be_bytes([bytes[b], bytes[b + 1], bytes[b + 2], bytes[b + 3]]) as f64
        })
        .collect();
    (out, n)
}

fn compare_rate(ref_path: &str, ours: &[f64], nrow: usize, ncol: usize) {
    let (refv, _) = read_plane(ref_path, "SCI");
    let npix = nrow * ncol;
    let bins = [(1.0, 5.0), (5.0, 20.0), (20.0, 100.0), (100.0, 500.0), (500.0, f64::INFINITY)];
    let mut ratios: Vec<Vec<f64>> = vec![Vec::new(); bins.len()];
    let mut bg_abs = Vec::new();
    let mut nan = 0usize;
    for p in 0..npix {
        let r = refv[p];
        if !r.is_finite() || !ours[p].is_finite() {
            nan += 1;
            continue;
        }
        if r.abs() < 1.0 {
            bg_abs.push((ours[p] - r).abs());
        }
        for (bi, (lo, hi)) in bins.iter().enumerate() {
            if r >= *lo && r < *hi {
                ratios[bi].push(ours[p] / r);
            }
        }
    }
    println!("\n--- Validation vs official rate.fits (SCI) ---");
    println!("  skipped (NaN either side): {nan}/{npix}");
    println!("  background (|ref|<1): median |ours-ref| = {:.4} DN/s", median(&mut bg_abs));
    println!("  median ours/ref by official-rate bin:");
    for (bi, (lo, hi)) in bins.iter().enumerate() {
        let n = ratios[bi].len();
        let hilbl = if hi.is_finite() { format!("{hi:.0}") } else { "inf".into() };
        println!("    {lo:>4.0}-{hilbl:<4} DN/s : {n:>6} px  ratio={:.4}", median(&mut ratios[bi]));
    }
}

fn compare_plane(ref_path: &str, ext: &str, ours: &[f64]) {
    let (refv, _) = read_plane(ref_path, ext);
    let mut ratios: Vec<f64> = (0..ours.len())
        .filter_map(|p| {
            let (o, r) = (ours[p], refv[p]);
            if o.is_finite() && r.is_finite() && r > 0.0 && o > 0.0 { Some(o / r) } else { None }
        })
        .collect();
    let n = ratios.len();
    println!("  {ext}: {n} px  median ours/ref = {:.4}", median(&mut ratios));
}
