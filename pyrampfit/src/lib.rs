//! Rust-accelerated JWST Stage-1 ramp fitting, exposed to Python via PyO3.
//!
//! This is the algorithm core validated in the parent project, reshaped to be a
//! drop-in accelerator for `stcal.ramp_fitting`: it takes the (already calibrated)
//! ramp cube plus the per-group DQ flags that earlier pipeline steps set, and
//! returns the same products — slope, var_poisson, var_rnoise, err.
//!
//! Inputs (numpy, C-contiguous):
//!   data    : float32 [nints, ngroups, nrows, ncols]  calibrated ramp (DN)
//!   groupdq : uint8   [nints, ngroups, nrows, ncols]  group DQ flags
//!   gain    : float32 [nrows, ncols]                  e-/DN
//!   readnoise: float32 [nrows, ncols]                 CDS read noise (DN)
//!   group_time : seconds between groups
//!
//! DQ bits (jwst dqflags): DO_NOT_USE=1, SATURATED=2, JUMP_DET=4. Groups flagged
//! DO_NOT_USE/SATURATED are dropped; a JUMP_DET group starts a new ramp segment.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyReadonlyArray2, PyReadonlyArray4};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

const DO_NOT_USE: u8 = 1;
const SATURATED: u8 = 2;
const JUMP_DET: u8 = 4;

/// Fixsen (2000) optimal-weighting power, selected by segment SNR.
fn power(data: f64, rn: f64, gain: f64) -> f64 {
    if data <= 0.0 {
        return 0.0;
    }
    let s = data * gain / (rn * rn + data * gain).sqrt();
    match s {
        s if s < 5.0 => 0.0,
        s if s < 10.0 => 0.4,
        s if s < 20.0 => 1.0,
        s if s < 50.0 => 3.0,
        s if s < 100.0 => 6.0,
        _ => 10.0,
    }
}

/// Optimal-weight fit of one contiguous segment. Returns (slope DN/s, var_R, var_P).
fn seg_fit(vals: &[f64], rn: f64, gain: f64, gt: f64) -> Option<(f64, f64, f64)> {
    let n = vals.len();
    if n < 2 {
        return None;
    }
    let p = power(vals[n - 1] - vals[0], rn, gain);
    let imid = (n as f64 - 1.0) / 2.0;
    let (mut sw, mut swx, mut swy, mut swxx, mut swxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (g, &y) in vals.iter().enumerate() {
        let x = g as f64;
        let w = if p == 0.0 { 1.0 } else { ((x - imid).abs() / imid).powf(p) };
        sw += w;
        swx += w * x;
        swy += w * y;
        swxx += w * x * x;
        swxy += w * x * y;
    }
    let denom = sw * swxx - swx * swx;
    if denom == 0.0 {
        return None;
    }
    let slope = (sw * swxy - swx * swy) / denom / gt;
    let m3 = (n * n * n - n) as f64;
    // CRDS read noise is CDS noise (= sqrt(2) x single-read), hence R^2 / 2.
    let var_r = 12.0 * (rn * rn / 2.0) / (m3 * gt * gt);
    let var_p = slope.max(0.0) / (gain * gt * (n as f64 - 1.0));
    Some((slope, var_r, var_p))
}

/// Accumulate one segment into inverse-variance sums (sw, sws, s2r, s2p).
fn accum(seg: &[f64], rn: f64, gain: f64, gt: f64, acc: &mut (f64, f64, f64, f64)) {
    if let Some((sl, vr, vp)) = seg_fit(seg, rn, gain, gt) {
        let vc = vr + vp;
        if vc.is_finite() && vc > 0.0 {
            let w = 1.0 / vc;
            acc.0 += w;
            acc.1 += w * sl;
            acc.2 += w * w * vr;
            acc.3 += w * w * vp;
        }
    }
}

#[pyfunction]
fn fit_ramps<'py>(
    py: Python<'py>,
    data: PyReadonlyArray4<'py, f32>,
    groupdq: PyReadonlyArray4<'py, u8>,
    gain: PyReadonlyArray2<'py, f32>,
    readnoise: PyReadonlyArray2<'py, f32>,
    group_time: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let d = data.as_array();
    let q = groupdq.as_array();
    let gainv = gain.as_array();
    let rnv = readnoise.as_array();

    let (nints, ngroups, nrows, ncols) = (d.shape()[0], d.shape()[1], d.shape()[2], d.shape()[3]);
    let npix = nrows * ncols;
    let gt = group_time;

    // Per-pixel ramp fit (parallel). Returns (slope, var_poisson, var_rnoise, err).
    let res: Vec<(f32, f32, f32, f32)> = (0..npix)
        .into_par_iter()
        .map(|pix| {
            let r = pix / ncols;
            let c = pix % ncols;
            let gain_p = gainv[[r, c]] as f64;
            let rn_p = rnv[[r, c]] as f64;
            let nan = (f32::NAN, f32::NAN, f32::NAN, f32::NAN);
            if !(gain_p > 0.0 && rn_p > 0.0) {
                return nan;
            }

            let mut e = (0.0f64, 0.0f64, 0.0f64, 0.0f64); // exposure-level sums
            let mut seg: Vec<f64> = Vec::with_capacity(ngroups);
            for i in 0..nints {
                let mut acc = (0.0f64, 0.0f64, 0.0f64, 0.0f64); // integration-level
                seg.clear();
                for g in 0..ngroups {
                    let flag = q[[i, g, r, c]];
                    if flag & (DO_NOT_USE | SATURATED) != 0 {
                        accum(&seg, rn_p, gain_p, gt, &mut acc);
                        seg.clear();
                        continue;
                    }
                    if flag & JUMP_DET != 0 && !seg.is_empty() {
                        accum(&seg, rn_p, gain_p, gt, &mut acc);
                        seg.clear();
                    }
                    seg.push(d[[i, g, r, c]] as f64);
                }
                accum(&seg, rn_p, gain_p, gt, &mut acc);
                seg.clear();

                if acc.0 > 0.0 {
                    let sl = acc.1 / acc.0;
                    let vr = acc.2 / (acc.0 * acc.0);
                    let vp = acc.3 / (acc.0 * acc.0);
                    let vc = vr + vp;
                    if vc > 0.0 {
                        let w = 1.0 / vc;
                        e.0 += w;
                        e.1 += w * sl;
                        e.2 += w * w * vr;
                        e.3 += w * w * vp;
                    }
                }
            }

            if e.0 > 0.0 {
                let slope = e.1 / e.0;
                let vr = e.2 / (e.0 * e.0);
                let vp = e.3 / (e.0 * e.0);
                (slope as f32, vp as f32, vr as f32, (vr + vp).sqrt() as f32)
            } else {
                nan
            }
        })
        .collect();

    let mut slope = Array2::<f32>::from_elem((nrows, ncols), f32::NAN);
    let mut var_p = Array2::<f32>::from_elem((nrows, ncols), f32::NAN);
    let mut var_r = Array2::<f32>::from_elem((nrows, ncols), f32::NAN);
    let mut err = Array2::<f32>::from_elem((nrows, ncols), f32::NAN);
    for (pix, &(s, vp, vr, e)) in res.iter().enumerate() {
        let (r, c) = (pix / ncols, pix % ncols);
        slope[[r, c]] = s;
        var_p[[r, c]] = vp;
        var_r[[r, c]] = vr;
        err[[r, c]] = e;
    }

    let out = PyDict::new_bound(py);
    out.set_item("slope", slope.into_pyarray_bound(py))?;
    out.set_item("var_poisson", var_p.into_pyarray_bound(py))?;
    out.set_item("var_rnoise", var_r.into_pyarray_bound(py))?;
    out.set_item("err", err.into_pyarray_bound(py))?;
    Ok(out)
}

#[pymodule]
fn jwst_rampfit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit_ramps, m)?)?;
    m.add("DO_NOT_USE", DO_NOT_USE)?;
    m.add("SATURATED", SATURATED)?;
    m.add("JUMP_DET", JUMP_DET)?;
    Ok(())
}
