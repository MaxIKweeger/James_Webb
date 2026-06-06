//! Output writers for the computed maps: a multi-extension FITS product, a PNG
//! visualization of the rate map, and a raw little-endian f32 dump for numpy.

use std::fs::File;
use std::io::{BufWriter, Write};

/// Write all three outputs: `<prefix>.fits`, `<prefix>.png`, `<prefix>.f32`.
/// `var_rn`/`var_po` may be empty (simple kernel without variances).
pub fn write_all(prefix: &str, rate: &[f64], var_rn: &[f64], var_po: &[f64], nrow: usize, ncol: usize) {
    let has_var = !var_rn.is_empty() && !var_po.is_empty();
    write_fits(&format!("{prefix}.fits"), rate, var_rn, var_po, nrow, ncol, has_var);
    write_png(&format!("{prefix}.png"), rate, nrow, ncol);
    write_raw(&format!("{prefix}.f32"), rate);
    println!(
        "\nWrote {prefix}.fits ({}), {prefix}.png ({nrow}x{ncol}), {prefix}.f32 (little-endian f32)",
        if has_var { "SCI+ERR+VAR_POISSON+VAR_RNOISE" } else { "SCI" }
    );
}

// --- FITS -----------------------------------------------------------------

fn card(buf: &mut Vec<u8>, s: &str) {
    let mut c = [b' '; 80];
    let b = s.as_bytes();
    let n = b.len().min(80);
    c[..n].copy_from_slice(&b[..n]);
    buf.extend_from_slice(&c);
}

fn pad_2880(buf: &mut Vec<u8>, fill: u8) {
    while buf.len() % 2880 != 0 {
        buf.push(fill);
    }
}

fn write_fits(path: &str, rate: &[f64], var_rn: &[f64], var_po: &[f64], nrow: usize, ncol: usize, has_var: bool) {
    let f = File::create(path).expect("create fits");
    let mut w = BufWriter::new(f);

    // Primary HDU: no data.
    let mut hdr = Vec::new();
    card(&mut hdr, "SIMPLE  =                    T / conforms to FITS standard");
    card(&mut hdr, "BITPIX  =                    8");
    card(&mut hdr, "NAXIS   =                    0");
    card(&mut hdr, "EXTEND  =                    T");
    card(&mut hdr, "ORIGIN  = 'James Webb Ramp-Fit (Rust)'");
    card(&mut hdr, "END");
    pad_2880(&mut hdr, b' ');
    w.write_all(&hdr).unwrap();

    // Image extensions.
    let err: Vec<f64> = if has_var {
        (0..rate.len()).map(|p| (var_rn[p] + var_po[p]).sqrt()).collect()
    } else {
        Vec::new()
    };
    let mut planes: Vec<(&str, &[f64])> = vec![("SCI", rate)];
    if has_var {
        planes.push(("ERR", &err));
        planes.push(("VAR_POISSON", var_po));
        planes.push(("VAR_RNOISE", var_rn));
    }

    for (name, data) in planes {
        let mut h = Vec::new();
        card(&mut h, "XTENSION= 'IMAGE   '           / Image extension");
        card(&mut h, "BITPIX  =                  -32");
        card(&mut h, "NAXIS   =                    2");
        card(&mut h, &format!("NAXIS1  = {:>20}", ncol));
        card(&mut h, &format!("NAXIS2  = {:>20}", nrow));
        card(&mut h, "PCOUNT  =                    0");
        card(&mut h, "GCOUNT  =                    1");
        card(&mut h, &format!("EXTNAME = '{:<8}'", name));
        card(&mut h, "BUNIT   = 'DN/s    '");
        card(&mut h, "END");
        pad_2880(&mut h, b' ');
        w.write_all(&h).unwrap();

        // Data: big-endian float32, row-major (NAXIS1 fastest).
        let mut d = Vec::with_capacity(data.len() * 4);
        for &v in data {
            d.extend_from_slice(&(v as f32).to_be_bytes());
        }
        pad_2880(&mut d, 0);
        w.write_all(&d).unwrap();
    }
    w.flush().unwrap();
}

// --- PNG (grayscale, percentile-stretched) --------------------------------

fn write_png(path: &str, rate: &[f64], nrow: usize, ncol: usize) {
    // Robust contrast: clip to the 1st/99th percentiles of finite pixels.
    let mut finite: Vec<f64> = rate.iter().copied().filter(|x| x.is_finite()).collect();
    finite.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (lo, hi) = if finite.len() > 1 {
        (finite[finite.len() / 100], finite[finite.len() * 99 / 100])
    } else {
        (0.0, 1.0)
    };
    let span = if hi > lo { hi - lo } else { 1.0 };

    let px: Vec<u8> = rate
        .iter()
        .map(|&v| {
            if !v.is_finite() {
                0
            } else {
                (((v - lo) / span).clamp(0.0, 1.0) * 255.0) as u8
            }
        })
        .collect();

    let f = File::create(path).expect("create png");
    let w = BufWriter::new(f);
    let mut enc = png::Encoder::new(w, ncol as u32, nrow as u32);
    enc.set_color(png::ColorType::Grayscale);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header().expect("png header");
    writer.write_image_data(&px).expect("png data");
}

// --- Raw little-endian f32 (numpy: np.fromfile(dtype='<f4')) ---------------

fn write_raw(path: &str, rate: &[f64]) {
    let f = File::create(path).expect("create raw");
    let mut w = BufWriter::new(f);
    let mut buf = Vec::with_capacity(rate.len() * 4);
    for &v in rate {
        buf.extend_from_slice(&(v as f32).to_le_bytes());
    }
    w.write_all(&buf).unwrap();
    w.flush().unwrap();
}
