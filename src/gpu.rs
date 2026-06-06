//! GPU port of the calibrated OLS slope (phase-2 equivalent) using wgpu compute.
//!
//! One thread per pixel. Consecutive threads read consecutive memory for a fixed
//! (integration, group), so the natural cube layout gives coalesced loads. The
//! 1.3 GB cube exceeds GPU buffer-binding limits, so we stream it in batches of
//! integrations, accumulating per-pixel slope sums across dispatches (dispatches
//! are serialized, so the read-modify-write of the accumulators needs no atomics).
//!
//! This first kernel matches the phase-2 CPU result (uniform-weight OLS, simple
//! mean over integrations); Fixsen weights / variances / jumps are not ported yet.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    npix: u32,
    ngroup: u32,
    batch_nint: u32,
    ncoeff: u32,
    has_lin: u32,
    has_dark: u32,
    tgroup: f32,
    jthresh: f32,
}

/// Params for the weighted/raw kernels, which decode packed i16 on the GPU and so
/// need BZERO/BSCALE. 48 bytes (16-byte aligned for a uniform buffer).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ParamsW {
    npix: u32,
    ngroup: u32,
    batch_nint: u32,
    ncoeff: u32,
    has_lin: u32,
    has_dark: u32,
    tgroup: f32,
    jthresh: f32,
    bzero: f32,
    bscale: f32,
    _p0: u32,
    _p1: u32,
}

const WGSL: &str = r#"
struct Params {
  npix: u32, ngroup: u32, batch_nint: u32, ncoeff: u32,
  has_lin: u32, has_dark: u32, tgroup: f32, _pad: u32,
};
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> sat: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read> coeffs: array<f32>;
@group(0) @binding(4) var<storage, read> dark: array<f32>;
@group(0) @binding(5) var<storage, read> ramp: array<f32>;
@group(0) @binding(6) var<storage, read_write> acc_slope: array<f32>;
@group(0) @binding(7) var<storage, read_write> acc_count: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let p = gid.x;
  if (p >= P.npix) { return; }
  let npix = P.npix;
  let ng = P.ngroup;
  let satp = sat[p];
  let biasp = bias[p];
  var acc = acc_slope[p];
  var cnt = acc_count[p];

  for (var it = 0u; it < P.batch_nint; it = it + 1u) {
    var m = 0u;
    var sy = 0.0;
    var sxy = 0.0;
    var done = false;
    for (var g = 0u; g < ng; g = g + 1u) {
      if (done) { continue; }
      let v = ramp[(it * ng + g) * npix + p];
      if (v >= satp) { done = true; continue; }
      let x = v - biasp;
      var y = x;
      if (P.has_lin == 1u) {
        let off = p * P.ncoeff;
        var a = coeffs[off + P.ncoeff - 1u];
        for (var k: i32 = i32(P.ncoeff) - 2; k >= 0; k = k - 1) {
          a = a * x + coeffs[off + u32(k)];
        }
        y = a;
      }
      if (P.has_dark == 1u) {
        y = y - dark[g * npix + p];
      }
      let fg = f32(m);
      sy = sy + y;
      sxy = sxy + fg * y;
      m = m + 1u;
    }
    if (m >= 2u) {
      let fm = f32(m);
      let sx = (fm - 1.0) * fm * 0.5;
      let sxx = (fm - 1.0) * fm * (2.0 * fm - 1.0) / 6.0;
      let denom = fm * sxx - sx * sx;
      let slope = (fm * sxy - sx * sy) / denom / P.tgroup;
      acc = acc + slope;
      cnt = cnt + 1u;
    }
  }
  acc_slope[p] = acc;
  acc_count[p] = cnt;
}
"#;

#[allow(clippy::too_many_arguments)]
pub fn run_gpu(
    data: &[u8], // SCI cube bytes (BITPIX=16, big-endian)
    npix: usize,
    nrow: usize,
    ncol: usize,
    ngroup: usize,
    nint: usize,
    bzero: f64,
    bscale: f64,
    tgroup: f64,
    gainfact: f64,
    saturation: &[f64],
    superbias: &[f64],
    coeffs: &[f64],
    ncoeff: usize,
    dark_ramp: &[f64],
) -> (Vec<f64>, f64) {
    let _ = (nrow, ncol);
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("no GPU adapter found");
    println!("[gpu] adapter: {}", adapter.get_info().name);

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("ramp-fit"),
        required_features: wgpu::Features::empty(),
        required_limits: adapter.limits(),
        memory_hints: wgpu::MemoryHints::Performance,
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        trace: wgpu::Trace::Off,
    }))
    .expect("failed to get GPU device");

    // Batch size: keep the streamed ramp buffer under ~256 MB.
    let cap_bytes = 256usize * 1024 * 1024;
    let per_int = ngroup * npix * 4;
    let batch_max = (cap_bytes / per_int).clamp(1, nint);
    let ramp_floats = batch_max * ngroup * npix;
    println!("[gpu] streaming {nint} integrations in batches of {batch_max} ({} MB/batch)", ramp_floats * 4 / (1024 * 1024));

    let jt = 0.0f32;
    let as_f32 = |v: &[f64]| -> Vec<f32> { v.iter().map(|&x| x as f32).collect() };
    let storage = wgpu::BufferUsages::STORAGE;

    let buf_sat = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sat"), contents: bytemuck::cast_slice(&as_f32(saturation)), usage: storage });
    let buf_bias = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("bias"), contents: bytemuck::cast_slice(&as_f32(superbias)), usage: storage });
    let coeffs_f32 = if coeffs.is_empty() { vec![0f32] } else { as_f32(coeffs) };
    let buf_coeffs = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("coeffs"), contents: bytemuck::cast_slice(&coeffs_f32), usage: storage });
    let dark_f32 = if dark_ramp.is_empty() { vec![0f32] } else { as_f32(dark_ramp) };
    let buf_dark = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("dark"), contents: bytemuck::cast_slice(&dark_f32), usage: storage });

    let buf_ramp = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ramp"), size: (ramp_floats * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

    let zeros_f = vec![0f32; npix];
    let zeros_u = vec![0u32; npix];
    let buf_acc_s = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("acc_slope"), contents: bytemuck::cast_slice(&zeros_f),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC });
    let buf_acc_c = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("acc_count"), contents: bytemuck::cast_slice(&zeros_u),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC });

    let buf_params = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("params"), size: std::mem::size_of::<Params>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("rampfit"), source: wgpu::ShaderSource::Wgsl(WGSL.into()) });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("rampfit"), layout: None, module: &shader,
        entry_point: Some("main"), compilation_options: Default::default(), cache: None });

    let bgl = pipeline.get_bind_group_layout(0);
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg"), layout: &bgl, entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_sat.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_bias.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: buf_coeffs.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: buf_dark.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: buf_ramp.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: buf_acc_s.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: buf_acc_c.as_entire_binding() },
        ] });

    // Convert one batch of integrations from raw big-endian i16 to physical f32.
    let raw = |idx: usize| -> f32 {
        let b = idx * 2;
        (bzero + bscale * (i16::from_be_bytes([data[b], data[b + 1]]) as f64)) as f32
    };
    let mut host = vec![0f32; ramp_floats];

    let t0 = std::time::Instant::now();
    let (mut t_fill, mut t_gpu) = (0f64, 0f64);
    let mut done = 0usize;
    while done < nint {
        let bn = batch_max.min(nint - done);
        // Fill host[(it*ngroup+g)*npix + p] (CPU i16 -> physical f32).
        let tf = std::time::Instant::now();
        host.par_fill_batch(done, bn, ngroup, npix, &raw);
        t_fill += tf.elapsed().as_secs_f64();
        let tg = std::time::Instant::now();
        queue.write_buffer(&buf_ramp, 0, bytemuck::cast_slice(&host[..bn * ngroup * npix]));
        let params = Params {
            npix: npix as u32, ngroup: ngroup as u32, batch_nint: bn as u32,
            ncoeff: ncoeff as u32, has_lin: (ncoeff > 0) as u32,
            has_dark: (!dark_ramp.is_empty()) as u32, tgroup: tgroup as f32, jthresh: jt };
        queue.write_buffer(&buf_params, 0, bytemuck::bytes_of(&params));

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind, &[]);
            let groups = npix.div_ceil(256) as u32;
            pass.dispatch_workgroups(groups, 1, 1);
        }
        queue.submit([enc.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        t_gpu += tg.elapsed().as_secs_f64();
        done += bn;
    }

    // Read back accumulators.
    let (rs, rc) = readback(&device, &queue, &buf_acc_s, &buf_acc_c, npix);
    let elapsed = t0.elapsed().as_secs_f64();
    println!("[gpu] breakdown: i16->f32 convert {t_fill:.2}s, upload+dispatch+poll {t_gpu:.2}s");

    let rate: Vec<f64> = (0..npix)
        .map(|p| if rc[p] > 0 { gainfact * rs[p] as f64 / rc[p] as f64 } else { f64::NAN })
        .collect();
    (rate, elapsed)
}

/// Phase-3a kernel: Fixsen optimal weighting + read-noise/Poisson variances +
/// inverse-variance combination. Two passes over the groups (the 2nd recomputes
/// calibration) avoid a per-thread local ramp buffer. No jump detection (3b).
const WGSL_W: &str = r#"
struct Params {
  npix: u32, ngroup: u32, batch_nint: u32, ncoeff: u32,
  has_lin: u32, has_dark: u32, tgroup: f32, jthresh: f32,
  bzero: f32, bscale: f32, _p0: u32, _p1: u32,
};
@group(0) @binding(0) var<uniform> U: Params;
@group(0) @binding(1) var<storage, read> sat: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read> coeffs: array<f32>;
@group(0) @binding(4) var<storage, read> dark: array<f32>;
@group(0) @binding(5) var<storage, read> raw: array<u32>;
@group(0) @binding(6) var<storage, read> rn: array<f32>;
@group(0) @binding(7) var<storage, read> gain: array<f32>;
@group(0) @binding(8) var<storage, read_write> acc_sw: array<f32>;
@group(0) @binding(9) var<storage, read_write> acc_sws: array<f32>;
@group(0) @binding(10) var<storage, read_write> acc_2r: array<f32>;
@group(0) @binding(11) var<storage, read_write> acc_2p: array<f32>;

fn calib(v: f32, p: u32, g: u32) -> f32 {
  let x = v - bias[p];
  var y = x;
  if (U.has_lin == 1u) {
    let off = p * U.ncoeff;
    var a = coeffs[off + U.ncoeff - 1u];
    for (var k: i32 = i32(U.ncoeff) - 2; k >= 0; k = k - 1) {
      a = a * x + coeffs[off + u32(k)];
    }
    y = a;
  }
  if (U.has_dark == 1u) { y = y - dark[g * U.npix + p]; }
  return y;
}

fn pexp(data: f32, r: f32, gn: f32) -> f32 {
  if (data <= 0.0) { return 0.0; }
  let s = data * gn / sqrt(r * r + data * gn);
  if (s < 5.0) { return 0.0; }
  if (s < 10.0) { return 0.4; }
  if (s < 20.0) { return 1.0; }
  if (s < 50.0) { return 3.0; }
  if (s < 100.0) { return 6.0; }
  return 10.0;
}

fn sample(s: u32) -> f32 {
  let word = raw[s >> 1u];
  var half: u32;
  if ((s & 1u) == 0u) { half = word & 0xffffu; } else { half = word >> 16u; }
  let be = ((half & 0xffu) << 8u) | ((half >> 8u) & 0xffu); // big-endian -> value
  var iv: i32 = i32(be);
  if (be >= 0x8000u) { iv = iv - 65536; } // sign-extend 16-bit
  return U.bzero + U.bscale * f32(iv);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let p = gid.x;
  if (p >= U.npix) { return; }
  let npix = U.npix;
  let ng = U.ngroup;
  let satp = sat[p];
  let rp = rn[p];
  let gp = gain[p];
  let tg = U.tgroup;
  if (!(rp > 0.0 && gp > 0.0)) { return; }

  for (var it = 0u; it < U.batch_nint; it = it + 1u) {
    // Pass A: count usable groups, first & last calibrated values.
    var m = 0u; var first = 0.0; var last = 0.0; var done = false;
    for (var g = 0u; g < ng; g = g + 1u) {
      if (done) { continue; }
      let v = sample((it * ng + g) * npix + p);
      if (v >= satp) { done = true; continue; }
      let y = calib(v, p, g);
      if (m == 0u) { first = y; }
      last = y;
      m = m + 1u;
    }
    if (m < 2u) { continue; }
    let pw = pexp(last - first, rp, gp);
    let imid = (f32(m) - 1.0) * 0.5;

    // Pass B: weighted regression sums.
    var sw = 0.0; var swx = 0.0; var swy = 0.0; var swxx = 0.0; var swxy = 0.0;
    var idx = 0u; done = false;
    for (var g = 0u; g < ng; g = g + 1u) {
      if (done) { continue; }
      let v = sample((it * ng + g) * npix + p);
      if (v >= satp) { done = true; continue; }
      let y = calib(v, p, g);
      let fg = f32(idx);
      var w = 1.0;
      if (pw != 0.0) { w = pow(abs(fg - imid) / imid, pw); }
      sw = sw + w; swx = swx + w * fg; swy = swy + w * y;
      swxx = swxx + w * fg * fg; swxy = swxy + w * fg * y;
      idx = idx + 1u;
    }
    let denom = sw * swxx - swx * swx;
    let slope = (sw * swxy - swx * swy) / denom / tg;
    let fm = f32(m);
    let m3 = fm * fm * fm - fm;
    let var_r = 12.0 * (rp * rp / 2.0) / (m3 * tg * tg);
    let var_p = max(slope, 0.0) / (gp * tg * (fm - 1.0));
    let varc = var_r + var_p;
    if (varc > 0.0) {
      let wv = 1.0 / varc;
      acc_sw[p] = acc_sw[p] + wv;
      acc_sws[p] = acc_sws[p] + wv * slope;
      acc_2r[p] = acc_2r[p] + wv * wv * var_r;
      acc_2p[p] = acc_2p[p] + wv * wv * var_p;
    }
  }
}
"#;

/// Phase-3b kernel: phase-3a + two-point-difference jump detection (MAD sigma)
/// with multi-segment fitting. Uses per-thread local arrays (≤80 groups).
const WGSL_J: &str = r#"
struct Params {
  npix: u32, ngroup: u32, batch_nint: u32, ncoeff: u32,
  has_lin: u32, has_dark: u32, tgroup: f32, jthresh: f32,
  bzero: f32, bscale: f32, _p0: u32, _p1: u32,
};
@group(0) @binding(0) var<uniform> U: Params;
@group(0) @binding(1) var<storage, read> sat: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read> coeffs: array<f32>;
@group(0) @binding(4) var<storage, read> dark: array<f32>;
@group(0) @binding(5) var<storage, read> raw: array<u32>;
@group(0) @binding(6) var<storage, read> rn: array<f32>;
@group(0) @binding(7) var<storage, read> gain: array<f32>;
@group(0) @binding(8) var<storage, read_write> acc_sw: array<f32>;
@group(0) @binding(9) var<storage, read_write> acc_sws: array<f32>;
@group(0) @binding(10) var<storage, read_write> acc_2r: array<f32>;
@group(0) @binding(11) var<storage, read_write> acc_2p: array<f32>;

fn calib(v: f32, p: u32, g: u32) -> f32 {
  let x = v - bias[p];
  var y = x;
  if (U.has_lin == 1u) {
    let off = p * U.ncoeff;
    var a = coeffs[off + U.ncoeff - 1u];
    for (var k: i32 = i32(U.ncoeff) - 2; k >= 0; k = k - 1) { a = a * x + coeffs[off + u32(k)]; }
    y = a;
  }
  if (U.has_dark == 1u) { y = y - dark[g * U.npix + p]; }
  return y;
}

fn pexp(data: f32, r: f32, gn: f32) -> f32 {
  if (data <= 0.0) { return 0.0; }
  let s = data * gn / sqrt(r * r + data * gn);
  if (s < 5.0) { return 0.0; }
  if (s < 10.0) { return 0.4; }
  if (s < 20.0) { return 1.0; }
  if (s < 50.0) { return 3.0; }
  if (s < 100.0) { return 6.0; }
  return 10.0;
}

fn isort(wk: ptr<function, array<f32, 80>>, na: u32) {
  for (var a = 1u; a < na; a = a + 1u) {
    let key = (*wk)[a];
    var b = a;
    loop {
      if (b == 0u) { break; }
      if ((*wk)[b - 1u] <= key) { break; }
      (*wk)[b] = (*wk)[b - 1u];
      b = b - 1u;
    }
    (*wk)[b] = key;
  }
}

fn sample(s: u32) -> f32 {
  let word = raw[s >> 1u];
  var half: u32;
  if ((s & 1u) == 0u) { half = word & 0xffffu; } else { half = word >> 16u; }
  let be = ((half & 0xffu) << 8u) | ((half >> 8u) & 0xffu);
  var iv: i32 = i32(be);
  if (be >= 0x8000u) { iv = iv - 65536; }
  return U.bzero + U.bscale * f32(iv);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let p = gid.x;
  if (p >= U.npix) { return; }
  let npix = U.npix;
  let ng = U.ngroup;
  let satp = sat[p];
  let rp = rn[p];
  let gp = gain[p];
  let tg = U.tgroup;
  let thr = U.jthresh;
  if (!(rp > 0.0 && gp > 0.0)) { return; }

  var ramp: array<f32, 80>;
  var excluded: array<u32, 80>;
  var wk: array<f32, 80>;

  for (var it = 0u; it < U.batch_nint; it = it + 1u) {
    // Pass A: build calibrated ramp.
    var m = 0u; var done = false;
    for (var g = 0u; g < ng; g = g + 1u) {
      if (done) { continue; }
      let v = sample((it * ng + g) * npix + p);
      if (v >= satp) { done = true; continue; }
      ramp[m] = calib(v, p, g);
      m = m + 1u;
    }
    if (m < 2u) { continue; }

    let nd = m - 1u;
    for (var i = 0u; i < nd; i = i + 1u) { excluded[i] = 0u; }

    // Jump detection (two-point difference, robust MAD sigma).
    if (m >= 3u && thr > 0.0) {
      for (var iter = 0u; iter < nd; iter = iter + 1u) {
        var na = 0u;
        for (var i = 0u; i < nd; i = i + 1u) {
          if (excluded[i] == 0u) { wk[na] = ramp[i + 1u] - ramp[i]; na = na + 1u; }
        }
        if (na < 2u) { break; }
        isort(&wk, na);
        let med = wk[na / 2u];
        for (var i = 0u; i < na; i = i + 1u) { wk[i] = abs(wk[i] - med); }
        isort(&wk, na);
        let mad = wk[na / 2u];
        let model = sqrt(rp * rp + max(med, 0.0) / gp);
        let sigma = max(1.4826 * mad, model);
        if (!(sigma > 0.0)) { break; }
        var bestr = thr;
        var besti = -1;
        for (var i = 0u; i < nd; i = i + 1u) {
          if (excluded[i] == 0u) {
            let ratio = (ramp[i + 1u] - ramp[i] - med) / sigma;
            if (ratio > bestr) { bestr = ratio; besti = i32(i); }
          }
        }
        if (besti < 0) { break; }
        excluded[u32(besti)] = 1u;
      }
    }

    // Fit each segment, combine segments by inverse variance.
    var segsw = 0.0; var segsws = 0.0; var seg2r = 0.0; var seg2p = 0.0;
    var segstart = 0u;
    for (var k = 1u; k <= m; k = k + 1u) {
      var boundary = (k == m);
      if (!boundary) { boundary = (excluded[k - 1u] == 1u); }
      if (boundary) {
        let seglen = k - segstart;
        if (seglen >= 2u) {
          let pw = pexp(ramp[k - 1u] - ramp[segstart], rp, gp);
          let imid = (f32(seglen) - 1.0) * 0.5;
          var sw = 0.0; var swx = 0.0; var swy = 0.0; var swxx = 0.0; var swxy = 0.0;
          for (var j = 0u; j < seglen; j = j + 1u) {
            let fg = f32(j); let y = ramp[segstart + j];
            var w = 1.0;
            if (pw != 0.0) { w = pow(abs(fg - imid) / imid, pw); }
            sw = sw + w; swx = swx + w * fg; swy = swy + w * y;
            swxx = swxx + w * fg * fg; swxy = swxy + w * fg * y;
          }
          let denom = sw * swxx - swx * swx;
          let slope = (sw * swxy - swx * swy) / denom / tg;
          let fl = f32(seglen); let m3 = fl * fl * fl - fl;
          let vr = 12.0 * (rp * rp / 2.0) / (m3 * tg * tg);
          let vp = max(slope, 0.0) / (gp * tg * (fl - 1.0));
          let vc = vr + vp;
          if (vc > 0.0) {
            let wv = 1.0 / vc;
            segsw = segsw + wv; segsws = segsws + wv * slope;
            seg2r = seg2r + wv * wv * vr; seg2p = seg2p + wv * wv * vp;
          }
        }
        segstart = k;
      }
    }
    if (segsw > 0.0) {
      let slope_int = segsws / segsw;
      let vr_int = seg2r / (segsw * segsw);
      let vp_int = seg2p / (segsw * segsw);
      let vc_int = vr_int + vp_int;
      if (vc_int > 0.0) {
        let wvi = 1.0 / vc_int;
        acc_sw[p] = acc_sw[p] + wvi;
        acc_sws[p] = acc_sws[p] + wvi * slope_int;
        acc_2r[p] = acc_2r[p] + wvi * wvi * vr_int;
        acc_2p[p] = acc_2p[p] + wvi * wvi * vp_int;
      }
    }
  }
}
"#;

/// Phase-3a on GPU. Returns (rate, var_rnoise, var_poisson, seconds).
#[allow(clippy::too_many_arguments)]
pub fn run_gpu_weighted(
    data: &[u8], npix: usize, ngroup: usize, nint: usize,
    bzero: f64, bscale: f64, tgroup: f64, gainfact: f64,
    saturation: &[f64], superbias: &[f64], coeffs: &[f64], ncoeff: usize, dark_ramp: &[f64],
    readnoise: &[f64], gainmap: &[f64], jthresh: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64) {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None, force_fallback_adapter: false,
    })).expect("no GPU adapter");
    println!("[gpu] adapter: {}", adapter.get_info().name);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("rampfit-w"), required_features: wgpu::Features::empty(),
        required_limits: adapter.limits(), memory_hints: wgpu::MemoryHints::Performance,
        experimental_features: wgpu::ExperimentalFeatures::disabled(), trace: wgpu::Trace::Off,
    })).expect("device");

    let cap_bytes = 256usize * 1024 * 1024;
    let per_int_bytes = ngroup * npix * 2; // packed i16, uploaded raw
    let batch_max = (cap_bytes / per_int_bytes).clamp(1, nint);
    let raw_bytes = batch_max * per_int_bytes;
    let jt = jthresh as f32;
    assert!(jt <= 0.0 || ngroup <= 80, "GPU jump kernel supports up to 80 groups (got {ngroup})");
    println!("[gpu] phase-3{} weighted (packed i16); batches of {batch_max} integrations ({} MB)",
        if jt > 0.0 { "b (jumps)" } else { "a" }, raw_bytes / (1024 * 1024));

    let as_f32 = |v: &[f64]| -> Vec<f32> { v.iter().map(|&x| x as f32).collect() };
    let st = wgpu::BufferUsages::STORAGE;
    let mk = |label, v: &[f32]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label), contents: bytemuck::cast_slice(v), usage: st });
    let buf_sat = mk("sat", &as_f32(saturation));
    let buf_bias = mk("bias", &as_f32(superbias));
    let buf_coeffs = mk("coeffs", &if coeffs.is_empty() { vec![0f32] } else { as_f32(coeffs) });
    let buf_dark = mk("dark", &if dark_ramp.is_empty() { vec![0f32] } else { as_f32(dark_ramp) });
    let buf_rn = mk("rn", &as_f32(readnoise).iter().map(|x| if x.is_finite() { *x } else { 0.0 }).collect::<Vec<f32>>());
    let buf_gain = mk("gain", &as_f32(gainmap).iter().map(|x| if x.is_finite() { *x } else { 0.0 }).collect::<Vec<f32>>());

    // Two raw buffers for double-buffering: the GPU computes from one while the
    // CPU/queue fills the other.
    let mk_raw = || device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("raw"), size: raw_bytes as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let buf_raw = [mk_raw(), mk_raw()];
    let acc = |label| device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label), contents: bytemuck::cast_slice(&vec![0f32; npix]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC });
    let buf_sw = acc("sw"); let buf_sws = acc("sws"); let buf_2r = acc("2r"); let buf_2p = acc("2p");

    let mk_par = || device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("params"), size: std::mem::size_of::<ParamsW>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let buf_params = [mk_par(), mk_par()];

    let src = if jt > 0.0 { WGSL_J } else { WGSL_W };
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("w"), source: wgpu::ShaderSource::Wgsl(src.into()) });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("w"), layout: None, module: &shader, entry_point: Some("main"),
        compilation_options: Default::default(), cache: None });
    let bgl = pipeline.get_bind_group_layout(0);
    let mk_bind = |k: usize| device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &bgl, entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_params[k].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_sat.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_bias.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: buf_coeffs.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: buf_dark.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: buf_raw[k].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: buf_rn.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: buf_gain.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: buf_sw.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 9, resource: buf_sws.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 10, resource: buf_2r.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 11, resource: buf_2p.as_entire_binding() },
        ] });
    let binds = [mk_bind(0), mk_bind(1)];

    let t0 = std::time::Instant::now();
    let mut done = 0usize;
    let mut i = 0usize;
    while done < nint {
        let k = i % 2; // alternate buffers so fill(N+1) overlaps compute(N)
        let bn = batch_max.min(nint - done);
        let byte_start = done * per_int_bytes;
        let byte_len = bn * per_int_bytes;
        // Upload the raw big-endian i16 bytes straight from the mmap (no CPU convert).
        queue.write_buffer(&buf_raw[k], 0, &data[byte_start..byte_start + byte_len]);
        let params = ParamsW {
            npix: npix as u32, ngroup: ngroup as u32, batch_nint: bn as u32,
            ncoeff: ncoeff as u32, has_lin: (ncoeff > 0) as u32,
            has_dark: (!dark_ramp.is_empty()) as u32, tgroup: tgroup as f32, jthresh: jt,
            bzero: bzero as f32, bscale: bscale as f32, _p0: 0, _p1: 0 };
        queue.write_buffer(&buf_params[k], 0, bytemuck::bytes_of(&params));
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &binds[k], &[]);
            pass.dispatch_workgroups(npix.div_ceil(64) as u32, 1, 1);
        }
        queue.submit([enc.finish()]);
        // Non-blocking: let the queue make progress without stalling the CPU.
        device.poll(wgpu::PollType::Poll).ok();
        done += bn;
        i += 1;
    }
    device.poll(wgpu::PollType::wait_indefinitely()).ok(); // drain all batches
    let t_compute = t0.elapsed().as_secs_f64();
    let sw = readback_f32(&device, &queue, &buf_sw, npix);
    let sws = readback_f32(&device, &queue, &buf_sws, npix);
    let s2r = readback_f32(&device, &queue, &buf_2r, npix);
    let s2p = readback_f32(&device, &queue, &buf_2p, npix);
    let elapsed = t0.elapsed().as_secs_f64();
    println!("[gpu] upload+compute (double-buffered) {t_compute:.2}s");

    let gf2 = gainfact * gainfact;
    let mut rate = vec![f64::NAN; npix];
    let mut vrn = vec![f64::NAN; npix];
    let mut vpo = vec![f64::NAN; npix];
    for p in 0..npix {
        if sw[p] > 0.0 {
            let w = sw[p] as f64;
            rate[p] = gainfact * sws[p] as f64 / w;
            vrn[p] = gf2 * s2r[p] as f64 / (w * w);
            vpo[p] = gf2 * s2p[p] as f64 / (w * w);
        }
    }
    (rate, vrn, vpo, elapsed)
}

fn readback_f32(device: &wgpu::Device, queue: &wgpu::Queue, buf: &wgpu::Buffer, npix: usize) -> Vec<f32> {
    let stage = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: (npix * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_buffer_to_buffer(buf, 0, &stage, 0, (npix * 4) as u64);
    queue.submit([enc.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    stage.slice(..).map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    rx.recv().unwrap().unwrap();
    bytemuck::cast_slice(&stage.slice(..).get_mapped_range()).to_vec()
}

fn readback(device: &wgpu::Device, queue: &wgpu::Queue, acc_s: &wgpu::Buffer, acc_c: &wgpu::Buffer, npix: usize) -> (Vec<f32>, Vec<u32>) {
    let stage_s = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: (npix * 4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let stage_c = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: (npix * 4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_buffer_to_buffer(acc_s, 0, &stage_s, 0, (npix * 4) as u64);
    enc.copy_buffer_to_buffer(acc_c, 0, &stage_c, 0, (npix * 4) as u64);
    queue.submit([enc.finish()]);

    let map = |buf: &wgpu::Buffer| {
        let (tx, rx) = std::sync::mpsc::channel();
        buf.slice(..).map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        rx.recv().unwrap().unwrap();
    };
    map(&stage_s);
    map(&stage_c);
    let s: Vec<f32> = bytemuck::cast_slice(&stage_s.slice(..).get_mapped_range()).to_vec();
    let c: Vec<u32> = bytemuck::cast_slice(&stage_c.slice(..).get_mapped_range()).to_vec();
    (s, c)
}

/// Helper trait to fill the host batch buffer in parallel.
trait FillBatch {
    fn par_fill_batch(&mut self, int_off: usize, bn: usize, ngroup: usize, npix: usize, raw: &(dyn Fn(usize) -> f32 + Sync));
}
impl FillBatch for Vec<f32> {
    fn par_fill_batch(&mut self, int_off: usize, bn: usize, ngroup: usize, npix: usize, raw: &(dyn Fn(usize) -> f32 + Sync)) {
        use rayon::prelude::*;
        self[..bn * ngroup * npix]
            .par_chunks_mut(npix)
            .enumerate()
            .for_each(|(row, chunk)| {
                // row = it*ngroup + g  -> global sample base
                let it = row / ngroup;
                let g = row % ngroup;
                let gi = int_off + it;
                let base = (gi * ngroup + g) * npix;
                for (p, out) in chunk.iter_mut().enumerate() {
                    *out = raw(base + p);
                }
            });
    }
}
