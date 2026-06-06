//! Minimal pure-Rust FITS reader — just enough to open a JWST `_uncal` ramp cube,
//! walk its HDUs, and expose header keywords + a byte slice for the SCI data.
//!
//! FITS layout recap: a file is a sequence of HDUs. Each HDU is a header made of
//! 2880-byte blocks containing 80-char "cards", terminated by an `END` card, then
//! (optionally) a data unit also padded to a 2880-byte boundary.

use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;

const BLOCK: usize = 2880;
const CARD: usize = 80;

pub struct Hdu {
    pub keys: HashMap<String, String>,
    /// Byte offset of this HDU's data unit within the mmap.
    pub data_offset: usize,
    /// Length in bytes of the (unpadded) data unit.
    pub data_len: usize,
}

impl Hdu {
    /// Parse a header keyword as i64.
    pub fn int(&self, key: &str) -> Option<i64> {
        self.keys.get(key)?.parse().ok()
    }
    /// Parse a header keyword as f64.
    pub fn float(&self, key: &str) -> Option<f64> {
        self.keys.get(key)?.parse().ok()
    }
}

pub struct Fits {
    pub mmap: Mmap,
    pub hdus: Vec<Hdu>,
}

impl Fits {
    pub fn open(path: &str) -> std::io::Result<Self> {
        let file = File::open(path)?;
        // SAFETY: file is read-only for the lifetime of the mmap; we never mutate it.
        let mmap = unsafe { Mmap::map(&file)? };
        let hdus = parse_hdus(&mmap);
        Ok(Self { mmap, hdus })
    }

    /// First HDU whose EXTNAME matches.
    pub fn find(&self, extname: &str) -> Option<&Hdu> {
        self.hdus
            .iter()
            .find(|h| h.keys.get("EXTNAME").map(|s| s == extname).unwrap_or(false))
    }
}

fn parse_hdus(buf: &[u8]) -> Vec<Hdu> {
    let mut hdus = Vec::new();
    let mut pos = 0usize;

    while pos + BLOCK <= buf.len() {
        let mut keys: HashMap<String, String> = HashMap::new();
        let mut p = pos;
        let mut found_end = false;

        // Read header blocks until the END card.
        while p + BLOCK <= buf.len() {
            for c in 0..(BLOCK / CARD) {
                let card = &buf[p + c * CARD..p + c * CARD + CARD];
                let kw = std::str::from_utf8(&card[0..8]).unwrap_or("").trim();
                if kw == "END" {
                    found_end = true;
                }
                // A value-carrying card has '=' in column 9.
                if !kw.is_empty() && card[8] == b'=' {
                    let rest = std::str::from_utf8(&card[10..]).unwrap_or("");
                    keys.insert(kw.to_string(), parse_value(rest));
                }
            }
            p += BLOCK;
            if found_end {
                break;
            }
        }
        if !found_end {
            break;
        }

        let data_start = p;
        let bitpix = keys.get("BITPIX").and_then(|s| s.parse::<i64>().ok()).unwrap_or(0);
        let naxis = keys.get("NAXIS").and_then(|s| s.parse::<i64>().ok()).unwrap_or(0);
        let mut nelem: i64 = if naxis > 0 { 1 } else { 0 };
        for n in 1..=naxis {
            let ax = keys
                .get(&format!("NAXIS{n}"))
                .and_then(|s| s.parse::<i64>().ok())
                .unwrap_or(0);
            nelem *= ax;
        }
        let data_bytes = (nelem * (bitpix.abs() / 8)) as usize;
        let data_padded = data_bytes.div_ceil(BLOCK) * BLOCK;

        hdus.push(Hdu {
            keys,
            data_offset: data_start,
            data_len: data_bytes,
        });
        pos = data_start + data_padded;
    }

    hdus
}

/// Parse the value part of a card (everything after `KEYWORD= `).
fn parse_value(s: &str) -> String {
    let s = s.trim_start();
    if let Some(stripped) = s.strip_prefix('\'') {
        // Quoted string: up to the closing quote.
        if let Some(end) = stripped.find('\'') {
            return stripped[..end].trim().to_string();
        }
        return stripped.trim().to_string();
    }
    // Numeric / logical value: strip the trailing `/ comment`.
    let v = match s.find('/') {
        Some(i) => &s[..i],
        None => s,
    };
    v.trim().to_string()
}
