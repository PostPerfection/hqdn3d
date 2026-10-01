/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! Denoise raw 8-bit frames on stdin and write them to stdout.
//!
//! The input is unpacked `gray` or `yuv420p` (4:2:0, chroma planes half
//! width and half height, no padding). This does not parse a container.

use std::env;
use std::io::{self, Read, Write};
use std::process::ExitCode;

use hqdn3d::{Hqdn3d, PlaneState, Strength};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("hqdn3d: {err}");
            ExitCode::from(1)
        }
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse(env::args().skip(1))?;
    let filter = Hqdn3d::new(args.strength).map_err(|err| err.to_string())?;
    let frame_len = args.format.frame_len(args.width, args.height)?;

    let mut input = io::stdin().lock();
    let mut output = io::stdout().lock();
    let mut incoming = vec![0u8; frame_len];
    let mut outgoing = vec![0u8; frame_len];
    let mut y_state = PlaneState::new();
    let mut u_state = PlaneState::new();
    let mut v_state = PlaneState::new();

    loop {
        if !read_frame(&mut input, &mut incoming)? {
            break;
        }
        denoise_frame(
            &filter,
            args.format,
            args.width,
            args.height,
            &incoming,
            &mut outgoing,
            &mut y_state,
            &mut u_state,
            &mut v_state,
        )?;
        output
            .write_all(&outgoing)
            .map_err(|err| format!("writing stdout: {err}"))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn denoise_frame(
    filter: &Hqdn3d,
    format: Format,
    width: usize,
    height: usize,
    src: &[u8],
    dst: &mut [u8],
    y_state: &mut PlaneState,
    u_state: &mut PlaneState,
    v_state: &mut PlaneState,
) -> Result<(), String> {
    let fail = |err| format!("{err}");
    match format {
        Format::Gray => filter
            .denoise_luma(y_state, src, dst, width, height, width, width)
            .map_err(fail),
        Format::Yuv420 => filter
            .denoise_yuv420(y_state, u_state, v_state, src, dst, width, height)
            .map_err(fail),
    }
}

/// `Ok(false)` is a clean end of input. A frame that ends mid-way is an error.
fn read_frame(input: &mut impl Read, buf: &mut [u8]) -> Result<bool, String> {
    let mut filled = 0;
    while filled < buf.len() {
        match input.read(&mut buf[filled..]) {
            Ok(0) if filled == 0 => return Ok(false),
            Ok(0) => return Err("truncated frame on stdin".to_string()),
            Ok(n) => filled += n,
            Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
            Err(err) => return Err(format!("reading stdin: {err}")),
        }
    }
    Ok(true)
}

#[derive(Clone, Copy)]
enum Format {
    Gray,
    Yuv420,
}

impl Format {
    fn frame_len(self, width: usize, height: usize) -> Result<usize, String> {
        let y = width
            .checked_mul(height)
            .ok_or_else(|| "frame is too large".to_string())?;
        match self {
            Format::Gray => Ok(y),
            Format::Yuv420 => {
                if !width.is_multiple_of(2) || !height.is_multiple_of(2) {
                    return Err("yuv420p requires even width and height".to_string());
                }
                let chroma = (width / 2) * (height / 2);
                y.checked_add(chroma)
                    .and_then(|n| n.checked_add(chroma))
                    .ok_or_else(|| "frame is too large".to_string())
            }
        }
    }
}

struct Args {
    width: usize,
    height: usize,
    format: Format,
    strength: Strength,
}

impl Args {
    fn parse(values: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut width = None;
        let mut height = None;
        let mut format = Format::Gray;
        let mut luma_spatial = None;
        let mut chroma_spatial = None;
        let mut luma_temporal = None;
        let mut chroma_temporal = None;

        let mut values = values.peekable();
        while let Some(arg) = values.next() {
            match arg.as_str() {
                "-h" | "--help" => {
                    print_usage();
                    std::process::exit(0);
                }
                "--width" => width = Some(parse_dim("--width", &mut values)?),
                "--height" => height = Some(parse_dim("--height", &mut values)?),
                "--format" => {
                    format = match values.next().as_deref() {
                        Some("gray") => Format::Gray,
                        Some("yuv420p") => Format::Yuv420,
                        Some(other) => {
                            return Err(format!("unknown format {other:?}; use gray or yuv420p"));
                        }
                        None => return Err("--format needs gray or yuv420p".to_string()),
                    }
                }
                "--luma-spatial" => {
                    luma_spatial = Some(parse_strength("--luma-spatial", &mut values)?)
                }
                "--chroma-spatial" => {
                    chroma_spatial = Some(parse_strength("--chroma-spatial", &mut values)?)
                }
                "--luma-temporal" => {
                    luma_temporal = Some(parse_strength("--luma-temporal", &mut values)?)
                }
                "--chroma-temporal" => {
                    chroma_temporal = Some(parse_strength("--chroma-temporal", &mut values)?)
                }
                other => return Err(format!("unknown argument {other}; try --help")),
            }
        }

        let width = width.ok_or("--width is required")?;
        let height = height.ok_or("--height is required")?;
        let base = Strength::from_luma_spatial(luma_spatial.unwrap_or(4.0))
            .map_err(|err| err.to_string())?;
        let strength = Strength::new(
            luma_spatial.unwrap_or(base.luma_spatial()),
            chroma_spatial.unwrap_or(base.chroma_spatial()),
            luma_temporal.unwrap_or(base.luma_temporal()),
            chroma_temporal.unwrap_or(base.chroma_temporal()),
        )
        .map_err(|err| err.to_string())?;

        Ok(Self {
            width,
            height,
            format,
            strength,
        })
    }
}

fn parse_dim(flag: &str, values: &mut impl Iterator<Item = String>) -> Result<usize, String> {
    let raw = values
        .next()
        .ok_or_else(|| format!("{flag} needs a positive integer"))?;
    let value: usize = raw
        .parse()
        .map_err(|_| format!("{flag} needs a positive integer"))?;
    if value == 0 {
        return Err(format!("{flag} needs a positive integer"));
    }
    Ok(value)
}

fn parse_strength(flag: &str, values: &mut impl Iterator<Item = String>) -> Result<f64, String> {
    let raw = values
        .next()
        .ok_or_else(|| format!("{flag} needs a number >= 0"))?;
    let value: f64 = raw
        .parse()
        .map_err(|_| format!("{flag} needs a number >= 0"))?;
    if !value.is_finite() || value < 0.0 {
        return Err(format!("{flag} needs a number >= 0"));
    }
    Ok(value)
}

fn print_usage() {
    eprintln!(
        "\
usage: hqdn3d --width W --height H [--format gray|yuv420p]
              [--luma-spatial F] [--chroma-spatial F]
              [--luma-temporal F] [--chroma-temporal F]

Read raw 8-bit frames from stdin and write denoised frames to stdout.
gray is the default format. yuv420p is 4:2:0 with no row padding.

Strengths are 8-bit level differences. At that difference the filter keeps
25% of its running average. With no strength flags the defaults are
4:3:6:4.5 (luma spatial, chroma spatial, luma temporal, chroma temporal).
Setting only --luma-spatial derives the other three (chroma spatial 3/4,
luma temporal 6/4, chroma temporal scaled to match). An explicit 0 turns
that pass off.

Larger planes are filtered on several threads. Y, U, and V run together."
    );
}
