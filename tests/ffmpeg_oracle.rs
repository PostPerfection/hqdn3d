/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! Black-box comparison with the system `ffmpeg` binary.
//!
//! Run with `cargo test -- --ignored`. FFmpeg treats an explicit strength of
//! zero as "use the default", so a disabled pass is requested there with a
//! tiny positive value. This crate treats zero as off.

use std::process::Command;

use hqdn3d::{Hqdn3d, PlaneState, Strength};

#[test]
#[ignore = "compares against the system ffmpeg binary"]
fn defaults_stay_close_to_ffmpeg_on_gray() {
    let (max_abs, mae) = compare_gray(32, 24, 4, Strength::default(), "hqdn3d=4:3:6:4.5");
    eprintln!("default gray max {max_abs} mae {mae:.4}");
    // The continuous curve and ffmpeg's coefficient table differ by a level
    // or two. Mean error stays far below one level.
    assert!(max_abs <= 2, "max abs diff {max_abs}, mae {mae:.4}");
    assert!(mae < 0.08, "mae {mae:.4}");
}

#[test]
#[ignore = "compares against the system ffmpeg binary"]
fn stronger_settings_stay_within_a_few_levels() {
    let strength = Strength::new(8.0, 0.0, 12.0, 0.0).unwrap();
    let (max_abs, mae) = compare_gray(
        32,
        24,
        3,
        strength,
        "hqdn3d=luma_spatial=8:chroma_spatial=0.0000001:luma_tmp=12:chroma_tmp=0.0000001",
    );
    eprintln!("8/12 gray max {max_abs} mae {mae:.4}");
    assert!(max_abs <= 3, "max abs diff {max_abs}, mae {mae:.4}");
    assert!(mae < 0.30, "mae {mae:.4}");
}

#[test]
#[ignore = "compares against the system ffmpeg binary"]
fn threaded_defaults_stay_close_to_ffmpeg_on_gray() {
    // 128×128 crosses the crate's parallel threshold, so this exercises the
    // row and column split rather than the single-thread fallback.
    let (max_abs, mae) = compare_gray(128, 128, 2, Strength::default(), "hqdn3d=4:3:6:4.5");
    eprintln!("threaded default gray max {max_abs} mae {mae:.4}");
    assert!(max_abs <= 2, "max abs diff {max_abs}, mae {mae:.4}");
    assert!(mae < 0.12, "mae {mae:.4}");
}

#[test]
#[ignore = "compares against the system ffmpeg binary"]
fn yuv420_defaults_stay_close_to_ffmpeg() {
    let width = 16;
    let height = 12;
    let frames = 3;
    let y = width * height;
    let c = (width / 2) * (height / 2);
    let frame_len = y + c + c;
    let input = synthetic(frame_len * frames, 7);
    let ours = denoise_yuv420(&input, width, height, frames, Strength::default());
    let theirs = ffmpeg_raw(
        &input,
        &format!("{width}x{height}"),
        "yuv420p",
        "hqdn3d=4:3:6:4.5",
    );
    assert_eq!(ours.len(), theirs.len());
    let (max_abs, mae) = error_stats(&ours, &theirs);
    eprintln!("default yuv420 max {max_abs} mae {mae:.4}");
    assert!(max_abs <= 1, "max abs diff {max_abs}, mae {mae:.4}");
    assert!(mae < 0.08, "mae {mae:.4}");
}

fn compare_gray(
    width: usize,
    height: usize,
    frames: usize,
    strength: Strength,
    vf: &str,
) -> (u8, f64) {
    let frame_len = width * height;
    let input = synthetic(frame_len * frames, 3);
    let ours = denoise_gray(&input, width, height, strength);
    let theirs = ffmpeg_raw(&input, &format!("{width}x{height}"), "gray", vf);
    assert_eq!(ours.len(), theirs.len());
    error_stats(&ours, &theirs)
}

fn denoise_gray(input: &[u8], width: usize, height: usize, strength: Strength) -> Vec<u8> {
    let filter = Hqdn3d::new(strength).unwrap();
    let mut state = PlaneState::new();
    let frame_len = width * height;
    let mut output = vec![0u8; input.len()];
    for (src, dst) in input.chunks(frame_len).zip(output.chunks_mut(frame_len)) {
        filter
            .denoise_luma(&mut state, src, dst, width, height, width, width)
            .unwrap();
    }
    output
}

fn denoise_yuv420(
    input: &[u8],
    width: usize,
    height: usize,
    _frames: usize,
    strength: Strength,
) -> Vec<u8> {
    let filter = Hqdn3d::new(strength).unwrap();
    let mut y_state = PlaneState::new();
    let mut u_state = PlaneState::new();
    let mut v_state = PlaneState::new();
    let y = width * height;
    let c = (width / 2) * (height / 2);
    let frame_len = y + c + c;
    let mut output = vec![0u8; input.len()];
    for (src, dst) in input.chunks(frame_len).zip(output.chunks_mut(frame_len)) {
        filter
            .denoise_yuv420(
                &mut y_state,
                &mut u_state,
                &mut v_state,
                src,
                dst,
                width,
                height,
            )
            .unwrap();
    }
    output
}

fn ffmpeg_raw(input: &[u8], size: &str, pix_fmt: &str, vf: &str) -> Vec<u8> {
    let mut child = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            pix_fmt,
            "-s",
            size,
            "-r",
            "1",
            "-i",
            "pipe:0",
            "-vf",
            vf,
            "-f",
            "rawvideo",
            "-pix_fmt",
            pix_fmt,
            "pipe:1",
        ])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("ffmpeg is on PATH");
    use std::io::Write;
    child.stdin.take().unwrap().write_all(input).unwrap();
    let gathered = child.wait_with_output().unwrap();
    assert!(
        gathered.status.success(),
        "ffmpeg failed: {}",
        String::from_utf8_lossy(&gathered.stderr)
    );
    gathered.stdout
}

fn synthetic(len: usize, seed: u32) -> Vec<u8> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 24) as u8
        })
        .collect()
}

fn error_stats(ours: &[u8], theirs: &[u8]) -> (u8, f64) {
    let mut max_abs = 0u8;
    let mut sum = 0u64;
    for (a, b) in ours.iter().zip(theirs) {
        let diff = a.abs_diff(*b);
        max_abs = max_abs.max(diff);
        sum += u64::from(diff);
    }
    (max_abs, sum as f64 / ours.len() as f64)
}
