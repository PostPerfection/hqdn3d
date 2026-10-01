/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! A clean-room implementation of the hqdn3d video denoiser.
//!
//! hqdn3d is the high-quality 3D denoiser known from MPlayer and FFmpeg. It
//! reduces film grain and sensor noise with a separable low-pass that runs
//! horizontally, vertically, and from one frame to the next, and it backs
//! off where samples differ enough to be an edge. This crate implements
//! that filter from the published parameter definitions and the shape of
//! its coefficient curve. It does not derive from the MPlayer, FFmpeg, or
//! VLC sources.
//!
//! # Model
//!
//! Each strength parameter is a distance on an 8-bit scale. At a difference
//! of zero the filter adopts the running average completely. At a difference
//! equal to the strength it keeps 25% of that average and 75% of the new
//! sample. Past that the weight falls off quickly, so real edges stay put.
//! [`weight`] evaluates this curve.
//!
//! Samples are lifted to 8.8 fixed point (eight fraction bits, centered in
//! the source code's bin) before filtering. The extra fraction is what lets
//! a sub-level average persist across pixels and frames instead of being
//! quantized away at every step.
//!
//! A plane of 128×128 or larger is split across a rayon pool. The horizontal
//! pass divides the rows across that pool, then the vertical and temporal
//! passes divide the columns into contiguous bands. [`Hqdn3d::denoise_yuv420`]
//! also runs Y, U, and V at the same time. The split is the same recurrence
//! as one thread.
//!
//! The three passes are:
//!
//! 1. Horizontal, left to right, within the row.
//! 2. Vertical, against the previous row's spatial result. The first row's
//!    reference is the row itself.
//! 3. Temporal, against the previous frame's denoised output. The first
//!    frame of a [`PlaneState`] has no reference and is filtered spatially
//!    only.
//!
//! A strength of zero disables that pass. In FFmpeg's option parser a zero
//! means "derive the documented default" instead; pass the defaults
//! explicitly here, or use [`Strength::from_luma_spatial`].
//!
//! # Example
//!
//! ```
//! use hqdn3d::{Hqdn3d, PlaneState, Strength};
//!
//! # fn main() -> Result<(), hqdn3d::Error> {
//! let filter = Hqdn3d::new(Strength::default())?;
//! let mut luma = PlaneState::new();
//! let src = [40u8, 44, 41, 70, 42, 39];
//! let mut dst = [0u8; 6];
//! filter.denoise_luma(&mut luma, &src, &mut dst, 3, 2, 3, 3)?;
//! # let _ = dst;
//! # Ok(())
//! # }
//! ```

#![deny(missing_docs)]

mod curve;
mod filter;
mod parallel;

pub use curve::weight;
pub use filter::{Filter, PlaneState};

use std::fmt;

/// Failure from building a filter or denoising a plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// A strength was negative or not a finite number.
    InvalidStrength,
    /// Width or height was zero, a stride was shorter than the width, or
    /// the plane dimensions overflowed.
    InvalidDimensions,
    /// A plane buffer was shorter than `(height - 1) * stride + width`.
    BufferTooSmall,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidStrength => write!(f, "strength must be finite and >= 0"),
            Error::InvalidDimensions => write!(f, "invalid plane dimensions"),
            Error::BufferTooSmall => write!(f, "plane buffer is shorter than its geometry"),
        }
    }
}

impl std::error::Error for Error {}

/// The four hqdn3d strengths, on an 8-bit sample scale.
///
/// Spatial strengths are the within-frame passes (horizontal and vertical
/// share one strength per plane). Temporal strengths blend each sample with
/// the same location in the previous denoised frame.
///
/// The documented defaults are luma spatial 4, chroma spatial 3, luma
/// temporal 6, and chroma temporal 4.5. [`Strength::default`] is those
/// values, and [`Strength::from_luma_spatial`] derives the other three from
/// one luma spatial setting the way the filter's documentation describes:
/// chroma spatial is `3/4` of luma spatial, luma temporal is `6/4` of luma
/// spatial, and chroma temporal is luma temporal scaled by the ratio of the
/// two spatial strengths.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Strength {
    luma_spatial: f64,
    chroma_spatial: f64,
    luma_temporal: f64,
    chroma_temporal: f64,
}

impl Default for Strength {
    fn default() -> Self {
        Self {
            luma_spatial: 4.0,
            chroma_spatial: 3.0,
            luma_temporal: 6.0,
            chroma_temporal: 4.5,
        }
    }
}

impl Strength {
    /// Explicit strengths. Zero disables a pass. Negative and non-finite
    /// values are rejected.
    pub fn new(
        luma_spatial: f64,
        chroma_spatial: f64,
        luma_temporal: f64,
        chroma_temporal: f64,
    ) -> Result<Self, Error> {
        Ok(Self {
            luma_spatial: finite_strength(luma_spatial)?,
            chroma_spatial: finite_strength(chroma_spatial)?,
            luma_temporal: finite_strength(luma_temporal)?,
            chroma_temporal: finite_strength(chroma_temporal)?,
        })
    }

    /// Derive all four strengths from the luma spatial setting.
    ///
    /// A luma spatial value of zero turns every pass off. Otherwise chroma
    /// spatial is `3/4` of luma, luma temporal is `6/4` of luma, and chroma
    /// temporal follows `luma_temporal * chroma_spatial / luma_spatial`.
    pub fn from_luma_spatial(luma_spatial: f64) -> Result<Self, Error> {
        let luma_spatial = finite_strength(luma_spatial)?;
        if luma_spatial == 0.0 {
            return Self::new(0.0, 0.0, 0.0, 0.0);
        }
        let chroma_spatial = 3.0 * luma_spatial / 4.0;
        let luma_temporal = 6.0 * luma_spatial / 4.0;
        let chroma_temporal = luma_temporal * chroma_spatial / luma_spatial;
        Self::new(luma_spatial, chroma_spatial, luma_temporal, chroma_temporal)
    }

    /// Luma spatial strength.
    pub fn luma_spatial(self) -> f64 {
        self.luma_spatial
    }

    /// Chroma spatial strength.
    pub fn chroma_spatial(self) -> f64 {
        self.chroma_spatial
    }

    /// Luma temporal strength.
    pub fn luma_temporal(self) -> f64 {
        self.luma_temporal
    }

    /// Chroma temporal strength.
    pub fn chroma_temporal(self) -> f64 {
        self.chroma_temporal
    }
}

fn finite_strength(value: f64) -> Result<f64, Error> {
    if value.is_finite() && value >= 0.0 {
        Ok(value)
    } else {
        Err(Error::InvalidStrength)
    }
}

/// Luma and chroma filters built from a [`Strength`].
///
/// Each plane needs its own [`PlaneState`]. U and V are both chroma, and
/// each of them keeps a separate state.
#[derive(Clone, Debug)]
pub struct Hqdn3d {
    luma: Filter,
    chroma: Filter,
}

impl Hqdn3d {
    /// Build both plane filters from `strength`.
    pub fn new(strength: Strength) -> Result<Self, Error> {
        Ok(Self {
            luma: Filter::new(strength.luma_spatial, strength.luma_temporal)?,
            chroma: Filter::new(strength.chroma_spatial, strength.chroma_temporal)?,
        })
    }

    /// The luma plane filter.
    pub fn luma(&self) -> &Filter {
        &self.luma
    }

    /// The chroma plane filter, shared by U and V. Each plane still has its
    /// own [`PlaneState`].
    pub fn chroma(&self) -> &Filter {
        &self.chroma
    }

    /// Denoise a luma plane. See [`Filter::denoise`].
    #[allow(clippy::too_many_arguments)]
    pub fn denoise_luma(
        &self,
        state: &mut PlaneState,
        src: &[u8],
        dst: &mut [u8],
        width: usize,
        height: usize,
        src_stride: usize,
        dst_stride: usize,
    ) -> Result<(), Error> {
        self.luma
            .denoise(state, src, dst, width, height, src_stride, dst_stride)
    }

    /// Denoise a packed 4:2:0 frame.
    ///
    /// `src` and `dst` are tightly packed `yuv420p`: the full luma plane,
    /// then U, then V, each with no row padding. Width and height must be
    /// even. Y, U, and V run at the same time, and each plane uses the same
    /// row and column split as [`Filter::denoise`]. U and V need separate
    /// state.
    #[allow(clippy::too_many_arguments)]
    pub fn denoise_yuv420(
        &self,
        y: &mut PlaneState,
        u: &mut PlaneState,
        v: &mut PlaneState,
        src: &[u8],
        dst: &mut [u8],
        width: usize,
        height: usize,
    ) -> Result<(), Error> {
        if !width.is_multiple_of(2) || !height.is_multiple_of(2) {
            return Err(Error::InvalidDimensions);
        }
        let y_bytes = width.checked_mul(height).ok_or(Error::InvalidDimensions)?;
        let chroma_w = width / 2;
        let chroma_h = height / 2;
        let c_bytes = chroma_w
            .checked_mul(chroma_h)
            .ok_or(Error::InvalidDimensions)?;
        let frame = y_bytes
            .checked_add(c_bytes)
            .and_then(|n| n.checked_add(c_bytes))
            .ok_or(Error::InvalidDimensions)?;
        if src.len() < frame || dst.len() < frame {
            return Err(Error::BufferTooSmall);
        }

        let (src_y, src_rest) = src.split_at(y_bytes);
        let (src_u, src_v) = src_rest.split_at(c_bytes);
        let (dst_y, dst_rest) = dst.split_at_mut(y_bytes);
        let (dst_u, dst_v) = dst_rest.split_at_mut(c_bytes);

        let (y_result, (u_result, v_result)) = rayon::join(
            || {
                self.luma
                    .denoise(y, src_y, dst_y, width, height, width, width)
            },
            || {
                rayon::join(
                    || {
                        self.chroma
                            .denoise(u, src_u, dst_u, chroma_w, chroma_h, chroma_w, chroma_w)
                    },
                    || {
                        self.chroma
                            .denoise(v, src_v, dst_v, chroma_w, chroma_h, chroma_w, chroma_w)
                    },
                )
            },
        );
        y_result?;
        u_result?;
        v_result?;
        Ok(())
    }

    /// Denoise a chroma plane. See [`Filter::denoise`].
    #[allow(clippy::too_many_arguments)]
    pub fn denoise_chroma(
        &self,
        state: &mut PlaneState,
        src: &[u8],
        dst: &mut [u8],
        width: usize,
        height: usize,
        src_stride: usize,
        dst_stride: usize,
    ) -> Result<(), Error> {
        self.chroma
            .denoise(state, src, dst, width, height, src_stride, dst_stride)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_documented_derivation() {
        let derived = Strength::from_luma_spatial(4.0).unwrap();
        assert_eq!(Strength::default(), derived);
        assert_eq!(derived.luma_spatial(), 4.0);
        assert_eq!(derived.chroma_spatial(), 3.0);
        assert_eq!(derived.luma_temporal(), 6.0);
        assert_eq!(derived.chroma_temporal(), 4.5);
    }

    #[test]
    fn luma_spatial_eight_derives_the_usual_triple() {
        let strength = Strength::from_luma_spatial(8.0).unwrap();
        assert_eq!(strength.chroma_spatial(), 6.0);
        assert_eq!(strength.luma_temporal(), 12.0);
        assert_eq!(strength.chroma_temporal(), 9.0);
    }

    #[test]
    fn zero_luma_spatial_turns_the_whole_filter_off() {
        let strength = Strength::from_luma_spatial(0.0).unwrap();
        assert_eq!(strength, Strength::new(0.0, 0.0, 0.0, 0.0).unwrap());
    }

    #[test]
    fn luma_and_chroma_use_different_temporal_strengths() {
        let filter = Hqdn3d::new(Strength::default()).unwrap();
        let mut luma_state = PlaneState::new();
        let mut chroma_state = PlaneState::new();
        let mut luma = [0u8; 1];
        let mut chroma = [0u8; 1];
        filter
            .luma()
            .denoise_packed(&mut luma_state, &[128], &mut luma, 1, 1)
            .unwrap();
        filter
            .chroma()
            .denoise_packed(&mut chroma_state, &[128], &mut chroma, 1, 1)
            .unwrap();
        filter
            .luma()
            .denoise_packed(&mut luma_state, &[140], &mut luma, 1, 1)
            .unwrap();
        filter
            .chroma()
            .denoise_packed(&mut chroma_state, &[140], &mut chroma, 1, 1)
            .unwrap();
        assert_ne!(luma, chroma);
    }

    #[test]
    fn yuv420_matches_filtering_the_planes_separately() {
        let filter = Hqdn3d::new(Strength::default()).unwrap();
        let width = 16;
        let height = 8;
        let y = width * height;
        let c = (width / 2) * (height / 2);
        let mut seed = 1u32;
        let mut frame = vec![0u8; y + c + c];
        for sample in &mut frame {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            *sample = (seed >> 24) as u8;
        }

        let mut together = frame.clone();
        let mut y_state = PlaneState::new();
        let mut u_state = PlaneState::new();
        let mut v_state = PlaneState::new();
        filter
            .denoise_yuv420(
                &mut y_state,
                &mut u_state,
                &mut v_state,
                &frame,
                &mut together,
                width,
                height,
            )
            .unwrap();

        let mut apart = frame.clone();
        let (src_y, src_rest) = frame.split_at(y);
        let (src_u, src_v) = src_rest.split_at(c);
        let (dst_y, dst_rest) = apart.split_at_mut(y);
        let (dst_u, dst_v) = dst_rest.split_at_mut(c);
        filter
            .denoise_luma(
                &mut PlaneState::new(),
                src_y,
                dst_y,
                width,
                height,
                width,
                width,
            )
            .unwrap();
        filter
            .denoise_chroma(
                &mut PlaneState::new(),
                src_u,
                dst_u,
                width / 2,
                height / 2,
                width / 2,
                width / 2,
            )
            .unwrap();
        filter
            .denoise_chroma(
                &mut PlaneState::new(),
                src_v,
                dst_v,
                width / 2,
                height / 2,
                width / 2,
                width / 2,
            )
            .unwrap();
        assert_eq!(together, apart);
    }

    #[test]
    fn threaded_yuv420_matches_one_thread() {
        // 320×240 is large enough that luma and both chroma planes cross
        // the parallel threshold when more than one worker is available.
        let filter = Hqdn3d::new(Strength::default()).unwrap();
        let width = 320;
        let height = 240;
        let y_bytes = width * height;
        let c_bytes = (width / 2) * (height / 2);
        let frame_len = y_bytes + c_bytes + c_bytes;
        let mut seed = 0xA5A5_5A5Au32;
        let mut frames = Vec::new();
        for _ in 0..2 {
            let mut frame = vec![0u8; frame_len];
            for sample in &mut frame {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                *sample = (seed >> 24) as u8;
            }
            frames.push(frame);
        }

        let mut threaded_y = PlaneState::new();
        let mut threaded_u = PlaneState::new();
        let mut threaded_v = PlaneState::new();
        let mut serial_y = PlaneState::new();
        let mut serial_u = PlaneState::new();
        let mut serial_v = PlaneState::new();
        for frame in &frames {
            let mut threaded = vec![0u8; frame_len];
            filter
                .denoise_yuv420(
                    &mut threaded_y,
                    &mut threaded_u,
                    &mut threaded_v,
                    frame,
                    &mut threaded,
                    width,
                    height,
                )
                .unwrap();

            let mut serial = vec![0u8; frame_len];
            let (src_y, src_rest) = frame.split_at(y_bytes);
            let (src_u, src_v) = src_rest.split_at(c_bytes);
            let (dst_y, dst_rest) = serial.split_at_mut(y_bytes);
            let (dst_u, dst_v) = dst_rest.split_at_mut(c_bytes);
            let chroma_w = width / 2;
            let chroma_h = height / 2;
            filter
                .luma()
                .denoise_with(&mut serial_y, src_y, dst_y, width, height, width, width, 1)
                .unwrap();
            filter
                .chroma()
                .denoise_with(
                    &mut serial_u,
                    src_u,
                    dst_u,
                    chroma_w,
                    chroma_h,
                    chroma_w,
                    chroma_w,
                    1,
                )
                .unwrap();
            filter
                .chroma()
                .denoise_with(
                    &mut serial_v,
                    src_v,
                    dst_v,
                    chroma_w,
                    chroma_h,
                    chroma_w,
                    chroma_w,
                    1,
                )
                .unwrap();
            assert_eq!(threaded, serial);
        }
    }

    #[test]
    fn odd_yuv420_is_rejected() {
        let filter = Hqdn3d::new(Strength::default()).unwrap();
        let src = [0u8; 6];
        let mut dst = [0u8; 6];
        let err = filter
            .denoise_yuv420(
                &mut PlaneState::new(),
                &mut PlaneState::new(),
                &mut PlaneState::new(),
                &src,
                &mut dst,
                3,
                2,
            )
            .unwrap_err();
        assert_eq!(err, Error::InvalidDimensions);
    }
}
