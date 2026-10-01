/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! Separable horizontal, vertical, and temporal low-pass.
//!
//! Larger planes are split across the rayon pool. Rows of the horizontal
//! pass are independent, and columns of the vertical and temporal passes
//! are independent, so the threaded schedule is the same recurrence as one
//! thread.

use crate::Error;
use crate::curve::Curve;
use crate::parallel::{auto_bands, column_ranges, for_bands, for_rows};

/// 8.8 fixed point. Sample `s` sits at the center of its code bin.
const SHIFT: i32 = 8;
const BIAS: i32 = 1 << (SHIFT - 1);
const DOMAIN_MAX: i32 = (1 << 16) - 1;

#[inline]
fn promote(sample: u8) -> i32 {
    ((sample as i32) << SHIFT) + BIAS
}

#[inline]
fn demote(value: i32) -> u8 {
    (value.clamp(0, DOMAIN_MAX) >> SHIFT) as u8
}

/// Temporal memory for one plane.
///
/// The state is the previous frame after denoising, kept at higher precision
/// than the 8-bit picture so a sub-level average can survive from frame to
/// frame. A new state has no reference, and the first frame is filtered
/// spatially only. [`PlaneState::reset`] drops the reference, which is the
/// right thing to do at a scene cut.
#[derive(Clone, Debug, Default)]
pub struct PlaneState {
    previous: Vec<u16>,
    horizontal: Vec<u16>,
    band_out: Vec<u8>,
    band_prev: Vec<u16>,
    width: usize,
    height: usize,
    ready: bool,
}

impl PlaneState {
    /// Empty state, with no temporal reference.
    pub fn new() -> Self {
        Self::default()
    }

    /// Forget the previous frame.
    pub fn reset(&mut self) {
        self.previous.clear();
        self.horizontal.clear();
        self.band_out.clear();
        self.band_prev.clear();
        self.width = 0;
        self.height = 0;
        self.ready = false;
    }

    /// Whether the next frame will be blended with a previous one.
    pub fn has_reference(&self) -> bool {
        self.ready
    }
}

/// One spatial curve and one temporal curve, applied to a single plane.
#[derive(Clone, Debug)]
pub struct Filter {
    spatial: Option<Curve>,
    temporal: Option<Curve>,
    spatial_strength: f64,
    temporal_strength: f64,
}

impl Filter {
    /// Build a plane filter. A strength of zero disables that pass.
    /// Negative and non-finite strengths are rejected.
    pub fn new(spatial: f64, temporal: f64) -> Result<Self, Error> {
        Ok(Self {
            spatial: optional_curve(spatial)?,
            temporal: optional_curve(temporal)?,
            spatial_strength: spatial,
            temporal_strength: temporal,
        })
    }

    /// Spatial strength, on the 8-bit scale. Zero means the pass is off.
    pub fn spatial_strength(&self) -> f64 {
        self.spatial_strength
    }

    /// Temporal strength, on the 8-bit scale. Zero means the pass is off.
    pub fn temporal_strength(&self) -> f64 {
        self.temporal_strength
    }

    /// Denoise one 8-bit plane into `dst`.
    ///
    /// `src` and `dst` may use different row strides. Each stride is the
    /// distance in bytes from one row to the next and must be at least
    /// `width`. The buffers must cover `(height - 1) * stride + width` bytes.
    ///
    /// `state` is updated with this frame's output. Use the same state for
    /// later frames of the same plane, and replace it or call
    /// [`PlaneState::reset`] when the picture dimensions change or the shot
    /// cuts. A dimension change discards the temporal reference.
    ///
    /// Planes of 128×128 and up are split across the rayon pool: the
    /// horizontal pass by rows, then the vertical and temporal passes by
    /// columns. Smaller planes stay on the calling thread. Both schedules
    /// implement the same recurrence.
    #[allow(clippy::too_many_arguments)]
    pub fn denoise(
        &self,
        state: &mut PlaneState,
        src: &[u8],
        dst: &mut [u8],
        width: usize,
        height: usize,
        src_stride: usize,
        dst_stride: usize,
    ) -> Result<(), Error> {
        self.denoise_with(
            state,
            src,
            dst,
            width,
            height,
            src_stride,
            dst_stride,
            auto_bands(width, height),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn denoise_with(
        &self,
        state: &mut PlaneState,
        src: &[u8],
        dst: &mut [u8],
        width: usize,
        height: usize,
        src_stride: usize,
        dst_stride: usize,
        bands: usize,
    ) -> Result<(), Error> {
        check_plane(src.len(), dst.len(), width, height, src_stride, dst_stride)?;
        let parallel = bands > 1;
        if self.spatial.is_none() && self.temporal.is_none() {
            copy_plane(src, dst, width, height, src_stride, dst_stride, parallel);
            return Ok(());
        }

        if state.width != width || state.height != height {
            state.previous.clear();
            state.ready = false;
            state.width = width;
            state.height = height;
        }
        let pixels = width.checked_mul(height).ok_or(Error::InvalidDimensions)?;

        let mut previous = std::mem::take(&mut state.previous);
        let mut horizontal = std::mem::take(&mut state.horizontal);
        let mut band_out = std::mem::take(&mut state.band_out);
        let mut band_prev = std::mem::take(&mut state.band_prev);
        if previous.len() != pixels {
            previous.resize(pixels, 0);
            state.ready = false;
        }
        band_out.resize(pixels, 0);
        band_prev.resize(pixels, 0);
        if self.spatial.is_some() {
            horizontal.resize(pixels, 0);
        }

        let reference_ready = state.ready;
        let ranges = column_ranges(width, bands.max(1));
        let parallel = ranges.len() > 1;
        let spatial = self.spatial.as_ref();
        let temporal = self.temporal.as_ref();

        if spatial.is_some() {
            for_rows(
                &mut horizontal,
                height,
                width,
                width,
                parallel,
                &|y, row| {
                    let start = y * src_stride;
                    horizontal_row(spatial, &src[start..start + width], row);
                },
            );
        }

        let job = Job {
            spatial,
            temporal,
            reference_ready,
            src,
            src_stride,
            width,
            height,
            horizontal: spatial.map(|_| horizontal.as_slice()),
            previous: previous.as_slice(),
        };
        for_bands(
            &mut band_out,
            &mut band_prev,
            &ranges,
            height,
            parallel,
            &|x0, x1, out, prev| job.filter_columns(x0, x1, out, prev),
        );

        for_rows(dst, height, dst_stride, width, parallel, &|y, row| {
            gather_band_row(&band_out, &ranges, height, y, row);
        });
        for_rows(&mut previous, height, width, width, parallel, &|y, row| {
            gather_band_row(&band_prev, &ranges, height, y, row);
        });

        state.previous = previous;
        state.horizontal = horizontal;
        state.band_out = band_out;
        state.band_prev = band_prev;
        state.ready = self.temporal.is_some();
        Ok(())
    }

    /// [`Filter::denoise`] for a tightly packed plane (`stride == width`).
    pub fn denoise_packed(
        &self,
        state: &mut PlaneState,
        src: &[u8],
        dst: &mut [u8],
        width: usize,
        height: usize,
    ) -> Result<(), Error> {
        self.denoise(state, src, dst, width, height, width, width)
    }

    /// Denoise `frame` in place. The plane may have row padding (`stride`).
    pub fn denoise_in_place(
        &self,
        state: &mut PlaneState,
        frame: &mut [u8],
        width: usize,
        height: usize,
        stride: usize,
    ) -> Result<(), Error> {
        let source = frame.to_vec();
        self.denoise(state, &source, frame, width, height, stride, stride)
    }
}

struct Job<'a> {
    spatial: Option<&'a Curve>,
    temporal: Option<&'a Curve>,
    reference_ready: bool,
    src: &'a [u8],
    src_stride: usize,
    width: usize,
    height: usize,
    horizontal: Option<&'a [u16]>,
    previous: &'a [u16],
}

impl Job<'_> {
    /// Vertical and temporal recurrence for columns `x0..x1`.
    ///
    /// The band is stored row by row: row `y` starts at `y * (x1 - x0)`.
    /// The first row's vertical reference is that row's own samples, so the
    /// horizontal result is mixed back toward the original pixels. Later
    /// rows use the previous row's spatial result, before the temporal pass.
    fn filter_columns(&self, x0: usize, x1: usize, out: &mut [u8], prev_out: &mut [u16]) {
        let band_width = x1 - x0;
        let mut vertical = vec![0i32; band_width];
        if self.spatial.is_some() {
            for (local, x) in (x0..x1).enumerate() {
                vertical[local] = promote(self.src[x]);
            }
        }

        for y in 0..self.height {
            let src_row = y * self.src_stride;
            let plane_row = y * self.width;
            let band_row = y * band_width;
            for (local, x) in (x0..x1).enumerate() {
                let current = promote(self.src[src_row + x]);
                let after_horizontal = match self.horizontal {
                    Some(rows) => i32::from(rows[plane_row + x]),
                    None => current,
                };
                let after_vertical = match self.spatial {
                    Some(curve) => {
                        let blended = curve.blend(vertical[local], after_horizontal);
                        vertical[local] = blended;
                        blended
                    }
                    None => after_horizontal,
                };
                let output = match (self.temporal, self.reference_ready) {
                    (Some(curve), true) => {
                        curve.blend(i32::from(self.previous[plane_row + x]), after_vertical)
                    }
                    _ => after_vertical,
                };
                out[band_row + local] = demote(output);
                prev_out[band_row + local] = output as u16;
            }
        }
    }
}

fn horizontal_row(spatial: Option<&Curve>, src_row: &[u8], dst: &mut [u16]) {
    let mut left: Option<i32> = None;
    for (sample, slot) in src_row.iter().zip(dst.iter_mut()) {
        let current = promote(*sample);
        let value = match (spatial, left) {
            (Some(curve), Some(previous)) => curve.blend(previous, current),
            _ => current,
        };
        *slot = value as u16;
        left = Some(value);
    }
}

fn gather_band_row<T: Copy>(
    bands: &[T],
    ranges: &[(usize, usize)],
    height: usize,
    y: usize,
    dst_row: &mut [T],
) {
    let mut offset = 0;
    for &(x0, x1) in ranges {
        let band_width = x1 - x0;
        let start = offset + y * band_width;
        dst_row[x0..x1].copy_from_slice(&bands[start..start + band_width]);
        offset += band_width * height;
    }
}

fn optional_curve(strength: f64) -> Result<Option<Curve>, Error> {
    if !strength.is_finite() || strength < 0.0 {
        return Err(Error::InvalidStrength);
    }
    Ok(Curve::new(strength))
}

fn check_plane(
    src_len: usize,
    dst_len: usize,
    width: usize,
    height: usize,
    src_stride: usize,
    dst_stride: usize,
) -> Result<(), Error> {
    if width == 0 || height == 0 || src_stride < width || dst_stride < width {
        return Err(Error::InvalidDimensions);
    }
    let src_need = plane_span(height, src_stride, width).ok_or(Error::InvalidDimensions)?;
    let dst_need = plane_span(height, dst_stride, width).ok_or(Error::InvalidDimensions)?;
    if src_len < src_need || dst_len < dst_need {
        return Err(Error::BufferTooSmall);
    }
    Ok(())
}

fn plane_span(height: usize, stride: usize, width: usize) -> Option<usize> {
    height
        .checked_sub(1)?
        .checked_mul(stride)?
        .checked_add(width)
}

fn copy_plane(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    src_stride: usize,
    dst_stride: usize,
    parallel: bool,
) {
    for_rows(dst, height, dst_stride, width, parallel, &|y, row| {
        let from = y * src_stride;
        row.copy_from_slice(&src[from..from + width]);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spatial(strength: f64) -> Filter {
        Filter::new(strength, 0.0).unwrap()
    }

    fn temporal(strength: f64) -> Filter {
        Filter::new(0.0, strength).unwrap()
    }

    #[test]
    fn zero_strength_copies_and_keeps_padding() {
        let filter = Filter::new(0.0, 0.0).unwrap();
        let src = [1u8, 2, 9, 3, 4, 9, 5, 6, 9];
        let mut dst = [0u8; 9];
        let mut state = PlaneState::new();
        filter
            .denoise(&mut state, &src, &mut dst, 2, 3, 3, 3)
            .unwrap();
        assert_eq!(dst, [1, 2, 0, 3, 4, 0, 5, 6, 0]);
        assert!(!state.has_reference());
    }

    #[test]
    fn flat_field_is_unchanged_across_frames() {
        let filter = Filter::new(12.0, 18.0).unwrap();
        let src = [90u8; 16];
        let mut dst = [0u8; 16];
        let mut state = PlaneState::new();
        for _ in 0..5 {
            filter
                .denoise_packed(&mut state, &src, &mut dst, 4, 4)
                .unwrap();
            assert_eq!(dst, src);
        }
    }

    #[test]
    fn top_left_of_the_first_frame_is_unchanged() {
        let filter = spatial(30.0);
        let src = [10u8, 80, 10, 80, 10, 40, 10, 40, 200];
        let mut dst = [0u8; 9];
        filter
            .denoise_packed(&mut PlaneState::new(), &src, &mut dst, 3, 3)
            .unwrap();
        assert_eq!(dst[0], src[0]);
    }

    #[test]
    fn past_pixels_do_not_see_a_later_impulse() {
        let filter = spatial(30.0);
        let mut src = [100u8; 49];
        src[3 * 7 + 3] = 116;
        let mut dst = [0u8; 49];
        filter
            .denoise_packed(&mut PlaneState::new(), &src, &mut dst, 7, 7)
            .unwrap();

        for y in 0..7 {
            for x in 0..7 {
                if y < 3 || (y == 3 && x < 3) {
                    let i = y * 7 + x;
                    assert_eq!(dst[i], src[i], "pixel {x},{y} changed");
                }
            }
        }
        assert_ne!(dst, src, "strength 30 should move a 16-level impulse");
    }

    #[test]
    fn single_column_is_one_vertical_blend() {
        // A one-pixel-wide plane has no horizontal neighbor. The first
        // sample is unchanged, and the next sample is one blend toward it.
        let filter = spatial(6.0);
        let src = [100u8, 104];
        let mut dst = [0u8; 2];
        filter
            .denoise_packed(&mut PlaneState::new(), &src, &mut dst, 1, 2)
            .unwrap();
        assert_eq!(dst[0], 100);
        assert_eq!(dst[1], blend_once(6.0, 100, 104));
    }

    #[test]
    fn temporal_second_frame_moves_toward_the_first() {
        let filter = temporal(6.0);
        let mut state = PlaneState::new();
        let mut first = [0u8; 1];
        let mut second = [0u8; 1];
        filter
            .denoise_packed(&mut state, &[100], &mut first, 1, 1)
            .unwrap();
        filter
            .denoise_packed(&mut state, &[104], &mut second, 1, 1)
            .unwrap();
        assert_eq!(first, [100]);
        assert_eq!(second, [blend_once(6.0, 100, 104)]);
        assert!(second[0] > 100 && second[0] < 104);
    }

    #[test]
    fn repeated_frame_stays_put_once_it_is_the_reference() {
        let filter = temporal(10.0);
        let mut state = PlaneState::new();
        let frame = [70u8; 4];
        let mut dst = [0u8; 4];
        filter
            .denoise_packed(&mut state, &frame, &mut dst, 2, 2)
            .unwrap();
        filter
            .denoise_packed(&mut state, &frame, &mut dst, 2, 2)
            .unwrap();
        assert_eq!(dst, frame);
    }

    #[test]
    fn reset_drops_the_temporal_pull() {
        let filter = temporal(20.0);
        let mut state = PlaneState::new();
        let mut out_a = [0u8; 1];
        let mut out_b = [0u8; 1];
        filter
            .denoise_packed(&mut state, &[20], &mut out_a, 1, 1)
            .unwrap();
        filter
            .denoise_packed(&mut state, &[80], &mut out_b, 1, 1)
            .unwrap();
        assert_ne!(out_b[0], 80);

        state.reset();
        let mut fresh = [0u8; 1];
        filter
            .denoise_packed(&mut state, &[80], &mut fresh, 1, 1)
            .unwrap();
        assert_eq!(fresh, [80]);
    }

    #[test]
    fn resize_discards_the_reference() {
        let filter = temporal(20.0);
        let mut state = PlaneState::new();
        let mut out = [0u8; 4];
        filter
            .denoise_packed(&mut state, &[20, 20, 20, 20], &mut out, 2, 2)
            .unwrap();
        let mut next = [0u8; 1];
        filter
            .denoise_packed(&mut state, &[80], &mut next, 1, 1)
            .unwrap();
        assert_eq!(next, [80]);
    }

    #[test]
    fn stride_matches_a_tightly_packed_plane() {
        let filter = Filter::new(8.0, 10.0).unwrap();
        let pixels = [
            12u8, 18, 40, 15, 90, 14, 16, 70, 11, 13, 19, 42, 17, 15, 80, 12,
        ];
        let mut tight_src = Vec::new();
        let mut padded = Vec::new();
        for y in 0..4 {
            for x in 0..3 {
                let sample = pixels[y * 3 + x];
                tight_src.push(sample);
                padded.push(sample);
            }
            padded.push(255);
        }
        let mut state_a = PlaneState::new();
        let mut state_b = PlaneState::new();
        let mut tight = [0u8; 12];
        let mut wide = [7u8; 16];
        for _ in 0..3 {
            filter
                .denoise_packed(&mut state_a, &tight_src, &mut tight, 3, 4)
                .unwrap();
            filter
                .denoise(&mut state_b, &padded, &mut wide, 3, 4, 4, 4)
                .unwrap();
            for y in 0..4 {
                assert_eq!(&wide[y * 4..y * 4 + 3], &tight[y * 3..y * 3 + 3], "row {y}");
                assert_eq!(wide[y * 4 + 3], 7, "padding written");
            }
        }
    }

    #[test]
    fn in_place_matches_out_of_place() {
        let filter = Filter::new(5.0, 9.0).unwrap();
        let frames = [[40u8, 44, 41, 60], [42, 47, 39, 58], [41, 45, 40, 55]];
        let mut separate = PlaneState::new();
        let mut inplace = PlaneState::new();
        for frame in frames {
            let mut dst = [0u8; 4];
            let mut buf = frame;
            filter
                .denoise_packed(&mut separate, &frame, &mut dst, 2, 2)
                .unwrap();
            filter
                .denoise_in_place(&mut inplace, &mut buf, 2, 2, 2)
                .unwrap();
            assert_eq!(buf, dst);
        }
    }

    #[test]
    fn output_stays_inside_the_samples_seen_so_far() {
        let filter = Filter::new(15.0, 25.0).unwrap();
        let mut state = PlaneState::new();
        let mut seen_min = 255u8;
        let mut seen_max = 0u8;
        let mut seed = 0x1234_5678u32;
        for _ in 0..6 {
            let mut src = [0u8; 25];
            for sample in &mut src {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                *sample = (seed >> 24) as u8;
                seen_min = seen_min.min(*sample);
                seen_max = seen_max.max(*sample);
            }
            let mut dst = [0u8; 25];
            filter
                .denoise_packed(&mut state, &src, &mut dst, 5, 5)
                .unwrap();
            for value in dst {
                assert!(value >= seen_min.saturating_sub(1));
                assert!(value <= seen_max.saturating_add(1));
            }
        }
    }

    #[test]
    fn short_buffer_and_bad_geometry_are_errors() {
        let filter = spatial(4.0);
        let src = [0u8; 4];
        let mut dst = [0u8; 4];
        let mut state = PlaneState::new();
        assert_eq!(
            filter
                .denoise_packed(&mut state, &src, &mut dst, 3, 2)
                .unwrap_err(),
            Error::BufferTooSmall
        );
        assert_eq!(
            filter
                .denoise_packed(&mut state, &src, &mut dst, 0, 2)
                .unwrap_err(),
            Error::InvalidDimensions
        );
        assert_eq!(
            filter
                .denoise(&mut state, &src, &mut dst, 2, 2, 1, 2)
                .unwrap_err(),
            Error::InvalidDimensions
        );
        assert!(!state.has_reference());
    }

    #[test]
    fn parallel_schedule_matches_one_thread() {
        let filter = Filter::new(4.0, 6.0).unwrap();
        let width = 48;
        let height = 30;
        let stride = width + 5;
        let mut seed = 0x9e37_79b9u32;
        let mut frames = Vec::new();
        for _ in 0..3 {
            let mut frame = vec![0u8; height * stride];
            for y in 0..height {
                for x in 0..width {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    frame[y * stride + x] = (seed >> 24) as u8;
                }
            }
            frames.push(frame);
        }

        let mut serial_state = PlaneState::new();
        let mut parallel_state = PlaneState::new();
        for frame in &frames {
            let mut serial = vec![7u8; height * stride];
            let mut parallel = vec![7u8; height * stride];
            filter
                .denoise_with(
                    &mut serial_state,
                    frame,
                    &mut serial,
                    width,
                    height,
                    stride,
                    stride,
                    1,
                )
                .unwrap();
            filter
                .denoise_with(
                    &mut parallel_state,
                    frame,
                    &mut parallel,
                    width,
                    height,
                    stride,
                    stride,
                    5,
                )
                .unwrap();
            assert_eq!(serial, parallel);
            assert!(serial_state.has_reference());
        }
    }

    #[test]
    fn negative_strength_is_rejected() {
        assert_eq!(Filter::new(-1.0, 0.0).unwrap_err(), Error::InvalidStrength);
        assert_eq!(
            Filter::new(1.0, f64::NAN).unwrap_err(),
            Error::InvalidStrength
        );
    }

    /// One low-pass step, duplicated here so the test states the contract
    /// instead of calling back into the curve table.
    fn blend_once(strength: f64, previous: u8, current: u8) -> u8 {
        let prev = i32::from(previous) * 256 + 128;
        let cur = i32::from(current) * 256 + 128;
        let delta = f64::from(prev - cur);
        let levels = delta.abs() / 256.0;
        let base = 1.0 - strength / 255.0;
        let gamma = 0.25f64.ln() / base.ln();
        let similarity = (1.0 - levels / 255.0).max(0.0);
        let mixed = f64::from(cur) + similarity.powf(gamma) * delta;
        ((mixed.round() as i32).clamp(0, 65535) >> 8) as u8
    }
}
