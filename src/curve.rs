/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! Coefficient curve for the edge-aware low-pass.
//!
//! A strength value is the sample difference, on an 8-bit scale, at which
//! the filter keeps 25% of its running average and 75% of the new sample.
//! Similarity between two samples is `1 - |difference| / 255`, and the
//! blend weight is that similarity raised to a gamma chosen to hit 0.25 at
//! the configured strength.

use std::fmt;

/// Largest strength at which `1 - strength / 255` is still safely above zero.
const MAX_DISTANCE: f64 = 254.0;
const FULL_SCALE: f64 = 255.0;
const QUARTER: f64 = 0.25;

/// Fraction of the previous filtered sample to keep.
///
/// `strength` is the 8-bit level difference at which the weight is 0.25.
/// `level_difference` is an absolute difference on that same scale.
/// A strength of zero or a non-finite input disables the contribution, so
/// the weight is zero. Strengths above 254 are treated as 254, which is as
/// far as this curve can go while its logarithm stays defined.
pub fn weight(strength: f64, level_difference: f64) -> f64 {
    let Some(gamma) = gamma(strength) else {
        return 0.0;
    };
    if !level_difference.is_finite() {
        return 0.0;
    }
    let similarity = (1.0 - level_difference.abs() / FULL_SCALE).max(0.0);
    if similarity == 0.0 {
        0.0
    } else {
        similarity.powf(gamma)
    }
}

pub(crate) fn gamma(strength: f64) -> Option<f64> {
    if !strength.is_finite() || strength <= 0.0 {
        return None;
    }
    let distance = strength.min(MAX_DISTANCE);
    let base = 1.0 - distance / FULL_SCALE;
    if base <= 0.0 || base >= 1.0 {
        return None;
    }
    Some(QUARTER.ln() / base.ln())
}

/// Precomputed weights indexed by absolute difference in the 8.8 domain.
#[derive(Clone)]
pub(crate) struct Curve {
    keep: Vec<f32>,
}

impl fmt::Debug for Curve {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Curve")
    }
}

impl Curve {
    pub(crate) fn new(strength: f64) -> Option<Self> {
        let gamma = gamma(strength)?;
        let mut keep = Vec::with_capacity(65536);
        for difference in 0..65536u32 {
            let levels = f64::from(difference) / 256.0;
            let similarity = (1.0 - levels / FULL_SCALE).max(0.0);
            let weight = if similarity == 0.0 {
                0.0
            } else {
                similarity.powf(gamma)
            };
            keep.push(weight as f32);
        }
        Some(Self { keep })
    }

    /// `previous` and `current` are 8.8 samples. The result stays in that domain.
    #[inline]
    pub(crate) fn blend(&self, previous: i32, current: i32) -> i32 {
        let delta = previous - current;
        let index = (delta.unsigned_abs() as usize).min(self.keep.len() - 1);
        let mixed = f64::from(current) + f64::from(self.keep[index]) * f64::from(delta);
        mixed.round().clamp(0.0, 65535.0) as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quarter_weight_lands_on_the_configured_distance() {
        for strength in [1.0, 4.0, 6.0, 12.5, 30.0, 80.0] {
            let at = weight(strength, strength);
            assert!(
                (at - 0.25).abs() < 1e-12,
                "strength {strength}: weight {at}"
            );
        }
    }

    #[test]
    fn identical_samples_keep_the_running_average() {
        assert_eq!(weight(4.0, 0.0), 1.0);
    }

    #[test]
    fn weight_falls_as_the_difference_grows() {
        let mut previous = weight(8.0, 0.0);
        for step in 1..=40 {
            let next = weight(8.0, f64::from(step));
            assert!(next < previous, "step {step}: {next} >= {previous}");
            previous = next;
        }
        assert_eq!(weight(8.0, 255.0), 0.0);
        assert_eq!(weight(8.0, 1000.0), 0.0);
    }

    #[test]
    fn disabled_or_unusable_strength_keeps_nothing() {
        assert_eq!(weight(0.0, 0.0), 0.0);
        assert_eq!(weight(-4.0, 1.0), 0.0);
        assert_eq!(weight(f64::NAN, 1.0), 0.0);
        assert_eq!(weight(4.0, f64::INFINITY), 0.0);
    }

    #[test]
    fn extreme_strength_still_has_a_finite_curve() {
        let at_cap = weight(10_000.0, 254.0);
        assert!(at_cap.is_finite());
        assert!((at_cap - weight(254.0, 254.0)).abs() < 1e-12);
    }
}
