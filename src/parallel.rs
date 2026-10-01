/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! Row and column scheduling for the rayon pool.
//!
//! The horizontal pass has no dependency between rows, so rows run side by
//! side. The vertical pass carries state down a column and nowhere else, so
//! columns run side by side. Both produce the same samples as one thread.

use rayon::join;

/// Run `f` on each row. `stride` is the row pitch in elements and may be
/// wider than `width`. The last row only needs `width` elements.
pub(crate) fn for_rows<T: Send>(
    buf: &mut [T],
    height: usize,
    stride: usize,
    width: usize,
    parallel: bool,
    f: &(impl Fn(usize, &mut [T]) + Sync),
) {
    if !parallel || height <= 1 {
        for y in 0..height {
            let offset = y * stride;
            f(y, &mut buf[offset..offset + width]);
        }
        return;
    }
    split_rows(buf, 0, height, stride, width, f);
}

fn split_rows<T: Send>(
    buf: &mut [T],
    y0: usize,
    height: usize,
    stride: usize,
    width: usize,
    f: &(impl Fn(usize, &mut [T]) + Sync),
) {
    if height <= 1 {
        if height == 1 {
            f(y0, &mut buf[..width]);
        }
        return;
    }
    let mid = height / 2;
    let (left, right) = buf.split_at_mut(mid * stride);
    join(
        || split_rows(left, y0, mid, stride, width, f),
        || split_rows(right, y0 + mid, height - mid, stride, width, f),
    );
}

/// Contiguous, non-empty column ranges covering `0..width`.
pub(crate) fn column_ranges(width: usize, bands: usize) -> Vec<(usize, usize)> {
    let bands = bands.max(1).min(width.max(1));
    (0..bands)
        .map(|band| {
            let start = width * band / bands;
            let end = width * (band + 1) / bands;
            (start, end)
        })
        .filter(|(start, end)| start < end)
        .collect()
}

/// Run `f` on each column range. `left` and `right` are column-band major:
/// band `(x0, x1)` occupies `(x1 - x0) * height` elements, rows stored
/// contiguously inside the band.
pub(crate) fn for_bands<T: Send, U: Send>(
    left: &mut [T],
    right: &mut [U],
    ranges: &[(usize, usize)],
    height: usize,
    parallel: bool,
    f: &(impl Fn(usize, usize, &mut [T], &mut [U]) + Sync),
) {
    if ranges.is_empty() {
        return;
    }
    if !parallel || ranges.len() == 1 {
        let (x0, x1) = ranges[0];
        f(x0, x1, left, right);
        return;
    }
    split_bands(left, right, ranges, height, f);
}

fn split_bands<T: Send, U: Send>(
    left: &mut [T],
    right: &mut [U],
    ranges: &[(usize, usize)],
    height: usize,
    f: &(impl Fn(usize, usize, &mut [T], &mut [U]) + Sync),
) {
    if ranges.len() <= 1 {
        if let Some(&(x0, x1)) = ranges.first() {
            f(x0, x1, left, right);
        }
        return;
    }
    let mid = ranges.len() / 2;
    let (low, high) = ranges.split_at(mid);
    let low_elems: usize = low.iter().map(|(start, end)| (end - start) * height).sum();
    let (left_low, left_high) = left.split_at_mut(low_elems);
    let (right_low, right_high) = right.split_at_mut(low_elems);
    join(
        || split_bands(left_low, right_low, low, height, f),
        || split_bands(left_high, right_high, high, height, f),
    );
}

pub(crate) fn thread_count() -> usize {
    // The live pool, so RAYON_NUM_THREADS is honored. available_parallelism
    // would keep splitting after the pool had been limited to one thread.
    rayon::current_num_threads().max(1)
}

/// Serial below this many pixels. A small plane spends more time waking
/// workers than filtering.
const PARALLEL_PIXELS: usize = 16_384;

pub(crate) fn auto_bands(width: usize, height: usize) -> usize {
    let threads = thread_count();
    let pixels = width.saturating_mul(height);
    if threads <= 1 || pixels < PARALLEL_PIXELS {
        1
    } else {
        threads.min(width).max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::Mutex;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn column_ranges_cover_the_width_without_gaps() {
        for width in 1..33 {
            for bands in 1..9 {
                let ranges = column_ranges(width, bands);
                assert_eq!(ranges.first().unwrap().0, 0);
                assert_eq!(ranges.last().unwrap().1, width);
                for pair in ranges.windows(2) {
                    assert_eq!(pair[0].1, pair[1].0);
                }
            }
        }
    }

    #[test]
    fn parallel_bands_run_on_more_than_one_thread() {
        if thread_count() < 2 {
            return;
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let seen = Mutex::new(HashSet::new());
        pool.install(|| {
            let ranges = column_ranges(4, 4);
            let mut left = vec![0u8; 4 * 8];
            let mut right = vec![0u16; 4 * 8];
            for_bands(&mut left, &mut right, &ranges, 8, true, &|_, _, _, _| {
                seen.lock().unwrap().insert(thread::current().id());
                thread::sleep(Duration::from_millis(30));
            });
        });
        assert!(
            seen.lock().unwrap().len() > 1,
            "column bands stayed on one thread"
        );
    }
}
