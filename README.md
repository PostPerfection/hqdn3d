# hqdn3d

A clean-room implementation, in Rust, of the hqdn3d video denoiser.

hqdn3d is the high-quality 3D denoiser known from MPlayer and FFmpeg. It
reduces grain and sensor noise, which also makes the picture easier to
compress. This crate implements that filter from its published parameters
and the shape of its coefficient curve. It is not derived from the
MPlayer, FFmpeg, or VLC sources (those are GPL). The code here is MPL 2.0.

Historical implementations quantize the coefficient curve into a lookup
table. This one evaluates the same curve directly and keeps an 8.8
fixed-point average between samples. At the ordinary defaults the two
agree to within a level or two, and the average pixel is much closer
than that. They are not bit-identical, especially at very high
strengths, where the table's bins show up.

## The filter

Each strength is a distance on an 8-bit scale (0–255). The blend weight
toward the running average is

```text
similarity = clamp(1 - |difference| / 255, 0, 1)
weight     = similarity ^ gamma
```

`gamma` is chosen so the weight is exactly 0.25 when `|difference|` equals
the strength. Equal samples are replaced by the running average. A real
edge, much larger than the strength, is left alone.

For every pixel the filter runs three causal passes:

1. Horizontal, left to right, within the row.
2. Vertical, against the previous row. The first row's reference is the
   row itself, so that row is pulled back toward its original samples
   after the horizontal pass.
3. Temporal, against the same pixel in the previous denoised frame. The
   first frame has no reference and is filtered spatially only.

Horizontal and vertical share the spatial strength. Samples are lifted to
8.8 fixed point before the passes, so a fraction of a level can accumulate
instead of being rounded off at every pixel.

A strength of zero turns that pass off. FFmpeg's option parser treats an
explicit zero as "derive the default" instead; to disable a pass there,
people pass a tiny positive number.

## Strengths

| Parameter | Default | Meaning |
| --- | --- | --- |
| luma spatial | 4 | within-frame luma |
| chroma spatial | 3 | within-frame chroma (`3/4` of luma spatial) |
| luma temporal | 6 | across-frame luma (`6/4` of luma spatial) |
| chroma temporal | 4.5 | across-frame chroma (luma temporal scaled by the spatial chroma/luma ratio) |

`Strength::from_luma_spatial` applies those ratios. `Strength::default` is
`4, 3, 6, 4.5`.

## Library

```rust
use hqdn3d::{Hqdn3d, PlaneState, Strength};

let filter = Hqdn3d::new(Strength::default())?;
let mut luma = PlaneState::new();
let src = [/* tightly packed 8-bit luma */];
let mut dst = vec![0u8; src.len()];
filter.denoise_luma(&mut luma, &src, &mut dst, width, height, width, width)?;
```

Keep one `PlaneState` per plane and reuse it for later frames. Call
`reset` on a scene cut, or whenever the old frame should stop pulling on
the new one. U and V each need their own state; they share the chroma
filter. Row strides may be wider than the picture.

`denoise_yuv420` runs the three planes at the same time. Inside a plane,
work of about 128×128 and up is split across a rayon thread pool. The
horizontal pass divides the rows across the pool, then the vertical and
temporal passes take a contiguous band of columns each. That is the same
recurrence as a single thread, so the picture does not change when more
cores are used. Smaller planes stay on the calling thread.

## Command line

`hqdn3d` reads raw frames from stdin and writes raw frames to stdout. It
does not parse a container.

```sh
# gray, defaults 4:3:6:4.5
hqdn3d --width 1920 --height 1080 < in.gray > out.gray

# 4:2:0, one knob
hqdn3d --width 1920 --height 1080 --format yuv420p --luma-spatial 8 \
  < in.yuv > out.yuv

# spatial only
hqdn3d --width 640 --height 360 --luma-temporal 0 --chroma-temporal 0 \
  < in.gray > out.gray
```

`--format` is `gray` (the default) or `yuv420p` (even dimensions, no row
padding).

## License

[Mozilla Public License 2.0](https://www.mozilla.org/MPL/2.0/). See `LICENSE`.
Each source file carries that notice. Files can be combined into a larger
work under other terms; changes to these files stay under MPL 2.0.
