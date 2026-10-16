# pointblitz-core changes

Base: pointblitz-core 0.2.1 (crates.io), unchanged except:

- `src/animated.rs`, `src/animated/book.rs`, `src/animated.wgsl`: `AnimatedPoints`, a point layer with
  per-point fade-in (150 ms), fade-out (150 ms) and glide (700 ms ease-out) interpolated in the vertex
  shader from a `now` uniform; keyed adds/moves/removes, partial buffer uploads, growable capacity.
- `src/lib.rs`: exports the module. `src/renderer.rs`: `focal_px` is `pub(crate)`.

Existing layers are untouched. Drop this patch when the layer lands in a pointblitz release.
