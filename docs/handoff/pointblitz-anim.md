# Handoff: pointblitz animated point layer (feat/pointblitz-anim)

## Goal
Per-point animation for the status board's point layer, interpolated on the GPU: fade-in 150 ms,
fade-out 150 ms, glide 700 ms ease-out, driven by a `now` uniform; partial buffer updates (<= 200k
points per frame); growable capacity. Kept inside a re-vendored pointblitz-core; existing layers untouched.

## State of the tree
develop had dropped `third_party/pointblitz` (6665f61, crates.io 0.2.1). This branch re-vendors only
`pointblitz-core` 0.2.1 at `third_party/pointblitz/pointblitz-core` and patches it in root `Cargo.toml`
`[patch.crates-io]` (`pointblitz-core = { path = ... }`). See `third_party/pointblitz/PATCHES.md`.

## API (`pointblitz_core::animated`, re-exported `AnimatedPoints`, `ColorMode`)
- `AnimatedPoints::new(device, color_format, origin:[f64;3], capacity)`; capacity doubles when full.
- `apply(device, queue, &[Add{key,pos:[f64;3],rgb}], &[Move{key,pos,rgb:Option}], &[u64] removes, now:f64) -> ApplyStats`
  (types in `animated::book`). Order: recycle, removes, moves, adds. Add of live key = move; unknown move/remove -> `missing`.
- `prepare(device, queue, size, &Camera, now, fog)` then `draw(&mut RenderPass)` inside `begin_pass`.
- `set_point_size(PointSize)`, `set_color_mode(ColorMode::{Rgb, Height{low,high}, Flat})`, `set_capacity`, `set_origin`.
- Full usage notes: scratchpad status file `status/pblitz.md` (may be gone); the above is the contract.

## Done
- `src/animated/book.rs`: CPU bookkeeping (slots, free set, fade-out recycle queue, dirty-range coalescing with
  16-slot gap bridging, capacity growth, origin shift, f32 time base with rebase after 4 h). 11 unit tests.
- `src/animated.rs`: wgpu layer (32 B/slot instance buffer, one opaque pipeline, uniform 176 B, partial
  `write_buffer` per dirty range, realloc + re-upload from CPU shadow on growth).
- `src/animated.wgsl`: vertex shader does mode/alpha/glide from `cam.params.w`; opacity is screen-door dithered
  (interleaved gradient noise) so the pipeline stays opaque + depth-correct. Modes 0 static,1 in,2 out,3 glide,4 dead.
  Durations come from the uniform (`durations`), constants in `book.rs`. Height colour = legacy ramp * 0.91.
- Tests: `tests/animated.rs` (GPU, headless; registered as `[[test]]` in the core `Cargo.toml`).
- Verified: `cargo test -p pointblitz-core`, `cargo check -p skylens-client`, `PROFILE=dev bash src/skylens_client/build-web.sh`,
  and WGSL -> GLSL ES 300 (WebGL) via naga for both entry points.

## Remaining
1. Board integration (other agent, branch feat/board-keyed-live): create `AnimatedPoints` in the viewer's GPU struct
   (`src/shared/viewer/view3d.rs`, next to `points: Renderer`), call `prepare` + `draw` in the frame pass
   (around `points.draw`, ~line 560-570), feed `apply` from the board's keyed point diff. No glue was added here.
2. Keep requesting frames while animating (book().fading() > 0 or last apply < 0.7 s ago).
3. Not browser-tested: the wasm build links, shader GLSL generation passes, but no run in a real WebGL2/WebGPU page.
4. Optional: per-point colour tween, PATCHES.md removal when upstreamed, enforce FRAME_BUDGET if desired.

## Test
    cargo test -p pointblitz-core                # unit + GPU tests (GPU tests skip without an adapter)
    cargo check -p skylens-client
    PROFILE=dev bash src/skylens_client/build-web.sh   # needs wasm-bindgen-cli matching Cargo.lock
