# deltamesh

[![CI](https://github.com/me0w2en/deltamesh/actions/workflows/ci.yml/badge.svg)](https://github.com/me0w2en/deltamesh/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/deltamesh.svg)](https://crates.io/crates/deltamesh)
[![docs.rs](https://docs.rs/deltamesh/badge.svg)](https://docs.rs/deltamesh)

Incremental meshing of streamed point-cloud segments.

deltamesh turns point clouds that arrive segment by segment into a triangle mesh split into fixed-size blocks.
When a new segment arrives, only the blocks it touches are rebuilt, and only blocks whose mesh actually changed are
reported. A quick, rough *preview* of a segment can be shown first and later replaced by its *refined* version; the
replacement is exact, so the result is bit-for-bit the same as if the preview had never been ingested.

It was written for live 3D reconstruction from drone imagery, where a full re-mesh after every segment takes seconds
to minutes while the incremental update takes tens of milliseconds (see [BENCHMARKS.md](BENCHMARKS.md)).

## How it works

1. **Binning.** Points are averaged into small cubic bins. Normals are taken from the input or estimated from
   neighbouring bins (PCA).
2. **Orientation.** Normal signs are made consistent by propagation over a neighbour graph, optionally checked against
   the already accumulated refined surface.
3. **Accumulation.** Each bin adds a Gaussian-weighted signed distance to a sparse voxel field stored in 32³-voxel
   blocks. Every preview segment writes to its own layer, so replacing it is a matter of dropping that layer.
4. **Extraction.** Surface Nets extracts triangles from each changed block. Vertices on block boundaries are
   bit-identical between neighbours, so blocks join without cracks.
5. **Simplification** (optional). meshoptimizer reduces each block while keeping boundary vertices locked.

All steps are deterministic: the same input produces the same bytes regardless of the number of threads.

## Usage

```toml
[dependencies]
deltamesh = "0.1"
```

```rust
use deltamesh::{sdf::SdfMesher, Config, Level, Mesher, Point};

let mut mesher = SdfMesher::new(Config::default());

// A preview segment, then the refined version of the same segment.
let preview: Vec<Point> = load_points("segment0_preview.ply");
let refined: Vec<Point> = load_points("segment0_refined.ply");

mesher.ingest(0, Level::Preview, &preview)?;
for id in mesher.extract() {
    let block = mesher.mesh(&id).unwrap();
    upload(block); // send to the viewer
}

mesher.ingest(0, Level::Refined, &refined)?; // replaces the preview exactly
for id in mesher.extract() {
    upload(mesher.mesh(&id).unwrap());
}
```

`extract` returns only the blocks whose content changed since the last call.

### Block output

Each `BlockMesh` has:

| Field | Type | Meaning |
|---|---|---|
| `id` | `[i32; 3]` | Block coordinate. The block origin is `id * block_dim * voxel` in input coordinates. |
| `version` | `u32` | Increases only when the block mesh actually changes. Keep a block only if its version is newer than yours. |
| `positions` | `Vec<f32>` | `x y z` per vertex, absolute coordinates. |
| `colors` | `Vec<u8>` | `r g b` per vertex. |
| `indices` | `Vec<u32>` | Three vertex indices per triangle, counter-clockwise when seen from outside. |

A block with zero triangles means "remove this block". Normals are not included; compute them on the client
(for example `computeVertexNormals()` in three.js).

### Configuration

`Config::default()` uses a 0.2 m voxel, 32³-voxel blocks, 0.1 m bins, a 0.4 m splat radius and no simplification.
Smaller voxels give more detail at the cost of memory and time. `simplify_error` (metres) enables per-block
simplification.

### GPU

Enable the `gpu` feature to run on the GPU through [wgpu](https://wgpu.rs) (Metal, Vulkan, DX12):

```toml
deltamesh = { version = "0.1", features = ["gpu"] }
```

```rust
use deltamesh::{gpu::GpuCtx, gpu_mesher::GpuMesher, Config};

let ctx = GpuCtx::shared()?; // one device per process
let mut mesher = GpuMesher::new(ctx, Config::default())?;
```

`GpuMesher` keeps the distance field in GPU memory and runs binning, normal estimation, accumulation and extraction on
the GPU; normal orientation and simplification stay on the CPU. Its output is not bit-identical to the CPU path
(triangle counts differ by about 0.001%) but is deterministic on a given device.

Open a single GPU device per process and share it between meshers (`GpuCtx::shared()`). Creating and destroying several
devices concurrently can deadlock inside some Vulkan drivers.

Software rasterizers such as llvmpipe are never selected. Set `DELTAMESH_GPU_ADAPTER=<part of the adapter name>` to
choose a specific adapter.

## Command-line tool

The `cli/` crate builds a `deltamesh` binary used for benchmarks. It is not published to crates.io.

```sh
cargo build --release
./target/release/deltamesh --layout segments --data path/to/segments --out out/run1
./target/release/deltamesh --list-gpus
```

Input layouts:

- `segments`: a folder with `r{k}_preview_new.ply` and `r{k}_refined.ply` for each segment `k`.
- `chunks`: a folder of `tc_*.ply` files; a synthetic preview is made from each chunk.

Input PLY files are binary little-endian with `x y z` as `float`, `nx ny nz` as `float` and `red green blue` as `uchar`.
The tool writes a per-step CSV (`bench.csv`) and, unless `--no-ply` is given, a merged PLY after each step.
Run `./target/release/deltamesh --help` for all options.

## Building and testing

```sh
cargo test --release                       # CPU tests; GPU tests are skipped when no GPU is found
cargo build --release --no-default-features # CLI without GPU support
```

Minimum supported Rust version: 1.87.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this work by you, as
defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
