# Internals, benchmarks, release engineering

## What this fork changes

The stock DFN3 ONNX export bakes GRU hidden state inside each `GRU` op and
leaves the time dim symbolic. `libDF/src/tract.rs` then unrolls that into a
per-frame, stateless graph at model-load time via
`tract_pulse::PulsedModel`. The Intel NPU plugin only accepts fully static
graphs with explicit state tensors, so this fork does three things:

1. **ONNX graph surgery** — `scripts/npu_export.py` rewrites each GRU op so
   `initial_h` (input[5]) and `Y_h` (output[1]) become graph-level tensors
   `h<N>_in` / `h<N>_out`, and pins the `S` sequence dim to 1. Weights are
   unchanged; no Python training dependency needed.
2. **Rust backend** — `libDF/src/openvino.rs` (behind the `openvino` cargo
   feature) mirrors the public surface of `DfTract` (`process`,
   `set_atten_lim`, `set_pf_beta`, pub fields `ch`, `sr`, `hop_size`, …) so
   `DfPlugin` in the LADSPA crate can alias either via a feature flag. Uses
   the `openvino` crate 0.9 with `runtime-linking`, so the `.so` has no
   compile-time link against OpenVINO.
3. **Feature split in ladspa/** — `tract-backend` (default, unchanged
   behaviour) vs `openvino-backend` (opt-in). Exactly one must be picked.
   The model tarballs are different because the two backends expect
   different ONNX shapes.

## Repository layout

```
libDF/src/openvino.rs                  OpenVINO backend
libDF/tests/openvino_smoke.rs          runtime smoke test (ch=1 + ch=2) + full-pipeline bench
libDF/tests/tract_bench.rs             CPU bench on the tract backend for comparison
ladspa/                                LADSPA plugin, backend-selection via cargo feature
scripts/npu_export.py                  ONNX graph surgery
models/DeepFilterNet3_ll_onnx_npu.tar.gz   NPU-ready low-latency DFN3
models/DeepFilterNet3_onnx_npu.tar.gz      NPU-ready DFN3
tooling/build_deb.sh                   .deb builder (depends on Intel OpenVINO apt repo)
tooling/ladspa_probe/                  dlopen-based smoke test for the built .so
```

## Running the backend directly

```bash
cargo test --release -p deep_filter \
    --no-default-features --features "openvino default-model-ll" \
    --test openvino_smoke openvino_backend_runs -- --nocapture
```

Force a device:

```bash
DFN_OV_DEVICE=NPU cargo test --release -p deep_filter \
    --no-default-features --features "openvino default-model-ll" \
    --test openvino_smoke openvino_backend_full_pipeline_bench -- --nocapture
```

Run the dlopen smoke test against the built `.so`:

```bash
cargo run --release -p ladspa-probe -- ~/.ladspa/libdeep_filter_ladspa.so
```

## OpenVINO runtime discovery

At first use the plugin calls `ensure_openvino_lib_path()` in
`libDF/src/openvino.rs`, which scans, in order:

1. `$DFN_OV_LIBS_DIR` (env override; dev against a pip wheel for example)
2. `option_env!("DFN_OV_LIBS_DIR")` baked at build time (unused by `build_deb.sh`, kept as an escape hatch)
3. `/usr/lib` (Intel APT layout)
4. `/usr/lib/x86_64-linux-gnu` (Debian multiarch)
5. `/opt/intel/openvino/runtime/lib/intel64` (Intel tarball)

First hit containing `libopenvino_c.so.*` wins. A per-process shim at
`$TMPDIR/dfn-ovshim-<pid>/` gets unversioned symlinks to the versioned libs
so `openvino-finder` accepts them. Shim + real dir are both prepended to
`LD_LIBRARY_PATH` for this process. No persistent symlinks, no env drop-ins,
nothing to clean up across OpenVINO upgrades.

## Device selection

`DfOpenVino::new` picks a device with this precedence (logged at `info`):

| `DFN_OV_DEVICE`         | Behaviour                                                         |
|-------------------------|-------------------------------------------------------------------|
| unset / `AUTO` / empty  | First available of `NPU`, `GPU`, `CPU`.                           |
| `CPU` / `GPU` / `NPU`   | Force; fails if not available.                                    |
| any other string        | Passed as-is to `ov::Core::compile_model` (e.g. `HETERO:NPU,CPU`).|

## Benchmarks

Core Ultra 7 165H (Meteor Lake), DFN3 low-latency, 200-hop average,
full-stage pipeline (min_db_thresh lowered so every hop runs encoder +
erb_dec + df_dec):

```
backend                   avg wall/hop   avg CPU/hop   CPU frac of audio (10 ms)
tract (baseline)          1.76 ms        1.76 ms        17.6 %
openvino CPU              1.51 ms        1.49 ms        14.9 %
openvino NPU              1.91 ms        0.16 ms         1.6 %
openvino HETERO:NPU,CPU   1.75 ms        0.15 ms         1.5 %
```

NPU wall time is comparable to tract, but **CPU time drops ~11×** —
the point of the fork. Reproduce:

```bash
# tract baseline
cargo test --release -p deep_filter --features "tract default-model-ll" \
    --test tract_bench -- --nocapture

# openvino full pipeline
DFN_OV_DEVICE=NPU cargo test --release -p deep_filter \
    --no-default-features --features "openvino default-model-ll" \
    --test openvino_smoke openvino_backend_full_pipeline_bench -- --nocapture
```

## Contributing upstream

The diff is designed to be accepted into
[Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) without
disturbing the tract backend. Sendable paths:

```
libDF/Cargo.toml                adds `openvino` feature + optional dep
libDF/src/lib.rs                gates `pub mod openvino;` under cfg
libDF/src/openvino.rs           new backend
libDF/tests/openvino_smoke.rs   feature-gated smoke + bench
libDF/tests/tract_bench.rs      baseline bench (tract only)
ladspa/Cargo.toml               splits `tract-backend` (default) / `openvino-backend`
ladspa/src/lib.rs               minimal diff, `DfTract as DfModel`
scripts/npu_export.py           ONNX surgery
```

Not sendable upstream:

```
tooling/**                      packaging glue + smoke-test host
models/*_npu.tar.gz             35 MB each — ask the maintainer to prefer a release asset
README.md / DEVELOPMENT.md      fork-specific docs
```

Suggested PR flow:

1. Open an issue first. Propose the backend, attach the bench table above.
2. Commit split: one for `scripts/npu_export.py`, one for the libDF
   backend + tests, one for the ladspa feature split.
3. CI: add `cargo check --features openvino` to the existing GH Actions
   matrix — `runtime-linking` means CI does not need OpenVINO installed
   to compile.
4. `cargo clippy --features openvino -- -D warnings`, `cargo fmt`.
5. Optional SNR parity test: render a reference wav through tract and
   OpenVINO-CPU, assert SNR above 60 dB.

## Known limitations

* **NPU op coverage.** Older NPU firmwares may reject a GRU op; use
  `DFN_OV_DEVICE=HETERO:NPU,CPU` so unsupported nodes fall back to CPU.
* **fp16 / quantisation.** Bundled tarballs are fp32. For best NPU
  performance run `ovc --compress_to_fp16` (or `nncf` int8) on the IR.
  Not automated here.
* **Multi-Core-per-process.** Creating more than one `ov::Core` in the
  same process has been observed to SIGSEGV on teardown under 2026.1.
  EasyEffects only makes one instance per plugin slot, so this does not
  bite in practice; the second smoke test is `#[ignore]`d to avoid
  clashing with the first.
* **Stereo.** Handled by serial per-channel inference (two passes per
  hop). Doubles inference wall time but stays well under the real-time
  budget. A batch=ch compile would be faster; left as future work.
