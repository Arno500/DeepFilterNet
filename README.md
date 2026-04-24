# deep-filter-openvino

Fork of [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) with an
OpenVINO inference backend so DeepFilterNet3 runs on the **Intel NPU** (Meteor
Lake / Core Ultra and newer), Intel GPU, or CPU. Same denoise quality as
upstream, same LADSPA controls, but about **11× less CPU time per audio
frame** on the NPU — designed for low-battery voice denoise in EasyEffects
and other LADSPA hosts.

## Quick install (Debian/Ubuntu, amd64)

On any Debian-derived distribution with the OpenVINO APT repo reachable:

```bash
# 1. Intel OpenVINO APT repo.
curl -sSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
    | sudo gpg --dearmor -o /usr/share/keyrings/intel-openvino.gpg
echo "deb [signed-by=/usr/share/keyrings/intel-openvino.gpg] https://apt.repos.intel.com/openvino ubuntu24 main" \
    | sudo tee /etc/apt/sources.list.d/intel-openvino.list
sudo apt update

# 2. Install the plugin. apt pulls in libopenvino and its device plugins.
sudo apt install ./build/deep-filter-openvino_0.5.7_amd64.deb

# 3. Any LADSPA host (e.g. EasyEffects) -> pick DeepFilter in an input slot.
sudo apt install easyeffects
```

The `ubuntu24 main` component is the one listed by the
[official docs](https://docs.openvino.ai/2026/get-started/install-openvino/install-openvino-apt.html)
and works across recent Debian/Ubuntu releases; Intel has not yet published
a separate component per release.

Build the `.deb` yourself:

```bash
tooling/build_deb.sh
# -> build/deep-filter-openvino_0.5.7_amd64.deb
```

## Optional: run on the NPU

For Meteor Lake / Core Ultra CPUs, three more steps unlock the NPU:

```bash
# a) Your user must be in the 'render' group to access /dev/accel/accel0.
sudo gpasswd -a $USER render

# b) Intel NPU user-space driver -- not in apt yet, fetched from GitHub.
mkdir -p /tmp/npu && cd /tmp/npu
curl -LO https://github.com/intel/linux-npu-driver/releases/download/v1.32.1/linux-npu-driver-v1.32.1.20260422-24767473183-ubuntu2404.tar.gz
tar xzf linux-npu-driver-*.tar.gz
sudo dpkg -i intel-fw-npu_*.deb intel-level-zero-npu_*.deb intel-driver-compiler-npu_*.deb

# c) Log out + in so the render group takes effect, then verify.
python3 -c "import openvino as ov; print(ov.Core().available_devices)"
# expected: ['CPU', 'GPU', 'NPU']
```

The plugin picks the NPU automatically when it is present.

## Selecting the device

`DFN_OV_DEVICE` overrides the default auto-selection:

| Command                                     | What it does                                |
|---------------------------------------------|---------------------------------------------|
| unset (or `AUTO`, empty)                    | First available of NPU → GPU → CPU          |
| `DFN_OV_DEVICE=NPU easyeffects`             | NPU only (fails if absent)                  |
| `DFN_OV_DEVICE=CPU easyeffects`             | Force CPU (useful for A/B tests)            |
| `DFN_OV_DEVICE=GPU easyeffects`             | Integrated GPU                              |
| `DFN_OV_DEVICE=HETERO:NPU,CPU easyeffects`  | NPU where supported, CPU fallback per op    |

Persistent setting:

```bash
mkdir -p ~/.config/environment.d
echo 'DFN_OV_DEVICE=NPU' > ~/.config/environment.d/50-deepfilter.conf
```

then log out and back in.

## How the plugin finds OpenVINO

Intel's OpenVINO APT packages install the core libraries into `/usr/lib/`
(flat, not multi-arch) and the device plugins into
`/usr/lib/openvino-<version>/`. The Rust `openvino-finder` used by the
plugin does not search `/usr/lib` by default and only matches unversioned
SONAMEs, so it would fail out of the box.

On first use the plugin calls `ensure_openvino_lib_path()`, which scans:

1. `$DFN_OV_LIBS_DIR` (env override; useful for dev against a pip wheel)
2. `/usr/lib` (Intel APT layout)
3. `/usr/lib/x86_64-linux-gnu` (Debian multiarch)
4. `/opt/intel/openvino/runtime/lib/intel64` (Intel tarball)

The first directory containing `libopenvino_c.so.*` wins. The plugin then
builds a per-process shim of unversioned symlinks at
`$TMPDIR/dfn-ovshim-<pid>/` and prepends it to `LD_LIBRARY_PATH` for this
process only — no persistent symlinks on the host, nothing to clean up on
OpenVINO upgrade. OpenVINO itself still auto-discovers its device plugins
from `/usr/lib/openvino-<version>/` relative to the resolved library path.

## Troubleshooting

**`OpenVINO available devices: ['CPU']` but you have an NPU.**
```bash
lsmod | grep intel_vpu           # kernel module loaded
ls -la /dev/accel/accel0         # device exists, group 'render'
groups                           # your user is in 'render'
dpkg -l intel-level-zero-npu     # user-space driver installed
```
Fix whichever is missing.

**EasyEffects says "libdeep_filter_ladspa is not installed".**
```bash
dpkg -L deep-filter-openvino | grep ladspa
# expected: /usr/lib/x86_64-linux-gnu/ladspa/libdeep_filter_ladspa.so
```

**"DfOpenVino: no OpenVINO libs found in any of …" in the log.**
The OpenVINO runtime is missing. Re-install:
```bash
sudo apt install libopenvino-2026.1.0 libopenvino-intel-cpu-plugin-2026.1.0 \
                 libopenvino-auto-plugin-2026.1.0 libopenvino-hetero-plugin-2026.1.0 \
                 libopenvino-onnx-frontend-2026.1.0 libopenvino-ir-frontend-2026.1.0
```
For NPU inference also install `libopenvino-intel-npu-plugin-2026.1.0`.

**"Underrun detected (RTF: X). Processing too slow!" right after activating.**
Harmless. The NPU compiles the graph on first use (~500 ms). The plugin
auto-bumps its internal buffer and stabilises within a few frames.

**Audible artefacts on NPU but not CPU.**
Compare against `DFN_OV_DEVICE=CPU`. If the artefacts are NPU-specific,
they are most likely a quantisation / driver issue; drop a sample and a
reproducer into an issue.

## Uninstall

```bash
sudo apt remove deep-filter-openvino
# optionally also:
sudo apt remove 'libopenvino-*' 'libopenvino-intel-*-plugin-*'
sudo apt remove intel-fw-npu intel-level-zero-npu intel-driver-compiler-npu
```

## Building from source

See [`DEVELOPMENT.md`](DEVELOPMENT.md) for the internals, ONNX graph surgery,
and benchmarks.

```bash
# Python env used by the ONNX surgery (one-off).
uv venv --python 3.11 pyenv/.venv
source pyenv/.venv/bin/activate
uv pip install openvino onnx numpy

# Regenerate the NPU-patched model tarballs from the stock upstream ones.
python scripts/npu_export.py models/DeepFilterNet3_ll_onnx.tar.gz
python scripts/npu_export.py models/DeepFilterNet3_onnx.tar.gz

# Build + package.
tooling/build_deb.sh
```

Pick the tract backend (CPU only, upstream-equivalent) instead:

```bash
cargo build --release -p deep-filter-ladspa
# Plugin .so lands at target/release/libdeep_filter_ladspa.so.
```

Pick the OpenVINO backend directly, without packaging:

```bash
cargo build --profile release-lto -p deep-filter-ladspa \
    --no-default-features --features "default-model-ll openvino-backend"
```

## Licences

* DeepFilterNet code and model weights: MIT or Apache-2.0, Hendrik Schröter
  and contributors (https://github.com/Rikorose/DeepFilterNet).
* OpenVINO runtime: Apache-2.0, Intel Corporation
  (https://github.com/openvinotoolkit/openvino).
