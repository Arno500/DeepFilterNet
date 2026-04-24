#!/usr/bin/env bash
# Build a slim .deb containing only the DeepFilterNet LADSPA plugin compiled
# against the OpenVINO backend. Depends on the official Intel OpenVINO
# packages from https://apt.repos.intel.com/openvino — does NOT bundle the
# runtime.
#
# Intel's debs ship the core libs under /usr/lib (flat, not multi-arch) and
# the plugins under /usr/lib/openvino-<version>/. `openvino-finder` only
# searches the multi-arch dir /usr/lib/x86_64-linux-gnu/, so our postinst
# builds a small dir of unversioned symlinks pointing at whichever
# libopenvino*.so.<version> is installed, and the .so is compiled with
# DFN_OV_LIBS_DIR=<that dir> baked in.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# OpenVINO package-version tag. Intel's debs name their packages e.g.
# libopenvino-2026.1.0 (one per minor version), so dependencies are pinned
# to a specific release rather than a meta name.
OV_VERSION="2026.1.0"
VERSION="0.5.7"
ARCH="amd64"
PKG="deep-filter-openvino"
DEB_NAME="${PKG}_${VERSION}_${ARCH}.deb"
STAGE="${REPO_ROOT}/build/${PKG}_${VERSION}"

echo "[1/4] Building LADSPA plugin (openvino-backend, release-lto)…"
source "${HOME}/.cargo/env"
# Runtime layout is discovered at load time by ensure_openvino_lib_path(); no
# build-time env is needed. OPENVINO_INSTALL_DIR is only consulted by the
# openvino-sys build.rs when the 'dynamic-linking' feature is active, which
# we do not use.
unset DFN_OV_LIBS_DIR OPENVINO_INSTALL_DIR || true
(cd "${REPO_ROOT}" && cargo build --profile release-lto \
    -p deep-filter-ladspa --no-default-features \
    --features "default-model-ll openvino-backend")

SO_BUILT="${REPO_ROOT}/target/release-lto/libdeep_filter_ladspa.so"
test -f "${SO_BUILT}"

echo "[2/4] Staging .deb tree at ${STAGE}…"
rm -rf "${STAGE}"
install -d -m 0755 "${STAGE}/DEBIAN"
install -d -m 0755 "${STAGE}/usr/lib/x86_64-linux-gnu/ladspa"
install -d -m 0755 "${STAGE}/usr/share/doc/${PKG}"

install -m 0644 "${SO_BUILT}" "${STAGE}/usr/lib/x86_64-linux-gnu/ladspa/libdeep_filter_ladspa.so"

cat > "${STAGE}/usr/share/doc/${PKG}/copyright" <<'EOF'
Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/
Upstream-Name: DeepFilterNet (OpenVINO fork)
Source: https://github.com/Rikorose/DeepFilterNet

Files: *
Copyright: 2022-2026 Hendrik Schröter
License: MIT OR Apache-2.0
EOF

INSTALLED_SIZE=$(du -sk "${STAGE}" | cut -f1)

cat > "${STAGE}/DEBIAN/control" <<EOF
Package: ${PKG}
Version: ${VERSION}
Section: sound
Priority: optional
Architecture: ${ARCH}
Installed-Size: ${INSTALLED_SIZE}
Maintainer: Local build <root@localhost>
Depends: libc6 (>= 2.38),
 libgcc-s1,
 libstdc++6 (>= 13),
 libopenvino-${OV_VERSION},
 libopenvino-intel-cpu-plugin-${OV_VERSION},
 libopenvino-auto-plugin-${OV_VERSION},
 libopenvino-hetero-plugin-${OV_VERSION},
 libopenvino-onnx-frontend-${OV_VERSION},
 libopenvino-ir-frontend-${OV_VERSION}
Recommends: libopenvino-intel-npu-plugin-${OV_VERSION},
 libopenvino-intel-gpu-plugin-${OV_VERSION},
 intel-level-zero-npu,
 intel-driver-compiler-npu,
 intel-fw-npu
Suggests: easyeffects
Description: DeepFilterNet noise suppression LADSPA plugin (OpenVINO backend)
 DeepFilterNet3 low-latency speech enhancement packaged as a LADSPA plugin
 (libdeep_filter_ladspa.so) that runs inference on the Intel NPU (or GPU, or
 CPU) via OpenVINO. Drop-in for the tract-based upstream plugin; identical
 control ports so EasyEffects' DeepFilterNet integration picks it up
 automatically.
 .
 Requires the OpenVINO ${OV_VERSION} runtime from Intel's APT repository
 (https://apt.repos.intel.com/openvino). For NPU inference, also install the
 intel-level-zero-npu, intel-driver-compiler-npu, intel-fw-npu packages
 (currently only available as direct downloads from
 https://github.com/intel/linux-npu-driver/releases).
 .
 Select the inference device at process start with the environment variable
 DFN_OV_DEVICE=NPU|GPU|CPU|AUTO (default AUTO picks NPU if available).
EOF

cat > "${STAGE}/DEBIAN/postinst" <<'POSTINST'
#!/bin/sh
# The plugin locates OpenVINO at runtime by scanning /usr/lib and a few
# other known dirs for libopenvino_c.so.* — nothing to do here apart from a
# friendly message.
set -e
if [ "$1" = "configure" ]; then
    cat <<'MSG'

deep-filter-openvino installed.

If an NPU is available on this machine, make sure you are in the 'render'
group (sudo gpasswd -a $USER render) and that the Intel NPU user-space
driver is installed (see README). Then launch EasyEffects — it picks the
plugin up automatically.

Verify with:
  python3 -c 'import openvino as ov; print(ov.Core().available_devices)'

MSG
fi
exit 0
POSTINST
chmod 0755 "${STAGE}/DEBIAN/postinst"

echo "[3/4] Building .deb…"
rm -f "${REPO_ROOT}/build/${DEB_NAME}"
dpkg-deb --build --root-owner-group "${STAGE}" "${REPO_ROOT}/build/${DEB_NAME}"

echo "[4/4] Done."
ls -lh "${REPO_ROOT}/build/${DEB_NAME}"
echo ""
echo "Install (needs Intel OpenVINO apt repo set up first):"
echo "  sudo apt install ${REPO_ROOT}/build/${DEB_NAME}"
