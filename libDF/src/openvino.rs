//! OpenVINO backend for DeepFilterNet.
//!
//! Mirrors the public API of the `tract` backend (see `libDF/src/tract.rs`) so
//! that the LADSPA plugin (and other consumers) can swap between them behind a
//! feature flag. Designed to run the three NPU-patched sub-models
//! (`enc.onnx`, `erb_dec.onnx`, `df_dec.onnx`) produced by
//! `tooling/npu_export.py`. The patched models expose every GRU hidden state
//! as explicit graph-level input/output (`h<N>_in` / `h<N>_out`) and pin all
//! dimensions to `1`, which is what the Intel NPU plugin requires.
//!
//! Inference device defaults to `NPU` if available, then `GPU`, then `CPU`.
//! Override with env var `DFN_OV_DEVICE=NPU|GPU|CPU|AUTO` (case-insensitive)
//! or a free-form OpenVINO device string (e.g. `HETERO:NPU,CPU`).

use std::collections::VecDeque;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
#[cfg(feature = "timings")]
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use flate2::read::GzDecoder;
use ini::Ini;
use itertools::izip;
use ndarray::{prelude::*, Axis};
use openvino::{CompiledModel, Core, DeviceType, ElementType, InferRequest, Model, Shape, Tensor};
use tar::Archive;

use crate::{Complex32, DFState};

// -----------------------------------------------------------------------------
// Model bundle
// -----------------------------------------------------------------------------

/// Parsed model tarball (three ONNX blobs + config.ini).
///
/// The layout matches what `tooling/npu_export.py` writes and what the stock
/// DeepFilterNet export scripts produce, minus the GRU-state surgery. Either
/// tarball shape can be loaded; only the NPU-patched one will compile for the
/// NPU device.
#[derive(Clone)]
pub struct DfParams {
    config: Ini,
    enc: Vec<u8>,
    erb_dec: Vec<u8>,
    df_dec: Vec<u8>,
}

impl DfParams {
    pub fn new(tar_file: PathBuf) -> Result<Self> {
        let file = File::open(tar_file).context("Could not open model tar file.")?;
        Self::from_targz(file)
    }
    pub fn from_bytes(tar_buf: &[u8]) -> Result<Self> {
        Self::from_targz(tar_buf)
    }
    fn from_targz<R: Read>(f: R) -> Result<Self> {
        let tar = GzDecoder::new(f);
        let mut archive = Archive::new(tar);
        let mut enc = Vec::new();
        let mut erb_dec = Vec::new();
        let mut df_dec = Vec::new();
        let mut config = Ini::new();
        for e in archive
            .entries()
            .context("Could not extract models from tar file.")?
        {
            let mut file = e.context("Could not open model tar entry.")?;
            let path = file.path().unwrap();
            if path.ends_with("enc.onnx") {
                file.read_to_end(&mut enc)?;
            } else if path.ends_with("erb_dec.onnx") {
                file.read_to_end(&mut erb_dec)?;
            } else if path.ends_with("df_dec.onnx") {
                file.read_to_end(&mut df_dec)?;
            } else if path.ends_with("config.ini") {
                config = Ini::read_from(&mut file).context("Could not parse config.ini")?;
            } else if path.ends_with("version.txt") {
                let mut version = String::new();
                file.read_to_string(&mut version).ok();
                log::info!("Loading model with id: {}", version.trim());
            }
        }
        if enc.is_empty() || erb_dec.is_empty() || df_dec.is_empty() {
            bail!("Model tarball missing one of enc.onnx / erb_dec.onnx / df_dec.onnx");
        }
        Ok(Self {
            config,
            enc,
            erb_dec,
            df_dec,
        })
    }
}

impl Default for DfParams {
    fn default() -> Self {
        #[cfg(feature = "default-model-ll")]
        {
            log::debug!("Loading model DeepFilterNet3_ll_onnx_npu.tar.gz");
            return DfParams::from_bytes(include_bytes!(
                "../../models/DeepFilterNet3_ll_onnx_npu.tar.gz"
            ))
            .expect("Could not load model config");
        }
        #[cfg(all(feature = "default-model", not(feature = "default-model-ll")))]
        {
            log::debug!("Loading model DeepFilterNet3_onnx_npu.tar.gz");
            return DfParams::from_bytes(include_bytes!(
                "../../models/DeepFilterNet3_onnx_npu.tar.gz"
            ))
            .expect("Could not load model config");
        }
        #[cfg(not(any(feature = "default-model", feature = "default-model-ll")))]
        panic!("libDF not compiled with a default NPU model");
    }
}

// -----------------------------------------------------------------------------
// Runtime configuration
// -----------------------------------------------------------------------------

#[derive(Clone)]
pub enum ReduceMask {
    NONE = 0,
    MAX = 1,
    MEAN = 2,
}
impl Default for ReduceMask {
    fn default() -> Self {
        ReduceMask::NONE
    }
}
impl TryFrom<i32> for ReduceMask {
    type Error = ();
    fn try_from(v: i32) -> std::result::Result<Self, Self::Error> {
        match v {
            x if x == ReduceMask::NONE as i32 => Ok(ReduceMask::NONE),
            x if x == ReduceMask::MAX as i32 => Ok(ReduceMask::MAX),
            x if x == ReduceMask::MEAN as i32 => Ok(ReduceMask::MEAN),
            _ => Err(()),
        }
    }
}

pub struct RuntimeParams {
    pub n_ch: usize,
    pub post_filter: bool,
    pub post_filter_beta: f32,
    pub atten_lim_db: f32,
    pub min_db_thresh: f32,
    pub max_db_erb_thresh: f32,
    pub max_db_df_thresh: f32,
    pub reduce_mask: ReduceMask,
}
impl RuntimeParams {
    pub fn new(
        n_ch: usize,
        post_filter_beta: f32,
        atten_lim_db: f32,
        min_db_thresh: f32,
        max_db_erb_thresh: f32,
        max_db_df_thresh: f32,
        reduce_mask: ReduceMask,
    ) -> Self {
        Self {
            n_ch,
            post_filter: post_filter_beta > 0.,
            post_filter_beta,
            atten_lim_db,
            min_db_thresh,
            max_db_erb_thresh,
            max_db_df_thresh,
            reduce_mask,
        }
    }
    pub fn default_with_ch(channels: usize) -> Self {
        Self {
            n_ch: channels,
            post_filter: false,
            post_filter_beta: 0.02,
            atten_lim_db: 100.,
            min_db_thresh: -10.,
            max_db_erb_thresh: 30.,
            max_db_df_thresh: 20.,
            reduce_mask: ReduceMask::MEAN,
        }
    }
}
impl Default for RuntimeParams {
    fn default() -> Self {
        Self::default_with_ch(1)
    }
}

// -----------------------------------------------------------------------------
// Shared compiled models
// -----------------------------------------------------------------------------

/// Compiled, device-resident OpenVINO models. Shared across plugin instances
/// via `Arc`; new `InferRequest`s are minted per instance on `Clone`.
struct OvShared {
    core: Mutex<Core>,
    enc: Mutex<CompiledModel>,
    erb_dec: Mutex<CompiledModel>,
    df_dec: Mutex<CompiledModel>,
    enc_gru_count: usize,
    erb_dec_gru_count: usize,
    df_dec_gru_count: usize,
    device: String,
}

/// Locate OpenVINO shared libraries on disk and expose them to
/// `openvino-finder`, which only searches a fixed set of dirs and only
/// matches unversioned SONAMEs (`libopenvino_c.so`). Intel's APT packages
/// install versioned files (`libopenvino_c.so.2610`) into `/usr/lib/` — a
/// non-multiarch dir the finder never looks at — so left to its own devices
/// `openvino-finder` fails to load.
///
/// Strategy:
///   1. Probe a known list of install layouts (Intel APT / tarball / pip /
///      user override via `DFN_OV_LIBS_DIR`) for any file matching
///      `libopenvino_c.so.<version>`.
///   2. On the first hit, build a process-local shim under
///      `$TMPDIR/dfn-ovshim-<pid>/` with unversioned symlinks pointing at
///      the versioned files (`libopenvino.so`, `libopenvino_c.so`, and the
///      two frontend libs).
///   3. Prepend the shim + the real libs dir to `LD_LIBRARY_PATH` so
///      `openvino-finder` matches on the shim and OpenVINO's plugin loader
///      still resolves any transitive `dlopen` by the real path.
///
/// Runs unconditionally: if the user has already configured a working
/// OpenVINO env (`OPENVINO_INSTALL_DIR` set, or the finder's default paths
/// happen to work), a second shim is harmless.
fn ensure_openvino_lib_path() {
    // Candidate layouts, in probe order.
    let mut candidate_dirs: Vec<String> = Vec::new();
    if let Ok(d) = std::env::var("DFN_OV_LIBS_DIR") {
        if !d.is_empty() {
            candidate_dirs.push(d);
        }
    }
    if let Some(baked) = option_env!("DFN_OV_LIBS_DIR") {
        if !baked.is_empty() {
            candidate_dirs.push(baked.to_string());
        }
    }
    candidate_dirs.extend(
        [
            "/usr/lib",                                   // Intel APT debs (flat)
            "/usr/lib/x86_64-linux-gnu",                  // Debian multiarch
            "/opt/intel/openvino/runtime/lib/intel64",    // Intel tarball
        ]
        .iter()
        .map(|s| s.to_string()),
    );

    for dir in &candidate_dirs {
        let dir_path = std::path::Path::new(dir);
        if !dir_path.is_dir() {
            continue;
        }
        // Need at least libopenvino_c.so.* present for this dir to matter.
        if find_versioned_lib(dir_path, "libopenvino_c.so").is_none() {
            continue;
        }
        // Build (or re-build) the per-process shim dir.
        let shim = shim_dir();
        if std::fs::create_dir_all(&shim).is_err() {
            log::warn!("DfOpenVino: could not create shim dir {}", shim.display());
            return;
        }
        for base in [
            "libopenvino.so",
            "libopenvino_c.so",
            "libopenvino_ir_frontend.so",
            "libopenvino_onnx_frontend.so",
        ] {
            if let Some(src) = find_versioned_lib(dir_path, base) {
                let dst = shim.join(base);
                let _ = std::fs::remove_file(&dst);
                if let Err(e) = std::os::unix::fs::symlink(&src, &dst) {
                    log::warn!(
                        "DfOpenVino: could not symlink {} → {}: {}",
                        dst.display(),
                        src.display(),
                        e
                    );
                }
            }
        }
        // Prepend the shim first (finder reads LD_LIBRARY_PATH left→right).
        prepend_env("LD_LIBRARY_PATH", &shim.to_string_lossy());
        // Also expose the real dir so transitive dlopens (plugins, TBB, etc.)
        // find everything they need even if they don't honour the shim.
        prepend_env("LD_LIBRARY_PATH", dir);
        log::debug!(
            "DfOpenVino: OpenVINO libs located at {}, shim at {}",
            dir,
            shim.display()
        );
        return;
    }

    log::warn!(
        "DfOpenVino: no OpenVINO libs found in any of {:?}; Core::new() will likely fail. \
         Install libopenvino-* from the Intel APT repo, or set DFN_OV_LIBS_DIR.",
        candidate_dirs
    );
}

/// Return `<dir>/<base>.<anything>` for some version suffix, if the
/// directory contains any file matching that pattern. Symlinks are
/// followed; we pick the first hit (lexically first entry returned by
/// `read_dir`) — the choice does not matter because Intel ships both
/// `libfoo.so.<SONAME>` and `libfoo.so.<full-version>` that resolve to the
/// same on-disk object.
fn find_versioned_lib(dir: &std::path::Path, base: &str) -> Option<std::path::PathBuf> {
    let prefix = format!("{}.", base);
    let read = std::fs::read_dir(dir).ok()?;
    for entry in read.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with(&prefix) {
            return Some(entry.path());
        }
    }
    None
}

fn shim_dir() -> std::path::PathBuf {
    let tmp = std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".to_string());
    std::path::PathBuf::from(tmp).join(format!("dfn-ovshim-{}", std::process::id()))
}

fn prepend_env(key: &str, value: &str) {
    let current = std::env::var(key).unwrap_or_default();
    if current.split(':').any(|p| p == value) {
        return;
    }
    let new = if current.is_empty() {
        value.to_string()
    } else {
        format!("{}:{}", value, current)
    };
    // SAFETY: called from DfOpenVino::new(), before any OpenVINO call, on
    // the thread that will own inference — no concurrent readers of env.
    unsafe {
        std::env::set_var(key, new);
    }
}

fn select_device(core: &Core) -> Result<DeviceType<'static>> {
    let requested = std::env::var("DFN_OV_DEVICE")
        .unwrap_or_default()
        .to_uppercase();
    let available: Vec<String> = core
        .available_devices()
        .context("Could not enumerate OpenVINO devices")?
        .into_iter()
        .map(|d| d.as_ref().to_string())
        .collect();
    log::info!("OpenVINO available devices: {:?}", available);
    let has = |name: &str| available.iter().any(|d| d == name || d.starts_with(name));
    let picked = match requested.as_str() {
        "" | "AUTO" => {
            if has("NPU") {
                "NPU".to_string()
            } else if has("GPU") {
                "GPU".to_string()
            } else {
                "CPU".to_string()
            }
        }
        explicit => explicit.to_string(),
    };
    log::info!("OpenVINO selected device: {}", picked);
    Ok(DeviceType::from(picked.as_str()).to_owned())
}

fn count_gru_sockets(model: &Model) -> Result<usize> {
    let n_in = model.get_inputs_len()?;
    let mut count = 0usize;
    for i in 0..n_in {
        let node = model.get_input_by_index(i)?;
        let name = node.get_name()?;
        if name.starts_with('h') && name.ends_with("_in") {
            count += 1;
        }
    }
    Ok(count)
}

impl OvShared {
    fn build(dfp: &DfParams) -> Result<Arc<Self>> {
        ensure_openvino_lib_path();
        let mut core = Core::new().map_err(|e| anyhow!("OpenVINO Core init failed: {e}"))?;
        let device = select_device(&core)?;
        let device_name: String = device.as_ref().to_string();
        let enc_model = core.read_model_from_buffer(&dfp.enc, None)?;
        let erb_model = core.read_model_from_buffer(&dfp.erb_dec, None)?;
        let df_model = core.read_model_from_buffer(&dfp.df_dec, None)?;
        let enc_gru = count_gru_sockets(&enc_model)?;
        let erb_gru = count_gru_sockets(&erb_model)?;
        let df_gru = count_gru_sockets(&df_model)?;
        if enc_gru == 0 || erb_gru == 0 || df_gru == 0 {
            bail!(
                "Model tarball appears non-NPU (no h<N>_in graph inputs found); \
                 run tooling/npu_export.py on the stock DFN3 ONNX bundle first."
            );
        }
        log::info!(
            "Compiling OpenVINO models on device '{}' (enc GRUs={}, erb_dec GRUs={}, df_dec GRUs={})",
            device_name,
            enc_gru,
            erb_gru,
            df_gru
        );
        let enc_cm = core.compile_model(&enc_model, device.to_owned())?;
        let erb_cm = core.compile_model(&erb_model, device.to_owned())?;
        let df_cm = core.compile_model(&df_model, device)?;
        Ok(Arc::new(Self {
            core: Mutex::new(core),
            enc: Mutex::new(enc_cm),
            erb_dec: Mutex::new(erb_cm),
            df_dec: Mutex::new(df_cm),
            enc_gru_count: enc_gru,
            erb_dec_gru_count: erb_gru,
            df_dec_gru_count: df_gru,
            device: device_name,
        }))
    }

    fn new_enc_request(&self) -> Result<InferRequest> {
        Ok(self.enc.lock().unwrap().create_infer_request()?)
    }
    fn new_erb_request(&self) -> Result<InferRequest> {
        Ok(self.erb_dec.lock().unwrap().create_infer_request()?)
    }
    fn new_df_request(&self) -> Result<InferRequest> {
        Ok(self.df_dec.lock().unwrap().create_infer_request()?)
    }
}

// -----------------------------------------------------------------------------
// DfOpenVino
// -----------------------------------------------------------------------------

/// Per-instance inference runtime.
///
/// The three inference requests, their persistent input/state tensors, and the
/// DSP rolling buffers all live here. Cloned instances share the compiled
/// models behind an `Arc<OvShared>` but allocate fresh inference requests and
/// state so they can run concurrently.
pub struct DfOpenVino {
    shared: Arc<OvShared>,
    // Inference requests.
    enc_req: InferRequest,
    erb_dec_req: InferRequest,
    df_dec_req: InferRequest,
    // Persistent input tensors (set_tensor'd once at init, mutated in place).
    t_feat_erb: Tensor,
    t_feat_spec: Tensor,
    t_erb_emb: Tensor,
    t_erb_e3: Tensor,
    t_erb_e2: Tensor,
    t_erb_e1: Tensor,
    t_erb_e0: Tensor,
    t_df_emb: Tensor,
    t_df_c0: Tensor,
    // Hidden-state in-tensors. Outer axis = channel (one set per input
    // channel, since the NPU-patched models are batch=1 and we serialise
    // inference across channels). Inner axis = GRU socket index; names are
    // `h<i>_in` / `h<i>_out`.
    t_enc_h_in: Vec<Vec<Tensor>>,
    t_erb_h_in: Vec<Vec<Tensor>>,
    t_df_h_in: Vec<Vec<Tensor>>,

    // Scratch for intermediate wiring between submodels.
    emb_buf: Vec<f32>, // [1, 1, emb_hidden_dim]
    c0_buf: Vec<f32>,  // [1, conv_ch, 1, nb_df]
    e0_buf: Vec<f32>,
    e1_buf: Vec<f32>,
    e2_buf: Vec<f32>,
    e3_buf: Vec<f32>,

    // Public runtime fields (API-compatible with DfTract).
    pub lookahead: usize,
    pub df_lookahead: usize,
    pub conv_lookahead: usize,
    pub sr: usize,
    pub ch: usize,
    pub fft_size: usize,
    pub hop_size: usize,
    pub nb_erb: usize,
    pub min_nb_erb_freqs: usize,
    pub nb_df: usize,
    pub n_freqs: usize,
    pub df_order: usize,
    pub post_filter: bool,
    pub post_filter_beta: f32,
    pub alpha: f32,
    pub min_db_thresh: f32,
    pub max_db_erb_thresh: f32,
    pub max_db_df_thresh: f32,
    pub reduce_mask: ReduceMask,
    pub atten_lim: Option<f32>,
    pub df_states: Vec<DFState>,
    // DFN internal DSP buffers.
    pub spec_buf: Array5<f32>, // [ch, 1, 1, n_freqs, 2]
    erb_buf: Array4<f32>,      // [ch, 1, 1, nb_erb]
    cplx_buf: Array4<f32>,     // [ch, 1, nb_df, 2] (laid out contiguous)
    rolling_spec_buf_y: VecDeque<Array5<f32>>, // Enhanced stage-1 buf
    rolling_spec_buf_x: VecDeque<Array5<f32>>, // Noisy buf
    m_zeros: Vec<f32>,
    skip_counter: usize,
}

// Cloning is required by the LADSPA plugin which clones a global `Option<Df*>`
// into the worker thread. We share the heavyweight compiled models and spin
// fresh inference requests + state tensors for the new instance.
impl Clone for DfOpenVino {
    fn clone(&self) -> Self {
        let shared = Arc::clone(&self.shared);
        let mut me = Self::new_from_shared(
            shared,
            self.ch,
            self.sr,
            self.hop_size,
            self.fft_size,
            self.nb_erb,
            self.min_nb_erb_freqs,
            self.nb_df,
            self.n_freqs,
            self.df_order,
            self.conv_lookahead,
            self.df_lookahead,
            self.lookahead,
            self.alpha,
            self.min_db_thresh,
            self.max_db_erb_thresh,
            self.max_db_df_thresh,
            self.reduce_mask.clone(),
            self.atten_lim,
            self.post_filter,
            self.post_filter_beta,
        )
        .expect("Could not clone DfOpenVino");
        // Copy rolling buffers + hidden state snapshots so the clone resumes
        // from the same point.
        me.rolling_spec_buf_x = self.rolling_spec_buf_x.clone();
        me.rolling_spec_buf_y = self.rolling_spec_buf_y.clone();
        me.spec_buf = self.spec_buf.clone();
        me.erb_buf = self.erb_buf.clone();
        me.cplx_buf = self.cplx_buf.clone();
        me.df_states = self.df_states.clone();
        me.skip_counter = self.skip_counter;
        // Copy hidden states tensor-by-tensor.
        for (dst_ch, src_ch) in me.t_enc_h_in.iter_mut().zip(self.t_enc_h_in.iter()) {
            for (dst, src) in dst_ch.iter_mut().zip(src_ch.iter()) {
                copy_tensor_data(src, dst);
            }
        }
        for (dst_ch, src_ch) in me.t_erb_h_in.iter_mut().zip(self.t_erb_h_in.iter()) {
            for (dst, src) in dst_ch.iter_mut().zip(src_ch.iter()) {
                copy_tensor_data(src, dst);
            }
        }
        for (dst_ch, src_ch) in me.t_df_h_in.iter_mut().zip(self.t_df_h_in.iter()) {
            for (dst, src) in dst_ch.iter_mut().zip(src_ch.iter()) {
                copy_tensor_data(src, dst);
            }
        }
        me
    }
}

impl DfOpenVino {
    pub fn new(dfp: DfParams, rp: &RuntimeParams) -> Result<Self> {
        #[cfg(feature = "timings")]
        let t0 = Instant::now();

        let config = dfp.config.clone();
        let model_cfg = config
            .section(Some("deepfilternet"))
            .context("Missing 'deepfilternet' section in config.ini")?;
        let df_cfg = config
            .section(Some("df"))
            .context("Missing 'df' section in config.ini")?;
        let train_cfg = config
            .section(Some("train"))
            .context("Missing 'train' section in config.ini")?;

        let model_type = train_cfg
            .get("model")
            .context("Missing 'model' key in [train] section")?;
        if model_type != "deepfilternet3" {
            bail!(
                "OpenVINO backend only supports model type deepfilternet3, got '{}'",
                model_type
            );
        }

        let sr = df_cfg.get("sr").unwrap().parse::<usize>()?;
        let hop_size = df_cfg.get("hop_size").unwrap().parse::<usize>()?;
        let fft_size = df_cfg.get("fft_size").unwrap().parse::<usize>()?;
        let nb_erb = df_cfg.get("nb_erb").unwrap().parse::<usize>()?;
        let nb_df = df_cfg.get("nb_df").unwrap().parse::<usize>()?;
        let min_nb_erb_freqs = df_cfg.get("min_nb_erb_freqs").unwrap().parse::<usize>()?;
        let df_order = df_cfg
            .get("df_order")
            .unwrap_or_else(|| model_cfg.get("df_order").unwrap())
            .parse::<usize>()?;
        let conv_lookahead = model_cfg
            .get("conv_lookahead")
            .unwrap()
            .parse::<usize>()?;
        let df_lookahead = df_cfg
            .get("df_lookahead")
            .unwrap_or_else(|| model_cfg.get("df_lookahead").unwrap())
            .parse::<usize>()?;
        let n_freqs = fft_size / 2 + 1;
        let alpha = if let Some(a) = df_cfg.get("norm_alpha") {
            a.parse::<f32>()?
        } else {
            let tau = df_cfg.get("norm_tau").unwrap().parse::<f32>()?;
            calc_norm_alpha(sr, hop_size, tau)
        };
        let lookahead = conv_lookahead.max(df_lookahead);

        let atten_lim_db = rp.atten_lim_db.abs();
        let atten_lim = if atten_lim_db >= 100. {
            None
        } else if atten_lim_db < 0.01 {
            log::warn!("Attenuation limit too strong. No noise reduction will be performed");
            Some(1.)
        } else {
            log::info!("Running with an attenuation limit of {:.0} dB", atten_lim_db);
            Some(10f32.powf(-atten_lim_db / 20.))
        };

        let shared = OvShared::build(&dfp)?;

        let this = Self::new_from_shared(
            shared,
            rp.n_ch,
            sr,
            hop_size,
            fft_size,
            nb_erb,
            min_nb_erb_freqs,
            nb_df,
            n_freqs,
            df_order,
            conv_lookahead,
            df_lookahead,
            lookahead,
            alpha,
            rp.min_db_thresh,
            rp.max_db_erb_thresh,
            rp.max_db_df_thresh,
            rp.reduce_mask.clone(),
            atten_lim,
            rp.post_filter,
            rp.post_filter_beta,
        )?;

        #[cfg(feature = "timings")]
        log::info!(
            "Init DfOpenVino in {:.1}ms on '{}'",
            t0.elapsed().as_secs_f32() * 1000.,
            this.shared.device
        );
        Ok(this)
    }

    #[allow(clippy::too_many_arguments)]
    fn new_from_shared(
        shared: Arc<OvShared>,
        ch: usize,
        sr: usize,
        hop_size: usize,
        fft_size: usize,
        nb_erb: usize,
        min_nb_erb_freqs: usize,
        nb_df: usize,
        n_freqs: usize,
        df_order: usize,
        conv_lookahead: usize,
        df_lookahead: usize,
        lookahead: usize,
        alpha: f32,
        min_db_thresh: f32,
        max_db_erb_thresh: f32,
        max_db_df_thresh: f32,
        reduce_mask: ReduceMask,
        atten_lim: Option<f32>,
        post_filter: bool,
        post_filter_beta: f32,
    ) -> Result<Self> {
        if ch == 0 {
            bail!("DfOpenVino requires at least one channel");
        }
        if ch > 1 {
            log::info!(
                "DfOpenVino ch={}: batch=1 models, inference will serialise per channel",
                ch
            );
        }

        // Shapes (batch=1, seq=1, static).
        let shape_erb = Shape::new(&[1, 1, 1, nb_erb as i64])?;
        let shape_spec = Shape::new(&[1, 2, 1, nb_df as i64])?;
        let shape_hidden = Shape::new(&[1, 1, 512])?;
        let shape_emb = Shape::new(&[1, 1, 512])?;
        let shape_c0 = Shape::new(&[1, 64, 1, nb_df as i64])?;
        let shape_e0 = Shape::new(&[1, 64, 1, nb_erb as i64])?;
        let shape_e1 = Shape::new(&[1, 64, 1, (nb_erb / 2) as i64])?;
        let shape_e2 = Shape::new(&[1, 64, 1, (nb_erb / 4) as i64])?;
        let shape_e3 = Shape::new(&[1, 64, 1, (nb_erb / 4) as i64])?;

        let mut t_feat_erb = Tensor::new(ElementType::F32, &shape_erb)?;
        let mut t_feat_spec = Tensor::new(ElementType::F32, &shape_spec)?;
        let mut t_erb_emb = Tensor::new(ElementType::F32, &shape_emb)?;
        let mut t_erb_e3 = Tensor::new(ElementType::F32, &shape_e3)?;
        let mut t_erb_e2 = Tensor::new(ElementType::F32, &shape_e2)?;
        let mut t_erb_e1 = Tensor::new(ElementType::F32, &shape_e1)?;
        let mut t_erb_e0 = Tensor::new(ElementType::F32, &shape_e0)?;
        let mut t_df_emb = Tensor::new(ElementType::F32, &shape_emb)?;
        let mut t_df_c0 = Tensor::new(ElementType::F32, &shape_c0)?;

        let mk_hidden_set = |count: usize| -> Result<Vec<Tensor>> {
            let mut v = Vec::with_capacity(count);
            for _ in 0..count {
                let mut t = Tensor::new(ElementType::F32, &shape_hidden)?;
                t.get_data_mut::<f32>()?.fill(0.);
                v.push(t);
            }
            Ok(v)
        };
        let mk_hidden_by_channel = |count: usize| -> Result<Vec<Vec<Tensor>>> {
            let mut v = Vec::with_capacity(ch);
            for _ in 0..ch {
                v.push(mk_hidden_set(count)?);
            }
            Ok(v)
        };
        let t_enc_h_in = mk_hidden_by_channel(shared.enc_gru_count)?;
        let t_erb_h_in = mk_hidden_by_channel(shared.erb_dec_gru_count)?;
        let t_df_h_in = mk_hidden_by_channel(shared.df_dec_gru_count)?;

        // Zero-fill main I/O tensors.
        t_feat_erb.get_data_mut::<f32>()?.fill(0.);
        t_feat_spec.get_data_mut::<f32>()?.fill(0.);
        t_erb_emb.get_data_mut::<f32>()?.fill(0.);
        t_erb_e3.get_data_mut::<f32>()?.fill(0.);
        t_erb_e2.get_data_mut::<f32>()?.fill(0.);
        t_erb_e1.get_data_mut::<f32>()?.fill(0.);
        t_erb_e0.get_data_mut::<f32>()?.fill(0.);
        t_df_emb.get_data_mut::<f32>()?.fill(0.);
        t_df_c0.get_data_mut::<f32>()?.fill(0.);

        // Create infer requests and bind persistent feature-input tensors.
        // Hidden-state tensors are re-bound per channel inside process_raw.
        let mut enc_req = shared.new_enc_request()?;
        enc_req.set_tensor("feat_erb", &t_feat_erb)?;
        enc_req.set_tensor("feat_spec", &t_feat_spec)?;

        let mut erb_dec_req = shared.new_erb_request()?;
        erb_dec_req.set_tensor("emb", &t_erb_emb)?;
        erb_dec_req.set_tensor("e3", &t_erb_e3)?;
        erb_dec_req.set_tensor("e2", &t_erb_e2)?;
        erb_dec_req.set_tensor("e1", &t_erb_e1)?;
        erb_dec_req.set_tensor("e0", &t_erb_e0)?;

        let mut df_dec_req = shared.new_df_request()?;
        df_dec_req.set_tensor("emb", &t_df_emb)?;
        df_dec_req.set_tensor("c0", &t_df_c0)?;

        let spec_shape = [ch, 1, 1, n_freqs, 2];
        let spec_buf = Array5::<f32>::zeros((ch, 1, 1, n_freqs, 2));
        let erb_buf = Array4::<f32>::zeros((ch, 1, 1, nb_erb));
        let cplx_buf = Array4::<f32>::zeros((ch, 1, nb_df, 2));
        let _ = spec_shape;

        let mut rolling_spec_buf_y = VecDeque::with_capacity(df_order + lookahead);
        let mut rolling_spec_buf_x = VecDeque::with_capacity(df_order.max(lookahead));
        for _ in 0..(df_order + conv_lookahead) {
            rolling_spec_buf_y.push_back(Array5::<f32>::zeros((ch, 1, 1, n_freqs, 2)));
        }
        for _ in 0..df_order.max(lookahead) {
            rolling_spec_buf_x.push_back(Array5::<f32>::zeros((ch, 1, 1, n_freqs, 2)));
        }

        let mut df_states = Vec::with_capacity(ch);
        for _ in 0..ch {
            let mut s = DFState::new(sr, fft_size, hop_size, nb_erb, min_nb_erb_freqs);
            s.init_norm_states(nb_df);
            df_states.push(s);
        }

        let m_zeros = vec![0.; nb_erb];

        Ok(Self {
            shared,
            enc_req,
            erb_dec_req,
            df_dec_req,
            t_feat_erb,
            t_feat_spec,
            t_erb_emb,
            t_erb_e3,
            t_erb_e2,
            t_erb_e1,
            t_erb_e0,
            t_df_emb,
            t_df_c0,
            t_enc_h_in,
            t_erb_h_in,
            t_df_h_in,
            emb_buf: vec![0.; 1 * 1 * 512],
            c0_buf: vec![0.; 1 * 64 * 1 * nb_df],
            e0_buf: vec![0.; 1 * 64 * 1 * nb_erb],
            e1_buf: vec![0.; 1 * 64 * 1 * (nb_erb / 2)],
            e2_buf: vec![0.; 1 * 64 * 1 * (nb_erb / 4)],
            e3_buf: vec![0.; 1 * 64 * 1 * (nb_erb / 4)],
            lookahead,
            df_lookahead,
            conv_lookahead,
            sr,
            ch,
            fft_size,
            hop_size,
            nb_erb,
            min_nb_erb_freqs,
            nb_df,
            n_freqs,
            df_order,
            post_filter,
            post_filter_beta,
            alpha,
            min_db_thresh,
            max_db_erb_thresh,
            max_db_df_thresh,
            reduce_mask,
            atten_lim,
            df_states,
            spec_buf,
            erb_buf,
            cplx_buf,
            rolling_spec_buf_y,
            rolling_spec_buf_x,
            m_zeros,
            skip_counter: 0,
        })
    }

    pub fn set_pf_beta(&mut self, beta: f32) {
        log::debug!("Setting post-filter beta to {beta}");
        self.post_filter_beta = beta;
        if beta > 0. {
            self.post_filter = true;
        } else if beta == 0. {
            self.post_filter = false;
        } else {
            log::warn!("Post-filter beta cannot be negative; disabling post filter.");
            self.post_filter = false;
            self.post_filter_beta = 0.;
        }
    }

    pub fn set_atten_lim(&mut self, db: f32) {
        let lim = db.abs();
        self.atten_lim = if lim >= 100. {
            None
        } else if lim < 0.01 {
            log::warn!("Attenuation limit too strong. No noise reduction will be performed");
            Some(1.)
        } else {
            log::debug!("Setting attenuation limit to {:.1} dB", lim);
            Some(10f32.powf(-lim / 20.))
        };
    }

    pub fn reset_state(&mut self) -> Result<()> {
        for ch_vec in self
            .t_enc_h_in
            .iter_mut()
            .chain(self.t_erb_h_in.iter_mut())
            .chain(self.t_df_h_in.iter_mut())
        {
            for t in ch_vec.iter_mut() {
                t.get_data_mut::<f32>()?.fill(0.);
            }
        }
        for s in self.spec_buf.iter_mut() {
            *s = 0.;
        }
        for b in self.rolling_spec_buf_x.iter_mut().chain(self.rolling_spec_buf_y.iter_mut()) {
            b.fill(0.);
        }
        for s in self.df_states.iter_mut() {
            s.reset();
        }
        self.skip_counter = 0;
        Ok(())
    }

    /// Run the three submodels on one frame of features, once per channel,
    /// and aggregate per-channel outputs.
    ///
    /// Returns `(lsnr, gains, coefs)` in the same shape as the tract backend.
    /// `gains` has shape `[ch_out, nb_erb]` where `ch_out` is either `1` if a
    /// mean/max reduction is configured, or `ch` otherwise. `coefs` has shape
    /// `[ch, nb_df, df_order, 2]`. Features must already be present for each
    /// channel inside the respective DFState (frame_analysis was already run
    /// in `process`).
    pub fn process_raw(&mut self) -> Result<(f32, Option<Array2<f32>>, Option<Array4<f32>>)> {
        let ch = self.ch;
        let nb_erb = self.nb_erb;
        let nb_df = self.nb_df;
        let df_order = self.df_order;

        // Per-channel results staging.
        let mut lsnr_sum = 0.0f32;
        let mut gains_per_ch = Array2::<f32>::zeros((ch, nb_erb));
        let mut coefs_per_ch = Array4::<f32>::zeros((ch, nb_df, df_order, 2));
        let mut any_apply_gains = false;
        let mut any_apply_gain_zeros = false;
        let mut any_apply_df = false;

        for c in 0..ch {
            // Build features into the encoder input tensors for this channel.
            let spec_view = self.spec_buf.index_axis(Axis(0), c);
            let spec_slice = spec_view.as_slice().unwrap();
            let spec_cplx = as_complex(spec_slice);
            {
                let erb_slice = self.t_feat_erb.get_data_mut::<f32>()?;
                self.df_states[c].feat_erb(spec_cplx, self.alpha, erb_slice);
            }
            {
                let mut tmp = vec![Complex32::default(); nb_df];
                self.df_states[c].feat_cplx(&spec_cplx[..nb_df], self.alpha, &mut tmp);
                let fs = self.t_feat_spec.get_data_mut::<f32>()?;
                let (re_plane, im_plane) = fs.split_at_mut(nb_df);
                for (i, cx) in tmp.iter().enumerate() {
                    re_plane[i] = cx.re;
                    im_plane[i] = cx.im;
                }
            }

            // Bind this channel's hidden states for the encoder.
            for i in 0..self.shared.enc_gru_count {
                self.enc_req
                    .set_tensor(&format!("h{}_in", i), &self.t_enc_h_in[c][i])?;
            }
            self.enc_req.infer()?;

            let emb_out = self.enc_req.get_tensor("emb")?;
            let c0_out = self.enc_req.get_tensor("c0")?;
            let e0_out = self.enc_req.get_tensor("e0")?;
            let e1_out = self.enc_req.get_tensor("e1")?;
            let e2_out = self.enc_req.get_tensor("e2")?;
            let e3_out = self.enc_req.get_tensor("e3")?;
            let lsnr_out = self.enc_req.get_tensor("lsnr")?;
            let lsnr_c = lsnr_out.get_data::<f32>()?[0];
            lsnr_sum += lsnr_c;

            for i in 0..self.shared.enc_gru_count {
                let hout = self.enc_req.get_tensor(&format!("h{}_out", i))?;
                let src = hout.get_data::<f32>()?;
                self.t_enc_h_in[c][i]
                    .get_data_mut::<f32>()?
                    .copy_from_slice(src);
            }

            self.emb_buf.copy_from_slice(emb_out.get_data::<f32>()?);
            self.c0_buf.copy_from_slice(c0_out.get_data::<f32>()?);
            self.e0_buf.copy_from_slice(e0_out.get_data::<f32>()?);
            self.e1_buf.copy_from_slice(e1_out.get_data::<f32>()?);
            self.e2_buf.copy_from_slice(e2_out.get_data::<f32>()?);
            self.e3_buf.copy_from_slice(e3_out.get_data::<f32>()?);

            let (apply_gains, apply_gain_zeros, apply_df) = self.apply_stages(lsnr_c);
            any_apply_gains |= apply_gains;
            any_apply_gain_zeros |= apply_gain_zeros;
            any_apply_df |= apply_df;

            if apply_gains {
                self.t_erb_emb
                    .get_data_mut::<f32>()?
                    .copy_from_slice(&self.emb_buf);
                self.t_erb_e3
                    .get_data_mut::<f32>()?
                    .copy_from_slice(&self.e3_buf);
                self.t_erb_e2
                    .get_data_mut::<f32>()?
                    .copy_from_slice(&self.e2_buf);
                self.t_erb_e1
                    .get_data_mut::<f32>()?
                    .copy_from_slice(&self.e1_buf);
                self.t_erb_e0
                    .get_data_mut::<f32>()?
                    .copy_from_slice(&self.e0_buf);
                for i in 0..self.shared.erb_dec_gru_count {
                    self.erb_dec_req
                        .set_tensor(&format!("h{}_in", i), &self.t_erb_h_in[c][i])?;
                }
                self.erb_dec_req.infer()?;
                let mask_out = self.erb_dec_req.get_tensor("m")?;
                let mask_slice = mask_out.get_data::<f32>()?;
                for (o, &v) in gains_per_ch
                    .index_axis_mut(Axis(0), c)
                    .iter_mut()
                    .zip(mask_slice.iter())
                {
                    *o = v;
                }
                for i in 0..self.shared.erb_dec_gru_count {
                    let hout = self.erb_dec_req.get_tensor(&format!("h{}_out", i))?;
                    let src = hout.get_data::<f32>()?;
                    self.t_erb_h_in[c][i]
                        .get_data_mut::<f32>()?
                        .copy_from_slice(src);
                }
            }

            if apply_df {
                self.t_df_emb
                    .get_data_mut::<f32>()?
                    .copy_from_slice(&self.emb_buf);
                self.t_df_c0
                    .get_data_mut::<f32>()?
                    .copy_from_slice(&self.c0_buf);
                for i in 0..self.shared.df_dec_gru_count {
                    self.df_dec_req
                        .set_tensor(&format!("h{}_in", i), &self.t_df_h_in[c][i])?;
                }
                self.df_dec_req.infer()?;
                let coefs_out = self.df_dec_req.get_tensor("coefs")?;
                let coefs_slice = coefs_out.get_data::<f32>()?;
                // Copy directly into the per-channel slice of coefs_per_ch.
                let mut dst = coefs_per_ch.index_axis_mut(Axis(0), c);
                dst.as_slice_mut().unwrap().copy_from_slice(coefs_slice);
                for i in 0..self.shared.df_dec_gru_count {
                    let hout = self.df_dec_req.get_tensor(&format!("h{}_out", i))?;
                    let src = hout.get_data::<f32>()?;
                    self.t_df_h_in[c][i]
                        .get_data_mut::<f32>()?
                        .copy_from_slice(src);
                }
            }
        }

        let lsnr = lsnr_sum / ch as f32;

        let gains = if any_apply_gains {
            // Replicate the tract `reduce_mask` behaviour: collapse multi-ch
            // masks to a single row using the configured reducer.
            Some(match self.reduce_mask {
                ReduceMask::MAX => {
                    let mut out = Array2::<f32>::zeros((1, nb_erb));
                    for mut row in out.rows_mut() {
                        for i in 0..nb_erb {
                            let mut m = f32::NEG_INFINITY;
                            for c in 0..ch {
                                m = m.max(gains_per_ch[[c, i]]);
                            }
                            row[i] = m;
                        }
                    }
                    out
                }
                ReduceMask::MEAN => {
                    let mut out = Array2::<f32>::zeros((1, nb_erb));
                    let inv = 1.0 / ch as f32;
                    for i in 0..nb_erb {
                        let mut s = 0.0;
                        for c in 0..ch {
                            s += gains_per_ch[[c, i]];
                        }
                        out[[0, i]] = s * inv;
                    }
                    out
                }
                ReduceMask::NONE => gains_per_ch.clone(),
            })
        } else if any_apply_gain_zeros {
            Some(Array2::<f32>::zeros((ch, nb_erb)))
        } else {
            None
        };

        let coefs = if any_apply_df {
            Some(coefs_per_ch)
        } else {
            None
        };

        log::trace!(
            "Enhancing frame (ch={}) with lsnr {:>5.1} dB. stage 1: {}, stage 2: {}",
            ch,
            lsnr,
            any_apply_gains,
            any_apply_df
        );
        Ok((lsnr, gains, coefs))
    }

    /// Process a time-domain frame. Matches `DfTract::process` semantics.
    pub fn process(
        &mut self,
        noisy: ArrayView2<f32>,
        mut enh: ArrayViewMut2<f32>,
    ) -> Result<f32> {
        debug_assert_eq!(noisy.len_of(Axis(0)), enh.len_of(Axis(0)));
        debug_assert_eq!(noisy.len_of(Axis(1)), enh.len_of(Axis(1)));
        debug_assert_eq!(noisy.len_of(Axis(1)), self.hop_size);

        let (max_a, e) = noisy
            .iter()
            .fold((0f32, 0f32), |acc, x| (acc.0.max(x.abs()), acc.1 + x.powi(2)));
        let rms = e / noisy.len() as f32;
        if rms < 1e-7 {
            self.skip_counter += 1;
        } else {
            self.skip_counter = 0;
        }
        if self.skip_counter > 5 {
            enh.fill(0.);
            return Ok(-15.);
        }
        if max_a > 0.9999 {
            log::warn!("Possible clipping detected ({:.3}).", max_a);
        }

        // Advance rolling spec buffers.
        self.rolling_spec_buf_y.pop_front();
        self.rolling_spec_buf_x.pop_front();

        // Frame analysis: time → spectrum, per channel.
        for (ns_ch, mut spec_ch, state) in izip!(
            noisy.axis_iter(Axis(0)),
            self.spec_buf.axis_iter_mut(Axis(0)),
            self.df_states.iter_mut()
        ) {
            let spec_slice = spec_ch.as_slice_mut().unwrap();
            let spec_cplx = as_mut_complex(spec_slice);
            state.analysis(ns_ch.as_slice().unwrap(), spec_cplx);
        }
        self.rolling_spec_buf_y.push_back(self.spec_buf.clone());
        self.rolling_spec_buf_x.push_back(self.spec_buf.clone());

        if self.atten_lim.unwrap_or_default() == 1. {
            enh.assign(&noisy);
            return Ok(35.);
        }

        let (lsnr, gains, coefs) = self.process_raw()?;
        let (apply_erb, _, _) = self.apply_stages(lsnr);

        // Apply ERB mask to the spectrum lagged by df_order-1 frames.
        {
            let spec_idx = self.df_order - 1;
            let spec = &mut self.rolling_spec_buf_y[spec_idx];
            if let Some(gains) = gains {
                if gains.shape()[0] == 1 {
                    let gain_slc = gains.as_slice().unwrap();
                    for mut spec_ch in spec.view_mut().axis_iter_mut(Axis(0)) {
                        let slice = spec_ch.as_slice_mut().unwrap();
                        let cplx = as_mut_complex(slice);
                        self.df_states[0].apply_mask(cplx, gain_slc);
                    }
                } else {
                    for (gain_row, mut spec_ch) in gains
                        .axis_iter(Axis(0))
                        .zip(spec.view_mut().axis_iter_mut(Axis(0)))
                    {
                        let slice = spec_ch.as_slice_mut().unwrap();
                        let cplx = as_mut_complex(slice);
                        self.df_states[0].apply_mask(cplx, gain_row.as_slice().unwrap());
                    }
                }
                self.skip_counter = 0;
            } else {
                self.skip_counter += 1;
            }
        }

        // Select output spec: masked (rolling_y[df_order-1]) → self.spec_buf.
        let src_spec = self.rolling_spec_buf_y[self.df_order - 1].clone();
        self.spec_buf.assign(&src_spec);

        // DF stage: combine with coefs over df_order frames.
        if let Some(coefs) = coefs {
            df_apply(
                &self.rolling_spec_buf_x,
                &coefs,
                self.nb_df,
                self.df_order,
                self.n_freqs,
                &mut self.spec_buf,
            );
        }

        // Post filter per channel.
        if apply_erb && self.post_filter {
            let lh_idx = self.lookahead.max(self.df_order) - self.lookahead - 1;
            let spec_noisy = self.rolling_spec_buf_x[lh_idx].clone();
            for (mut enh_ch, noisy_ch) in self
                .spec_buf
                .axis_iter_mut(Axis(0))
                .zip(spec_noisy.axis_iter(Axis(0)))
            {
                let n_slice = noisy_ch.as_slice().unwrap();
                let e_slice = enh_ch.as_slice_mut().unwrap();
                crate::post_filter(
                    as_complex(n_slice),
                    as_mut_complex(e_slice),
                    self.post_filter_beta,
                );
            }
        }

        // Limit noise attenuation by mixing back some of the noisy signal.
        if let Some(lim) = self.atten_lim {
            let lh_idx = self.lookahead.max(self.df_order) - self.lookahead - 1;
            let spec_noisy = self.rolling_spec_buf_x[lh_idx].clone();
            let noisy_slice = spec_noisy.as_slice().unwrap();
            let enh_slice = self.spec_buf.as_slice_mut().unwrap();
            for (e, &n) in enh_slice.iter_mut().zip(noisy_slice.iter()) {
                *e = *e * (1. - lim) + n * lim;
            }
        }

        // Frame synthesis per channel.
        for (state, mut spec_ch, mut enh_out_ch) in izip!(
            self.df_states.iter_mut(),
            self.spec_buf.axis_iter_mut(Axis(0)),
            enh.axis_iter_mut(Axis(0))
        ) {
            let slice = spec_ch.as_slice_mut().unwrap();
            let cplx = as_mut_complex(slice);
            state.synthesis(cplx, enh_out_ch.as_slice_mut().unwrap());
        }
        Ok(lsnr)
    }

    pub fn apply_stages(&self, lsnr: f32) -> (bool, bool, bool) {
        if lsnr < self.min_db_thresh {
            (false, true, false)
        } else if lsnr > self.max_db_erb_thresh {
            (false, false, false)
        } else if lsnr > self.max_db_df_thresh {
            (true, false, false)
        } else {
            (true, false, true)
        }
    }
}

// -----------------------------------------------------------------------------
// DSP helpers (ported from tract.rs — operate on ndarray instead of tract
// Tensor so we don't pull tract into the openvino feature set).
// -----------------------------------------------------------------------------

fn calc_norm_alpha(sr: usize, hop_size: usize, tau: f32) -> f32 {
    let dt = hop_size as f32 / sr as f32;
    let alpha = f32::exp(-dt / tau);
    let mut a = 1.0;
    let mut precision = 3;
    while a >= 1.0 {
        a = (alpha * 10i32.pow(precision) as f32).round() / 10i32.pow(precision) as f32;
        precision += 1;
    }
    a
}

fn as_complex(buffer: &[f32]) -> &[Complex32] {
    unsafe {
        let ptr = buffer.as_ptr() as *const Complex32;
        std::slice::from_raw_parts(ptr, buffer.len() / 2)
    }
}
fn as_mut_complex(buffer: &mut [f32]) -> &mut [Complex32] {
    unsafe {
        let ptr = buffer.as_ptr() as *mut Complex32;
        std::slice::from_raw_parts_mut(ptr, buffer.len() / 2)
    }
}

fn copy_tensor_data(src: &Tensor, dst: &mut Tensor) {
    let src_data = src.get_data::<f32>().expect("tensor f32");
    let dst_data = dst.get_data_mut::<f32>().expect("tensor f32 mut");
    dst_data.copy_from_slice(src_data);
}

/// DF stage: `spec_out[c, f] += sum_o spec[o][c, f] * coefs[c, f, o]` for
/// frequency bins below `nb_df`. Upper bins are preserved from the stage-1
/// enhanced spectrum already staged in `spec_out`.
fn df_apply(
    spec_buf: &VecDeque<Array5<f32>>,
    coefs: &Array4<f32>,
    nb_df: usize,
    df_order: usize,
    n_freqs: usize,
    spec_out: &mut Array5<f32>,
) {
    let ch = spec_out.len_of(Axis(0));
    debug_assert_eq!(n_freqs, spec_out.shape()[3]);
    debug_assert_eq!(ch, coefs.shape()[0]);
    debug_assert_eq!(nb_df, coefs.shape()[1]);
    debug_assert_eq!(df_order, coefs.shape()[2]);
    debug_assert!(spec_buf.len() >= df_order);

    // Zero the low-freq band of spec_out (we are about to accumulate there).
    {
        let mut out_slice = spec_out.view_mut();
        for mut ch_view in out_slice.axis_iter_mut(Axis(0)) {
            let data = ch_view.as_slice_mut().unwrap();
            // data layout [1, 1, n_freqs, 2] flat.
            for f in 0..nb_df {
                data[f * 2] = 0.;
                data[f * 2 + 1] = 0.;
            }
        }
    }

    // Iterate over DF frames (oldest .. newest, df_order entries).
    for (t, s_frame) in spec_buf.iter().take(df_order).enumerate() {
        for c in 0..ch {
            let s_slice = s_frame.index_axis(Axis(0), c);
            let s_flat = s_slice.as_slice().unwrap();
            let c_flat = coefs.index_axis(Axis(0), c);
            let mut out_ch = spec_out.index_axis_mut(Axis(0), c);
            let o_flat = out_ch.as_slice_mut().unwrap();
            for f in 0..nb_df {
                let s_re = s_flat[f * 2];
                let s_im = s_flat[f * 2 + 1];
                let c_re = c_flat[[f, t, 0]];
                let c_im = c_flat[[f, t, 1]];
                // Complex multiply-accumulate: (s * c)
                o_flat[f * 2] += s_re * c_re - s_im * c_im;
                o_flat[f * 2 + 1] += s_re * c_im + s_im * c_re;
            }
        }
    }
}
