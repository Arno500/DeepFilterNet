//! Sanity check: run a few hops of 48 kHz white noise through the OpenVINO
//! backend on whichever device is available, make sure we get finite output,
//! and print wall-clock timings so we can spot obvious regressions.
//!
//! Override the backend device with `DFN_OV_DEVICE=CPU|GPU|NPU`.

#![cfg(feature = "openvino")]

use std::time::Instant;

use df::openvino::{DfOpenVino, DfParams, RuntimeParams};
use ndarray::Array2;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

#[test]
fn openvino_backend_runs() {
    let _ = env_logger::builder().is_test(true).try_init();

    let dfp = DfParams::default();
    let rp = RuntimeParams::default_with_ch(1);
    let t0 = Instant::now();
    let mut df = DfOpenVino::new(dfp, &rp).expect("DfOpenVino::new failed");
    eprintln!(
        "init: {:.1}ms  (hop={}, sr={})",
        t0.elapsed().as_secs_f32() * 1000.,
        df.hop_size,
        df.sr
    );

    let hop = df.hop_size;
    let ch = df.ch;
    let mut rng = StdRng::seed_from_u64(42);
    let mut input = Array2::<f32>::zeros((ch, hop));
    let mut output = Array2::<f32>::zeros((ch, hop));

    // Warm-up: let the recurrent state settle + JIT caches warm.
    for _ in 0..3 {
        for v in input.iter_mut() {
            *v = rng.gen_range(-0.2f32..0.2f32);
        }
        df.process(input.view(), output.view_mut()).expect("process");
    }

    // Measure a small batch of hops.
    const N: usize = 50;
    let t0 = Instant::now();
    let mut last_lsnr = 0.0;
    for _ in 0..N {
        for v in input.iter_mut() {
            *v = rng.gen_range(-0.2f32..0.2f32);
        }
        last_lsnr = df.process(input.view(), output.view_mut()).expect("process");
    }
    let total_ms = t0.elapsed().as_secs_f32() * 1000.;
    let per_hop_ms = total_ms / N as f32;
    let t_audio_ms = hop as f32 / df.sr as f32 * 1000.;
    let rtf = per_hop_ms / t_audio_ms;
    eprintln!(
        "{} hops in {:.1}ms  ({:.2}ms/hop, audio {:.2}ms → RTF {:.3})",
        N, total_ms, per_hop_ms, t_audio_ms, rtf
    );
    eprintln!("final lsnr: {:.1} dB", last_lsnr);

    // Output must be finite. LSNR values are capped to [-15, 35] by config.
    for &v in output.iter() {
        assert!(v.is_finite(), "non-finite sample: {}", v);
    }
    assert!(last_lsnr.is_finite(), "non-finite lsnr");
}

/// Full-pipeline benchmark. Lowers thresholds so every hop runs encoder +
/// erb_dec + df_dec. Not wrapped in `#[ignore]` because it IS the benchmark
/// we care about — but skipped unless invoked explicitly by name.
#[test]
fn openvino_backend_full_pipeline_bench() {
    let _ = env_logger::builder().is_test(true).try_init();

    let dfp = DfParams::default();
    let mut rp = RuntimeParams::default_with_ch(1);
    rp.min_db_thresh = -20.0;     // always above → stage 1 runs
    rp.max_db_erb_thresh = 35.0;  // always below → stage 1 runs
    rp.max_db_df_thresh = 35.0;   // always below → stage 2 runs
    let t0 = Instant::now();
    let mut df = DfOpenVino::new(dfp, &rp).expect("init");
    let init_ms = t0.elapsed().as_secs_f32() * 1000.;

    let hop = df.hop_size;
    let mut rng = StdRng::seed_from_u64(42);
    let mut input = Array2::<f32>::zeros((1, hop));
    let mut output = Array2::<f32>::zeros((1, hop));

    // Warm-up (encoder/decoder state + plugin caches).
    for _ in 0..5 {
        for v in input.iter_mut() {
            *v = rng.gen_range(-0.2f32..0.2f32);
        }
        df.process(input.view(), output.view_mut()).unwrap();
    }

    fn thread_cpu_ns() -> u64 {
        unsafe {
            let mut ts = std::mem::zeroed::<libc::timespec>();
            libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts);
            ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
        }
    }

    const N: usize = 200;
    let cpu0 = thread_cpu_ns();
    let t0 = Instant::now();
    let mut min_ms = f32::INFINITY;
    let mut max_ms = 0.0f32;
    for _ in 0..N {
        for v in input.iter_mut() {
            *v = rng.gen_range(-0.2f32..0.2f32);
        }
        let t = Instant::now();
        df.process(input.view(), output.view_mut()).unwrap();
        let ms = t.elapsed().as_secs_f32() * 1000.;
        min_ms = min_ms.min(ms);
        max_ms = max_ms.max(ms);
    }
    let total_ms = t0.elapsed().as_secs_f32() * 1000.;
    let cpu_ms = (thread_cpu_ns() - cpu0) as f32 / 1_000_000.;
    let avg_ms = total_ms / N as f32;
    let avg_cpu_ms = cpu_ms / N as f32;
    let audio_ms = hop as f32 / df.sr as f32 * 1000.;
    eprintln!(
        "\n=== full pipeline, {} hops ===\n  init:         {:.1} ms\n  avg wall/hop: {:.2} ms  (audio {:.2} ms → RTF {:.3})\n  avg CPU/hop:  {:.2} ms  (CPU frac {:.3})\n  min wall/hop: {:.2} ms\n  max wall/hop: {:.2} ms",
        N, init_ms, avg_ms, audio_ms, avg_ms / audio_ms, avg_cpu_ms, avg_cpu_ms / audio_ms, min_ms, max_ms
    );
    for &v in output.iter() {
        assert!(v.is_finite());
    }
}

// OpenVINO's Core / plugin loader keeps some global state, and creating two
// `Core`s in the same process can segfault on teardown. Run this test with
// `--test openvino_smoke openvino_backend_stereo_runs` in isolation.
#[test]
#[ignore]
fn openvino_backend_stereo_runs() {
    let _ = env_logger::builder().is_test(true).try_init();

    let dfp = DfParams::default();
    let rp = RuntimeParams::default_with_ch(2);
    let mut df = DfOpenVino::new(dfp, &rp).expect("DfOpenVino::new(ch=2) failed");
    assert_eq!(df.ch, 2);

    let hop = df.hop_size;
    let mut rng = StdRng::seed_from_u64(42);
    let mut input = Array2::<f32>::zeros((2, hop));
    let mut output = Array2::<f32>::zeros((2, hop));
    for _ in 0..5 {
        for v in input.iter_mut() {
            *v = rng.gen_range(-0.2f32..0.2f32);
        }
        df.process(input.view(), output.view_mut())
            .expect("process ch=2");
    }
    for &v in output.iter() {
        assert!(v.is_finite());
    }
}
