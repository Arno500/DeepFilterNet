//! Mirror of openvino full-pipeline bench but using the tract backend, so the
//! numbers are directly comparable.

#![cfg(feature = "tract")]

use std::time::Instant;

use df::tract::{DfParams, DfTract, RuntimeParams};
use ndarray::Array2;
use rand::{rngs::StdRng, Rng, SeedableRng};

#[test]
fn tract_backend_full_pipeline_bench() {
    let _ = env_logger::builder().is_test(true).try_init();

    let dfp = DfParams::default();
    let mut rp = RuntimeParams::default_with_ch(1);
    rp.min_db_thresh = -20.0;
    rp.max_db_erb_thresh = 35.0;
    rp.max_db_df_thresh = 35.0;
    let t0 = Instant::now();
    let mut df = DfTract::new(dfp, &rp).expect("init");
    let init_ms = t0.elapsed().as_secs_f32() * 1000.;

    let hop = df.hop_size;
    let mut rng = StdRng::seed_from_u64(42);
    let mut input = Array2::<f32>::zeros((1, hop));
    let mut output = Array2::<f32>::zeros((1, hop));

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
        "\n=== tract full pipeline, {} hops ===\n  init:         {:.1} ms\n  avg wall/hop: {:.2} ms  (audio {:.2} ms → RTF {:.3})\n  avg CPU/hop:  {:.2} ms  (CPU frac {:.3})\n  min wall/hop: {:.2} ms\n  max wall/hop: {:.2} ms",
        N, init_ms, avg_ms, audio_ms, avg_ms / audio_ms, avg_cpu_ms, avg_cpu_ms / audio_ms, min_ms, max_ms
    );
    for &v in output.iter() {
        assert!(v.is_finite());
    }
}
