//! Minimal LADSPA host probe.
//!
//! Loads `libdeep_filter_ladspa.so`, walks each descriptor exported via
//! `ladspa_descriptor`, prints plugin metadata, and runs a single buffer of
//! zeros through the mono descriptor to make sure nothing panics. This is the
//! cheapest smoke-test that actually exercises the same dlopen + entry-point
//! path EasyEffects uses.
//!
//! Run with:
//!   LD_LIBRARY_PATH=<openvino/libs> DFN_OV_DEVICE=CPU \
//!     cargo run --manifest-path tooling/Cargo.toml --release
use std::ffi::{c_char, c_ulong, c_void, CStr};
use std::os::raw::c_int;
use std::ptr;

#[repr(C)]
#[derive(Debug)]
#[allow(non_snake_case)]
struct LadspaDescriptor {
    UniqueID: c_ulong,
    Label: *const c_char,
    Properties: c_int,
    Name: *const c_char,
    Maker: *const c_char,
    Copyright: *const c_char,
    PortCount: c_ulong,
    PortDescriptors: *const c_int,
    PortNames: *const *const c_char,
    PortRangeHints: *const c_void,
    ImplementationData: *const c_void,
    instantiate: Option<extern "C" fn(*const LadspaDescriptor, c_ulong) -> *mut c_void>,
    connect_port: Option<extern "C" fn(*mut c_void, c_ulong, *mut f32)>,
    activate: Option<extern "C" fn(*mut c_void)>,
    run: Option<extern "C" fn(*mut c_void, c_ulong)>,
    run_adding: Option<extern "C" fn(*mut c_void, c_ulong)>,
    set_run_adding_gain: Option<extern "C" fn(*mut c_void, f32)>,
    deactivate: Option<extern "C" fn(*mut c_void)>,
    cleanup: Option<extern "C" fn(*mut c_void)>,
}

fn main() {
    let so = std::env::args()
        .nth(1)
        .unwrap_or_else(|| format!("{}/.ladspa/libdeep_filter_ladspa.so", std::env::var("HOME").unwrap()));
    eprintln!("loading {}", so);
    let lib = unsafe { libloading::Library::new(&so) }.expect("dlopen");
    let get_desc: libloading::Symbol<
        unsafe extern "C" fn(c_ulong) -> *const LadspaDescriptor,
    > = unsafe { lib.get(b"ladspa_descriptor") }.expect("ladspa_descriptor symbol");

    let mut mono_desc: *const LadspaDescriptor = ptr::null();
    for i in 0..4 {
        let d = unsafe { get_desc(i) };
        if d.is_null() {
            break;
        }
        let desc = unsafe { &*d };
        let name = unsafe { CStr::from_ptr(desc.Name) }.to_string_lossy();
        let label = unsafe { CStr::from_ptr(desc.Label) }.to_string_lossy();
        eprintln!(
            "  [{}] id={} label={} name={} ports={}",
            i, desc.UniqueID, label, name, desc.PortCount
        );
        if mono_desc.is_null() && label == "deep_filter_mono" {
            mono_desc = d;
        }
    }
    assert!(!mono_desc.is_null(), "mono descriptor not found");

    let desc = unsafe { &*mono_desc };
    let instantiate = desc.instantiate.unwrap();
    let connect_port = desc.connect_port.unwrap();
    let activate = desc.activate.unwrap();
    let run = desc.run.unwrap();
    let cleanup = desc.cleanup.unwrap();

    eprintln!("instantiating mono at 48k…");
    let handle = instantiate(mono_desc, 48_000);
    assert!(!handle.is_null());

    // Wire: port 0 = audio in, 1 = audio out, 2.. = control inputs with
    // (mostly) sane defaults. LADSPA host is responsible for filling those.
    let sample_count: usize = 480; // one hop at 48k
    let mut in_buf = vec![0.0f32; sample_count];
    let mut out_buf = vec![0.0f32; sample_count];
    let mut ctrl_vals = [100.0f32, -10.0, 30.0, 20.0, 0.0, 0.0]; // atten_lim, min, max_erb, max_df, min_proc_buf, pf_beta

    unsafe {
        connect_port(handle, 0, in_buf.as_mut_ptr());
        connect_port(handle, 1, out_buf.as_mut_ptr());
        for (i, v) in ctrl_vals.iter_mut().enumerate() {
            connect_port(handle, (2 + i) as c_ulong, v as *mut f32);
        }
    }

    activate(handle);
    eprintln!("running 5 hops of silence…");
    for _ in 0..5 {
        run(handle, sample_count as c_ulong);
    }
    eprintln!("first 4 output samples: {:?}", &out_buf[..4]);
    cleanup(handle);
    eprintln!("ok");
}
