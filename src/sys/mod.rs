//! Auto-generated bindings to NVIDIA Video Codec SDK.
//!
//! The bindings were generated using [bindgen](https://github.com/rust-lang/rust-bindgen)
//! using the scripts `sys/linux_sys/bindgen.sh` and
//! `sys/windows_sys/bindgen.ps1` for the respective operating system.

mod guid;
mod version;

#[allow(warnings)]
#[rustfmt::skip]
#[cfg(target_os = "linux")]
mod linux_sys;
#[cfg(target_os = "linux")]
pub use linux_sys::*;

#[allow(warnings)]
#[rustfmt::skip]
#[cfg(target_os = "windows")]
mod windows_sys;
#[cfg(target_os = "windows")]
pub use windows_sys::*;

// `version.rs` and `guid.rs` follow the SDK 13.0 header, but the Windows
// bindings were generated from SDK 12.1: their struct layouts do not match.
#[cfg(target_os = "windows")]
compile_error!(
    "the Windows bindings have not been regenerated for NVENC API 13.0; run \
     src/sys/windows_sys/bindgen.ps1 against src/sys/headers/nvEncodeAPI.h"
);
