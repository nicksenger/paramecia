//! WIT interface definitions for paramecia controller components.
//!
//! This crate provides the WIT (WebAssembly Interface Types) definitions
//! that define the contract between a paramecia host and guest WASM controllers.
//!
//! # Usage
//!
//! ## As `&'static str`
//!
//! ```rust
//! let wit = paramecia_wit::WORLD_WIT;
//! ```
//!
//! ## For bindgen via `build.rs`
//!
//! Add `paramecia-wit` as a `[build-dependencies]` entry, then in your
//! `build.rs` write the WIT files to `OUT_DIR` and generate a bindgen
//! invocation file that the proc macro can read:
//!
//! ```rust,ignore
//! // build.rs
//! fn main() {
//!     let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
//!     let wit_dir = out.join("paramecia-wit");
//!     paramecia_wit::write_wit_to(&wit_dir).unwrap();
//!
//!     // Write a file that invokes the bindgen macro with the absolute path
//!     std::fs::write(
//!         out.join("paramecia_bindgen.rs"),
//!         format!(
//!             "wit_bindgen::generate!({{ world: \"controller\", path: \"{}\", generate_all, }});",
//!             wit_dir.display(),
//!         ),
//!     ).unwrap();
//! }
//! ```
//!
//! Then in your `lib.rs`:
//!
//! ```rust,ignore
//! mod bindings {
//!     include!(concat!(env!("OUT_DIR"), "/paramecia_bindgen.rs"));
//! }
//! ```

/// The main world.wit definition.
pub const WORLD_WIT: &str = include_str!("../wit/world.wit");

/// WASI CLI dependency.
pub const WASI_CLI_WIT: &str = include_str!("../wit/deps/cli.wit");

/// WASI clocks dependency.
pub const WASI_CLOCKS_WIT: &str = include_str!("../wit/deps/clocks.wit");

/// WASI filesystem dependency.
pub const WASI_FILESYSTEM_WIT: &str = include_str!("../wit/deps/filesystem.wit");

/// WASI random dependency.
pub const WASI_RANDOM_WIT: &str = include_str!("../wit/deps/random.wit");

/// WASI sockets dependency.
pub const WASI_SOCKETS_WIT: &str = include_str!("../wit/deps/sockets.wit");

/// Write all WIT files to the given directory, creating subdirectories as needed.
///
/// This is intended for use in `build.rs` scripts. The resulting directory
/// layout matches what `wasmtime::component::bindgen!` and
/// `wit_bindgen::generate!` expect for their `path` parameter.
pub fn write_wit_to(dir: &std::path::Path) -> std::io::Result<()> {
    let deps = dir.join("deps");
    std::fs::create_dir_all(&deps)?;
    std::fs::write(dir.join("world.wit"), WORLD_WIT)?;
    std::fs::write(deps.join("cli.wit"), WASI_CLI_WIT)?;
    std::fs::write(deps.join("clocks.wit"), WASI_CLOCKS_WIT)?;
    std::fs::write(deps.join("filesystem.wit"), WASI_FILESYSTEM_WIT)?;
    std::fs::write(deps.join("random.wit"), WASI_RANDOM_WIT)?;
    std::fs::write(deps.join("sockets.wit"), WASI_SOCKETS_WIT)?;
    Ok(())
}
