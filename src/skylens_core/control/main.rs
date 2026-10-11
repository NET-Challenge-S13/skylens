//! Native control tower window.
//!
//!     skylens-control --core ws://127.0.0.1:8080/control
//!
//! The web build has no binary; see build-web.sh.

#[cfg(not(target_arch = "wasm32"))]
fn main() -> std::process::ExitCode {
    let opts = match skylens_control::native::parse_args(std::env::args().skip(1), |k| std::env::var(k).ok()) {
        Ok(o) => o,
        Err(msg) => {
            eprintln!("{msg}");
            return std::process::ExitCode::from(2);
        }
    };
    match skylens_control::native::run(opts) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("skylens-control: {e:#}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn main() {}
