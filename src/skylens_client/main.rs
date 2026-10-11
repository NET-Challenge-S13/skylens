//! Native window of the status board.
//!
//!     skylens-client --core ws://127.0.0.1:8080/viewer
//!     skylens-client --core gateway.local:8080 --points off
//!
//! The web build has no binary; see build-web.sh.

#[cfg(not(target_arch = "wasm32"))]
fn main() -> std::process::ExitCode {
    let opts = match skylens_client::parse_args(std::env::args().skip(1)) {
        Ok(o) => o,
        Err(msg) => {
            eprintln!("{msg}");
            return std::process::ExitCode::from(2);
        }
    };
    match skylens_client::run_native(&opts) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("skylens-client: {e:#}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn main() {}
