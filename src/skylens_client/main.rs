//! Native window of a SkyLens screen.
//!
//!     skylens-client --screen control --core ws://127.0.0.1:8080/viewer
//!     skylens-client --screen status  --core core.local:8080 --anchor 36.3685,127.3475,30
//!
//! The web build has no binary; see build-web.sh.

#[cfg(not(target_arch = "wasm32"))]
fn main() -> std::process::ExitCode {
    let cfg = match skylens_client::Config::from_args(std::env::args().skip(1)) {
        Ok(cfg) => cfg,
        Err(msg) => {
            eprintln!("{msg}");
            return std::process::ExitCode::from(2);
        }
    };
    match skylens_client::native::run(cfg) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("skylens-client: {e:#}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn main() {}
