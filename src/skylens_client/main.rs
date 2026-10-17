//! Native window of the status board.
//!
//!     skylens-client --core ws://127.0.0.1:8080/viewer
//!     skylens-client --core gateway.local:8080 --points off
//!
//! The web build has no binary; see build-web.sh.

// A release build on Windows is a GUI program, so starting it does not open a console window
// behind it. Debug builds keep the console for the logs; a parent that hands over pipes (the demo
// launcher) still gets the output either way.
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

#[cfg(not(target_arch = "wasm32"))]
fn main() -> std::process::ExitCode {
    let opts = match status_screen::parse_args(std::env::args().skip(1)) {
        Ok(o) => o,
        Err(msg) => {
            eprintln!("{msg}");
            return std::process::ExitCode::from(2);
        }
    };
    match status_screen::run_native(&opts) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("skylens-client: {e:#}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn main() {}
