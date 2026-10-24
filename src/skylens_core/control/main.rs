//! Native control tower window.
//!
//!     skylens-control --core ws://127.0.0.1:8080/control
//!
//! The web build has no binary; see build-web.sh.

// A release build on Windows is a GUI program, so starting it does not open a console window
// behind it. Debug builds keep the console for the logs; a parent that hands over pipes (the demo
// launcher) still gets the output either way.
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

#[cfg(not(target_arch = "wasm32"))]
fn main() -> std::process::ExitCode {
    let opts = match control_screen::native::parse_args(std::env::args().skip(1), |k| std::env::var(k).ok()) {
        Ok(o) => o,
        Err(msg) => {
            eprintln!("{msg}");
            return std::process::ExitCode::from(2);
        }
    };
    match control_screen::native::run(opts) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("skylens-control: {e:#}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn main() {}
