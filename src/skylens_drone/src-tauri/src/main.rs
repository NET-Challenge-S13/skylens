// Thin Tauri shell.
//
// It opens a window on the SAME page a browser would load and does nothing else:
// no #[tauri::command], no state, no IPC. Every behaviour the drone has —
// gateway link, flight model, slicing, telemetry, the operator panel — lives in
// TypeScript under src/skylens_drone/ and runs unchanged in a plain tab. That is
// the point: `npm run demo` never needs a Rust toolchain, and this shell is only
// what you ship to a real airframe's ground unit.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    tauri::Builder::default()
        .run(tauri::generate_context!())
        .expect("skylens_drone: failed to start the Tauri window");
}
