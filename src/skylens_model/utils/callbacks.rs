//! Rust twin of `utils/callbacks.py`.
//!
//! Training-loop callbacks for stop and resume. The Python side hangs off HuggingFace's
//! `TrainerCallback`; there is no such framework in Rust, so this module also defines the
//! small callback protocol ([`TrainerCallback`], [`TrainerState`], [`TrainerControl`]) that
//! `trainer.rs` drives, with the same hook names.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use tch::nn::VarStore;

// --------------------------------------------------------------------------- //
// callback protocol
// --------------------------------------------------------------------------- //

/// Read-only view of training progress (`TrainerState`).
#[derive(Debug, Clone, Copy, Default)]
pub struct TrainerState {
    /// Fractional epoch: `completed_steps / steps_per_epoch`.
    pub epoch: f64,
    pub global_step: u64,
}

/// Flags a callback can raise (`TrainerControl`).
#[derive(Debug, Clone, Copy, Default)]
pub struct TrainerControl {
    pub should_save: bool,
    pub should_training_stop: bool,
}

/// Hook points of the training loop. Every hook has a no-op default.
pub trait TrainerCallback {
    fn on_train_begin(&mut self, _state: &TrainerState, _control: &mut TrainerControl, _vs: &VarStore) {}
    fn on_epoch_begin(&mut self, _state: &TrainerState, _control: &mut TrainerControl, _vs: &VarStore) {}
    fn on_step_end(&mut self, _state: &TrainerState, _control: &mut TrainerControl, _vs: &VarStore) {}
    fn on_evaluate(&mut self, _state: &TrainerState, _control: &mut TrainerControl, _vs: &VarStore) {}
    fn on_train_end(&mut self, _state: &TrainerState, _control: &mut TrainerControl, _vs: &VarStore) {}
}

// --------------------------------------------------------------------------- //
// graceful interrupt
// --------------------------------------------------------------------------- //

/// Number of termination signals received since the last `on_train_begin`.
/// Process-wide because a signal handler can only be installed once per process.
static SIGNALS_SEEN: AtomicUsize = AtomicUsize::new(0);

/// Finishes the current step, saves and stops when a termination signal arrives. A second
/// signal terminates immediately.
///
/// Python restores the previous handlers and re-raises the signal on the second hit; with
/// `ctrlc` there is no previous handler to restore, so the second signal exits the process
/// with status 130 instead.
pub struct GracefulInterruptCallback {
    installed: bool,
}

impl Default for GracefulInterruptCallback {
    fn default() -> Self {
        Self::new()
    }
}

impl GracefulInterruptCallback {
    pub fn new() -> Self {
        GracefulInterruptCallback { installed: false }
    }

    fn install(&mut self) {
        if self.installed {
            return;
        }
        let result = ctrlc::set_handler(|| {
            if SIGNALS_SEEN.fetch_add(1, Ordering::SeqCst) >= 1 {
                std::process::exit(130);
            }
            tracing::warn!(
                "[SkyLens] termination signal received: finishing the current step, saving a \
                 checkpoint and stopping. Send it once more to abort immediately."
            );
        });
        match result {
            Ok(()) => self.installed = true,
            // already installed by an earlier callback instance: the shared counter still works
            Err(ctrlc::Error::MultipleHandlers) => self.installed = true,
            Err(e) => tracing::warn!("[SkyLens] could not install the signal hook: {e}"),
        }
    }

    pub fn interrupted(&self) -> bool {
        SIGNALS_SEEN.load(Ordering::SeqCst) > 0
    }

    fn maybe_stop(&self, control: &mut TrainerControl) {
        if self.interrupted() {
            control.should_save = true;
            control.should_training_stop = true;
        }
    }
}

impl TrainerCallback for GracefulInterruptCallback {
    fn on_train_begin(&mut self, _s: &TrainerState, _c: &mut TrainerControl, _vs: &VarStore) {
        SIGNALS_SEEN.store(0, Ordering::SeqCst);
        self.install();
    }

    fn on_step_end(&mut self, _s: &TrainerState, c: &mut TrainerControl, _vs: &VarStore) {
        self.maybe_stop(c);
    }

    fn on_evaluate(&mut self, _s: &TrainerState, c: &mut TrainerControl, _vs: &VarStore) {
        self.maybe_stop(c);
    }

    fn on_train_end(&mut self, s: &TrainerState, _c: &mut TrainerControl, _vs: &VarStore) {
        if self.interrupted() {
            tracing::warn!(
                "[SkyLens] stopped on request (global_step={}). Point resume_from_checkpoint at \
                 the same output_dir to continue.",
                s.global_step
            );
        }
    }
}

// --------------------------------------------------------------------------- //
// resume
// --------------------------------------------------------------------------- //

/// Step number of a `checkpoint-<int>` directory name.
fn checkpoint_step(name: &str) -> Option<u64> {
    name.strip_prefix("checkpoint-")?.parse::<u64>().ok()
}

/// Latest `checkpoint-<step>` directory under `output_dir`, or `None`
/// (`transformers.trainer_utils.get_last_checkpoint`).
pub fn find_resume_checkpoint(output_dir: impl AsRef<Path>) -> Option<PathBuf> {
    let dir = output_dir.as_ref();
    let entries = std::fs::read_dir(dir).ok()?;
    entries
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .filter_map(|e| {
            let step = checkpoint_step(&e.file_name().to_string_lossy())?;
            Some((step, e.path()))
        })
        .max_by_key(|(step, _)| *step)
        .map(|(_, path)| path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_name_parsing() {
        assert_eq!(checkpoint_step("checkpoint-500"), Some(500));
        assert_eq!(checkpoint_step("checkpoint-"), None);
        assert_eq!(checkpoint_step("checkpoint-abc"), None);
        assert_eq!(checkpoint_step("runs"), None);
    }

    #[test]
    fn latest_checkpoint_is_numeric_max() {
        let dir = std::env::temp_dir().join("skylens_resume_test");
        let _ = std::fs::remove_dir_all(&dir);
        for n in ["checkpoint-9", "checkpoint-100", "checkpoint-20", "logs"] {
            std::fs::create_dir_all(dir.join(n)).unwrap();
        }
        // a stray file named like a checkpoint is ignored
        std::fs::write(dir.join("checkpoint-999"), b"x").unwrap();
        assert_eq!(find_resume_checkpoint(&dir), Some(dir.join("checkpoint-100")));
        assert_eq!(find_resume_checkpoint(dir.join("missing")), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn control_defaults_are_off() {
        let c = TrainerControl::default();
        assert!(!c.should_save && !c.should_training_stop);
    }
}
