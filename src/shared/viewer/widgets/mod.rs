//! Shared widgets ported from the legacy `shared/viewer/ui/*.ts` (and the board's overlay and
//! badge), building exactly the DOM those produced so the legacy CSS styles them unchanged.
//!
//! The legacy code used `setTimeout` for lifetimes; here each widget that expires has a `tick`
//! to call from the screen's poll with the current time.

pub mod detection_card;
pub mod loading;
pub mod net_badge;
pub mod surface;
pub mod toast;

pub use detection_card::{CardKind, DetectionCard};
pub use loading::LoadingScreen;
pub use net_badge::NetBadge;
pub use surface::{Surface, SurfaceVariant, create_surface, mount_pane_label};
pub use toast::{TOAST_LIFETIME_MS, ToastKind, Toasts};
