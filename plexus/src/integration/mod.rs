//! Integration of external crates and foreign types.
//!
//! This module provides implementations of traits in Plexus for foreign types.
//! Integrated crates are re-exported within a sub-module, which can be used to
//! avoid versioning conflicts.
//!
//! Re-exported types are hidden in the documention for Plexus.

pub mod theon {
    #[doc(hidden)]
    pub use ::theon::*;
}

// Feature modules. These are empty unless Cargo features are enabled.
pub mod cgmath;
pub mod mint;
pub mod nalgebra;
