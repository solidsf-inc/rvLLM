//! Model + runtime configuration per `v3/specs/02-config.md`.
//!
//! Entry points:
//! - [`ModelConfig::load_hf`] — parse HF `config.json` field-by-field.
//! - [`RuntimeConfigBuilder::build`] — the only path to a `RuntimeConfig`.

mod builder;
mod hf;
mod model;
mod runtime;

pub use builder::RuntimeConfigBuilder;
pub use model::{ModelArch, ModelConfig};
pub use runtime::{GraphMode, LogLevel, PreemptionMode, RuntimeConfig};
