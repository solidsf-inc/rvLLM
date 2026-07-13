//! HF `config.json` parsing helpers.
//!
//! Strict: no `serde(default)`, no `unwrap_or`. Missing field = named error.

use std::path::Path;

use crate::error::{ConfigError, Result, RvllmError};

pub(super) fn usize_field(
    v: &serde_json::Value,
    field: &'static str,
    file: &Path,
) -> Result<usize> {
    match v.get(field) {
        Some(x) if x.is_u64() => {
            let Some(raw) = x.as_u64() else {
                unreachable!("is_u64 and as_u64 must agree")
            };
            usize::try_from(raw).map_err(|_| {
                RvllmError::config(
                    ConfigError::InvalidField {
                        name: field,
                        reason: "value exceeds platform usize".into(),
                    },
                    field,
                )
            })
        }
        Some(_) => Err(RvllmError::config(
            ConfigError::HfTypeMismatch {
                name: field,
                expected: "non-negative integer",
            },
            field,
        )),
        None => Err(RvllmError::config(
            ConfigError::MissingHfField {
                name: field,
                file: file.to_path_buf(),
            },
            field,
        )),
    }
}

pub(super) fn f32_field(v: &serde_json::Value, field: &'static str, file: &Path) -> Result<f32> {
    match v.get(field) {
        Some(value) => {
            let x = value.as_f64().ok_or_else(|| {
                RvllmError::config(
                    ConfigError::HfTypeMismatch {
                        name: field,
                        expected: "finite number",
                    },
                    field,
                )
            })?;
            let value = x as f32;
            if !x.is_finite() || !value.is_finite() {
                return Err(RvllmError::config(
                    ConfigError::InvalidField {
                        name: field,
                        reason: "must be finite and representable as f32".into(),
                    },
                    field,
                ));
            }
            Ok(value)
        }
        None => Err(RvllmError::config(
            ConfigError::MissingHfField {
                name: field,
                file: file.to_path_buf(),
            },
            field,
        )),
    }
}

pub(super) fn bool_field_opt(v: &serde_json::Value, field: &'static str) -> Option<bool> {
    v.get(field).and_then(|x| x.as_bool())
}

/// String field supporting dotted paths like `architectures.0`.
pub(super) fn str_field(
    v: &serde_json::Value,
    dotted: &'static str,
    file: &Path,
) -> Result<String> {
    let mut cur = v;
    for part in dotted.split('.') {
        let next = if let Ok(idx) = part.parse::<usize>() {
            cur.get(idx)
        } else {
            cur.get(part)
        };
        cur = match next {
            Some(x) => x,
            None => {
                return Err(RvllmError::config(
                    ConfigError::MissingHfField {
                        name: dotted,
                        file: file.to_path_buf(),
                    },
                    dotted,
                ));
            }
        };
    }
    match cur.as_str() {
        Some(s) => Ok(s.to_string()),
        None => Err(RvllmError::config(
            ConfigError::HfTypeMismatch {
                name: dotted,
                expected: "string",
            },
            dotted,
        )),
    }
}
