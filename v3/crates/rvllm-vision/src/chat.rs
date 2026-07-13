// Ported from huggingface/transformers revision
// 10555512868d663ee1ff627e4f5c5c260114235b:
// src/transformers/models/gemma4/modular_gemma4.py
// Apache-2.0 License, Copyright (c) HuggingFace and Google.
// Source concept: per-image chat-template expansion
//   `<boi>{<image_soft_token> * count}<eoi>`.
// Modifications: bounded Rust string builder with token strings supplied
// by the caller from the tokenizer's added_tokens.

pub const MAX_IMAGE_SOFT_TOKENS: usize = 1_120;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ChatTemplateError {
    #[error("image soft-token count {0} exceeds {MAX_IMAGE_SOFT_TOKENS}")]
    TooManySoftTokens(usize),
    #[error("image token expansion length overflow")]
    LengthOverflow,
}

/// Returns the literal special-token text sequence that should be
/// substituted into a user message at each image position, e.g.:
///   "<start_of_image>"  +  "<image_soft_token>" * count  +  "<end_of_image>"
///
/// The actual string forms (e.g. "<start_of_image>") must come from
/// the tokenizer's added_tokens; the serving layer looks them up at startup
/// and passes them in here.
pub fn build_image_token_string(
    boi_str: &str,
    soft_token_str: &str,
    eoi_str: &str,
    count: usize,
) -> Result<String, ChatTemplateError> {
    if count > MAX_IMAGE_SOFT_TOKENS {
        return Err(ChatTemplateError::TooManySoftTokens(count));
    }
    let capacity = soft_token_str
        .len()
        .checked_mul(count)
        .and_then(|n| n.checked_add(boi_str.len()))
        .and_then(|n| n.checked_add(eoi_str.len()))
        .ok_or(ChatTemplateError::LengthOverflow)?;
    let mut out = String::with_capacity(capacity);
    out.push_str(boi_str);
    for _ in 0..count {
        out.push_str(soft_token_str);
    }
    out.push_str(eoi_str);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn three_soft_tokens() {
        assert_eq!(
            build_image_token_string("<boi>", "<img>", "<eoi>", 3).unwrap(),
            "<boi><img><img><img><eoi>"
        );
    }
    #[test]
    fn zero_soft_tokens() {
        assert_eq!(
            build_image_token_string("<boi>", "<img>", "<eoi>", 0).unwrap(),
            "<boi><eoi>"
        );
    }
    #[test]
    fn rejects_unbounded_expansion() {
        assert_eq!(
            build_image_token_string("<boi>", "<img>", "<eoi>", 1_121),
            Err(ChatTemplateError::TooManySoftTokens(1_121))
        );
    }
}
