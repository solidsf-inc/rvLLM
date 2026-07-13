// Portions are adapted from mistral.rs at commit
// 31c13eb4587d3e4a5204870c98b70c05a1e5c943 under the MIT License.

//! Bounded image loading for vision requests.
//!
//! `parse_image_url` accepts only `data:` URLs. Local files and HTTP are
//! opt-in crate features and still require an explicit [`ImagePolicy`] allowlist.

use std::io::Cursor;

use image::{DynamicImage, ImageReader, Limits};
use thiserror::Error;

const DEFAULT_MAX_ENCODED_BYTES: usize = 20 * 1024 * 1024;
const DEFAULT_MAX_DIMENSION: u32 = 8_192;
const DEFAULT_MAX_DECODE_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum ImageioError {
    #[error("image source is not permitted: {0}")]
    NotPermitted(String),
    #[error("invalid image source: {0}")]
    InvalidSource(String),
    #[error("image payload exceeds the configured byte limit")]
    TooLarge,
    #[error("image transport failed: {0}")]
    Transport(String),
    #[error("image decode failed: {0}")]
    Decode(String),
}

#[derive(Clone, Debug)]
pub struct ImagePolicy {
    max_encoded_bytes: usize,
    max_dimension: u32,
    max_decode_bytes: u64,
    #[cfg(feature = "local-files")]
    local_roots: Vec<std::path::PathBuf>,
    #[cfg(feature = "http")]
    allowed_hosts: std::collections::BTreeSet<String>,
    #[cfg(feature = "http")]
    connect_timeout: std::time::Duration,
    #[cfg(feature = "http")]
    request_timeout: std::time::Duration,
}

impl Default for ImagePolicy {
    fn default() -> Self {
        Self {
            max_encoded_bytes: DEFAULT_MAX_ENCODED_BYTES,
            max_dimension: DEFAULT_MAX_DIMENSION,
            max_decode_bytes: DEFAULT_MAX_DECODE_BYTES,
            #[cfg(feature = "local-files")]
            local_roots: Vec::new(),
            #[cfg(feature = "http")]
            allowed_hosts: std::collections::BTreeSet::new(),
            #[cfg(feature = "http")]
            connect_timeout: std::time::Duration::from_secs(3),
            #[cfg(feature = "http")]
            request_timeout: std::time::Duration::from_secs(10),
        }
    }
}

impl ImagePolicy {
    pub fn with_limits(
        mut self,
        max_encoded_bytes: usize,
        max_dimension: u32,
        max_decode_bytes: u64,
    ) -> Result<Self, ImageioError> {
        if max_encoded_bytes == 0 || max_dimension == 0 || max_decode_bytes == 0 {
            return Err(ImageioError::InvalidSource(
                "image limits must be non-zero".into(),
            ));
        }
        self.max_encoded_bytes = max_encoded_bytes;
        self.max_dimension = max_dimension;
        self.max_decode_bytes = max_decode_bytes;
        Ok(self)
    }

    #[cfg(feature = "local-files")]
    pub fn allow_local_root(
        mut self,
        root: impl AsRef<std::path::Path>,
    ) -> Result<Self, ImageioError> {
        let root = root
            .as_ref()
            .canonicalize()
            .map_err(|e| ImageioError::InvalidSource(format!("invalid local root: {e}")))?;
        if !root.is_dir() {
            return Err(ImageioError::InvalidSource(
                "local root is not a directory".into(),
            ));
        }
        self.local_roots.push(root);
        Ok(self)
    }

    #[cfg(feature = "http")]
    pub fn allow_http_host(mut self, host: &str) -> Result<Self, ImageioError> {
        let host = host.trim().trim_end_matches('.').to_ascii_lowercase();
        if host.is_empty()
            || host
                .bytes()
                .any(|b| !(b.is_ascii_alphanumeric() || b == b'.' || b == b'-' || b == b':'))
        {
            return Err(ImageioError::InvalidSource("invalid HTTP host".into()));
        }
        self.allowed_hosts.insert(host);
        Ok(self)
    }

    pub fn load(&self, source: &str) -> Result<DynamicImage, ImageioError> {
        let source_cap = self
            .max_encoded_bytes
            .checked_mul(2)
            .and_then(|n| n.checked_add(1_024))
            .unwrap_or(usize::MAX);
        if source.len() > source_cap {
            return Err(ImageioError::TooLarge);
        }

        let bytes = if starts_with_ignore_ascii_case(source, "data:") {
            self.decode_data_url(source)?
        } else if starts_with_ignore_ascii_case(source, "http://")
            || starts_with_ignore_ascii_case(source, "https://")
        {
            #[cfg(feature = "http")]
            {
                self.fetch_http(source)?
            }
            #[cfg(not(feature = "http"))]
            {
                return Err(ImageioError::NotPermitted(
                    "HTTP loading is disabled at build time".into(),
                ));
            }
        } else if starts_with_ignore_ascii_case(source, "file://") || !source.contains("://") {
            #[cfg(feature = "local-files")]
            {
                self.read_local(source)?
            }
            #[cfg(not(feature = "local-files"))]
            {
                return Err(ImageioError::NotPermitted(
                    "local file loading is disabled at build time".into(),
                ));
            }
        } else {
            return Err(ImageioError::InvalidSource(
                "unsupported image URL scheme".into(),
            ));
        };

        self.decode_image(bytes)
    }

    fn decode_data_url(&self, source: &str) -> Result<Vec<u8>, ImageioError> {
        let parsed = data_url::DataUrl::process(source)
            .map_err(|e| ImageioError::InvalidSource(format!("data URL parse: {e:?}")))?;
        let (bytes, _) = parsed
            .decode_to_vec()
            .map_err(|e| ImageioError::InvalidSource(format!("data URL decode: {e:?}")))?;
        self.check_encoded_len(bytes.len())?;
        Ok(bytes)
    }

    fn check_encoded_len(&self, len: usize) -> Result<(), ImageioError> {
        if len > self.max_encoded_bytes {
            Err(ImageioError::TooLarge)
        } else {
            Ok(())
        }
    }

    fn decode_image(&self, bytes: Vec<u8>) -> Result<DynamicImage, ImageioError> {
        self.check_encoded_len(bytes.len())?;
        let mut reader = ImageReader::new(Cursor::new(bytes))
            .with_guessed_format()
            .map_err(|e| ImageioError::Decode(e.to_string()))?;
        let mut limits = Limits::default();
        limits.max_image_width = Some(self.max_dimension);
        limits.max_image_height = Some(self.max_dimension);
        limits.max_alloc = Some(self.max_decode_bytes);
        reader.limits(limits);
        reader
            .decode()
            .map_err(|e| ImageioError::Decode(e.to_string()))
    }

    #[cfg(feature = "local-files")]
    fn read_local(&self, source: &str) -> Result<Vec<u8>, ImageioError> {
        use std::io::Read as _;

        if self.local_roots.is_empty() {
            return Err(ImageioError::NotPermitted(
                "no local image roots are configured".into(),
            ));
        }
        let requested = if starts_with_ignore_ascii_case(source, "file://") {
            let url = url::Url::parse(source)
                .map_err(|_| ImageioError::InvalidSource("invalid file URL".into()))?;
            if url.host_str().is_some() {
                return Err(ImageioError::NotPermitted(
                    "file URL authorities are not allowed".into(),
                ));
            }
            url.to_file_path()
                .map_err(|_| ImageioError::InvalidSource("invalid file URL path".into()))?
        } else {
            std::path::PathBuf::from(source)
        };
        let canonical = requested
            .canonicalize()
            .map_err(|e| ImageioError::Transport(format!("local image: {e}")))?;
        if !self
            .local_roots
            .iter()
            .any(|root| canonical.starts_with(root))
        {
            return Err(ImageioError::NotPermitted(
                "local image is outside the configured roots".into(),
            ));
        }
        let file = std::fs::File::open(&canonical)
            .map_err(|e| ImageioError::Transport(format!("local image: {e}")))?;
        let metadata = file
            .metadata()
            .map_err(|e| ImageioError::Transport(format!("local image: {e}")))?;
        if !metadata.is_file() || metadata.len() > self.max_encoded_bytes as u64 {
            return Err(ImageioError::TooLarge);
        }
        let mut bytes = Vec::with_capacity(metadata.len() as usize);
        file.take(self.max_encoded_bytes as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| ImageioError::Transport(format!("local image: {e}")))?;
        self.check_encoded_len(bytes.len())?;
        Ok(bytes)
    }

    #[cfg(feature = "http")]
    fn fetch_http(&self, source: &str) -> Result<Vec<u8>, ImageioError> {
        use std::io::Read as _;
        use std::net::ToSocketAddrs as _;

        let url = url::Url::parse(source)
            .map_err(|_| ImageioError::InvalidSource("invalid HTTP URL".into()))?;
        if !matches!(url.scheme(), "http" | "https")
            || !url.username().is_empty()
            || url.password().is_some()
        {
            return Err(ImageioError::InvalidSource(
                "HTTP URL must not contain credentials".into(),
            ));
        }
        let host = url
            .host_str()
            .ok_or_else(|| ImageioError::InvalidSource("HTTP URL has no host".into()))?
            .trim_end_matches('.')
            .to_ascii_lowercase();
        if !self.allowed_hosts.contains(&host) {
            return Err(ImageioError::NotPermitted(
                "HTTP host is not allowlisted".into(),
            ));
        }
        let port = url
            .port_or_known_default()
            .ok_or_else(|| ImageioError::InvalidSource("HTTP URL has no port".into()))?;
        let addrs: Vec<_> = (host.as_str(), port)
            .to_socket_addrs()
            .map_err(|e| ImageioError::Transport(format!("DNS lookup: {e}")))?
            .collect();
        if addrs.is_empty() || addrs.iter().any(|addr| !is_public_ip(addr.ip())) {
            return Err(ImageioError::NotPermitted(
                "HTTP host resolves to a non-public address".into(),
            ));
        }

        let client = reqwest::blocking::Client::builder()
            .connect_timeout(self.connect_timeout)
            .timeout(self.request_timeout)
            .redirect(reqwest::redirect::Policy::none())
            .resolve_to_addrs(&host, &addrs)
            .build()
            .map_err(|e| ImageioError::Transport(e.to_string()))?;
        let response = client
            .get(url)
            .send()
            .map_err(|e| ImageioError::Transport(e.to_string()))?;
        if !response.status().is_success() {
            return Err(ImageioError::Transport(format!(
                "HTTP status {}",
                response.status()
            )));
        }
        if response
            .content_length()
            .is_some_and(|n| n > self.max_encoded_bytes as u64)
        {
            return Err(ImageioError::TooLarge);
        }
        let mut bytes = Vec::new();
        response
            .take(self.max_encoded_bytes as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| ImageioError::Transport(e.to_string()))?;
        self.check_encoded_len(bytes.len())?;
        Ok(bytes)
    }
}

pub fn parse_image_url(source: &str) -> Result<DynamicImage, ImageioError> {
    ImagePolicy::default().load(source)
}

fn starts_with_ignore_ascii_case(value: &str, prefix: &str) -> bool {
    value
        .as_bytes()
        .get(..prefix.len())
        .is_some_and(|head| head.eq_ignore_ascii_case(prefix.as_bytes()))
}

#[cfg(feature = "http")]
fn is_public_ip(ip: std::net::IpAddr) -> bool {
    match ip {
        std::net::IpAddr::V4(ip) => {
            let [a, b, c, _] = ip.octets();
            !(a == 0
                || a == 10
                || a == 127
                || (a == 100 && (64..=127).contains(&b))
                || (a == 169 && b == 254)
                || (a == 172 && (16..=31).contains(&b))
                || (a == 192 && b == 0 && c == 0)
                || (a == 192 && b == 0 && c == 2)
                || (a == 192 && b == 168)
                || (a == 198 && (b == 18 || b == 19))
                || (a == 198 && b == 51 && c == 100)
                || (a == 203 && b == 0 && c == 113)
                || a >= 224)
        }
        std::net::IpAddr::V6(ip) => {
            if let Some(v4) = ip.to_ipv4_mapped() {
                return is_public_ip(v4.into());
            }
            let segments = ip.segments();
            !(ip.is_unspecified()
                || ip.is_loopback()
                || ip.is_multicast()
                || segments[0] & 0xfe00 == 0xfc00
                || segments[0] & 0xffc0 == 0xfe80
                || (segments[0] == 0x2001 && segments[1] == 0x0db8))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::{engine::general_purpose::STANDARD, Engine as _};
    use image::{ImageFormat, Rgb, RgbImage};

    fn tiny_png_bytes(width: u32, height: u32) -> Vec<u8> {
        let image = RgbImage::from_pixel(width, height, Rgb([1, 2, 3]));
        let mut bytes = Vec::new();
        image
            .write_to(&mut Cursor::new(&mut bytes), ImageFormat::Png)
            .expect("encode PNG");
        bytes
    }

    #[test]
    fn data_url_roundtrip() {
        let url = format!(
            "data:image/png;base64,{}",
            STANDARD.encode(tiny_png_bytes(2, 2))
        );
        let image = parse_image_url(&url).expect("decode data URL");
        assert_eq!((image.width(), image.height()), (2, 2));
    }

    #[test]
    fn default_policy_rejects_files_and_http() {
        assert!(matches!(
            parse_image_url("file:///tmp/a.png"),
            Err(ImageioError::NotPermitted(_))
        ));
        assert!(matches!(
            parse_image_url("https://example.com/a.png"),
            Err(ImageioError::NotPermitted(_))
        ));
    }

    #[test]
    fn byte_and_dimension_limits_are_enforced() {
        let bytes = tiny_png_bytes(3, 2);
        let url = format!("data:image/png;base64,{}", STANDARD.encode(&bytes));
        let byte_limited = ImagePolicy::default()
            .with_limits(bytes.len() - 1, 10, 1024)
            .expect("limits");
        assert!(matches!(
            byte_limited.load(&url),
            Err(ImageioError::TooLarge)
        ));

        let dimension_limited = ImagePolicy::default()
            .with_limits(bytes.len(), 2, 1024)
            .expect("limits");
        assert!(matches!(
            dimension_limited.load(&url),
            Err(ImageioError::Decode(_))
        ));
    }

    #[test]
    fn unknown_scheme_is_rejected() {
        assert!(matches!(
            parse_image_url("gopher://example.com/a.png"),
            Err(ImageioError::InvalidSource(_))
        ));
    }

    #[cfg(feature = "http")]
    #[test]
    fn private_networks_are_rejected() {
        for ip in ["127.0.0.1", "10.1.2.3", "169.254.1.2", "::1", "fd00::1"] {
            assert!(!is_public_ip(ip.parse().expect("IP")), "{ip}");
        }
        for ip in ["1.1.1.1", "8.8.8.8", "2606:4700:4700::1111"] {
            assert!(is_public_ip(ip.parse().expect("IP")), "{ip}");
        }
    }
}
