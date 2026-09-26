//! At-rest encryption for the record log: XChaCha20-Poly1305, one random
//! 24-byte nonce per line, so lines can be appended independently.

use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine as _;
use chacha20poly1305::aead::{Aead, AeadCore, KeyInit, OsRng, rand_core::RngCore};
use chacha20poly1305::{XChaCha20Poly1305, XNonce};

use crate::error::Error;

const NONCE_LEN: usize = 24;
/// Marks an encrypted line, so a plaintext line in an encrypted log (or the
/// reverse) is detected rather than misread.
pub const PREFIX: &str = "v1:";

#[derive(Clone)]
pub struct Cipher(XChaCha20Poly1305);

impl Cipher {
    /// `key_hex` is 64 hex characters (32 bytes), e.g. from `chigutiro keygen`.
    pub fn from_hex(key_hex: &str) -> Result<Cipher, Error> {
        let bytes = hex::decode(key_hex.trim()).map_err(|e| Error::Crypto(format!("key is not hex: {e}")))?;
        if bytes.len() != 32 {
            return Err(Error::Crypto(format!("key must be 32 bytes (64 hex chars), got {}", bytes.len())));
        }
        let cipher = XChaCha20Poly1305::new_from_slice(&bytes).map_err(|e| Error::Crypto(e.to_string()))?;
        Ok(Cipher(cipher))
    }

    pub fn seal(&self, plaintext: &[u8]) -> Result<String, Error> {
        let nonce = XChaCha20Poly1305::generate_nonce(&mut OsRng);
        let mut out = nonce.to_vec();
        out.extend(self.0.encrypt(&nonce, plaintext).map_err(|e| Error::Crypto(e.to_string()))?);
        Ok(format!("{PREFIX}{}", B64.encode(out)))
    }

    pub fn open(&self, line: &str) -> Result<Vec<u8>, Error> {
        let body = line
            .strip_prefix(PREFIX)
            .ok_or_else(|| Error::Crypto("line is not encrypted, but a key is configured".into()))?;
        let raw = B64.decode(body).map_err(|e| Error::Crypto(format!("base64: {e}")))?;
        if raw.len() <= NONCE_LEN {
            return Err(Error::Crypto("ciphertext too short".into()));
        }
        let (nonce, ct) = raw.split_at(NONCE_LEN);
        self.0
            .decrypt(XNonce::from_slice(nonce), ct)
            .map_err(|_| Error::Crypto("authentication failed (wrong key or tampered line)".into()))
    }
}

/// `n` random bytes as hex, from the OS generator.
pub fn random_hex(n: usize) -> String {
    let mut buf = vec![0u8; n];
    OsRng.fill_bytes(&mut buf);
    hex::encode(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_detects_tampering() {
        let c = Cipher::from_hex(&random_hex(32)).unwrap();
        let line = c.seal(b"hello").unwrap();
        assert_eq!(c.open(&line).unwrap(), b"hello");
        let other = Cipher::from_hex(&random_hex(32)).unwrap();
        assert!(other.open(&line).is_err());
        assert!(c.open("{\"plain\":true}").is_err());
    }
}
