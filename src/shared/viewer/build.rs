//! Fetches the bridge's board certificate at build time, so a native `wss` link can trust it
//! (`link::PINNED_CERTS`). The bridge makes its own self-signed certificate (setup-bridge.sh),
//! which no root store knows, and it may make a new one on a reinstall: so nothing is checked in,
//! the build asks the bridge.
//!
//! | env | default | meaning |
//! |---|---|---|
//! | `SKYLENS_BRIDGE_TLS`  | `175.126.98.44:8443` | `host:port` whose certificate is pinned; `off` pins nothing |
//! | `SKYLENS_BRIDGE_CERT` | unset | PEM file to pin instead of asking the bridge (offline builds) |
//!
//! A bridge that does not answer is a warning, not an error: the build then pins nothing and the
//! screens reach the bridge only if it has a certificate the web roots accept. The script reruns
//! only when this file or those variables change; after the bridge makes a new certificate,
//! `cargo clean -p skylens-viewer` (or touching this file) fetches it again.

use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::crypto::{CryptoProvider, verify_tls12_signature, verify_tls13_signature};
use rustls::pki_types::pem::PemObject as _;
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::{ClientConfig, ClientConnection, DigitallySignedStruct, Error, SignatureScheme};
use std::net::{TcpStream, ToSocketAddrs as _};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

const DEFAULT_BRIDGE: &str = "175.126.98.44:8443";
const TIMEOUT: Duration = Duration::from_secs(5);

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=SKYLENS_BRIDGE_TLS");
    println!("cargo:rerun-if-env-changed=SKYLENS_BRIDGE_CERT");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    // The web build talks through the browser, which has its own trust.
    let certs = if std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("wasm32") { Vec::new() } else { obtain() };

    let mut list = String::from("&[");
    for (i, der) in certs.iter().enumerate() {
        let path = out_dir.join(format!("pinned-{i}.der"));
        std::fs::write(&path, der).expect("write pinned certificate");
        list.push_str(&format!("include_bytes!({:?}),", path.display().to_string()));
    }
    list.push(']');
    std::fs::write(out_dir.join("pinned_certs.rs"), list).expect("write pinned_certs.rs");
}

/// DER certificates to pin, from `SKYLENS_BRIDGE_CERT` or from the bridge itself.
fn obtain() -> Vec<Vec<u8>> {
    if let Ok(path) = std::env::var("SKYLENS_BRIDGE_CERT") {
        println!("cargo:rerun-if-changed={path}");
        return match CertificateDer::from_pem_file(&path) {
            Ok(der) => vec![der.to_vec()],
            Err(e) => panic!("SKYLENS_BRIDGE_CERT={path}: {e}"),
        };
    }
    let addr = std::env::var("SKYLENS_BRIDGE_TLS").unwrap_or_else(|_| DEFAULT_BRIDGE.to_string());
    let addr = addr.trim();
    if addr.is_empty() || addr.eq_ignore_ascii_case("off") {
        return Vec::new();
    }
    match fetch(addr) {
        Ok(der) => vec![der],
        Err(e) => {
            println!("cargo:warning=bridge certificate not pinned ({addr}: {e}); wss to the bridge needs a publicly trusted certificate");
            Vec::new()
        }
    }
}

/// The certificate `addr` presents, read by completing a TLS handshake without judging it.
fn fetch(addr: &str) -> Result<Vec<u8>, String> {
    let socket = addr.to_socket_addrs().map_err(|e| e.to_string())?.next().ok_or("no address")?;
    let mut tcp = TcpStream::connect_timeout(&socket, TIMEOUT).map_err(|e| e.to_string())?;
    tcp.set_read_timeout(Some(TIMEOUT)).map_err(|e| e.to_string())?;
    tcp.set_write_timeout(Some(TIMEOUT)).map_err(|e| e.to_string())?;

    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let seen = Arc::new(Mutex::new(None));
    let config = ClientConfig::builder_with_provider(provider.clone())
        .with_safe_default_protocol_versions()
        .map_err(|e| e.to_string())?
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(Record { seen: seen.clone(), provider }))
        .with_no_client_auth();
    let host = addr.rsplit_once(':').map_or(addr, |(h, _)| h).trim_matches(['[', ']']);
    let name = ServerName::try_from(host.to_string()).map_err(|e| e.to_string())?;
    let mut conn = ClientConnection::new(Arc::new(config), name).map_err(|e| e.to_string())?;
    while conn.is_handshaking() {
        conn.complete_io(&mut tcp).map_err(|e| e.to_string())?;
    }
    seen.lock().map_err(|e| e.to_string())?.take().ok_or_else(|| "no certificate".to_string())
}

/// Accepts any certificate and keeps the server's own (the end entity).
#[derive(Debug)]
struct Record {
    seen: Arc<Mutex<Option<Vec<u8>>>>,
    provider: Arc<CryptoProvider>,
}

impl ServerCertVerifier for Record {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, Error> {
        if let Ok(mut seen) = self.seen.lock() {
            *seen = Some(end_entity.to_vec());
        }
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, Error> {
        verify_tls12_signature(message, cert, dss, &self.provider.signature_verification_algorithms)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, Error> {
        verify_tls13_signature(message, cert, dss, &self.provider.signature_verification_algorithms)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.provider.signature_verification_algorithms.supported_schemes()
    }
}
