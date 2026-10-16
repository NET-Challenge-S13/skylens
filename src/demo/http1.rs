//! A deliberately small HTTP/1.0 client over a raw TCP socket: health probes
//! and the board server's `/live/` (WHEP signalling) proxy. Requests say
//! HTTP/1.0 with `Connection: close`, so the response body runs to EOF and
//! servers do not chunk it (chunked bodies are still decoded, to be safe).

use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

#[derive(Debug, Clone, PartialEq)]
pub struct Response {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

#[cfg(test)]
impl Response {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }
}

/// Headers a proxy must not pass along (RFC 9110 hop-by-hop, plus the ones
/// this client sets itself).
pub fn is_hop_by_hop(name: &str) -> bool {
    [
        "connection", "keep-alive", "proxy-connection", "transfer-encoding", "te", "trailer",
        "upgrade", "host", "content-length",
    ]
    .iter()
    .any(|h| name.eq_ignore_ascii_case(h))
}

/// The request bytes. `headers` must not contain hop-by-hop headers.
pub fn encode_request(method: &str, host: &str, path: &str, headers: &[(String, String)], body: &[u8]) -> Vec<u8> {
    let mut out = format!("{method} {path} HTTP/1.0\r\nHost: {host}\r\nConnection: close\r\n");
    for (k, v) in headers {
        out.push_str(&format!("{k}: {v}\r\n"));
    }
    if !body.is_empty() || matches!(method, "POST" | "PUT" | "PATCH") {
        out.push_str(&format!("Content-Length: {}\r\n", body.len()));
    }
    out.push_str("\r\n");
    let mut bytes = out.into_bytes();
    bytes.extend_from_slice(body);
    bytes
}

/// Parse a complete response (everything up to EOF).
pub fn parse_response(raw: &[u8]) -> Result<Response, String> {
    let split = raw.windows(4).position(|w| w == b"\r\n\r\n").ok_or("no end of headers")?;
    let head = std::str::from_utf8(&raw[..split]).map_err(|_| "headers are not UTF-8")?;
    let mut lines = head.split("\r\n");
    let status_line = lines.next().ok_or("empty response")?;
    let mut parts = status_line.splitn(3, ' ');
    let version = parts.next().unwrap_or("");
    if !version.starts_with("HTTP/") {
        return Err(format!("not an HTTP response: {status_line}"));
    }
    let status: u16 = parts.next().and_then(|s| s.parse().ok()).ok_or_else(|| format!("bad status line: {status_line}"))?;
    let mut headers = Vec::new();
    for line in lines {
        if let Some((k, v)) = line.split_once(':') {
            headers.push((k.trim().to_string(), v.trim().to_string()));
        }
    }
    let mut body = raw[split + 4..].to_vec();
    let chunked = headers
        .iter()
        .any(|(k, v)| k.eq_ignore_ascii_case("transfer-encoding") && v.to_ascii_lowercase().contains("chunked"));
    if chunked {
        body = dechunk(&body)?;
        headers.retain(|(k, _)| !k.eq_ignore_ascii_case("transfer-encoding"));
    } else if let Some(len) = headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
        .and_then(|(_, v)| v.parse::<usize>().ok())
    {
        body.truncate(len);
    }
    Ok(Response { status, headers, body })
}

/// Decode a chunked transfer-coded body.
pub fn dechunk(mut data: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    loop {
        let eol = data.windows(2).position(|w| w == b"\r\n").ok_or("truncated chunk size")?;
        let size_line = std::str::from_utf8(&data[..eol]).map_err(|_| "bad chunk size")?;
        let size_hex = size_line.split(';').next().unwrap_or("").trim();
        let size = usize::from_str_radix(size_hex, 16).map_err(|_| format!("bad chunk size: {size_hex}"))?;
        data = &data[eol + 2..];
        if size == 0 {
            return Ok(out);
        }
        if data.len() < size {
            return Err("truncated chunk".into());
        }
        out.extend_from_slice(&data[..size]);
        data = &data[size..];
        if data.starts_with(b"\r\n") {
            data = &data[2..];
        }
    }
}

/// Send one request to `addr` (`host:port`) and read the whole response.
pub async fn request(
    addr: &str,
    method: &str,
    path: &str,
    headers: &[(String, String)],
    body: &[u8],
    timeout: Duration,
) -> std::io::Result<Response> {
    let work = async {
        let mut stream = TcpStream::connect(addr).await?;
        stream.write_all(&encode_request(method, addr, path, headers, body)).await?;
        let mut raw = Vec::new();
        stream.read_to_end(&mut raw).await?;
        parse_response(&raw).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    };
    tokio::time::timeout(timeout, work)
        .await
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::TimedOut, format!("{method} {addr}{path} timed out")))?
}

/// `GET path` answers 2xx.
pub async fn ok(addr: &str, path: &str) -> bool {
    matches!(request(addr, "GET", path, &[], &[], Duration::from_millis(1500)).await, Ok(r) if (200..300).contains(&r.status))
}

/// Something accepts TCP connections on `addr`.
pub async fn listening(addr: &str) -> bool {
    matches!(tokio::time::timeout(Duration::from_millis(500), TcpStream::connect(addr)).await, Ok(Ok(_)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_bytes() {
        let req = encode_request("POST", "127.0.0.1:10889", "/drone1/whep", &[("Content-Type".into(), "application/sdp".into())], b"v=0");
        let text = String::from_utf8(req).unwrap();
        assert!(text.starts_with("POST /drone1/whep HTTP/1.0\r\nHost: 127.0.0.1:10889\r\nConnection: close\r\n"));
        assert!(text.contains("Content-Type: application/sdp\r\n"));
        assert!(text.ends_with("Content-Length: 3\r\n\r\nv=0"));
        let get = String::from_utf8(encode_request("GET", "h", "/health", &[], b"")).unwrap();
        assert!(!get.contains("Content-Length"));
    }

    #[test]
    fn parses_plain_response() {
        let raw = b"HTTP/1.1 201 Created\r\nLocation: /drone1/whep/abc\r\nContent-Length: 4\r\n\r\nv=0\nEXTRA";
        let r = parse_response(raw).unwrap();
        assert_eq!(r.status, 201);
        assert_eq!(r.header("location"), Some("/drone1/whep/abc"));
        assert_eq!(r.body, b"v=0\n");
    }

    #[test]
    fn parses_chunked_response() {
        let raw = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n2;x=y\r\nde\r\n0\r\n\r\n";
        let r = parse_response(raw).unwrap();
        assert_eq!(r.body, b"abcde");
        assert_eq!(r.header("transfer-encoding"), None);
    }

    #[test]
    fn rejects_garbage() {
        assert!(parse_response(b"hello").is_err());
        assert!(parse_response(b"SSH-2.0\r\n\r\n").is_err());
        assert!(dechunk(b"zz\r\n").is_err());
        assert!(dechunk(b"5\r\nab").is_err());
    }

    #[test]
    fn hop_by_hop_headers() {
        for h in ["Connection", "upgrade", "Host", "Content-Length", "Transfer-Encoding"] {
            assert!(is_hop_by_hop(h), "{h}");
        }
        for h in ["Content-Type", "Authorization", "If-Match", "Location"] {
            assert!(!is_hop_by_hop(h), "{h}");
        }
    }
}
