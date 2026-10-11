//! Replay-server client: SSE events and chunk deliveries over plain HTTP/1.1 (std only). Shared by
//! the native viewer and the video server (P3.2); not built for wasm (the browser uses fetch).
//!
//! Two threads, so a delivery in progress never delays the timestamp of the next event:
//! - the event thread reads `/events` and stamps each snapshot the moment its line arrives;
//! - the fetch thread takes snapshots in order and streams `/chunks/<seq>`, forwarding every chunk
//!   as soon as its bytes are complete (decision 0026) — or, with a local data directory (decision
//!   0050), reads the snapshot's PLY there and converts it in this process, without HTTP.

use crate::chunk::{FLAG_LAST_IN_GENERATION, HEADER_LEN, MAGIC, decode_header};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct Snapshot {
    pub seq: u32,
    pub kind: String,
    pub points: usize,
}

/// Messages to the window thread.
#[derive(Debug)]
pub enum Msg {
    Received {
        snap: Snapshot,
        at: Instant,
    },
    Chunk {
        seq: u32,
        bytes: Vec<u8>,
        last: bool,
        /// When the chunk's last byte arrived.
        at: Instant,
    },
    Delivered {
        seq: u32,
        delivery: String,
        generation: u32,
        bytes: u64,
        fetch_start: Instant,
        /// Response headers received: the server has read the snapshot and starts converting.
        headers_at: Instant,
    },
    End,
    Error(String),
}

/// `http://127.0.0.1:8700` → `127.0.0.1:8700`.
pub fn host_port(url: &str) -> Result<String, String> {
    let rest = url
        .strip_prefix("http://")
        .ok_or("server URL must start with http://")?;
    Ok(rest.trim_end_matches('/').to_string())
}

type Headers = Vec<(String, String)>;

/// Sends a GET and returns the response headers (lower-case names) and the body reader.
pub fn get(host: &str, path: &str) -> std::io::Result<(Headers, BufReader<TcpStream>)> {
    let mut stream = TcpStream::connect(host)?;
    stream.set_nodelay(true)?;
    write!(
        stream,
        "GET {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n"
    )?;
    stream.flush()?;
    let mut r = BufReader::with_capacity(1 << 20, stream);
    let mut line = String::new();
    r.read_line(&mut line)?;
    if line.split_whitespace().nth(1) != Some("200") {
        return Err(std::io::Error::other(format!(
            "GET {path}: {}",
            line.trim_end()
        )));
    }
    let mut headers = Vec::new();
    loop {
        line.clear();
        if r.read_line(&mut line)? <= 2 {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            headers.push((k.trim().to_ascii_lowercase(), v.trim().to_string()));
        }
    }
    Ok((headers, r))
}

fn header<'a>(h: &'a [(String, String)], name: &str) -> Option<&'a str> {
    h.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
}

pub fn parse_snapshot(json: &str) -> Option<Snapshot> {
    let v: serde_json::Value = serde_json::from_str(json).ok()?;
    Some(Snapshot {
        seq: u32::try_from(v["seq"].as_u64()?).ok()?,
        kind: v["kind"].as_str()?.to_string(),
        points: usize::try_from(v["points"].as_u64()?).ok()?,
    })
}

/// Reads one chunk (header, then body) from a delivery stream; None at a clean end of stream.
pub fn read_chunk(r: &mut impl Read) -> std::io::Result<Option<Vec<u8>>> {
    let mut buf = vec![0u8; HEADER_LEN];
    let mut got = 0;
    while got < HEADER_LEN {
        match r.read(&mut buf[got..])? {
            0 if got == 0 => return Ok(None),
            0 => return Err(std::io::ErrorKind::UnexpectedEof.into()),
            n => got += n,
        }
    }
    // point_count sits at bytes 16..20 and the point stride at 72..74 (decisions 0022, 0051); the
    // whole chunk is validated once read.
    if buf[..4] != MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "not a chunk",
        ));
    }
    let n = u32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]) as usize;
    let stride = usize::from(u16::from_le_bytes([buf[72], buf[73]]));
    if !crate::chunk::STRIDES.contains(&stride) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("point stride {stride}"),
        ));
    }
    buf.resize(HEADER_LEN + n * stride, 0);
    r.read_exact(&mut buf[HEADER_LEN..])?;
    decode_header(&buf).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    Ok(Some(buf))
}

/// Streams one snapshot's chunks to `send`. Returns (delivery kind, generation, bytes, when the
/// response headers arrived).
fn fetch(
    host: &str,
    seq: u32,
    have: Option<(u32, usize)>,
    send: &impl Fn(Msg),
) -> std::io::Result<(String, u32, u64, Instant)> {
    let path = match have {
        Some((g, n)) => format!("/chunks/{seq}?have={g}.{n}"),
        None => format!("/chunks/{seq}"),
    };
    let (headers, mut r) = get(host, &path)?;
    let headers_at = Instant::now();
    let delivery = header(&headers, "x-pb-delivery").unwrap_or("?").to_string();
    let generation = header(&headers, "x-pb-generation")
        .and_then(|g| g.parse().ok())
        .unwrap_or(0);
    let mut bytes = 0u64;
    while let Some(c) = read_chunk(&mut r)? {
        bytes += c.len() as u64;
        let last = decode_header(&c).map(|h| h.flags & FLAG_LAST_IN_GENERATION != 0) == Ok(true);
        send(Msg::Chunk {
            seq,
            bytes: c,
            last,
            at: Instant::now(),
        });
        if last {
            break;
        }
    }
    Ok((delivery, generation, bytes, headers_at))
}

/// Takes snapshots in arrival order and fetches each one; `have` tracks what the client holds.
/// Where the fetch thread gets a snapshot's chunks from.
enum Source {
    /// `/chunks/<seq>` of the replay server.
    Http,
    /// The PLY in a local directory, converted here (decision 0050).
    Local(Local),
}

/// What converting locally needs from the replay server's manifest.
struct Local {
    dir: PathBuf,
    steps: Vec<crate::convert::Step>,
    files: Vec<String>,
    appends: Vec<bool>,
}

impl Local {
    fn load(host: &str, dir: PathBuf) -> std::io::Result<Self> {
        let (_, mut r) = get(host, "/manifest.json")?;
        let mut text = String::new();
        r.read_to_string(&mut text)?;
        let bad =
            |what: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, what.to_string());
        let v: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let events = v["events"]
            .as_array()
            .ok_or_else(|| bad("manifest without events"))?;
        let mut local = Self {
            dir,
            steps: Vec::new(),
            files: Vec::new(),
            appends: Vec::new(),
        };
        for e in events {
            let snap = parse_snapshot(&e.to_string()).ok_or_else(|| bad("bad manifest event"))?;
            local.steps.push(crate::convert::Step {
                seq: snap.seq,
                preview: snap.kind == "preview",
                points: snap.points,
            });
            local.files.push(
                e["file"]
                    .as_str()
                    .ok_or_else(|| bad("event without file"))?
                    .to_string(),
            );
            // Older replay servers do not say; then every preview is delivered whole.
            local.appends.push(e["appends"].as_bool().unwrap_or(false));
        }
        Ok(local)
    }

    /// Converts the delivery for `seq` (same plan as the replay server) and forwards its chunks as
    /// each is encoded. Returns (delivery, generation, bytes, conversion started).
    fn fetch(
        &self,
        seq: u32,
        have: Option<(u32, usize)>,
        send: &impl Fn(Msg),
    ) -> std::io::Result<(String, u32, u64, Instant)> {
        use crate::convert::{Conversion, Plan, plan};
        let idx = self
            .steps
            .iter()
            .position(|s| s.seq == seq)
            .ok_or_else(|| std::io::Error::other(format!("snapshot {seq} not in the manifest")))?;
        let (generation, skip, delivery) = match plan(&self.steps, idx, have, |k| self.appends[k]) {
            Plan::Full { generation } => (generation, 0, "full"),
            Plan::Delta { generation, skip } => (generation, skip, "delta"),
        };
        let started = Instant::now();
        let c = Conversion::start(self.dir.join(&self.files[idx]), skip, generation);
        let n = c.len().map_err(std::io::Error::other)?;
        let mut bytes = 0u64;
        for i in 0..n {
            let chunk = c.take(i).map_err(std::io::Error::other)?;
            bytes += chunk.len() as u64;
            send(Msg::Chunk {
                seq,
                bytes: chunk,
                last: i + 1 == n,
                at: Instant::now(),
            });
        }
        Ok((delivery.to_string(), generation, bytes, started))
    }
}

fn fetch_loop(host: &str, source: &Source, rx: mpsc::Receiver<Snapshot>, send: &impl Fn(Msg)) {
    let mut have: Option<(u32, usize)> = None;
    for snap in rx {
        let fetch_start = Instant::now();
        let got = match source {
            Source::Http => fetch(host, snap.seq, have, send),
            Source::Local(local) => local.fetch(snap.seq, have, send),
        };
        match got {
            Ok((delivery, generation, bytes, headers_at)) => {
                have = Some((generation, snap.points));
                send(Msg::Delivered {
                    seq: snap.seq,
                    delivery,
                    generation,
                    bytes,
                    fetch_start,
                    headers_at,
                });
            }
            Err(e) => {
                send(Msg::Error(format!("snapshot {}: {e}", snap.seq)));
                have = None; // next delivery is full
            }
        }
    }
    send(Msg::End);
}

/// Replay: follows `/events?speed=…`.
pub fn spawn_replay(host: String, speed: f64, send: impl Fn(Msg) + Send + Sync + 'static) {
    follow(host, None, speed, send);
}

/// Like [`spawn_replay`], but snapshots are read from `dir` (the replay server's data directory on
/// this machine) and converted in this process instead of fetched over HTTP (decision 0050).
pub fn spawn_replay_local(
    host: String,
    dir: PathBuf,
    speed: f64,
    send: impl Fn(Msg) + Send + Sync + 'static,
) {
    follow(host, Some(dir), speed, send);
}

fn source(host: &str, dir: Option<PathBuf>) -> std::io::Result<Source> {
    Ok(match dir {
        Some(dir) => Source::Local(Local::load(host, dir)?),
        None => Source::Http,
    })
}

fn follow(
    host: String,
    dir: Option<PathBuf>,
    speed: f64,
    send: impl Fn(Msg) + Send + Sync + 'static,
) {
    let send = std::sync::Arc::new(send);
    let src = match source(&host, dir) {
        Ok(src) => src,
        Err(e) => {
            send(Msg::Error(format!("manifest: {e}")));
            send(Msg::End);
            return;
        }
    };
    let (tx, rx) = mpsc::channel::<Snapshot>();
    let (h2, s2) = (host.clone(), send.clone());
    std::thread::spawn(move || fetch_loop(&h2, &src, rx, &|m| s2(m)));
    std::thread::spawn(move || {
        let run = || -> std::io::Result<()> {
            let (_, r) = get(&host, &format!("/events?speed={speed}"))?;
            let mut event = String::new();
            for line in r.lines() {
                let line = line?;
                let at = Instant::now();
                if let Some(e) = line.strip_prefix("event: ") {
                    event = e.to_string();
                } else if let Some(data) = line.strip_prefix("data: ") {
                    match event.as_str() {
                        "snapshot" => {
                            if let Some(snap) = parse_snapshot(data) {
                                send(Msg::Received {
                                    snap: snap.clone(),
                                    at,
                                });
                                let _ = tx.send(snap);
                            }
                        }
                        "end" => return Ok(()),
                        _ => {}
                    }
                }
            }
            Ok(())
        };
        if let Err(e) = run() {
            send(Msg::Error(format!("events: {e}")));
        }
        // Dropping tx ends the fetch loop, which sends End after the last delivery.
    });
}

/// Cold start: only the last snapshot of the manifest, as one full delivery.
pub fn spawn_cold(host: String, send: impl Fn(Msg) + Send + Sync + 'static) {
    cold(host, None, send);
}

/// Like [`spawn_cold`], reading the snapshot from `dir` (decision 0050).
pub fn spawn_cold_local(host: String, dir: PathBuf, send: impl Fn(Msg) + Send + Sync + 'static) {
    cold(host, Some(dir), send);
}

fn cold(host: String, dir: Option<PathBuf>, send: impl Fn(Msg) + Send + Sync + 'static) {
    std::thread::spawn(move || {
        let run = || -> std::io::Result<()> {
            let src = source(&host, dir.clone())?;
            let (_, mut r) = get(&host, "/manifest.json")?;
            let mut text = String::new();
            r.read_to_string(&mut text)?;
            let v: serde_json::Value = serde_json::from_str(&text)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            let last = v["events"]
                .as_array()
                .and_then(|a| a.last())
                .and_then(|e| parse_snapshot(&e.to_string()))
                .ok_or_else(|| std::io::Error::other("empty manifest"))?;
            send(Msg::Received {
                snap: last.clone(),
                at: Instant::now(),
            });
            let (tx, rx) = mpsc::channel();
            tx.send(last).ok();
            drop(tx);
            fetch_loop(&host, &src, rx, &send);
            Ok(())
        };
        if let Err(e) = run() {
            send(Msg::Error(format!("cold: {e}")));
            send(Msg::End);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Point;
    use crate::chunk::encode;

    #[test]
    fn reads_back_to_back_chunks_until_the_end() {
        let p = |x: f32| Point {
            position: [x, 0.0, 0.0],
            color: [1, 2, 3],
        };
        let a = encode(4, 0, 0, &[p(1.0), p(2.0)]);
        let b = encode(4, 1, FLAG_LAST_IN_GENERATION, &[]);
        let stream: Vec<u8> = [a.clone(), b.clone()].concat();
        let mut r = &stream[..];
        assert_eq!(read_chunk(&mut r).unwrap(), Some(a));
        assert_eq!(read_chunk(&mut r).unwrap(), Some(b));
        assert_eq!(read_chunk(&mut r).unwrap(), None);
        // Cut inside a chunk: an error, not a silent end.
        let mut cut = &stream[..90];
        assert!(read_chunk(&mut cut).is_err());
    }

    #[test]
    fn parses_server_urls_and_snapshots() {
        assert_eq!(
            host_port("http://127.0.0.1:8700/").unwrap(),
            "127.0.0.1:8700"
        );
        assert!(host_port("https://x").is_err());
        let s = parse_snapshot(
            r#"{"seq":3,"time_s":298.9,"kind":"preview","region":1,"file":"f","url":"/data/f","bytes":1,"points":371585}"#,
        )
        .unwrap();
        assert_eq!((s.seq, s.kind.as_str(), s.points), (3, "preview", 371_585));
    }
}
