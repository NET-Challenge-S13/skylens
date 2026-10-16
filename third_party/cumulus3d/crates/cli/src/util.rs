/*
 * util.rs
 *
 * Copyright (c) 2026 ParkSangWoo
 *
 * Author(s):
 *
 *      ParkSangWoo <dev.parksangwoo@gmail.com>
 *
 *
 * This file is part of cumulus3d.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

//! Logging, time measurement and small serialization helpers.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// run.log (+ standard output) recorder. Corresponds to `exec > run.log` in the script.
pub struct Logger {
    file: Option<Mutex<File>>,
    echo: bool,
    tz_offset_s: i64,
}

impl Logger {
    /// Create the log file at `path` (if `None`, no file; if `echo`, also standard output).
    pub fn new(path: Option<&Path>, echo: bool) -> std::io::Result<Self> {
        let file = match path {
            Some(p) => Some(Mutex::new(File::create(p)?)),
            None => None,
        };
        Ok(Self { file, echo, tz_offset_s: local_tz_offset() })
    }

    /// Record one line.
    pub fn line(&self, s: &str) {
        if let Some(f) = &self.file {
            if let Ok(mut f) = f.lock() {
                let _ = writeln!(f, "{s}");
                let _ = f.flush();
            }
        }
        if self.echo {
            println!("{s}");
        }
    }

    /// `[HH:MM:SS] msg` (local time).
    pub fn stamped(&self, s: &str) {
        self.line(&format!("[{}] {s}", self.clock()));
    }

    /// `[HH:MM:SS] msg` (given time, local time).
    pub fn stamped_at(&self, at: SystemTime, s: &str) {
        self.line(&format!("[{}] {s}", self.clock_at(at)));
    }

    /// Current local time `HH:MM:SS`.
    pub fn clock(&self) -> String {
        self.clock_at(SystemTime::now())
    }

    /// `HH:MM:SS` of the given time (local time).
    pub fn clock_at(&self, at: SystemTime) -> String {
        let t = at.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64 + self.tz_offset_s;
        let d = t.rem_euclid(86400);
        format!("{:02}:{:02}:{:02}", d / 3600, (d / 60) % 60, d % 60)
    }
}

/// Local timezone offset (seconds). Read once via `date +%z` without external dependencies; UTC on failure.
fn local_tz_offset() -> i64 {
    let out = std::process::Command::new("date").arg("+%z").output();
    let Ok(out) = out else { return 0 };
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.len() != 5 {
        return 0;
    }
    let sign = if s.starts_with('-') { -1 } else { 1 };
    let h: i64 = s[1..3].parse().unwrap_or(0);
    let m: i64 = s[3..5].parse().unwrap_or(0);
    sign * (h * 3600 + m * 60)
}

/// Event record (timeline.txt): `epoch seconds.nanoseconds event` lines. Also kept in memory as (relative time, string).
pub struct Timeline {
    file: Mutex<File>,
    events: Mutex<Vec<(f64, String)>>,
}

impl Timeline {
    /// Create the timeline file at `path`.
    pub fn new(path: &Path) -> std::io::Result<Self> {
        Ok(Self { file: Mutex::new(File::create(path)?), events: Mutex::new(Vec::new()) })
    }

    /// Event record + `[time] event` in run.log.
    pub fn ev(&self, log: &Logger, s: &str) {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
        let line = format!("{}.{:09} {s}", now.as_secs(), now.subsec_nanos());
        if let Ok(mut f) = self.file.lock() {
            let _ = writeln!(f, "{line}");
            let _ = f.flush();
        }
        if let Ok(mut e) = self.events.lock() {
            e.push((now.as_secs_f64(), s.to_string()));
        }
        log.stamped(s);
    }

    /// (seconds from the first event, string).
    pub fn relative(&self) -> Vec<(f64, String)> {
        let e = self.events.lock().map(|e| e.clone()).unwrap_or_default();
        let t0 = e.first().map(|x| x.0).unwrap_or(0.0);
        e.into_iter().map(|(t, s)| (t - t0, s)).collect()
    }
}

/// Per-stage accumulated time.
#[derive(Default)]
pub struct StageTimes {
    m: Mutex<BTreeMap<String, (Duration, usize)>>,
}

impl StageTimes {
    /// Accumulate stage time.
    pub fn add(&self, stage: &str, d: Duration) {
        if let Ok(mut m) = self.m.lock() {
            let e = m.entry(stage.to_string()).or_default();
            e.0 += d;
            e.1 += 1;
        }
    }

    /// Copy of (stage -> accumulated time, count).
    pub fn snapshot(&self) -> BTreeMap<String, (Duration, usize)> {
        self.m.lock().map(|m| m.clone()).unwrap_or_default()
    }
}

/// Time one stage and record it in run.log.
pub fn timed<T>(log: &Logger, times: &StageTimes, stage: &str, what: &str, f: impl FnOnce() -> T) -> T {
    let t = Instant::now();
    let r = f();
    let d = t.elapsed();
    times.add(stage, d);
    log.stamped(&format!("[time] {stage} {what}: {:.3}s", d.as_secs_f64()));
    r
}

/// Same shape as Python `round(x, n)` followed by `repr` (`.0` for integers).
pub fn py_float(x: f64, digits: i32) -> String {
    let p = 10f64.powi(digits);
    let r = (x * p).round() / p;
    let r = if r == 0.0 { 0.0 } else { r };
    if !r.is_finite() {
        return if r.is_nan() {
            "NaN".into()
        } else if r > 0.0 {
            "Infinity".into()
        } else {
            "-Infinity".into()
        };
    }
    let s = format!("{r}");
    if s.contains('.') || s.contains('e') {
        s
    } else {
        format!("{s}.0")
    }
}

/// Small JSON value (for the manifest).
#[derive(Clone, Debug)]
pub enum Json {
    /// `null`.
    Null,
    /// Boolean.
    Bool(bool),
    /// Integer.
    Int(i64),
    /// An already formatted number (Python round result).
    Num(String),
    /// String.
    Str(String),
    /// Array.
    Arr(Vec<Json>),
    /// Object (key order preserved).
    Obj(Vec<(String, Json)>),
}

impl Json {
    /// Object from a (key, value) list.
    pub fn obj(items: Vec<(&str, Json)>) -> Json {
        Json::Obj(items.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }
    /// Number rounded to `digits` decimal places.
    pub fn f(x: f64, digits: i32) -> Json {
        Json::Num(py_float(x, digits))
    }
    /// Number or null.
    pub fn of(x: Option<f64>, digits: i32) -> Json {
        x.map_or(Json::Null, |v| Json::f(v, digits))
    }
    /// Integer array.
    pub fn ints(v: &[usize]) -> Json {
        Json::Arr(v.iter().map(|x| Json::Int(*x as i64)).collect())
    }

    /// Shape of `json.dump(..., indent=1, ensure_ascii=False)`.
    pub fn dump(&self) -> String {
        let mut s = String::new();
        self.write(&mut s, 0);
        s
    }

    fn write(&self, s: &mut String, ind: usize) {
        match self {
            Json::Null => s.push_str("null"),
            Json::Bool(b) => s.push_str(if *b { "true" } else { "false" }),
            Json::Int(i) => s.push_str(&i.to_string()),
            Json::Num(n) => s.push_str(n),
            Json::Str(t) => push_str_lit(s, t),
            Json::Arr(v) => {
                if v.is_empty() {
                    s.push_str("[]");
                    return;
                }
                s.push_str("[\n");
                for (i, x) in v.iter().enumerate() {
                    s.push_str(&" ".repeat(ind + 1));
                    x.write(s, ind + 1);
                    if i + 1 < v.len() {
                        s.push(',');
                    }
                    s.push('\n');
                }
                s.push_str(&" ".repeat(ind));
                s.push(']');
            }
            Json::Obj(v) => {
                if v.is_empty() {
                    s.push_str("{}");
                    return;
                }
                s.push_str("{\n");
                for (i, (k, x)) in v.iter().enumerate() {
                    s.push_str(&" ".repeat(ind + 1));
                    push_str_lit(s, k);
                    s.push_str(": ");
                    x.write(s, ind + 1);
                    if i + 1 < v.len() {
                        s.push(',');
                    }
                    s.push('\n');
                }
                s.push_str(&" ".repeat(ind));
                s.push('}');
            }
        }
    }

    /// Shape of Python dict/list `repr` (for summary output).
    pub fn py_repr(&self) -> String {
        match self {
            Json::Null => "None".into(),
            Json::Bool(b) => (if *b { "True" } else { "False" }).into(),
            Json::Int(i) => i.to_string(),
            Json::Num(n) => n.clone(),
            Json::Str(t) => format!("'{}'", t.replace('\\', "\\\\").replace('\'', "\\'")),
            Json::Arr(v) => format!("[{}]", v.iter().map(|x| x.py_repr()).collect::<Vec<_>>().join(", ")),
            Json::Obj(v) => {
                format!("{{{}}}", v.iter().map(|(k, x)| format!("'{k}': {}", x.py_repr())).collect::<Vec<_>>().join(", "))
            }
        }
    }

    /// Look up an object's key.
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(v) => v.iter().find(|(k, _)| k == key).map(|(_, x)| x),
            _ => None,
        }
    }

    /// f64 if a number (integer or float).
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Int(i) => Some(*i as f64),
            Json::Num(n) => n.parse().ok(),
            _ => None,
        }
    }

    /// Read JSON text (standard syntax; numbers keep their original text in [`Json::Num`], or [`Json::Int`] if integer-shaped).
    pub fn parse(text: &str) -> Result<Json, String> {
        let mut p = JsonParser { s: text.as_bytes(), i: 0 };
        let v = p.value()?;
        p.ws();
        if p.i != p.s.len() {
            return Err(format!("JSON: {} 바이트에 뒤따르는 내용", p.i));
        }
        Ok(v)
    }
}

struct JsonParser<'a> {
    s: &'a [u8],
    i: usize,
}

impl JsonParser<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }

    fn eat(&mut self, c: u8) -> Result<(), String> {
        self.ws();
        if self.s.get(self.i) == Some(&c) {
            self.i += 1;
            Ok(())
        } else {
            Err(format!("JSON: {} 바이트에서 '{}' 필요", self.i, c as char))
        }
    }

    fn lit(&mut self, w: &str, v: Json) -> Result<Json, String> {
        if self.s[self.i..].starts_with(w.as_bytes()) {
            self.i += w.len();
            Ok(v)
        } else {
            Err(format!("JSON: {} 바이트의 값을 읽을 수 없음", self.i))
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        self.ws();
        match self.s.get(self.i) {
            None => Err("JSON: 내용이 끝남".into()),
            Some(b'{') => {
                self.i += 1;
                let mut items = Vec::new();
                self.ws();
                if self.s.get(self.i) == Some(&b'}') {
                    self.i += 1;
                    return Ok(Json::Obj(items));
                }
                loop {
                    self.ws();
                    let k = self.string()?;
                    self.eat(b':')?;
                    items.push((k, self.value()?));
                    self.ws();
                    match self.s.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b'}') => {
                            self.i += 1;
                            return Ok(Json::Obj(items));
                        }
                        _ => return Err(format!("JSON: {} 바이트에서 ',' 또는 '}}' 필요", self.i)),
                    }
                }
            }
            Some(b'[') => {
                self.i += 1;
                let mut items = Vec::new();
                self.ws();
                if self.s.get(self.i) == Some(&b']') {
                    self.i += 1;
                    return Ok(Json::Arr(items));
                }
                loop {
                    items.push(self.value()?);
                    self.ws();
                    match self.s.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b']') => {
                            self.i += 1;
                            return Ok(Json::Arr(items));
                        }
                        _ => return Err(format!("JSON: {} 바이트에서 ',' 또는 ']' 필요", self.i)),
                    }
                }
            }
            Some(b'"') => self.string().map(Json::Str),
            Some(b't') => self.lit("true", Json::Bool(true)),
            Some(b'f') => self.lit("false", Json::Bool(false)),
            Some(b'n') => self.lit("null", Json::Null),
            Some(_) => {
                let st = self.i;
                while self.i < self.s.len() && matches!(self.s[self.i], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E') {
                    self.i += 1;
                }
                let t = std::str::from_utf8(&self.s[st..self.i]).unwrap_or("");
                if t.is_empty() || t.parse::<f64>().is_err() {
                    return Err(format!("JSON: {st} 바이트의 값을 읽을 수 없음"));
                }
                Ok(t.parse::<i64>().map_or_else(|_| Json::Num(t.to_string()), Json::Int))
            }
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.eat(b'"')?;
        let mut out: Vec<u8> = Vec::new();
        loop {
            let Some(&c) = self.s.get(self.i) else { return Err("JSON: 문자열이 끝나지 않음".into()) };
            self.i += 1;
            match c {
                b'"' => return String::from_utf8(out).map_err(|e| e.to_string()),
                b'\\' => {
                    let Some(&e) = self.s.get(self.i) else { return Err("JSON: 문자열이 끝나지 않음".into()) };
                    self.i += 1;
                    match e {
                        b'n' => out.push(b'\n'),
                        b't' => out.push(b'\t'),
                        b'r' => out.push(b'\r'),
                        b'b' => out.push(8),
                        b'f' => out.push(12),
                        b'u' => {
                            let h = self.s.get(self.i..self.i + 4).and_then(|h| std::str::from_utf8(h).ok());
                            let c = h.and_then(|h| u32::from_str_radix(h, 16).ok()).and_then(char::from_u32).unwrap_or('\u{fffd}');
                            self.i += 4;
                            let mut b = [0u8; 4];
                            out.extend_from_slice(c.encode_utf8(&mut b).as_bytes());
                        }
                        other => out.push(other),
                    }
                }
                c => out.push(c),
            }
        }
    }
}

fn push_str_lit(s: &mut String, t: &str) {
    s.push('"');
    for c in t.chars() {
        match c {
            '"' => s.push_str("\\\""),
            '\\' => s.push_str("\\\\"),
            '\n' => s.push_str("\\n"),
            '\t' => s.push_str("\\t"),
            c if (c as u32) < 0x20 => s.push_str(&format!("\\u{:04x}", c as u32)),
            c => s.push(c),
        }
    }
    s.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn py_float_matches_python_repr() {
        assert_eq!(py_float(1.0, 3), "1.0");
        assert_eq!(py_float(12.345, 1), "12.3");
        assert_eq!(py_float(-0.0001, 2), "0.0");
        assert_eq!(py_float(0.987654, 3), "0.988");
    }

    #[test]
    fn json_parse_reads_nested_values() {
        let j = Json::parse(r#" {"note": "a\"b\u00e9", "cameras": {"1": [1607.5, 1611, 1e2, -0.08], "2": []}, "ok": true, "n": null} "#)
            .unwrap();
        assert!(matches!(j.get("note"), Some(Json::Str(s)) if s == "a\"bé"));
        let Some(Json::Arr(v)) = j.get("cameras").and_then(|c| c.get("1")) else { panic!() };
        let f: Vec<f64> = v.iter().filter_map(Json::as_f64).collect();
        assert_eq!(f, vec![1607.5, 1611.0, 100.0, -0.08]);
        assert!(matches!(j.get("ok"), Some(Json::Bool(true))) && matches!(j.get("n"), Some(Json::Null)));
        assert!(Json::parse("{\"a\": }").is_err() && Json::parse("[1, 2").is_err() && Json::parse("1 2").is_err());
        // Round trip through the writer.
        let d = Json::obj(vec![("a", Json::Int(1)), ("b", Json::Arr(vec![Json::f(0.5, 2), Json::Null]))]);
        assert_eq!(Json::parse(&d.dump()).unwrap().dump(), d.dump());
    }

    #[test]
    fn json_dump_indent1() {
        let j = Json::obj(vec![("a", Json::Int(1)), ("b", Json::Arr(vec![])), ("c", Json::Arr(vec![Json::Null]))]);
        assert_eq!(j.dump(), "{\n \"a\": 1,\n \"b\": [],\n \"c\": [\n  null\n ]\n}");
        assert_eq!(j.py_repr(), "{'a': 1, 'b': [], 'c': [None]}");
    }
}
