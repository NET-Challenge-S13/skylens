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

//! 기록·시간 측정·작은 직렬화 도우미.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// run.log(+ 표준 출력) 기록기. 스크립트의 `exec > run.log` 에 해당.
pub struct Logger {
    file: Option<Mutex<File>>,
    echo: bool,
    tz_offset_s: i64,
}

impl Logger {
    /// `path` 에 기록 파일을 만든다(`None` 이면 파일 없음; `echo` 면 표준 출력에도).
    pub fn new(path: Option<&Path>, echo: bool) -> std::io::Result<Self> {
        let file = match path {
            Some(p) => Some(Mutex::new(File::create(p)?)),
            None => None,
        };
        Ok(Self { file, echo, tz_offset_s: local_tz_offset() })
    }

    /// 한 줄 기록.
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

    /// `[HH:MM:SS] msg` (현지 시각).
    pub fn stamped(&self, s: &str) {
        self.line(&format!("[{}] {s}", self.clock()));
    }

    /// `[HH:MM:SS] msg` (주어진 시각, 현지 시각).
    pub fn stamped_at(&self, at: SystemTime, s: &str) {
        self.line(&format!("[{}] {s}", self.clock_at(at)));
    }

    /// 현재 현지 시각 `HH:MM:SS`.
    pub fn clock(&self) -> String {
        self.clock_at(SystemTime::now())
    }

    /// 주어진 시각의 `HH:MM:SS`(현지 시각).
    pub fn clock_at(&self, at: SystemTime) -> String {
        let t = at.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64 + self.tz_offset_s;
        let d = t.rem_euclid(86400);
        format!("{:02}:{:02}:{:02}", d / 3600, (d / 60) % 60, d % 60)
    }
}

/// 현지 시간대 오프셋(초). 외부 의존성 없이 `date +%z` 로 한 번 읽고, 실패하면 UTC.
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

/// 사건 기록(timeline.txt): `epoch초.나노 사건` 줄. 메모리에도 (상대 시각, 문자열) 로 보관.
pub struct Timeline {
    file: Mutex<File>,
    events: Mutex<Vec<(f64, String)>>,
}

impl Timeline {
    /// `path` 에 timeline 파일을 만든다.
    pub fn new(path: &Path) -> std::io::Result<Self> {
        Ok(Self { file: Mutex::new(File::create(path)?), events: Mutex::new(Vec::new()) })
    }

    /// 사건 기록 + run.log 에 `[시각] 사건`.
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

    /// (첫 사건 기준 초, 문자열).
    pub fn relative(&self) -> Vec<(f64, String)> {
        let e = self.events.lock().map(|e| e.clone()).unwrap_or_default();
        let t0 = e.first().map(|x| x.0).unwrap_or(0.0);
        e.into_iter().map(|(t, s)| (t - t0, s)).collect()
    }
}

/// 단계별 누적 시간.
#[derive(Default)]
pub struct StageTimes {
    m: Mutex<BTreeMap<String, (Duration, usize)>>,
}

impl StageTimes {
    /// 단계 시간 누적.
    pub fn add(&self, stage: &str, d: Duration) {
        if let Ok(mut m) = self.m.lock() {
            let e = m.entry(stage.to_string()).or_default();
            e.0 += d;
            e.1 += 1;
        }
    }

    /// (단계 → 누적 시간, 횟수) 사본.
    pub fn snapshot(&self) -> BTreeMap<String, (Duration, usize)> {
        self.m.lock().map(|m| m.clone()).unwrap_or_default()
    }
}

/// 단계 하나를 재고 run.log 에 남긴다.
pub fn timed<T>(log: &Logger, times: &StageTimes, stage: &str, what: &str, f: impl FnOnce() -> T) -> T {
    let t = Instant::now();
    let r = f();
    let d = t.elapsed();
    times.add(stage, d);
    log.stamped(&format!("[time] {stage} {what}: {:.3}s", d.as_secs_f64()));
    r
}

/// 파이썬 `round(x, n)` 뒤 `repr` 과 같은 모양(정수면 `.0`).
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

/// 작은 JSON 값(매니페스트용).
#[derive(Clone, Debug)]
pub enum Json {
    /// `null`.
    Null,
    /// 불린.
    Bool(bool),
    /// 정수.
    Int(i64),
    /// 이미 서식화한 수(파이썬 round 결과).
    Num(String),
    /// 문자열.
    Str(String),
    /// 배열.
    Arr(Vec<Json>),
    /// 객체(키 순서 유지).
    Obj(Vec<(String, Json)>),
}

impl Json {
    /// (키, 값) 목록으로 객체.
    pub fn obj(items: Vec<(&str, Json)>) -> Json {
        Json::Obj(items.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }
    /// 소수 `digits` 자리 반올림 수.
    pub fn f(x: f64, digits: i32) -> Json {
        Json::Num(py_float(x, digits))
    }
    /// 수 또는 null.
    pub fn of(x: Option<f64>, digits: i32) -> Json {
        x.map_or(Json::Null, |v| Json::f(v, digits))
    }
    /// 정수 배열.
    pub fn ints(v: &[usize]) -> Json {
        Json::Arr(v.iter().map(|x| Json::Int(*x as i64)).collect())
    }

    /// `json.dump(..., indent=1, ensure_ascii=False)` 모양.
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

    /// 파이썬 dict/list `repr` 모양(요약 출력용).
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

    /// 객체의 키 조회.
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(v) => v.iter().find(|(k, _)| k == key).map(|(_, x)| x),
            _ => None,
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
    fn json_dump_indent1() {
        let j = Json::obj(vec![("a", Json::Int(1)), ("b", Json::Arr(vec![])), ("c", Json::Arr(vec![Json::Null]))]);
        assert_eq!(j.dump(), "{\n \"a\": 1,\n \"b\": [],\n \"c\": [\n  null\n ]\n}");
        assert_eq!(j.py_repr(), "{'a': 1, 'b': [], 'c': [None]}");
    }
}
