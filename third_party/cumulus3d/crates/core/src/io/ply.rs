/*
 * ply.rs
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

//! PLY 점군 읽기·쓰기.

use crate::error::{Error, Result};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;

/// 점군. `normals`/`colors` 는 비어 있거나 `positions` 와 길이가 같다.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PointCloud {
    /// 점 위치.
    pub positions: Vec<[f32; 3]>,
    /// 법선(비어 있을 수 있음).
    pub normals: Vec<[f32; 3]>,
    /// RGB 색(비어 있을 수 있음).
    pub colors: Vec<[u8; 3]>,
}

impl PointCloud {
    /// 점 수.
    pub fn len(&self) -> usize {
        self.positions.len()
    }
    /// 점이 없는지.
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }
    /// 법선이 있는지.
    pub fn has_normals(&self) -> bool {
        !self.normals.is_empty()
    }
    /// 색이 있는지.
    pub fn has_colors(&self) -> bool {
        !self.colors.is_empty()
    }
}

/// 쓰기 속성 배치.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PlyLayout {
    /// 조밀 점군(dense.ply) 배치: x y z red green blue nx ny nz, 타입 이름 float32/uint8 (27바이트/점).
    #[default]
    XyzRgbNormal,
    /// 자체 출력: x y z nx ny nz red green blue, 타입 이름 float/uchar (27바이트/점).
    XyzNormalRgb,
}

/// 이진 리틀 엔디언 PLY 쓰기. 법선이 없으면 nx ny nz 생략, 색이 없으면 흰색(255).
pub fn write_ply(path: impl AsRef<Path>, cloud: &PointCloud, layout: PlyLayout) -> Result<()> {
    let n = cloud.len();
    if (cloud.has_normals() && cloud.normals.len() != n) || (cloud.has_colors() && cloud.colors.len() != n) {
        return Err(Error::InvalidArgument("PLY: 법선/색 길이가 점 수와 다름".into()));
    }
    let (ft, ut) = match layout {
        PlyLayout::XyzRgbNormal => ("float32", "uint8"),
        PlyLayout::XyzNormalRgb => ("float", "uchar"),
    };
    let mut w = BufWriter::new(std::fs::File::create(path)?);
    let mut h = String::new();
    h += "ply\nformat binary_little_endian 1.0\n";
    h += &format!("element vertex {n}\n");
    let xyz = format!("property {ft} x\nproperty {ft} y\nproperty {ft} z\n");
    let nrm = format!("property {ft} nx\nproperty {ft} ny\nproperty {ft} nz\n");
    let rgb = format!("property {ut} red\nproperty {ut} green\nproperty {ut} blue\n");
    h += &xyz;
    match layout {
        PlyLayout::XyzRgbNormal => {
            h += &rgb;
            if cloud.has_normals() {
                h += &nrm;
            }
        }
        PlyLayout::XyzNormalRgb => {
            if cloud.has_normals() {
                h += &nrm;
            }
            h += &rgb;
        }
    }
    h += "end_header\n";
    w.write_all(h.as_bytes())?;
    let rec = 12 + 3 + if cloud.has_normals() { 12 } else { 0 };
    let mut buf = Vec::with_capacity(rec * n.min(1 << 20));
    for i in 0..n {
        for v in cloud.positions[i] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        let c = if cloud.has_colors() { cloud.colors[i] } else { [255, 255, 255] };
        let put_n = |buf: &mut Vec<u8>| {
            if cloud.has_normals() {
                for v in cloud.normals[i] {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
            }
        };
        match layout {
            PlyLayout::XyzRgbNormal => {
                buf.extend_from_slice(&c);
                put_n(&mut buf);
            }
            PlyLayout::XyzNormalRgb => {
                put_n(&mut buf);
                buf.extend_from_slice(&c);
            }
        }
        if buf.len() >= rec << 20 {
            w.write_all(&buf)?;
            buf.clear();
        }
    }
    w.write_all(&buf)?;
    w.flush()?;
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Scalar {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl Scalar {
    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "char" | "int8" => Self::I8,
            "uchar" | "uint8" => Self::U8,
            "short" | "int16" => Self::I16,
            "ushort" | "uint16" => Self::U16,
            "int" | "int32" => Self::I32,
            "uint" | "uint32" => Self::U32,
            "float" | "float32" => Self::F32,
            "double" | "float64" => Self::F64,
            _ => return None,
        })
    }
    fn size(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }
    fn read(self, b: &[u8], le: bool) -> f64 {
        macro_rules! rd {
            ($t:ty, $n:expr) => {{
                let a: [u8; $n] = b[..$n].try_into().expect("크기");
                (if le { <$t>::from_le_bytes(a) } else { <$t>::from_be_bytes(a) }) as f64
            }};
        }
        match self {
            Self::I8 => b[0] as i8 as f64,
            Self::U8 => b[0] as f64,
            Self::I16 => rd!(i16, 2),
            Self::U16 => rd!(u16, 2),
            Self::I32 => rd!(i32, 4),
            Self::U32 => rd!(u32, 4),
            Self::F32 => rd!(f32, 4),
            Self::F64 => rd!(f64, 8),
        }
    }
}

#[derive(Debug)]
struct Element {
    name: String,
    count: usize,
    props: Vec<(String, Scalar)>,
    has_list: bool,
}

/// PLY 읽기(ascii / binary_little_endian / binary_big_endian). vertex 의 속성을 이름으로 찾는다:
/// x y z 필수, nx ny nz 와 red green blue 는 있으면 읽는다.
pub fn read_ply(path: impl AsRef<Path>) -> Result<PointCloud> {
    let path = path.as_ref();
    let mut r = BufReader::new(std::fs::File::open(path)?);
    let mut line = String::new();
    let read_line = |r: &mut BufReader<std::fs::File>, line: &mut String| -> Result<()> {
        line.clear();
        if r.read_line(line)? == 0 {
            return Err(Error::Format("PLY 헤더가 끝나지 않음".into()));
        }
        Ok(())
    };
    read_line(&mut r, &mut line)?;
    if line.trim() != "ply" {
        return Err(Error::Format("PLY 표지 없음".into()));
    }
    let mut format = None;
    let mut elements: Vec<Element> = Vec::new();
    loop {
        read_line(&mut r, &mut line)?;
        let t: Vec<&str> = line.split_whitespace().collect();
        match t.first().copied() {
            Some("format") => format = t.get(1).map(|s| s.to_string()),
            Some("element") => {
                let count = t.get(2).and_then(|s| s.parse().ok()).ok_or_else(|| Error::Format("element 개수".into()))?;
                elements.push(Element { name: t.get(1).unwrap_or(&"").to_string(), count, props: vec![], has_list: false });
            }
            Some("property") => {
                let el = elements.last_mut().ok_or_else(|| Error::Format("element 없는 property".into()))?;
                if t.get(1) == Some(&"list") {
                    el.has_list = true;
                } else {
                    let ty = t.get(1).and_then(|s| Scalar::parse(s)).ok_or_else(|| Error::Format(format!("PLY 타입: {}", line.trim())))?;
                    el.props.push((t.get(2).unwrap_or(&"").to_string(), ty));
                }
            }
            Some("end_header") => break,
            _ => {}
        }
    }
    let format = format.ok_or_else(|| Error::Format("PLY format 줄 없음".into()))?;
    let mut cloud = PointCloud::default();
    let mut skip_bytes = 0usize;
    let mut vertex = None;
    for el in &elements {
        if el.name == "vertex" {
            vertex = Some(el);
            break;
        }
        if (el.has_list || format == "ascii") && el.count > 0 {
            return Err(Error::Unsupported("vertex 앞의 list/ascii element".into()));
        }
        skip_bytes += el.count * el.props.iter().map(|p| p.1.size()).sum::<usize>();
    }
    let v = vertex.ok_or_else(|| Error::Format("vertex element 없음".into()))?;
    if v.has_list {
        return Err(Error::Unsupported("vertex 의 list 속성".into()));
    }
    let idx = |n: &str| v.props.iter().position(|p| p.0 == n);
    let pos = [idx("x"), idx("y"), idx("z")];
    if pos.iter().any(|p| p.is_none()) {
        return Err(Error::Format("x y z 속성 없음".into()));
    }
    let pos = pos.map(|p| p.expect("검사됨"));
    let nrm = [idx("nx"), idx("ny"), idx("nz")];
    let has_n = nrm.iter().all(|p| p.is_some());
    let col = [idx("red").or(idx("r")), idx("green").or(idx("g")), idx("blue").or(idx("b"))];
    let has_c = col.iter().all(|p| p.is_some());
    let mut vals = vec![0.0f64; v.props.len()];
    let push = |vals: &[f64], cloud: &mut PointCloud| {
        cloud.positions.push(pos.map(|i| vals[i] as f32));
        if has_n {
            cloud.normals.push(nrm.map(|i| vals[i.expect("검사됨")] as f32));
        }
        if has_c {
            cloud.colors.push(col.map(|i| vals[i.expect("검사됨")].clamp(0.0, 255.0) as u8));
        }
    };
    cloud.positions.reserve(v.count);
    match format.as_str() {
        "ascii" => {
            let mut text = String::new();
            r.read_to_string(&mut text)?;
            let mut toks = text.split_whitespace();
            for _ in 0..v.count {
                for x in vals.iter_mut() {
                    *x = toks.next().and_then(|s| s.parse().ok()).ok_or_else(|| Error::Format("PLY ascii 값 부족".into()))?;
                }
                push(&vals, &mut cloud);
            }
        }
        "binary_little_endian" | "binary_big_endian" => {
            let le = format == "binary_little_endian";
            std::io::copy(&mut (&mut r).take(skip_bytes as u64), &mut std::io::sink())?;
            let stride: usize = v.props.iter().map(|p| p.1.size()).sum();
            let mut offs = Vec::with_capacity(v.props.len());
            let mut o = 0;
            for p in &v.props {
                offs.push(o);
                o += p.1.size();
            }
            let mut rec = vec![0u8; stride];
            for _ in 0..v.count {
                r.read_exact(&mut rec).map_err(|_| Error::Format("PLY 자료가 짧음".into()))?;
                for (k, p) in v.props.iter().enumerate() {
                    vals[k] = p.1.read(&rec[offs[k]..], le);
                }
                push(&vals, &mut cloud);
            }
        }
        f => return Err(Error::Unsupported(format!("PLY 형식 {f}"))),
    }
    Ok(cloud)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(n: usize, normals: bool) -> PointCloud {
        PointCloud {
            positions: (0..n).map(|i| [i as f32 * 0.5, -(i as f32), 1e-3 * i as f32]).collect(),
            normals: if normals { (0..n).map(|i| [0.0, (i % 2) as f32, 1.0]).collect() } else { vec![] },
            colors: (0..n).map(|i| [(i % 256) as u8, 7, 255 - (i % 256) as u8]).collect(),
        }
    }

    #[test]
    fn ply_roundtrip_both_layouts() {
        let dir = std::env::temp_dir().join(format!("cumulus3d_ply_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for layout in [PlyLayout::XyzRgbNormal, PlyLayout::XyzNormalRgb] {
            for normals in [true, false] {
                let c = sample(1000, normals);
                let p = dir.join(format!("{layout:?}{normals}.ply"));
                write_ply(&p, &c, layout).unwrap();
                let bytes = std::fs::read(&p).unwrap();
                let hdr_end = bytes.windows(11).position(|w| w == b"end_header\n").unwrap() + 11;
                assert_eq!(bytes.len() - hdr_end, 1000 * if normals { 27 } else { 15 });
                let back = read_ply(&p).unwrap();
                assert_eq!(back, c);
            }
        }
        let p = dir.join("hdr.ply");
        write_ply(&p, &sample(2, true), PlyLayout::XyzRgbNormal).unwrap();
        let bytes = std::fs::read(&p).unwrap();
        let s = String::from_utf8_lossy(&bytes[..bytes.windows(11).position(|w| w == b"end_header\n").unwrap() + 11]).to_string();
        assert_eq!(
            s,
            "ply\nformat binary_little_endian 1.0\nelement vertex 2\nproperty float32 x\nproperty float32 y\nproperty float32 z\nproperty uint8 red\nproperty uint8 green\nproperty uint8 blue\nproperty float32 nx\nproperty float32 ny\nproperty float32 nz\nend_header\n"
        );
        // ascii with extra props and no color
        let p = dir.join("a.ply");
        std::fs::write(&p, "ply\nformat ascii 1.0\nelement vertex 2\nproperty double z\nproperty float x\nproperty float y\nproperty int extra\nend_header\n1 2 3 9\n4 5 6 9\n").unwrap();
        let c = read_ply(&p).unwrap();
        assert_eq!(c.positions, vec![[2.0, 3.0, 1.0], [5.0, 6.0, 4.0]]);
        assert!(!c.has_colors());
        std::fs::remove_dir_all(&dir).ok();
    }
}
