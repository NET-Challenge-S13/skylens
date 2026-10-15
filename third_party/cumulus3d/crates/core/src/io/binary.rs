/*
 * binary.rs
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

//! 리틀 엔디언 이진 읽기/쓰기 도우미.

use crate::error::{Error, Result};
use std::io::{Read, Write};

pub(crate) trait WriteLe: Write {
    fn put_u8(&mut self, v: u8) -> Result<()> {
        Ok(self.write_all(&[v])?)
    }
    fn put_u32(&mut self, v: u32) -> Result<()> {
        Ok(self.write_all(&v.to_le_bytes())?)
    }
    fn put_i32(&mut self, v: i32) -> Result<()> {
        Ok(self.write_all(&v.to_le_bytes())?)
    }
    fn put_u64(&mut self, v: u64) -> Result<()> {
        Ok(self.write_all(&v.to_le_bytes())?)
    }
    fn put_f32(&mut self, v: f32) -> Result<()> {
        Ok(self.write_all(&v.to_le_bytes())?)
    }
    fn put_f64(&mut self, v: f64) -> Result<()> {
        Ok(self.write_all(&v.to_le_bytes())?)
    }
    fn put_f64s(&mut self, v: &[f64]) -> Result<()> {
        for x in v {
            self.put_f64(*x)?;
        }
        Ok(())
    }
}
impl<W: Write + ?Sized> WriteLe for W {}

pub(crate) trait ReadLe: Read {
    fn get_bytes<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut b = [0u8; N];
        self.read_exact(&mut b).map_err(|e| {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                Error::Format("파일이 예상보다 일찍 끝남".into())
            } else {
                Error::Io(e)
            }
        })?;
        Ok(b)
    }
    fn get_u8(&mut self) -> Result<u8> {
        Ok(self.get_bytes::<1>()?[0])
    }
    fn get_u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.get_bytes()?))
    }
    fn get_i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.get_bytes()?))
    }
    fn get_u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.get_bytes()?))
    }
    fn get_f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.get_bytes()?))
    }
    fn get_f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.get_bytes()?))
    }
    fn get_f64_array<const N: usize>(&mut self) -> Result<[f64; N]> {
        let mut a = [0.0; N];
        for x in a.iter_mut() {
            *x = self.get_f64()?;
        }
        Ok(a)
    }
    /// 0 바이트로 끝나는 문자열.
    fn get_cstring(&mut self) -> Result<String> {
        let mut v = Vec::new();
        loop {
            let b = self.get_u8()?;
            if b == 0 {
                break;
            }
            v.push(b);
        }
        String::from_utf8(v).map_err(|_| Error::Format("UTF-8 이 아닌 이름".into()))
    }
    /// 길이 접두(u64) 바이트열.
    fn get_vec_u8(&mut self, n: usize) -> Result<Vec<u8>> {
        let mut v = vec![0u8; n];
        self.read_exact(&mut v)?;
        Ok(v)
    }
}
impl<R: Read + ?Sized> ReadLe for R {}

/// 손상 파일에서 거대한 할당을 막기 위한 상한 검사.
pub(crate) fn check_count(n: u64, what: &str) -> Result<usize> {
    if n > (1u64 << 40) {
        return Err(Error::Format(format!("{what} 개수가 비정상적으로 큼: {n}")));
    }
    Ok(n as usize)
}
