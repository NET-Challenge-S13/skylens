/*
 * fmt.rs
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

//! printf 호환 숫자 서식(로캘 무관).

/// C `printf("%.{precision}g")` 와 같은 결과. 모델 텍스트 형식은 precision = 17.
pub fn format_g(v: f64, precision: usize) -> String {
    if v.is_nan() {
        return if v.is_sign_negative() { "-nan".into() } else { "nan".into() };
    }
    if v.is_infinite() {
        return if v < 0.0 { "-inf".into() } else { "inf".into() };
    }
    let p = precision.max(1);
    if v == 0.0 {
        return if v.is_sign_negative() { "-0".into() } else { "0".into() };
    }
    // %e 변환으로 지수 X 를 정한다(반올림 후 지수).
    let e_str = format!("{:.*e}", p - 1, v);
    let (mant, exp) = e_str.split_once('e').expect("e 표기");
    let x: i32 = exp.parse().expect("지수");
    if x < -4 || x >= p as i32 {
        let mant = strip_zeros(mant);
        let sign = if x < 0 { '-' } else { '+' };
        format!("{mant}e{sign}{:02}", x.abs())
    } else {
        let prec = (p as i32 - 1 - x) as usize;
        strip_zeros(&format!("{:.*}", prec, v)).to_string()
    }
}

fn strip_zeros(s: &str) -> &str {
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.')
    } else {
        s
    }
}

/// `%.17g`.
pub fn g17(v: f64) -> String {
    format_g(v, 17)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn g_format() {
        assert_eq!(g17(1.0), "1");
        assert_eq!(g17(0.1), "0.10000000000000001");
        assert_eq!(g17(-2.5), "-2.5");
        assert_eq!(g17(0.0), "0");
        assert_eq!(g17(1e-5), "1.0000000000000001e-05");
        assert_eq!(g17(1e20), "1e+20");
        assert_eq!(g17(123456789.0), "123456789");
        assert_eq!(g17(1e16), "10000000000000000");
        assert_eq!(g17(1e17), "1e+17");
        assert_eq!(g17(0.0001), "0.0001");
        assert_eq!(format_g(1234.56789, 6), "1234.57");
        assert_eq!(format_g(1e-300, 6), "1e-300");
        for v in [0.1, 1.0 / 3.0, 6378137.0, -1.2345e-200, 9.99999999999999e22, f64::MIN_POSITIVE, f64::MAX] {
            assert_eq!(g17(v).parse::<f64>().unwrap(), v);
        }
    }
}
