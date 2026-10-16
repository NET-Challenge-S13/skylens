/*
 * pairs.rs
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
 * Third-party notices: parts of the algorithms, default parameters and data
 * formats in this file follow other open-source projects. Their copyright
 * notices and licenses are reproduced in THIRD_PARTY_NOTICES.md.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

//! Pair list file (two image names per line).

use cumulus3d_core::{pair_id_of, Error, ImageId, Result};
use std::collections::HashSet;
use std::path::Path;

/// Pair list parsing result.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PairList {
    /// File order; duplicate unordered pairs keep only the first-seen direction; self pairs excluded.
    pub pairs: Vec<(ImageId, ImageId)>,
    /// Names not in the store (that line is skipped).
    pub missing_names: Vec<String>,
}

/// Text parsing. Trims whitespace around lines, ignores blank lines and `#` lines; the separator is a single space character (' ').
/// Before the first space = name1, up to the next space = name2, the rest ignored. A tab is not a separator.
pub fn parse_pair_list(text: &str, lookup: impl Fn(&str) -> Option<ImageId>) -> PairList {
    let mut out = PairList::default();
    let mut seen = HashSet::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split(' ');
        let n1 = it.next().unwrap_or("");
        let n2 = it.next().unwrap_or("");
        let (Some(id1), Some(id2)) = (lookup(n1), lookup(n2)) else {
            for n in [n1, n2] {
                if lookup(n).is_none() {
                    out.missing_names.push(n.to_string());
                }
            }
            continue;
        };
        if id1 == id2 {
            continue;
        }
        let Ok(pid) = pair_id_of(id1, id2) else { continue };
        if seen.insert(pid) {
            out.pairs.push((id1, id2));
        }
    }
    out
}

/// File reading + parsing.
pub fn read_pair_list(path: impl AsRef<Path>, lookup: impl Fn(&str) -> Option<ImageId>) -> Result<PairList> {
    let text = std::fs::read_to_string(path.as_ref()).map_err(Error::Io)?;
    Ok(parse_pair_list(&text, lookup))
}
