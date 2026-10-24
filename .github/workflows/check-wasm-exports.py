"""Checks a wasm-bindgen bundle for the table mix-up that stops a screen from starting.

Usage: check-wasm-exports.py <file_bg.wasm> [...]

wasm-bindgen's glue grows `__wbindgen_externrefs` by four at startup. The module
has two tables, a funcref one that is fixed at its initial size and a growable
externref one, and the export has to point at the second. wasm-opt renumbers
tables and, without --enable-reference-types, has pointed the export at the
funcref table instead. The page then dies on its first line:

    RangeError: WebAssembly.Table.grow(): failed to grow table by 4

It is invisible in the build log and in the file size, and it reaches the
screens as a blank page, so it is checked here instead.
"""

import sys

FUNCREF, EXTERNREF = 0x70, 0x6F
NAMES = {FUNCREF: "funcref", EXTERNREF: "externref"}


def leb(b, i):
    r = s = 0
    while True:
        x = b[i]
        i += 1
        r |= (x & 0x7F) << s
        if not x & 0x80:
            return r, i
        s += 7


def name(b, i):
    n, i = leb(b, i)
    return b[i : i + n].decode("utf8", "replace"), i + n


def read(path):
    """Tables in index order (imported first), and the exported tables by name."""
    b = open(path, "rb").read()
    if b[:4] != b"\x00asm":
        raise SystemExit(f"{path}: not a wasm module")
    i, tables, exports = 8, [], {}
    while i < len(b):
        sid = b[i]
        i += 1
        size, i = leb(b, i)
        end = i + size
        if sid == 2:  # imports
            n, j = leb(b, i)
            for _ in range(n):
                _, j = name(b, j)
                _, j = name(b, j)
                kind = b[j]
                j += 1
                if kind == 0:
                    _, j = leb(b, j)
                elif kind == 1:
                    et = b[j]
                    fl = b[j + 1]
                    j += 2
                    mn, j = leb(b, j)
                    mx = None
                    if fl & 1:
                        mx, j = leb(b, j)
                    tables.append((et, mn, mx))
                elif kind == 2:
                    fl = b[j]
                    j += 1
                    _, j = leb(b, j)
                    if fl & 1:
                        _, j = leb(b, j)
                elif kind == 3:
                    j += 2
        elif sid == 4:  # tables
            n, j = leb(b, i)
            for _ in range(n):
                et = b[j]
                fl = b[j + 1]
                j += 2
                mn, j = leb(b, j)
                mx = None
                if fl & 1:
                    mx, j = leb(b, j)
                tables.append((et, mn, mx))
        elif sid == 7:  # exports
            n, j = leb(b, i)
            for _ in range(n):
                nm, j = name(b, j)
                kind = b[j]
                j += 1
                idx, j = leb(b, j)
                if kind == 1:
                    exports[nm] = idx
        i = end
    return tables, exports


def check(path):
    tables, exports = read(path)
    idx = exports.get("__wbindgen_externrefs")
    if idx is None:
        return [f"{path}: no __wbindgen_externrefs export; is this a wasm-bindgen bundle?"]
    if idx >= len(tables):
        return [f"{path}: __wbindgen_externrefs points at table {idx}, which does not exist"]
    et, mn, mx = tables[idx]
    if et != EXTERNREF:
        return [
            f"{path}: __wbindgen_externrefs points at table {idx}, a "
            f"{NAMES.get(et, hex(et))} table (min {mn}, max {mx}), not the externref one. "
            "The screen will die on startup in Table.grow. Check the wasm-opt step."
        ]
    if mx is not None and mx <= mn:
        return [f"{path}: the externref table cannot grow (min {mn}, max {mx})"]
    print(f"  ok  {path}: __wbindgen_externrefs -> table {idx} ({NAMES[et]}, min {mn}, growable)")
    return []


problems = [p for path in sys.argv[1:] for p in check(path)]
if problems:
    print("\n".join(problems), file=sys.stderr)
    raise SystemExit(1)
