#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Publish every workspace crate to crates.io in dependency order; safe to re-run.
# Needs CARGO_REGISTRY_TOKEN in the environment (or a logged-in cargo).
set -uo pipefail
cd "$(dirname "$0")/.."

out=$(mktemp)
trap 'rm -f "$out"' EXIT

if cargo publish --workspace --locked 2>&1 | tee "$out"; [ "${PIPESTATUS[0]}" -eq 0 ]; then
  exit 0
fi
if ! grep -q "already exists" "$out"; then
  echo "publish failed" >&2
  exit 1
fi

# Part of the workspace is already published (an interrupted earlier run): go crate by crate, skipping those.
echo "some crates already exist; publishing the rest one by one"
order=$(cargo metadata --no-deps --format-version 1 | python3 -I -c '
import json, sys
m = json.load(sys.stdin)
names = {p["name"] for p in m["packages"]}
deps = {p["name"]: {d["name"] for d in p["dependencies"] if d["name"] in names} for p in m["packages"]}
done = []
while len(done) < len(names):
    for n in sorted(names - set(done)):
        if deps[n] <= set(done):
            done.append(n)
            break
    else:
        sys.exit("dependency cycle")
print("\n".join(done))
')
for crate in $order; do
  if cargo publish -p "$crate" --locked 2>&1 | tee "$out"; [ "${PIPESTATUS[0]}" -eq 0 ]; then
    continue
  fi
  if grep -q "already exists" "$out"; then
    echo "$crate: already published, skipped"
  else
    echo "$crate: publish failed" >&2
    exit 1
  fi
done
