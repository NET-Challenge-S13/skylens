<#
  Start the SkyLens DJI mission bridge with DJI cloud delivery enabled.

  Cloud mission list/delete and cloud delivery need two secrets. Neither is
  stored in this script; both stay on the local machine and out of git:

    - SKYLENS_DJI_WK_KEY : the HMAC signing key for X-Wk-SecretId=uav.
      Taken from the environment if already set, otherwise read from
      config/.dji_wk_key (gitignored, one line, the key only).

    - the DJI Fly account token: read by the bridge from output/.dji_token
      (gitignored) through SKYLENS_DJI_TOKEN_FILE, re-read on every request so
      a refresher can update it without a restart.

  The token EXPIRES. When the cloud list starts failing with an auth error,
  refresh output/.dji_token: run `python scripts/dji_token_refresh.py` with WSA
  (or any rooted Android running a logged-in DJI Fly), or paste a fresh token in.

  Without either secret the bridge still starts, in download-only mode.

  Usage:  powershell -ExecutionPolicy Bypass -File scripts\run-mission-bridge.ps1
#>

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

if (-not $env:SKYLENS_DJI_WK_KEY) {
  $keyFile = Join-Path $repo 'config\.dji_wk_key'
  if (Test-Path $keyFile) {
    $env:SKYLENS_DJI_WK_KEY = (Get-Content $keyFile -Raw).Trim()
  } else {
    Write-Warning "no signing key: set SKYLENS_DJI_WK_KEY or write it to $keyFile. Cloud delivery stays disabled."
  }
}

if (-not $env:SKYLENS_DJI_TOKEN_FILE) {
  $env:SKYLENS_DJI_TOKEN_FILE = Join-Path $repo 'output\.dji_token'
}
if (-not (Test-Path $env:SKYLENS_DJI_TOKEN_FILE)) {
  Write-Warning "no token at $($env:SKYLENS_DJI_TOKEN_FILE). Cloud delivery stays disabled until one is written."
}

python src/skylens_mission_bridge/server.py
