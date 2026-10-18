//! Lito X1 Mission Bridge: SkyLens GPS route -> DJI Fly KMZ missions.
//!
//! Port of `legacy/mission_bridge/server.py`. Serves the same localhost HTTP
//! surface (`/health`, `/missions`, `/formation`, `/cloud/*` on :8091) and,
//! when `SKYLENS_CORE_UPLINK` is set, also connects to the core's `ws /uplink`.

mod cloud;
mod envfile;
mod formation;
mod kmz;
mod mission;
mod paths;
mod server;
mod uplink;

use std::sync::Arc;

use anyhow::Context as _;
use tracing_subscriber::EnvFilter;

use crate::cloud::{CloudClient, CloudConfig};
use crate::envfile::EnvFile;
use crate::server::{App, Settings};
use crate::uplink::UplinkConfig;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .init();

    let root = paths::repo_root();
    let process_env = |k: &str| std::env::var(k).ok();
    // Settings file at the repository root (`.env.mission`, like the core's `.env.vworld`).
    // A non-empty process variable still wins over the file.
    let file_path = EnvFile::location(&root, process_env);
    let file = EnvFile::load(&file_path)?;
    match &file.path {
        Some(p) => tracing::info!("[mission-bridge] settings file: {} ({} entries)", p.display(), file.len()),
        None => tracing::info!(
            "[mission-bridge] settings file: {} not found (environment and defaults only; see .env.mission.example)",
            file_path.display()
        ),
    }
    let env = file.lookup(process_env);
    let settings = Settings::from_lookup(&root, &env)?;
    // Also falls back to config/.dji_wk_key and output/.dji_token under the repository root.
    let cloud = CloudConfig::from_lookup_with_launcher_defaults(&root, &env).map(CloudClient::new);
    let uplink = UplinkConfig::from_lookup(&env)?;

    std::fs::create_dir_all(&settings.output)
        .with_context(|| format!("creating {}", settings.output.display()))?;
    let addr = format!("{}:{}", settings.host, settings.port);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("binding {addr}"))?;

    tracing::info!("[mission-bridge] listening on http://{addr}");
    tracing::info!(
        "[mission-bridge] Lito template: {} ({})",
        settings.template.display(),
        if settings.template.is_file() { "ready" } else { "MISSING" }
    );
    tracing::info!("[mission-bridge] output: {}", settings.output.display());
    match &settings.install_target {
        Some(target) => tracing::info!("[mission-bridge] install target: {target}"),
        None => tracing::info!("[mission-bridge] install target: not configured (download-only mode)"),
    }
    if cloud.is_some() {
        tracing::info!("[mission-bridge] DJI cloud: configured (deliver:'cloud' enabled)");
    } else {
        tracing::info!(
            "[mission-bridge] DJI cloud: not configured (set SKYLENS_DJI_MC_TOKEN + SKYLENS_DJI_WK_KEY)"
        );
    }

    let app = Arc::new(App::new(settings, cloud));
    match uplink {
        Some(cfg) => {
            tracing::info!(
                "[mission-bridge] core uplink: {} station(s), formation spacing {} m",
                cfg.stations.len(),
                cfg.spacing_m
            );
            tokio::spawn(uplink::run(app.clone(), cfg));
        }
        None => tracing::info!("[mission-bridge] core uplink: off (set SKYLENS_CORE_UPLINK)"),
    }

    tokio::select! {
        result = server::serve(app, listener) => result,
        _ = tokio::signal::ctrl_c() => Ok(()),
    }
}
