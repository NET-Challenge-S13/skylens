//! Rust twin of `utils/serving_registry.py`: in-memory job registry with a
//! single background worker.
//!
//! Deliberately has NO database (COMPONENTS.md §7): jobs disappear when the
//! process exits and the core re-issues anything it still cares about.
//!
//! Jobs run STRICTLY ONE AT A TIME. Both real workloads want the whole GPU, so
//! overlapping them would only trade throughput for latency. That also makes
//! `queued` in `JobAccepted` an honest number: how many jobs are ahead of yours.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use super::serving_pipeline::PipelineUnavailable;
use super::serving_schemas::{JobResult, JobState, JobStatus};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobKind {
    Recon,
    Detect,
}

impl JobKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Recon => "recon",
            Self::Detect => "detect",
        }
    }
}

/// Progress sink handed to a runner; values are clamped to 0..=1.
pub type Report = Arc<dyn Fn(f64) + Send + Sync>;
pub type RunnerFuture = Pin<Box<dyn Future<Output = anyhow::Result<JobResult>> + Send>>;
/// A runner gets a progress sink and returns the finished result.
pub type Runner = Box<dyn FnOnce(Report) -> RunnerFuture + Send>;

fn now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

#[derive(Debug, Clone)]
pub struct Job {
    pub id: String,
    pub kind: JobKind,
    pub state: JobState,
    pub progress: f64,
    pub result: Option<JobResult>,
    pub error: Option<String>,
    pub created_at: f64,
    pub started_at: Option<f64>,
    pub finished_at: Option<f64>,
}

impl Job {
    fn new(id: String, kind: JobKind) -> Self {
        Job {
            id,
            kind,
            state: JobState::Queued,
            progress: 0.0,
            result: None,
            error: None,
            created_at: now(),
            started_at: None,
            finished_at: None,
        }
    }

    pub fn status(&self) -> JobStatus {
        JobStatus {
            job_id: self.id.clone(),
            state: self.state,
            progress: (self.progress * 10_000.0).round() / 10_000.0,
            result: self.result.clone(),
            error: self.error.clone(),
        }
    }
}

struct Inner {
    jobs: HashMap<String, Job>,
    order: Vec<String>,
    runners: HashMap<String, Runner>,
}

/// Accepts jobs, runs them one at a time, remembers the recent ones.
pub struct JobRegistry {
    inner: Arc<Mutex<Inner>>,
    tx: mpsc::UnboundedSender<String>,
    rx: Mutex<Option<mpsc::UnboundedReceiver<String>>>,
    worker: Mutex<Option<JoinHandle<()>>>,
    history: usize,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    // A panic while holding the lock leaves plain data behind; keep serving.
    m.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// `"<Type>: <message>"`, the shape `serving_registry.py` stores in `error`.
pub fn describe_error(err: &anyhow::Error) -> String {
    if err.downcast_ref::<PipelineUnavailable>().is_some() {
        format!("{err:#}")
    } else {
        format!("Error: {err:#}")
    }
}

impl Default for JobRegistry {
    fn default() -> Self {
        Self::new(512)
    }
}

impl JobRegistry {
    pub fn new(history: usize) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        JobRegistry {
            inner: Arc::new(Mutex::new(Inner {
                jobs: HashMap::new(),
                order: Vec::new(),
                runners: HashMap::new(),
            })),
            tx,
            rx: Mutex::new(Some(rx)),
            worker: Mutex::new(None),
            history,
        }
    }

    // -- lifecycle ---------------------------------------------------------

    /// Spawn the worker on the current tokio runtime. Idempotent.
    pub fn start(&self) {
        let mut worker = lock(&self.worker);
        if worker.is_some() {
            return;
        }
        let Some(rx) = lock(&self.rx).take() else {
            return; // stopped before; a registry runs one worker in its life
        };
        let inner = self.inner.clone();
        *worker = Some(tokio::spawn(run_forever(inner, rx)));
    }

    /// Stop the worker. A job that was running is marked failed ("cancelled").
    pub fn stop(&self) {
        let Some(worker) = lock(&self.worker).take() else {
            return;
        };
        worker.abort();
        let mut inner = lock(&self.inner);
        for job in inner.jobs.values_mut() {
            if job.state == JobState::Running {
                job.state = JobState::Failed;
                job.error = Some("cancelled".into());
                job.finished_at = Some(now());
            }
        }
    }

    // -- api ---------------------------------------------------------------

    /// Register a job and hand back its id and queue position at accept time.
    pub fn submit(&self, kind: JobKind, runner: Runner) -> (String, u64) {
        let mut inner = lock(&self.inner);
        let ahead = inner
            .jobs
            .values()
            .filter(|j| matches!(j.state, JobState::Queued | JobState::Running))
            .count() as u64;
        let hex = uuid::Uuid::new_v4().simple().to_string();
        let id = format!("{}-{}", kind.as_str(), &hex[..12]);
        inner.jobs.insert(id.clone(), Job::new(id.clone(), kind));
        inner.order.push(id.clone());
        inner.runners.insert(id.clone(), runner);
        // The receiver lives as long as the registry unless stop() ran; a job
        // submitted after that simply stays queued, like the Python queue.
        let _ = self.tx.send(id.clone());
        evict(&mut inner, self.history);
        drop(inner);
        tracing::info!("job {id} accepted ({}), {ahead} ahead", kind.as_str());
        (id, ahead)
    }

    pub fn get(&self, job_id: &str) -> Option<Job> {
        lock(&self.inner).jobs.get(job_id).cloned()
    }

    pub fn status(&self, job_id: &str) -> Option<JobStatus> {
        lock(&self.inner).jobs.get(job_id).map(Job::status)
    }

    /// Jobs per state, always with all four keys.
    pub fn counts(&self) -> std::collections::BTreeMap<String, u64> {
        let mut out: std::collections::BTreeMap<String, u64> =
            JobState::ALL.iter().map(|s| (s.as_str().to_string(), 0)).collect();
        for job in lock(&self.inner).jobs.values() {
            *out.entry(job.state.as_str().to_string()).or_default() += 1;
        }
        out
    }
}

// -- worker ----------------------------------------------------------------

async fn run_forever(inner: Arc<Mutex<Inner>>, mut rx: mpsc::UnboundedReceiver<String>) {
    while let Some(job_id) = rx.recv().await {
        let runner = {
            let mut guard = lock(&inner);
            let runner = guard.runners.remove(&job_id);
            if !guard.jobs.contains_key(&job_id) {
                continue; // evicted before it ever ran
            }
            match runner {
                Some(r) => r,
                None => continue,
            }
        };
        run_one(&inner, job_id, runner).await;
    }
}

async fn run_one(inner: &Arc<Mutex<Inner>>, job_id: String, runner: Runner) {
    let started = now();
    {
        let mut guard = lock(inner);
        if let Some(job) = guard.jobs.get_mut(&job_id) {
            job.state = JobState::Running;
            job.started_at = Some(started);
        }
    }

    let sink = inner.clone();
    let id = job_id.clone();
    let report: Report = Arc::new(move |value: f64| {
        // min(1, max(0, v)) in Python maps NaN to 0.
        let v = if value.is_nan() { 0.0 } else { value.clamp(0.0, 1.0) };
        if let Some(job) = lock(&sink).jobs.get_mut(&id) {
            job.progress = v;
        }
    });

    // Run on its own task so a panic fails this job instead of the worker.
    let outcome = tokio::spawn(runner(report)).await;

    let mut guard = lock(inner);
    let Some(job) = guard.jobs.get_mut(&job_id) else {
        return;
    };
    match outcome {
        Ok(Ok(result)) => {
            job.result = Some(result);
            job.progress = 1.0;
            job.state = JobState::Done;
            tracing::info!("job {job_id} done in {:.2}s", now() - started);
        }
        Ok(Err(err)) => {
            job.state = JobState::Failed;
            job.error = Some(describe_error(&err));
            tracing::error!("job {job_id} failed: {err:#}");
        }
        Err(join) => {
            job.state = JobState::Failed;
            job.error = Some(if join.is_cancelled() {
                "cancelled".to_string()
            } else {
                format!("panic: {}", panic_message(join))
            });
            tracing::error!("job {job_id} failed: {:?}", job.error);
        }
    }
    job.finished_at = Some(now());
}

fn panic_message(join: tokio::task::JoinError) -> String {
    match join.try_into_panic() {
        Ok(payload) => {
            if let Some(s) = payload.downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "unknown panic".to_string()
            }
        }
        Err(other) => other.to_string(),
    }
}

// -- housekeeping ----------------------------------------------------------

/// Drop the oldest FINISHED jobs once the table grows past `history`.
fn evict(inner: &mut Inner, history: usize) {
    while inner.order.len() > history {
        let victim = inner.order.iter().position(|id| match inner.jobs.get(id) {
            None => true,
            Some(job) => matches!(job.state, JobState::Done | JobState::Failed),
        });
        let Some(i) = victim else {
            return; // everything still pending; let the table grow
        };
        let id = inner.order.remove(i);
        inner.jobs.remove(&id);
        inner.runners.remove(&id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::serving_schemas::{DetectJobResult, DetectResultKind};
    use std::time::Duration;

    fn detect_result(segment: i64) -> JobResult {
        JobResult::Detect(DetectJobResult {
            kind: DetectResultKind::Tag,
            segment,
            detections: vec![],
        })
    }

    fn ok_runner(segment: i64, delay_ms: u64) -> Runner {
        Box::new(move |report: Report| -> RunnerFuture {
            Box::pin(async move {
                report(0.5);
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                Ok::<JobResult, anyhow::Error>(detect_result(segment))
            })
        })
    }

    async fn wait_for(reg: &JobRegistry, id: &str, state: JobState) -> JobStatus {
        for _ in 0..500 {
            let s = reg.status(id).expect("job exists");
            if s.state == state {
                return s;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        panic!("job {id} never reached {state:?}");
    }

    #[tokio::test]
    async fn lifecycle_queued_running_done() {
        let reg = JobRegistry::new(16);
        let (id, ahead) = reg.submit(JobKind::Detect, ok_runner(1, 50));
        assert!(id.starts_with("detect-") && id.len() == "detect-".len() + 12);
        assert_eq!(ahead, 0);
        assert_eq!(reg.status(&id).unwrap().state, JobState::Queued);

        reg.start();
        let running = wait_for(&reg, &id, JobState::Running).await;
        assert!(running.result.is_none());
        let done = wait_for(&reg, &id, JobState::Done).await;
        assert_eq!(done.progress, 1.0);
        assert_eq!(done.result, Some(detect_result(1)));
        assert_eq!(done.error, None);
        assert!(reg.get(&id).unwrap().finished_at.is_some());
        assert_eq!(reg.counts()["done"], 1);
        reg.stop();
    }

    #[tokio::test]
    async fn queue_position_and_serial_execution() {
        let reg = JobRegistry::new(16);
        let (a, qa) = reg.submit(JobKind::Recon, ok_runner(0, 80));
        let (b, qb) = reg.submit(JobKind::Recon, ok_runner(1, 10));
        assert_eq!((qa, qb), (0, 1));
        reg.start();
        wait_for(&reg, &a, JobState::Running).await;
        // One at a time: b waits while a runs.
        assert_eq!(reg.status(&b).unwrap().state, JobState::Queued);
        wait_for(&reg, &b, JobState::Done).await;
        let ja = reg.get(&a).unwrap();
        let jb = reg.get(&b).unwrap();
        assert!(jb.started_at.unwrap() >= ja.finished_at.unwrap());
        reg.stop();
    }

    #[tokio::test]
    async fn failure_and_panic_do_not_kill_the_worker() {
        let reg = JobRegistry::new(16);
        reg.start();
        let (bad, _) = reg.submit(
            JobKind::Recon,
            Box::new(|_r: Report| -> RunnerFuture {
                Box::pin(async {
                    Err::<JobResult, anyhow::Error>(PipelineUnavailable("nope".into()).into())
                })
            }),
        );
        let (boom, _) = reg.submit(
            JobKind::Recon,
            Box::new(|_r: Report| -> RunnerFuture {
                Box::pin(async {
                    if true {
                        panic!("kaboom");
                    }
                    Ok::<JobResult, anyhow::Error>(detect_result(0))
                })
            }),
        );
        let (good, _) = reg.submit(JobKind::Detect, ok_runner(3, 1));

        let s = wait_for(&reg, &bad, JobState::Failed).await;
        assert_eq!(s.error.as_deref(), Some("PipelineUnavailable: nope"));
        let s = wait_for(&reg, &boom, JobState::Failed).await;
        assert!(s.error.unwrap().contains("kaboom"));
        wait_for(&reg, &good, JobState::Done).await;
        reg.stop();
    }

    #[tokio::test]
    async fn progress_is_clamped_and_rounded() {
        let reg = JobRegistry::new(16);
        let (gate_tx, gate_rx) = tokio::sync::oneshot::channel::<()>();
        let (id, _) = reg.submit(
            JobKind::Detect,
            Box::new(move |report: Report| -> RunnerFuture {
                Box::pin(async move {
                    report(0.123456);
                    let _ = gate_rx.await;
                    Ok::<JobResult, anyhow::Error>(detect_result(0))
                })
            }),
        );
        reg.start();
        wait_for(&reg, &id, JobState::Running).await;
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(reg.status(&id).unwrap().progress, 0.1235);
        let _ = gate_tx.send(());
        wait_for(&reg, &id, JobState::Done).await;
        reg.stop();
    }

    #[tokio::test]
    async fn eviction_drops_oldest_finished_only() {
        let reg = JobRegistry::new(2);
        reg.start();
        let (first, _) = reg.submit(JobKind::Detect, ok_runner(0, 1));
        wait_for(&reg, &first, JobState::Done).await;
        let (second, _) = reg.submit(JobKind::Detect, ok_runner(1, 1));
        wait_for(&reg, &second, JobState::Done).await;
        let (third, _) = reg.submit(JobKind::Detect, ok_runner(2, 1));
        assert!(reg.get(&first).is_none(), "oldest finished job evicted");
        assert!(reg.get(&second).is_some());
        wait_for(&reg, &third, JobState::Done).await;
        reg.stop();
    }

    #[test]
    fn unknown_job_is_none_and_counts_have_all_states() {
        let reg = JobRegistry::default();
        assert!(reg.get("recon-nope").is_none());
        let counts = reg.counts();
        for k in ["queued", "running", "done", "failed"] {
            assert_eq!(counts[k], 0);
        }
    }
}
