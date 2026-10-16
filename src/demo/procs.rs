//! Child processes: start, prefix their output, report exits, stop them.
//!
//! On Windows the launcher puts itself in a Job Object with kill-on-close
//! before it starts anything. Every child and grandchild inherits the job
//! (the core's ffmpeg readers included), and when the launcher goes away for
//! any reason the job handle closes and Windows ends all of them. Children
//! also get their own process group so a console Ctrl-C reaches only the
//! launcher, which then stops them in order instead of racing them.

use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};
use tokio::process::Command;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;

static START: OnceLock<Instant> = OnceLock::new();

pub fn start_clock() {
    let _ = START.get_or_init(Instant::now);
}

/// `  12.3s tag    | line`. Time is seconds since the launcher started.
pub fn format_line(elapsed: Duration, tag: &str, line: &str) -> String {
    format!("{:>7.1}s {:<6} | {}", elapsed.as_secs_f64(), tag, line)
}

pub fn log(tag: &str, line: &str) {
    let elapsed = START.get_or_init(Instant::now).elapsed();
    println!("{}", format_line(elapsed, tag, line));
}

/// Split child output into lines on `\n` or `\r` (ffmpeg redraws with `\r`),
/// dropping blank ones.
pub fn split_lines(buf: &[u8]) -> Vec<String> {
    buf.split(|&b| b == b'\n' || b == b'\r')
        .map(|l| String::from_utf8_lossy(l).trim_end().to_string())
        .filter(|l| !l.trim().is_empty())
        .collect()
}

/// Something the supervisor has to react to.
#[derive(Debug)]
pub enum Event {
    /// A child process ended on its own.
    Exited { name: String, critical: bool, code: Option<i32> },
    /// An in-process part (board server, uplink simulator) failed.
    Failed { name: String, error: String },
    /// The operator pressed Ctrl-C.
    Interrupt,
}

pub type Events = mpsc::UnboundedSender<Event>;

pub struct Spec {
    pub name: String,
    pub tag: String,
    pub program: PathBuf,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
    pub cwd: PathBuf,
    /// Exiting stops the whole demo (everything except the native windows).
    pub critical: bool,
    /// Wait this long before starting the process (zero: start right away).
    pub delay: Duration,
}

pub struct Running {
    pub name: String,
    kill: Option<oneshot::Sender<()>>,
    task: JoinHandle<()>,
}

impl Running {
    /// Ask the child to stop and wait (bounded) until it has.
    pub async fn stop(mut self) {
        if let Some(k) = self.kill.take() {
            let _ = k.send(());
        }
        let _ = tokio::time::timeout(Duration::from_secs(5), &mut self.task).await;
    }
}

async fn pipe<R: AsyncRead + Unpin>(tag: String, stream: R) {
    let mut reader = BufReader::new(stream);
    let mut buf = Vec::new();
    loop {
        buf.clear();
        match reader.read_until(b'\n', &mut buf).await {
            Ok(0) | Err(_) => break,
            Ok(_) => {
                for line in split_lines(&buf) {
                    log(&tag, &line);
                }
            }
        }
    }
}

/// Start a child. Its stdout and stderr are logged under its tag; when it
/// exits on its own an [`Event::Exited`] goes to `events`.
pub fn spawn(spec: Spec, events: Events) -> anyhow::Result<Running> {
    if !spec.delay.is_zero() {
        return Ok(spawn_later(spec, events));
    }
    let child = start(&spec)?;
    Ok(watch(spec.name, spec.critical, child, events))
}

/// A child that starts after `spec.delay`; a start failure is reported as [`Event::Failed`].
fn spawn_later(spec: Spec, events: Events) -> Running {
    let (kill_tx, mut kill_rx) = oneshot::channel::<()>();
    let name = spec.name.clone();
    let task = tokio::spawn(async move {
        tokio::select! {
            _ = tokio::time::sleep(spec.delay) => {}
            _ = &mut kill_rx => return,
        }
        match start(&spec) {
            Ok(mut child) => {
                let exited = tokio::select! {
                    status = child.wait() => Some(status),
                    _ = &mut kill_rx => None,
                };
                match exited {
                    Some(status) => {
                        let code = status.ok().and_then(|s| s.code());
                        let _ = events.send(Event::Exited { name: spec.name, critical: spec.critical, code });
                    }
                    None => terminate(&mut child).await,
                }
            }
            Err(e) => {
                let _ = events.send(Event::Failed { name: spec.name, error: format!("{e:#}") });
            }
        }
    });
    Running { name, kill: Some(kill_tx), task }
}

/// Launch the process and attach its output to the log.
fn start(spec: &Spec) -> anyhow::Result<tokio::process::Child> {
    let mut cmd = Command::new(&spec.program);
    cmd.args(&spec.args)
        .current_dir(&spec.cwd)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    for (k, v) in &spec.env {
        cmd.env(k, v);
    }
    #[cfg(windows)]
    {
        // CREATE_NEW_PROCESS_GROUP: console Ctrl-C goes to the launcher only.
        cmd.creation_flags(0x0000_0200);
    }
    #[cfg(unix)]
    {
        cmd.process_group(0);
    }
    let mut child = cmd
        .spawn()
        .map_err(|e| anyhow::anyhow!("{}: could not start {}: {e}", spec.name, spec.program.display()))?;
    #[cfg(windows)]
    job::adopt(&child);
    if let Some(out) = child.stdout.take() {
        tokio::spawn(pipe(spec.tag.clone(), out));
    }
    if let Some(err) = child.stderr.take() {
        tokio::spawn(pipe(spec.tag.clone(), err));
    }
    Ok(child)
}

fn watch(name: String, critical: bool, mut child: tokio::process::Child, events: Events) -> Running {
    let (kill_tx, kill_rx) = oneshot::channel::<()>();
    let run_name = name.clone();
    let task = tokio::spawn(async move {
        let exited = tokio::select! {
            status = child.wait() => Some(status),
            _ = kill_rx => None,
        };
        match exited {
            Some(status) => {
                let code = status.ok().and_then(|s| s.code());
                let _ = events.send(Event::Exited { name: run_name, critical, code });
            }
            None => terminate(&mut child).await,
        }
    });
    Running { name, kill: Some(kill_tx), task }
}

async fn terminate(child: &mut tokio::process::Child) {
    #[cfg(unix)]
    {
        if let Some(pid) = child.id() {
            // The whole process group, so helpers the child started go too.
            let _ = std::process::Command::new("kill").arg("-TERM").arg(format!("-{pid}")).status();
            if tokio::time::timeout(Duration::from_secs(3), child.wait()).await.is_ok() {
                return;
            }
        }
    }
    let _ = child.start_kill();
    let _ = tokio::time::timeout(Duration::from_secs(3), child.wait()).await;
}

/// Stop children in reverse start order.
pub async fn stop_all(children: Vec<Running>) {
    for child in children.into_iter().rev() {
        log("launch", &format!("{} 종료", child.name));
        child.stop().await;
    }
}

#[cfg(windows)]
pub mod job {
    //! A kill-on-close Job Object holding the launcher and everything it starts.

    use std::sync::OnceLock;
    use windows_sys::Win32::Foundation::HANDLE;
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation, SetInformationJobObject,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    /// The job handle, kept open for the launcher's lifetime (closing it, at
    /// process exit, is what kills the children). `self_in_job` says whether
    /// the launcher itself is in the job, in which case children inherit it.
    struct Job {
        handle: usize,
        self_in_job: bool,
    }

    static JOB: OnceLock<Option<Job>> = OnceLock::new();

    /// Create the job and put the launcher in it. Returns a line for the log.
    pub fn init() -> String {
        let job = JOB.get_or_init(|| unsafe {
            let handle: HANDLE = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if handle.is_null() {
                return None;
            }
            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            let ok = SetInformationJobObject(
                handle,
                JobObjectExtendedLimitInformation,
                (&info as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(),
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            );
            if ok == 0 {
                return None;
            }
            let self_in_job = AssignProcessToJobObject(handle, GetCurrentProcess()) != 0;
            Some(Job { handle: handle as usize, self_in_job })
        });
        match job {
            Some(j) if j.self_in_job => "Job Object 준비: 런처가 끝나면 모든 하위 프로세스가 함께 종료됩니다".into(),
            Some(_) => "Job Object 준비: 자식마다 따로 등록합니다 (런처를 잡에 넣지 못함)".into(),
            None => "Job Object를 만들지 못했습니다. 비정상 종료 시 ffmpeg 등이 남을 수 있습니다".into(),
        }
    }

    /// Put a freshly spawned child in the job when it did not inherit it.
    pub fn adopt(child: &tokio::process::Child) {
        let Some(Some(job)) = JOB.get() else { return };
        if job.self_in_job {
            return;
        }
        if let Some(raw) = child.raw_handle() {
            unsafe {
                AssignProcessToJobObject(job.handle as HANDLE, raw as HANDLE);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_are_prefixed_and_aligned() {
        let l = format_line(Duration::from_millis(12_345), "core", "listening on :8080");
        assert_eq!(l, "   12.3s core   | listening on :8080");
        let l = format_line(Duration::from_secs(0), "pub-L", "x");
        assert_eq!(l, "    0.0s pub-L  | x");
    }

    #[test]
    fn output_splits_on_cr_and_lf_and_drops_blanks() {
        assert_eq!(split_lines(b"a\r\nb\rc\n\n  \n"), vec!["a", "b", "c"]);
        assert_eq!(split_lines(b"tail without newline"), vec!["tail without newline"]);
        assert_eq!(split_lines(b"bad \xff byte\n"), vec!["bad \u{fffd} byte"]);
    }

    #[tokio::test]
    async fn exits_are_reported_and_stop_kills() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let (quick, slow) = if cfg!(windows) {
            (
                ("cmd".to_string(), vec!["/C".to_string(), "echo hi".to_string()]),
                ("ping".to_string(), vec!["-n".to_string(), "30".to_string(), "127.0.0.1".to_string()]),
            )
        } else {
            (("sh".to_string(), vec!["-c".to_string(), "echo hi".to_string()]), ("sleep".to_string(), vec!["30".to_string()]))
        };
        let spec = |name: &str, (program, args): (String, Vec<String>)| Spec {
            name: name.to_string(),
            tag: name.to_string(),
            program: PathBuf::from(program),
            args,
            env: vec![],
            cwd: std::env::temp_dir(),
            critical: true,
            delay: Duration::ZERO,
        };
        let _quick = spawn(spec("quick", quick), tx.clone()).unwrap();
        match tokio::time::timeout(Duration::from_secs(10), rx.recv()).await.unwrap().unwrap() {
            Event::Exited { name, critical, code } => {
                assert_eq!(name, "quick");
                assert!(critical);
                assert_eq!(code, Some(0));
            }
            other => panic!("unexpected {other:?}"),
        }
        let slow = spawn(spec("slow", slow), tx).unwrap();
        let started = Instant::now();
        slow.stop().await;
        assert!(started.elapsed() < Duration::from_secs(10));
        // a stopped child is not reported as having exited on its own
        assert!(rx.try_recv().is_err());
    }
}
