// Free the demo's ports.
//
//   npm run demo:clean
//
// A demo that was killed hard — the terminal closed, the launcher crashed,
// Ctrl+C pressed twice — can leave a component behind. Those survivors keep
// answering their health URL, and the next `npm run demo` would take them for
// its own components and run the demo against yesterday's code. The launcher
// now refuses to start when a port is busy (see preflight in index.ts) and
// points here.
//
// This kills by PORT rather than by process name on purpose: `node` and
// `python` on this machine are not only ours.

import { execFileSync } from 'node:child_process';
import process from 'node:process';

/** Every port a demo component listens on, in the order they start. */
const PORTS = [8100, 8080, 8082, 8081, 8090, 5173];

/** PIDs listening on `port`, asked of the platform's own tooling. */
function listeners(port: number): number[] {
  try {
    if (process.platform === 'win32') {
      const out = execFileSync('netstat', ['-ano', '-p', 'TCP'], { encoding: 'utf8' });
      const pids = new Set<number>();
      for (const line of out.split(/\r?\n/)) {
        // "  TCP    127.0.0.1:8100   0.0.0.0:0   LISTENING   1234"
        const m = /^\s*TCP\s+\S+:(\d+)\s+\S+\s+LISTENING\s+(\d+)\s*$/.exec(line);
        if (m && Number(m[1]) === port) pids.add(Number(m[2]));
      }
      return [...pids];
    }
    const out = execFileSync('lsof', ['-ti', `tcp:${port}`, '-sTCP:LISTEN'], { encoding: 'utf8' });
    return out
      .split(/\s+/)
      .map(Number)
      .filter((n) => Number.isFinite(n) && n > 0);
  } catch {
    // Nothing listening: lsof exits non-zero, netstat simply matches nothing.
    return [];
  }
}

/** Kill a pid and its children — a component may be a launcher of launchers. */
function killTree(pid: number): boolean {
  try {
    if (process.platform === 'win32') {
      execFileSync('taskkill', ['/pid', String(pid), '/T', '/F'], { stdio: 'ignore' });
    } else {
      process.kill(pid, 'SIGKILL');
    }
    return true;
  } catch {
    return false;
  }
}

let killed = 0;
for (const port of PORTS) {
  for (const pid of listeners(port)) {
    // The launcher itself owns no port, so anything found here is a component.
    if (pid === process.pid) continue;
    if (killTree(pid)) {
      console.log(`  포트 ${port} — pid ${pid} 종료`);
      killed += 1;
    } else if (listeners(port).includes(pid)) {
      console.log(`  포트 ${port} — pid ${pid} 종료 실패 (권한을 확인하십시오)`);
    }
    // taskkill also fails when the process is ALREADY gone — a component that
    // died with its parent moments ago. Saying "종료 실패" for that sends the
    // operator hunting a process that no longer exists, so it only counts as a
    // failure if the port is still held.
  }
}

// Report what is still held rather than claiming success: a port kept by
// another user's process is exactly the case the operator has to see.
const stillBusy = PORTS.filter((port) => listeners(port).length > 0);
if (stillBusy.length > 0) {
  console.log(`  아직 사용 중인 포트: ${stillBusy.join(', ')}`);
  process.exit(1);
}
console.log(killed === 0 ? '  정리할 프로세스가 없습니다.' : '  데모 포트가 모두 비었습니다.');
