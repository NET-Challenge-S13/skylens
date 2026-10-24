// Fetch + decode point segments off the main thread.
//
// A refined segment is ~13 MB and ~500k points; decoding it on the render
// thread would stall the board for a frame or two per arrival. Here the bytes
// are fetched, decoded by the shared decoder (shared/pointPly.ts) and the
// typed arrays handed back as transferables (no copy).

import { parsePointPly, shufflePoints } from '../../shared/pointPly.ts';
import type { PointCloud } from '../../shared/pointPly.ts';

export type PointWorkerRequest =
  | { type: 'load'; id: number; url: string }
  | { type: 'cancel'; id: number };

export type PointWorkerReply =
  | {
      type: 'done';
      id: number;
      cloud: PointCloud;
      bytes: number;
      fetchMs: number;
      parseMs: number;
    }
  | { type: 'error'; id: number; message: string; aborted: boolean };

const scope = self as unknown as {
  onmessage: ((ev: MessageEvent<PointWorkerRequest>) => void) | null;
  postMessage(msg: PointWorkerReply, transfer?: Transferable[]): void;
};

const pending = new Map<number, AbortController>();

async function load(id: number, url: string): Promise<void> {
  const ctrl = new AbortController();
  pending.set(id, ctrl);
  try {
    const t0 = performance.now();
    const res = await fetch(url, { signal: ctrl.signal });
    if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
    const buf = await res.arrayBuffer();
    const t1 = performance.now();
    if (ctrl.signal.aborted) throw new DOMException('aborted', 'AbortError');
    const cloud = parsePointPly(buf);
    // Random order: the layer thins over-budget segments by drawing a prefix.
    shufflePoints(cloud, id);
    const t2 = performance.now();
    const transfer: Transferable[] = [cloud.positions.buffer as ArrayBuffer];
    if (cloud.colors) transfer.push(cloud.colors.buffer as ArrayBuffer);
    if (cloud.normals) transfer.push(cloud.normals.buffer as ArrayBuffer);
    scope.postMessage(
      { type: 'done', id, cloud, bytes: buf.byteLength, fetchMs: t1 - t0, parseMs: t2 - t1 },
      transfer,
    );
  } catch (err) {
    scope.postMessage({
      type: 'error',
      id,
      message: err instanceof Error ? err.message : String(err),
      aborted: ctrl.signal.aborted,
    });
  } finally {
    pending.delete(id);
  }
}

scope.onmessage = (ev) => {
  const msg = ev.data;
  if (msg.type === 'load') void load(msg.id, msg.url);
  else pending.get(msg.id)?.abort();
};
