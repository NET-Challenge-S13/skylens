// Browser camera -> LiveCapture wiring.
//
// Kept out of core/ because it touches navigator/fetch/URL. core/capture.ts owns
// the encode; this file only opens a camera and decides where the encoded bytes
// go.
//
// PUBLISHING IS THE HONEST WEAK POINT of the browser live path: a page has
// nowhere to put a file. If SKYLENS_DRONE_UPLOAD_URL is configured we POST the
// slice there and use the URL the server returns; otherwise we mint a `blob:`
// URL, which is only resolvable inside THIS page — the core would not be able to
// fetch it. The fallback logs that loudly rather than pretending it shipped.

import { LiveCapture, type CaptureSource, type SliceRequest } from './core/capture.ts';

export interface LiveCameraOptions {
  width: number;
  height: number;
  framerate: number;
  bitrate: number;
  /** Endpoint that accepts the raw slice body and answers with a fetchable URL. */
  uploadUrl: string | null;
  droneId: number;
  onLog: (line: string) => void;
}

async function publishTo(
  uploadUrl: string,
  bytes: Uint8Array,
  req: SliceRequest,
  droneId: number,
): Promise<string> {
  const url = `${uploadUrl}${uploadUrl.includes('?') ? '&' : '?'}drone=${droneId}&slice=${req.index}`;
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'content-type': 'video/H265' },
    body: new Blob([bytes as BlobPart], { type: 'video/H265' }),
  });
  if (!res.ok) throw new Error(`slice upload failed: ${res.status} ${res.statusText}`);
  const body = (await res.json()) as { uri?: string; url?: string };
  const uri = body.uri ?? body.url;
  if (!uri) throw new Error('slice upload returned no uri');
  return uri;
}

/**
 * Open the camera and build a real H.265 encoder around it. Throws when the
 * runtime has no WebCodecs HEVC encoder — the caller falls back to DemoCapture
 * and says so, rather than shipping something that is not H.265.
 */
export async function openLiveCamera(opts: LiveCameraOptions): Promise<CaptureSource> {
  if (!LiveCapture.available()) {
    throw new Error('WebCodecs VideoEncoder / MediaStreamTrackProcessor unavailable in this browser');
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: opts.width, height: opts.height, frameRate: opts.framerate },
    audio: false,
  });
  const track = stream.getVideoTracks()[0];
  if (!track) throw new Error('camera produced no video track');

  let warned = false;
  return new LiveCapture({
    track,
    width: opts.width,
    height: opts.height,
    framerate: opts.framerate,
    bitrate: opts.bitrate,
    publish: async (bytes, req) => {
      if (opts.uploadUrl) return publishTo(opts.uploadUrl, bytes, req, opts.droneId);
      if (!warned) {
        warned = true;
        opts.onLog(
          'no SKYLENS_DRONE_UPLOAD_URL: slice URIs are blob: handles reachable only inside this ' +
            'page. The core cannot fetch them — configure an upload endpoint for a real uplink.',
        );
      }
      return URL.createObjectURL(new Blob([bytes as BlobPart], { type: 'video/H265' }));
    },
  });
}
